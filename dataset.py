# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple
from pathlib import Path  # Use the modern pathlib for handling file paths
import sys
from typing import Set
from scipy.sparse import coo_matrix
import h5py


#from train import GLOBAL_MAX
GLOBAL_MAX = 2000000

H5_RECIDX_DTYPE = np.dtype([("file", "i4"), ("row", "i8")])

class H5Indexed(Dataset):
    """
    Lazy, indexed dataset over one or more HDF5 files produced by XMage JHDF5 writer.

    Expected datasets per file:
      /indices     (int32, shape [NNZ])
      /offsets     (int64, shape [N+1], offsets[0]==0, offsets[-1]==len(indices))
      /action      (float32 or float64, shape [N, 128])
      /isPlayer    (uint8, shape [N])
      /actionType  (int32, shape [N])
      /stateScore  (float32 or float64, shape [N])
      /resultLabel (float32 or float64, shape [N])
    """

    def __init__(self, dir_path: str):
        p = Path(dir_path)
        # pick up .h5 / .hdf5
        h5_paths = sorted(list(p.glob("*.h5")) + list(p.glob("*.hdf5")))
        self.files: List[str] = [str(pp) for pp in h5_paths]

        self.fhs: List[h5py.File] = []
        self.ignore_list = set()

        if not self.files:
            print(f"No .h5/.hdf5 files in {dir_path}")
            return

        # Build a flat (file,row) index
        rows: List[np.ndarray] = []
        total = 0
        for fid, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                n, _= f["/action"].shape
                if n > 0:
                    a = np.empty(n, dtype=H5_RECIDX_DTYPE)
                    a["file"].fill(fid)
                    a["row"] = np.arange(n, dtype=np.int64)
                    rows.append(a)
                    total += n

        self.idx = np.concatenate(rows) if rows else np.empty(0, dtype=H5_RECIDX_DTYPE)
        print(f"Found {len(self.idx)} rows across {len(self.files)} file(s)")

    def __len__(self) -> int:
        if not self.files:
            return 0
        return int(self.idx.shape[0])

    def __del__(self):
        if self.fhs:
            for fh in self.fhs:
                try:
                    fh.close()
                except Exception:
                    pass

    def ensure_open(self):
        if not self.fhs:
            # Big raw data chunk cache (rdcc_*), no SWMR for faster reads
            self.fhs = [
                h5py.File(
                    p, "r",
                    swmr=False,  # faster if you don't need live-writes
                    rdcc_nbytes=128 * 1024 * 1024*10,  # 128MB cache
                    rdcc_nslots=1_000_003,  # large hash table to reduce collisions
                    rdcc_w0=0.75
                )
                for p in self.files
            ]

    def _fetch_row(self, fid: int, row: int):
        f = self.fhs[fid]

        off = f["/offsets"]
        a = int(off[row]);
        b = int(off[row + 1])
        sv_idx = f["/indices"][a:b]  # 1 read

        row_full = f["/row"][row, :]  # 1 read (A+4)
        A = row_full.shape[0] - 4
        action_row = row_full[:A]
        resultLbl = float(row_full[A + 0])
        stateScore = float(row_full[A + 1])
        isP = int(row_full[A + 2] > 0.5)
        actionType = int(row_full[A + 3])

        return sv_idx, action_row, resultLbl, isP, actionType, stateScore

    def __getitem__(self, k: int):
        self.ensure_open()
        ent = self.idx[k]
        fid = int(ent["file"])
        row = int(ent["row"])

        sv_idx, action_row, resultLbl, isP, aTy, stateScore = self._fetch_row(fid, row)

        # apply ignore_list (same as your H5 path)
        if self.ignore_list:
            sv_idx = [j for j in sv_idx.tolist() if j not in self.ignore_list]
        else:
            sv_idx = sv_idx.tolist()

        # map to tensors with same shapes/dtypes you used before
        return (
            torch.tensor(sv_idx, dtype=torch.long),                       # ragged indices
            torch.tensor(action_row, dtype=torch.float32),                # policy [A]
            torch.tensor([resultLbl], dtype=torch.float32),               # value [1]
            torch.tensor([1.0 if isP else 0.0], dtype=torch.float32),     # is_player [1]
            torch.tensor([aTy], dtype=torch.long),                        # action_type [1]
        )
def collate_batch(batch):
    """
    This collate function is essential for batching variable-length data.
    """
    indices_list, policy_list, value_list, is_player_list, action_type_list = [], [], [], [], []
    for (idxs, policy, value, player, action_type) in batch:
        indices_list.append(idxs)
        policy_list.append(policy)
        value_list.append(value)
        is_player_list.append(player)
        action_type_list.append(action_type)


    offsets = [0] + [len(idxs) for idxs in indices_list]
    offsets = torch.cumsum(torch.LongTensor(offsets), dim=0)[:-1]

    idxs = torch.cat(indices_list)
    policies = torch.stack(policy_list)
    values = torch.stack(value_list)
    is_player_list = torch.stack(is_player_list)
    action_types = torch.stack(action_type_list)

    return idxs, offsets, policies, values, is_player_list, action_types

def create_redundancy_ignore_list(ds) -> Set[int]:
    """
    Find perfectly co-occurring (identical) binary features across the dataset
    and return all but one index from each duplicate group. Unseen features aren't touched
    """
    #make sparse matrix
    rows_list, cols_list = [], []

    sample_id = 0

    for (idxs, _, _, _, _) in ds:
        idxs_np = idxs.cpu().numpy().astype(np.int32, copy=False)
        rows_list.append(np.full(idxs_np.shape, sample_id, dtype=np.int32))
        cols_list.append(idxs_np)
        sample_id += 1

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)


    data = np.ones_like(cols, dtype=np.uint8)
    num_samples = sample_id

    x_csc = coo_matrix(
        (data, (rows, cols)),
        shape=(num_samples, GLOBAL_MAX),
        dtype=np.uint8
    ).tocsc()

    #group identical columns
    groups = {}
    indptr = x_csc.indptr
    idxs = x_csc.indices

    for j in range(GLOBAL_MAX):
        start, end = indptr[j], indptr[j + 1]
        key = tuple(idxs[start:end])  # () means unseen/empty
        groups.setdefault(key, []).append(j)

    #create ignore list
    ignore = set()

    unseen = groups.get((), [])
    #ignore.update(unseen)
    for key, js in groups.items():
        if key == () or len(js) == 1:
            continue
        js.sort()
        ignore.update(js[1:])

    num_duplicate_sets = sum(1 for k, v in groups.items() if k != () and len(v) > 1)
    kept = GLOBAL_MAX - len(ignore)
    print(f"Analysis complete. Found {num_duplicate_sets} sets of redundant features.")
    print(f"A total of {len(ignore)} feature indices will be ignored.")
    print(f"{kept} feature indices were kept.")
    print(f"{kept - len(unseen)} seen feature indices were kept.")

    return ignore

def remove_one_hot_labels(dataset):
    """
    Return a torch.utils.data.Subset that excludes samples whose policy label is one-hot.
    """
    keep_indices = []
    n = len(dataset)


    for i in range(n):
        _, policy, _, _, _ = dataset[i]  # (indices, policy, value)
        # policy is a FloatTensor of shape [A]
        # treat near-zeros as zero via eps
        nonzero = (policy > 0).sum().item()
        is_one_hot = (nonzero == 1)
        if not is_one_hot:
            keep_indices.append(i)


    removed = n - len(keep_indices)
    print(f"[remove_one_hot_labels] scanned {n} samples "
          f" kept {len(keep_indices)} (removed {removed} one-hot policies).")

    return Subset(dataset, keep_indices)


if __name__ == "__main__":
    # Define the directory where you save your game data files.
    data_directory = "data/MTGA_MonoU/ver7/training"

    try:
        # Load dataset from the specified folder
        full_dataset = H5Indexed(data_directory)

        winning = 0
        num_samples_to_process = 0

        dl = DataLoader(full_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

        for i, (indices_batch, offsets_batch, policies_batch, values_batch, players_batch, action_types_batch) in enumerate(dl):
            num_samples_to_process += 1

            current_sample_indices = indices_batch
            indices_to_print = current_sample_indices.tolist()

            if len(indices_to_print) > 100:
                sb = " ".join(map(str, indices_to_print[:100])) + " ..."
            else:
                sb = " ".join(map(str, indices_to_print))
            if not indices_to_print:
                sb = "[No active features]"

            action = policies_batch[0]
            av = action.tolist()

            label_tensor = values_batch[0]
            lbl = label_tensor.item()

            player = players_batch[0]

            action_type = action_types_batch[0]

            print(f"State: {sb}, Action: {av}, Result: {lbl}, isPlayer: {player}, ActionType: {action_type}")

            if lbl > 0:
                winning += 1
            if i >= 1000:
                break

        print(f"\nDataset size: {len(full_dataset)}\n")

        # Calculate the number of unique feature indices across the entire dataset
        if len(full_dataset) > 0:
            all_feature_indices = set()
            # We iterate through the dataset directly to access each sample's indices
            for sample_idx in range(len(full_dataset)):
                indices, _, _, _, _ = full_dataset[sample_idx]  # Unpack the sample tuple
                all_feature_indices.update(indices.tolist())
            print(f"Total unique feature indices in dataset: {len(all_feature_indices)}")

        if num_samples_to_process > 0:
            print(f"Winrate (over {num_samples_to_process} samples): {winning / num_samples_to_process:.3f}")
        else:
            print("No samples were processed.")


    except (FileNotFoundError, ValueError, IOError, IndexError) as e:
        print(f"An error occurred: {e}", file=sys.stderr)

