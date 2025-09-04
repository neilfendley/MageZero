# dataset.py
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from typing import Set, List, Tuple
from pathlib import Path  # Use the modern pathlib for handling file paths
import sys
import matplotlib
import matplotlib.pyplot as plt  # uses default backend; switch to 'Agg' if running headless
import os

from scipy.sparse import coo_matrix

MAX_FEATURES = 100000

class LabeledStateDataset(Dataset):
    """
    CORRECTED: Handles lazy loading correctly for multiprocessing.
    It opens the file handle *inside* each worker process to avoid pickling errors.
    """

    def __init__(self, path):
        self.path = path
        self.offsets = []
        self.ignore_list = set()
        self.n = 0
        self.A = 0
        self.S = 0

        # --- PRE-CALCULATION STEP ---
        # Open the file *temporarily* in the main process just to build the index of byte offsets.
        # The 'with' statement ensures it's closed immediately after.
        with open(self.path, "rb") as f:
            header = f.read(12)
            if len(header) < 12:
                raise IOError(f"File too small for header: {path}")
            self.n, self.S, self.A = struct.unpack(">iii", header)

            current_offset = 12  # Start after the header
            for _ in range(self.n):
                self.offsets.append(current_offset)
                (num_indices,) = struct.unpack(">i", f.read(4))
                record_size = 4 + (num_indices * 4) + (self.A * 8) + 8
                current_offset += record_size
                f.seek(current_offset)

        # IMPORTANT: self.file is NOT created here.

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # The first time a worker process calls __getitem__, it won't have the 'file' attribute.
        # We create it here, so each worker gets its own file handle.
        if not hasattr(self, 'file'):
            self.file = open(self.path, "rb")

        # Go to the pre-calculated position for this sample
        offset = self.offsets[idx]
        self.file.seek(offset)

        # Read ONLY the data for this one sample
        (num_indices,) = struct.unpack(">i", self.file.read(4))
        indices = struct.unpack(f">{num_indices}i", self.file.read(num_indices * 4))
        actions = struct.unpack(f">{self.A}d", self.file.read(self.A * 8))
        (label,) = struct.unpack(">d", self.file.read(8))

        # --- MODIFIED: Filtering now happens here ---
        # It's more efficient to filter here than in collate_fn because it happens
        # inside the worker process before data is sent to the main process.
        if self.ignore_list:
            indices = [idx for idx in indices if idx not in self.ignore_list]

        return (
            torch.LongTensor(indices),
            torch.FloatTensor(actions),
            torch.FloatTensor([label])
        )

    # --- NEW: make pickling safe for DataLoader(num_workers>0), esp. on Windows
    def __getstate__(self):
        state = self.__dict__.copy()
        f = state.pop('file', None)
        if f:
            try:
                f.close()
            except:
                pass
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # --- NEW: iterate indices *only* (no actions/label, no self.file side-effects)
    def iter_indices_only(self):
        with open(self.path, "rb") as f:
            for off in self.offsets:
                f.seek(off)
                (k,) = struct.unpack(">i", f.read(4))
                if k == 0:
                    yield np.empty(0, dtype=np.int32)
                    # skip actions+label
                    f.seek(self.A * 8 + 8, 1)
                    continue
                raw = f.read(k * 4)
                # big-endian int32 -> native int32 (no extra copy)
                idx = np.frombuffer(raw, dtype=">i4").astype(np.int32, copy=False)
                # skip actions + label
                f.seek(self.A * 8 + 8, 1)
                yield idx
    def __del__(self):
        # When a worker process is destroyed, this will close its file handle
        if hasattr(self, 'file'):
            self.file.close()


def collate_batch(batch):
    """
    This collate function is essential for batching variable-length data.
    It remains unchanged.
    """
    indices_list, policy_list, value_list = [], [], []
    for (_indices, _policy, _value) in batch:
        indices_list.append(_indices)
        policy_list.append(_policy)
        value_list.append(_value)

    offsets = [0] + [len(indices) for indices in indices_list]
    offsets = torch.cumsum(torch.LongTensor(offsets), dim=0)[:-1]

    indices = torch.cat(indices_list)
    policies = torch.stack(policy_list)
    values = torch.stack(value_list)

    return indices, offsets, policies, values


def load_dataset_from_directory(directory_path: str) -> ConcatDataset:
    """
    NEW: Finds all .bin files, creates a LabeledStateDataset for each,
    and combines them.

    Returns:
        A ConcatDataset object containing all samples from all .bin files.
    """
    print(f"Searching for dataset files in: {directory_path}")
    data_dir = Path(directory_path)

    bin_files = sorted(list(data_dir.glob('*.bin')))

    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in '{directory_path}'.")

    print(f"Found {len(bin_files)} dataset files.")

    datasets = [LabeledStateDataset(str(path)) for path in bin_files]

    print("Combining into a single dataset.")
    combined_dataset = ConcatDataset(datasets)

    return combined_dataset


from typing import Set
import numpy as np
from scipy.sparse import coo_matrix

def create_redundancy_ignore_list(dataset, num_global_features: int) -> Set[int]:
    """
    Find perfectly co-occurring (identical) binary features across the dataset
    and return all but one index from each duplicate group. Also drop unseen features.
    """
    print("Step 1: Building sparse matrix...")
    rows_list, cols_list = [], []

    # Support plain dataset or ConcatDataset
    dsets = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]

    sample_id = 0
    for ds in dsets:
        for idx in ds.iter_indices_only():  # <-- no __getitem__ call
            if idx.size:
                rows_list.append(np.full(idx.shape, sample_id, dtype=np.int32))
                cols_list.append(idx.astype(np.int32, copy=False))
            sample_id += 1

    if rows_list:
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
    else:
        rows = np.empty(0, dtype=np.int32)
        cols = np.empty(0, dtype=np.int32)

    data = np.ones_like(cols, dtype=np.uint8)
    num_samples = sample_id

    X_csc = coo_matrix(
        (data, (rows, cols)),
        shape=(num_samples, num_global_features),
        dtype=np.uint8
    ).tocsc()

    print("Step 2: Grouping identical columns...")
    groups = {}
    indptr = X_csc.indptr
    indices = X_csc.indices

    for j in range(num_global_features):
        start, end = indptr[j], indptr[j + 1]
        key = tuple(indices[start:end])  # () means unseen/empty
        groups.setdefault(key, []).append(j)

    print("Step 3: Generating ignore list...")
    ignore = set()

    unseen = groups.get((), [])
    ignore.update(unseen)

    for key, js in groups.items():
        if key == () or len(js) == 1:
            continue
        js.sort()
        ignore.update(js[1:])

    num_duplicate_sets = sum(1 for k, v in groups.items() if k != () and len(v) > 1)
    kept = num_global_features - len(ignore)
    print(f"Analysis complete. Found {num_duplicate_sets} sets of redundant features.")
    print(f"A total of {len(ignore)} feature indices will be ignored.")
    print(f"{kept} feature indices were kept.")

    return ignore

def remove_one_hot_labels(dataset, eps: float = 1e-8, verbose: bool = True):
    """
    Return a torch.utils.data.Subset that excludes samples whose policy label is one-hot.

    Args:
        dataset: LabeledStateDataset or ConcatDataset of them.
        eps: small threshold for treating values as zero (robust to float noise).
        verbose: print summary counts.

    One-hot detection rule:
      - count of entries > eps is exactly 1 (we don't assume exact 0/1).
    """
    keep_indices = []
    n = len(dataset)

    # cheap sequential scan; using dataset.__getitem__ so we can inspect the policy tensor
    # no gradients involved
    for i in range(n):
        _, policy, _ = dataset[i]  # (indices, policy, value)
        # policy is a FloatTensor of shape [A]
        # treat near-zeros as zero via eps
        nonzero = (policy > eps).sum().item()
        is_one_hot = (nonzero == 1)
        if not is_one_hot:
            keep_indices.append(i)

    if verbose:
        removed = n - len(keep_indices)
        print(f"[remove_one_hot_labels] scanned {n} samples "
              f"→ kept {len(keep_indices)} (removed {removed} one-hot policies).")

    return Subset(dataset, keep_indices)


# --- Main execution block ---
if __name__ == "__main__":
    # Define the directory where you save your game data files.
    data_directory = "data/UWTempo/ver7/training"

    try:
        # Load all datasets from the specified folder
        full_dataset = load_dataset_from_directory(data_directory)

        wins = 0
        num_samples_to_process = 0

        # The DataLoader MUST be given the custom collate function
        dl = DataLoader(full_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

        for i, (indices_batch, offsets_batch, policies_batch, values_batch) in enumerate(dl):
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

            print(f"State: {sb}, Action: {av}, Result: {lbl}")

            if lbl > 0:
                wins += 1
            if i >= 100:  # Stop after 10,000 samples like the original
                break

        print(f"\nDataset size: {len(full_dataset)}\n")

        # Calculate the number of unique feature indices across the entire dataset
        if len(full_dataset) > 0:
            all_feature_indices = set()
            # We iterate through the dataset directly to access each sample's indices
            for sample_idx in range(len(full_dataset)):
                indices, _, _ = full_dataset[sample_idx]  # Unpack the sample tuple
                all_feature_indices.update(indices.tolist())
            print(f"Total unique feature indices in dataset: {len(all_feature_indices)}")

        if num_samples_to_process > 0:
            print(f"Winrate (over {num_samples_to_process} samples): {wins / num_samples_to_process:.3f}")
        else:
            print("No samples were processed.")


    except (FileNotFoundError, ValueError, IOError, IndexError) as e:
        print(f"An error occurred: {e}", file=sys.stderr)

