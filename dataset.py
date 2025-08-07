# dataset.py
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Set, List, Tuple
from pathlib import Path  # Use the modern pathlib for handling file paths
import sys


class LabeledStateDataset(Dataset):
    """
    Loads labeled state data from a single binary file.
    This class remains unchanged and still operates on a single file path.
    """

    def __init__(self, path):
        self.samples = []
        self.S = 0  # Global vocabulary size (total number of possible features)
        self.A = 0  # Action dimension

        with open(path, "rb") as f:
            # Read the 12-byte header
            header = f.read(12)
            if len(header) < 12:
                raise IOError(f"File too small for header: {path}")

            # Unpack the 3 integers
            n, self.S, self.A = struct.unpack(">iii", header)

            for i in range(n):
                # 1) Read the number of active indices
                num_indices_bytes = f.read(4)
                if not num_indices_bytes: break  # Handles empty files or trailing data
                (num_indices,) = struct.unpack(">i", num_indices_bytes)

                # 2) Read the indices
                indices_bytes = f.read(num_indices * 4)
                if len(indices_bytes) < num_indices * 4:
                    raise IOError(f"Unexpected EOF when reading indices at record {i} in {path}")
                indices = struct.unpack(f">{num_indices}i", indices_bytes)

                # 3) Read action-distribution
                action_bytes = f.read(self.A * 8)
                if len(action_bytes) < self.A * 8:
                    raise IOError(f"Unexpected EOF when reading action-vector at record {i} in {path}")
                actions = struct.unpack(f">{self.A}d", action_bytes)

                # 4) Read result label
                label_bytes = f.read(8)
                if len(label_bytes) < 8:
                    raise IOError(f"Unexpected EOF when reading label at record {i} in {path}")
                (label,) = struct.unpack(">d", label_bytes)

                # Store the processed sample
                self.samples.append({
                    "indices": torch.LongTensor(indices),
                    "policy": torch.FloatTensor(actions),
                    "value": torch.FloatTensor([label])
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return the pre-processed sample dictionary
        sample = self.samples[idx]
        return sample["indices"], sample["policy"], sample["value"]


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


# --- Main execution block ---
if __name__ == "__main__":
    # Define the directory where you save your game data files.
    data_directory = "data/UWTempo/ver1/training"

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
            if i >= 200000:  # Stop after 10,000 samples like the original
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

