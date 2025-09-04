"""
mz_dataset_stats_simple.py

Simple, hardcoded script to inspect MageZero .bin datasets using dataset.py.
- No command line args.
- Edit the CONFIG section and run.

It prints:
- total samples
- unique active feature count
- redundancy stats (unseen, duplicate groups, non-redundant present, kept global)
- shows/saves: value label histogram ([-1,1]), average policy bar chart (first K actions)
- preview of the first N samples
"""

import os
import sys
from typing import Set, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

# =========================
# CONFIG (edit these)
# =========================
DATA_DIR = "data/UWTempo/ver15/training"  # <-- your folder of .bin files
OUT_DIR = "stats_out"                     # where to save figures
SHOW_PLOTS = True                         # False if running headless
SAVE_PLOTS = True                         # save images to OUT_DIR
POLICY_TOP_K = 50                         # first K actions in bar chart
HIST_BINS = 21                            # histogram bins for values
APPLY_IGNORE = False                      # apply ignore set (unseen + redundant) before stats
NUM_GLOBAL_FEATURES_OVERRIDE = None       # e.g. 100000; None uses dataset.MAX_FEATURES
PREVIEW_N = 50                            # how many samples to preview

# If headless, choose a non-GUI backend before importing pyplot
if not SHOW_PLOTS:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import your dataset module (must be next to this script or on PYTHONPATH)
from dataset import (
    LabeledStateDataset,
    load_dataset_from_directory,
    collate_batch,
    MAX_FEATURES, remove_one_hot_labels,
)

# ---------- Helpers (pure Python; do not modify dataset.py) ----------

def set_ignore_list(ds, ignore: Set[int]):
    """Apply a global ignore set to all sub-datasets."""
    if isinstance(ds, ConcatDataset):
        for sub in ds.datasets:
            set_ignore_list(sub, ignore)
    elif isinstance(ds, LabeledStateDataset):
        ds.ignore_list = set(ignore)

def unique_active_feature_count(dataset) -> int:
    """
    Count unique active feature indices across *any* Dataset (Subset, ConcatDataset, etc.).
    Uses __getitem__ so filters are respected.
    """
    seen = set()
    n = len(dataset)
    for i in range(n):
        indices, _, _ = dataset[i]   # (indices, policy, value)
        seen.update(indices.tolist())
    return len(seen)


def redundancy_analysis(dataset, num_global_features: int):
    """
    Find redundant/unseen features across *any* Dataset (Subset, ConcatDataset, etc.)
    using __getitem__, so Subset filtering is respected.
    """
    from scipy.sparse import coo_matrix

    rows_list, cols_list = [], []
    sample_id = 0
    n = len(dataset)

    for i in range(n):
        indices, _, _ = dataset[i]
        if indices.numel() > 0:
            rows_list.append(np.full(indices.shape, sample_id, dtype=np.int32))
            cols_list.append(indices.numpy().astype(np.int32, copy=False))
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

    groups = {}
    indptr, indices = X_csc.indptr, X_csc.indices
    for j in range(num_global_features):
        start, end = indptr[j], indptr[j + 1]
        key = tuple(indices[start:end])  # () means unseen
        groups.setdefault(key, []).append(j)

    unseen_cols = groups.get((), [])
    num_unseen = len(unseen_cols)
    num_present_cols = sum(1 for k in groups.keys() if k != ())
    num_duplicate_groups = sum(1 for k, v in groups.items() if k != () and len(v) > 1)
    num_redundant_present = sum((len(v) - 1) for k, v in groups.items() if k != () and len(v) > 1)
    num_nonredundant_present = num_present_cols
    kept_global = num_global_features - (num_unseen + num_redundant_present)

    ignore = set(unseen_cols)
    for k, v in groups.items():
        if k == () or len(v) <= 1:
            continue
        ignore.update(sorted(v)[1:])

    return {
        "num_samples": num_samples,
        "num_unseen": num_unseen,
        "num_present_cols": num_present_cols,
        "num_duplicate_groups": num_duplicate_groups,
        "num_redundant_present": num_redundant_present,
        "num_nonredundant_present": num_nonredundant_present,
        "kept_global": kept_global,
        "ignore_set": ignore,
    }
def stream_policy_value_stats(dataset):
    """
    Stream the dataset to collect:
      - avg_policy (mean over samples)
      - label_vals (raw values for histogram)
      - num_samples (count)
    """
    dl = DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=collate_batch)
    policy_sum = None
    n = 0
    label_vals = []

    for indices, offsets, policies, values in dl:
        if policies.numel() == 0:
            continue
        B, A = policies.shape
        if policy_sum is None:
            policy_sum = torch.zeros(A, dtype=torch.float32)
        policy_sum += policies.sum(dim=0)
        n += B
        label_vals.extend(torch.atleast_1d(values).squeeze(-1).tolist())

    avg_policy = (policy_sum / max(n, 1)).cpu().numpy() if policy_sum is not None else np.array([], dtype=np.float32)
    label_vals = np.array(label_vals, dtype=np.float32)
    return {
        "avg_policy": avg_policy,
        "label_vals": label_vals,
        "num_samples": n,
    }

def plot_value_histogram(label_vals: np.ndarray, bins: int, title: str, save_path: str | None):
    edges = np.linspace(-1.0, 1.0, bins + 1)
    plt.figure()
    plt.hist(label_vals, bins=edges)
    plt.xlabel("Value label")
    plt.ylabel("Count")
    plt.title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()

def plot_avg_policy(avg_policy: np.ndarray, top_k: int, title: str, save_path: str | None):
    A = len(avg_policy)
    if A == 0:
        print("Average policy plot skipped (no actions found).")
        return
    k = max(1, min(A, top_k))
    xs = np.arange(k)
    ys = avg_policy[:k]
    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel("Action index (0..K-1)")
    plt.ylabel("Mean policy value")
    plt.title(f"{title} | A={A}, showing first {k}")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()

def preview_samples(dataset, N: int = 50, max_indices_to_show: int = 100) -> str:
    dl = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
    out_lines = []
    for i, (indices_batch, offsets_batch, policies_batch, values_batch) in enumerate(dl):
        if i >= N:
            break
        idxs = indices_batch.tolist()
        if len(idxs) > max_indices_to_show:
            sb = " ".join(map(str, idxs[:max_indices_to_show])) + " ..."
        else:
            sb = " ".join(map(str, idxs)) if idxs else "[No active features]"
        pol = policies_batch[0].tolist()
        try:
            val = float(torch.atleast_1d(values_batch).squeeze(-1)[0].item())
        except Exception:
            val = float(values_batch[0].item())
        out_lines.append(f"State: {sb}\nAction: {pol}\nValue: {val:+.4f}\n")
    return "\n".join(out_lines)

def main():
    # Resolve global feature count
    num_global = int(MAX_FEATURES)

    # 1) Load and combine datasets
    full_dataset = load_dataset_from_directory(DATA_DIR)
    full_dataset = remove_one_hot_labels(full_dataset)
    total_samples = len(full_dataset)
    print(f"\n=== Loaded dataset ===")
    print(f"Path: {DATA_DIR}")
    print(f"Total samples: {total_samples}")

    # 2) Unique active feature count
    uniq_count = unique_active_feature_count(full_dataset)
    print(f"Unique active feature indices (present in data): {uniq_count}")

    # 3) Redundancy analysis
    print("\n=== Redundancy analysis ===")
    ra = redundancy_analysis(full_dataset, num_global)
    print(f"Samples scanned: {ra['num_samples']}")
    print(f"Unseen features (never active): {ra['num_unseen']}")
    print(f"Present feature columns: {ra['num_present_cols']}")
    print(f"Duplicate groups among present: {ra['num_duplicate_groups']}")
    print(f"Redundant present features (to drop): {ra['num_redundant_present']}")
    print(f"Non-redundant features among PRESENT: {ra['num_nonredundant_present']}")
    print(f"Non-redundant features GLOBAL (kept after dropping unseen & dups): {ra['kept_global']}")

    if APPLY_IGNORE:
        set_ignore_list(full_dataset, ra["ignore_set"])
        print("Applied ignore set to dataset (unseen + redundant).")

    # 4) Stream once for value histogram + average policy
    print("\n=== Value label histogram & average policy ===")
    sv = stream_policy_value_stats(full_dataset)
    print(f"Samples aggregated: {sv['num_samples']}")

    # 5) Plot figures
    if SAVE_PLOTS and not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)

    hist_path = os.path.join(OUT_DIR, "value_histogram.png") if SAVE_PLOTS else None
    plot_value_histogram(
        sv["label_vals"],
        bins=HIST_BINS,
        title="Value label distribution (-1..1)",
        save_path=hist_path
    )
    if SAVE_PLOTS:
        print(f"Saved value histogram → {hist_path}")

    policy_path = os.path.join(OUT_DIR, "avg_policy_firstK.png") if SAVE_PLOTS else None
    plot_avg_policy(
        sv["avg_policy"],
        top_k=POLICY_TOP_K,
        title="Average policy (mean over samples)",
        save_path=policy_path
    )
    if SAVE_PLOTS:
        print(f"Saved average policy bar chart → {policy_path}")

    # 6) Sample preview
    print(f"\n=== First {PREVIEW_N} samples ===")
    print(preview_samples(full_dataset, N=PREVIEW_N, max_indices_to_show=100))

if __name__ == "__main__":
    main()