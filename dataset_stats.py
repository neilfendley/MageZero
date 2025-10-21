# mz_dataset_stats_simple.py

import os
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import H5Indexed, collate_batch, create_redundancy_ignore_list
from train import (
    DECK_NAME, VER_NUMBER, GLOBAL_MAX,
    ActionType,
    PRIORITY_A_MAX, PRIORITY_B_MAX, TARGETS_MAX, BINARY_MAX,
)

# --------------------
# Config (edit here)
# --------------------
SPLIT = "training"          # "training" or "testing"
OUT_DIR = "stats_out"       # where to save figures
SHOW_PLOTS = True           # headless? set False
SAVE_PLOTS = True           # save PNGs to OUT_DIR
TOP_K = 50                  # show first K bars for each head
HIST_BINS = 21              # value histogram bins
APPLY_IGNORE = True         # apply ignore list while streaming stats
PREVIEW_N = 30              # preview count

# Choose a non-GUI backend if headless
if not SHOW_PLOTS:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = f"data/{DECK_NAME}/ver{VER_NUMBER}/{SPLIT}"
MODEL_DIR = f"models/{DECK_NAME}/ver{VER_NUMBER}"
IGNORE_PATH = os.path.join(MODEL_DIR, "ignore.roar")


# --------------------
# Small helpers
# --------------------
def _dataloader(ds, bs=512):
    return DataLoader(
        ds, batch_size=bs, shuffle=False,
        collate_fn=collate_batch, num_workers=0, pin_memory=True
    )

def _maybe_load_ignore() -> Iterable[int] | None:
    if os.path.exists(IGNORE_PATH):
        try:
            from pyroaring import BitMap
            with open(IGNORE_PATH, "rb") as f:
                bm = BitMap.deserialize(f.read())
            print(f"[info] loaded ignore.roar: {len(bm)} indices")
            return bm
        except Exception as e:
            print(f"[warn] failed to load ignore.roar: {e}")
    return None

def _apply_ignore(ds: H5Indexed, ignore: Iterable[int] | None):
    if ignore is not None:
        ds.ignore_list = ignore

def _unique_active_feature_count(ds: H5Indexed) -> int:
    seen = set()
    for batch_indices, *_ in _dataloader(ds, bs=1024):
        if batch_indices.numel():
            seen.update(batch_indices.tolist())
    return len(seen)


# --------------------
# Streaming stats
# --------------------
def stream_stats(ds: H5Indexed):
    """Stream once and collect per-head average policy + value histogram data."""
    dl = _dataloader(ds, bs=512)

    pA_sum = torch.zeros(PRIORITY_A_MAX, dtype=torch.float32)
    pB_sum = torch.zeros(PRIORITY_B_MAX, dtype=torch.float32)
    t_sum  = torch.zeros(TARGETS_MAX,   dtype=torch.float32)
    b_sum  = torch.zeros(BINARY_MAX,    dtype=torch.float32)

    nA = nB = nT = nB2 = 0
    vals = []

    with torch.no_grad():
        for batch in dl:
            idx, off, policy, value, is_player, action_type = batch
            # decision masks (same semantics as train.py)
            nonzero = (policy > 0).sum(dim=1)
            decision_mask = nonzero > 1
            if decision_mask.any():
                is_p = is_player.squeeze(-1) > 0
                a_t  = action_type.squeeze(-1).to(torch.long)
                mask_pA = (a_t == ActionType.PRIORITY.value) & is_p & decision_mask
                mask_pB = (a_t == ActionType.PRIORITY.value) & (~is_p) & decision_mask
                mask_t  = (a_t == ActionType.CHOOSE_TARGET.value) & decision_mask
                mask_b  = (a_t == ActionType.CHOOSE_USE.value) & decision_mask

                if mask_pA.any():
                    pA_sum += policy[mask_pA, :PRIORITY_A_MAX].sum(dim=0)
                    nA += int(mask_pA.sum().item())
                if mask_pB.any():
                    pB_sum += policy[mask_pB, :PRIORITY_B_MAX].sum(dim=0)
                    nB += int(mask_pB.sum().item())
                if mask_t.any():
                    t_sum  += policy[mask_t, :TARGETS_MAX].sum(dim=0)
                    nT += int(mask_t.sum().item())
                if mask_b.any():
                    b_sum  += policy[mask_b, :BINARY_MAX].sum(dim=0)
                    nB2 += int(mask_b.sum().item())

            vals.extend(torch.atleast_1d(value).squeeze(-1).tolist())

    return {
        "avg_player_priority": (pA_sum / max(nA, 1)).cpu().numpy(),
        "avg_opponent_priority": (pB_sum / max(nB, 1)).cpu().numpy(),
        "avg_targets": (t_sum / max(nT, 1)).cpu().numpy(),
        "avg_binary": (b_sum / max(nB2, 1)).cpu().numpy(),
        "values": np.array(vals, dtype=np.float32),
        "num_samples": len(ds),
        "counts": {"pA": nA, "pB": nB, "t": nT, "b": nB2},
    }


# --------------------
# Plotting
# --------------------
def plot_value_hist(vals: np.ndarray, bins: int, title: str, out: str | None):
    edges = np.linspace(-1.0, 1.0, bins + 1)
    plt.figure()
    plt.hist(vals, bins=edges)
    plt.xlabel("Value label")
    plt.ylabel("Count")
    plt.title(title)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()

def plot_avg_bar(arr: np.ndarray, k: int, title: str, out: str | None):
    A = len(arr)
    if A == 0:
        print(f"[skip] {title} (no actions present)")
        return
    kk = max(1, min(A, k))
    xs = np.arange(kk); ys = arr[:kk]
    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel("Action index (0..K-1)")
    plt.ylabel("Mean policy label (train target)")
    plt.title(f"{title} | A={A}, first {kk}")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# --------------------
# Preview
# --------------------
def preview(ds: H5Indexed, n=PREVIEW_N, max_idx=96) -> str:
    dl = _dataloader(ds, bs=1)
    lines = []
    for i, batch in enumerate(dl):
        if i >= n: break
        idxs, off, pol, val, is_p, a_t = batch
        idxs = idxs.tolist()
        idx_str = " ".join(map(str, idxs[:max_idx])) + (" ..." if len(idxs) > max_idx else "")
        valf = float(torch.atleast_1d(val).squeeze(-1)[0].item())
        ispf = bool(int(is_p.squeeze(-1)[0].item()))
        aty  = int(a_t.squeeze(-1)[0].item())
        lines.append(
            f"State[{i}]: {idx_str}\n"
            f"  actionType={aty} isPlayer={ispf}  value={valf:+.4f}\n"
            f"  policy[:16]={pol[0, :16].tolist()}\n"
        )
    return "\n".join(lines)


# --------------------
# Main
# --------------------
def main():
    print(f"[load] {DATA_DIR}")
    ds = H5Indexed(DATA_DIR)
    print(f"[stats] samples={len(ds)}")

    # Ignore list: prefer saved ignore.roar, otherwise (optionally) compute
    ignore = _maybe_load_ignore()
    if ignore is None and APPLY_IGNORE:
        print("[info] computing ignore list (redundancy) from dataset…")
        ignore_list = create_redundancy_ignore_list(ds, GLOBAL_MAX)
        print(f"[info] ignore computed: {len(ignore_list)} indices")
        ignore = ignore_list

    if APPLY_IGNORE and ignore is not None:
        _apply_ignore(ds, ignore)
        print("[info] ignore applied")

    uniq = _unique_active_feature_count(ds)
    print(f"[stats] unique active feature indices={uniq}")

    sv = stream_stats(ds)
    print(f"[stats] aggregated samples={sv['num_samples']} "
          f"(pA={sv['counts']['pA']}, pB={sv['counts']['pB']}, "
          f"t={sv['counts']['t']}, b={sv['counts']['b']})")

    if SAVE_PLOTS and not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)

    # value histogram
    plot_value_hist(
        sv["values"], HIST_BINS, f"Value labels ({DECK_NAME} v{VER_NUMBER} {SPLIT})",
        os.path.join(OUT_DIR, f"value_hist_{DECK_NAME}_v{VER_NUMBER}_{SPLIT}.png") if SAVE_PLOTS else None
    )

    # per-head avg policy bars
    plot_avg_bar(
        sv["avg_player_priority"], TOP_K, "Avg policy – Player PRIORITY",
        os.path.join(OUT_DIR, f"avg_policy_pA_{DECK_NAME}_v{VER_NUMBER}_{SPLIT}.png") if SAVE_PLOTS else None
    )
    plot_avg_bar(
        sv["avg_opponent_priority"], TOP_K, "Avg policy – Opponent PRIORITY",
        os.path.join(OUT_DIR, f"avg_policy_pB_{DECK_NAME}_v{VER_NUMBER}_{SPLIT}.png") if SAVE_PLOTS else None
    )
    plot_avg_bar(
        sv["avg_targets"], TOP_K, "Avg policy – CHOOSE_TARGET",
        os.path.join(OUT_DIR, f"avg_policy_target_{DECK_NAME}_v{VER_NUMBER}_{SPLIT}.png") if SAVE_PLOTS else None
    )
    plot_avg_bar(
        sv["avg_binary"], TOP_K, "Avg policy – CHOOSE_USE",
        os.path.join(OUT_DIR, f"avg_policy_binary_{DECK_NAME}_v{VER_NUMBER}_{SPLIT}.png") if SAVE_PLOTS else None
    )

    print("\n=== Preview ===")
    print(preview(ds, PREVIEW_N))


if __name__ == "__main__":
    main()
