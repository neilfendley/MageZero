"""Compress a training checkpoint to a sparse inference-only model.

Drops optimizer state, casts the embedding to fp16, and stores only the
embedding rows whose indices appear in the training data for the deck.

Usage:
    python -m magezero.compress_model <input.pt> [output.pt] \
        [--data <data_dir>] [--deck <deck_name>]

Defaults: --data data/ver{N}/training and --deck are inferred from
<input.pt> when it follows the standard models/<DECK>/ver<N>/model.pt layout.
"""
import argparse
import os
import re
from pathlib import Path

import h5py
import numpy as np
import torch


def _infer_deck_and_ver(input_path: str) -> tuple[str | None, int | None]:
    m = re.search(r"models/([^/]+)/ver(\d+)/", input_path)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def _scan_used_indices(data_dir: str, deck_name: str) -> set[int]:
    files = sorted(
        p for p in Path(data_dir).glob("**/*.hdf5")
        if p.name.split(".")[0].split("_")[0] == deck_name
    )
    print(f"  scanning {len(files)} hdf5 files under {data_dir} for deck '{deck_name}'")
    used: set[int] = set()
    for p in files:
        with h5py.File(p, "r") as f:
            idx = f["/indices"][...]
            used.update(np.unique(idx).tolist())
    print(f"  found {len(used)} unique used indices")
    return used


def compress(input_path: str, output_path: str, data_dir: str | None, deck: str | None) -> None:
    inferred_deck, inferred_ver = _infer_deck_and_ver(input_path)
    if deck is None:
        deck = inferred_deck
    if data_dir is None and inferred_ver is not None:
        data_dir = f"data/ver{inferred_ver}/training"
    if not deck or not data_dir:
        raise ValueError("need --deck and --data (or a standard models/<DECK>/ver<N>/ path)")

    # Scan first so a bad path/deck fails before we read the 10 GB checkpoint.
    used = _scan_used_indices(data_dir, deck)
    if not used:
        raise RuntimeError(f"no indices found scanning {data_dir} for {deck}")

    print(f"Loading {input_path} ...")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"checkpoint at {input_path} has no 'model_state_dict' key")

    sd = ckpt["model_state_dict"]
    emb = sd.pop("embedding_bag.weight")
    print(f"  embedding: {tuple(emb.shape)} {emb.dtype} ({emb.numel() * emb.element_size() / 1e9:.2f} GB)")

    used_sorted = torch.tensor(sorted(used), dtype=torch.int32)
    rows = emb[used_sorted.long()].half()
    print(f"  storing {used_sorted.numel()} rows × {rows.shape[1]} dim fp16 = "
          f"{rows.numel() * rows.element_size() / 1e6:.2f} MB")

    slim = {
        "model_state_dict": sd,
        "sparse_embedding": {
            "indices": used_sorted,        # int32 [K]
            "values": rows,                # fp16 [K, D]
            "shape": tuple(emb.shape),     # (num_embeddings, embedding_dim)
        },
    }
    for k in ("epoch", "avg_p_loss", "avg_v_loss"):
        if k in ckpt:
            slim[k] = ckpt[k]

    print(f"Saving to {output_path} ...")
    torch.save(slim, output_path)

    in_size = os.path.getsize(input_path)
    out_size = os.path.getsize(output_path)
    print(f"  in : {in_size / 1e9:.2f} GB")
    print(f"  out: {out_size / 1e6:.2f} MB  ({out_size / in_size * 100:.2f}% of original)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output", nargs="?", default=None)
    ap.add_argument("--data", default=None, help="dataset directory (default: data/ver{N}/training)")
    ap.add_argument("--deck", default=None, help="deck name filter (default: inferred from path)")
    args = ap.parse_args()

    out = args.output or str(Path(args.input).with_suffix(".sparse.pt"))
    compress(args.input, out, args.data, args.deck)
