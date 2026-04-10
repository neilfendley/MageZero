"""Quick inspection of HDF5 training data.

Usage:
    inspect-data data/ver0/training
    inspect-data data/ver0/training --samples 5
    inspect-data data/ver0/training --feature-table ../mage/FeatureTable.txt
"""
import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np

ACTION_TYPES = {0: "PRIORITY", 1: "CHOOSE_NUM", 2: "BLANK", 3: "CHOOSE_TARGET", 4: "MAKE_CHOICE", 5: "CHOOSE_USE"}
ACTION_NAMES = {0: "Pass", 1: "Tap:B", 2: "Tap:G", 3: "Tap:R", 4: "Tap:U", 5: "Tap:W", 6: "Tap:C"}

# regex: "114: [994151/{1}_dynamic#1]" -> group(1)=114, group(2)="{1}_dynamic"
_FT_RE = re.compile(r"^(\d+):\s+\[\S+/(.+?)#\d+\]")


def load_feature_table(path: Path) -> dict[int, str]:
    table: dict[int, str] = {}
    for line in path.read_text().splitlines():
        m = _FT_RE.match(line)
        if m:
            table[int(m.group(1))] = m.group(2)
    return table


def fmt_feature(idx: int, ft: dict[int, str] | None) -> str:
    if ft and idx in ft:
        return ft[idx]
    return str(idx)


def fmt_action(idx: int) -> str:
    return ACTION_NAMES.get(idx, f"ability[{idx}]")


def main():
    parser = argparse.ArgumentParser(description="Inspect HDF5 training data")
    parser.add_argument("dir", help="Data directory containing .hdf5/.h5 files")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to print per file (default 10)")
    parser.add_argument("--feature-table", type=str, default=None,
                        help="Path to FeatureTable.txt (auto-detected from ../mage/FeatureTable.txt if present)")
    args = parser.parse_args()

    p = Path(args.dir)
    files = sorted(list(p.rglob("*.hdf5")) + list(p.rglob("*.h5")))
    if not files:
        print(f"No .hdf5/.h5 files found in {p}", file=sys.stderr)
        sys.exit(1)

    # Load feature table
    ft = None
    ft_path = Path(args.feature_table) if args.feature_table else None
    if ft_path is None:
        # auto-detect relative to common locations
        candidates = [
            Path("../mage/FeatureTable.txt"),
            Path("../../mage/FeatureTable.txt"),
            Path(__file__).resolve().parent.parent.parent.parent / "mage" / "FeatureTable.txt",
        ]
        for c in candidates:
            if c.exists():
                ft_path = c
                break
    if ft_path and ft_path.exists():
        ft = load_feature_table(ft_path)
        print(f"Loaded {len(ft)} feature names from {ft_path}")
    else:
        print("(No FeatureTable.txt found — showing raw indices. Use --feature-table to specify.)")

    total_samples = 0
    total_nnz = 0
    all_results = []
    action_type_counts: dict[int, int] = {}

    for f in files:
        with h5py.File(f, "r") as h:
            off = h["/offsets"][...]
            row = h["/row"][...]
            N = off.shape[0] - 1
            nnz = int(off[-1])
            A = row.shape[1] - 4

            results = row[:, A]  # resultLabel column
            action_types = row[:, A + 3].astype(int)

            print(f"\n--- {f.relative_to(p)} ---")
            print(f"  samples: {N}, actions: {A}, nnz features: {nnz}")
            print(f"  avg features/sample: {nnz / N:.1f}" if N else "  (empty)")
            print(f"  result distribution: wins={int((results > 0).sum())}, losses={int((results <= 0).sum())}")

            for at in np.unique(action_types):
                action_type_counts[int(at)] = action_type_counts.get(int(at), 0) + int((action_types == at).sum())

            total_samples += N
            total_nnz += nnz
            all_results.extend(results.tolist())

            idx = h["/indices"][...]
            for i in range(min(N, args.samples)):
                a, b = int(off[i]), int(off[i + 1])
                feat_ids = idx[a:b]
                policy = row[i, :A]
                result = row[i, A]
                is_player = row[i, A + 2]
                atype = int(row[i, A + 3])

                # Format features
                feat_names = [fmt_feature(int(x), ft) for x in feat_ids[:20]]
                feat_str = ", ".join(feat_names)
                if len(feat_ids) > 20:
                    feat_str += f" ... ({len(feat_ids)} total)"

                # Format actions
                nonzero_actions = np.where(policy > 1e-6)[0]
                action_parts = []
                for j in nonzero_actions[:5]:
                    action_parts.append(f"{fmt_action(int(j))}:{policy[j]:.0f}")
                if len(nonzero_actions) > 5:
                    action_parts.append(f"... ({len(nonzero_actions)} total)")
                action_str = ", ".join(action_parts)

                atype_name = ACTION_TYPES.get(atype, str(atype))
                print(f"  [{i}] result={result:+.2f} type={atype_name} player={int(is_player)}")
                print(f"       actions: [{action_str}]")
                print(f"       features: [{feat_str}]")

    if total_samples == 0:
        print("\nNo samples found.")
        return

    results_arr = np.array(all_results)
    at_str = ", ".join(f"{ACTION_TYPES.get(k, str(k))}={v}" for k, v in sorted(action_type_counts.items()))
    print(f"\n=== Summary ===")
    print(f"  files: {len(files)}")
    print(f"  total samples: {total_samples}")
    print(f"  avg features/sample: {total_nnz / total_samples:.1f}")
    print(f"  win rate: {(results_arr > 0).mean():.3f}")
    print(f"  action types: {at_str}")


if __name__ == "__main__":
    main()
