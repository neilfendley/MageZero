from pathlib import Path
import h5py
import numpy as np

data_dir = "data/UWTempo/ver18/testing"

for f in sorted(Path(data_dir).glob("*.hdf5*")):
    with h5py.File(f, "r") as h:
        off = h["/offsets"][...]
        diffs = off[1:] - off[:-1]
        bad = (diffs < 0).any()
        bad_start = off[0] != 0

        if bad or bad_start:
            print(f"❌ {f.name}")
            if bad:
                idx = np.where(diffs < 0)[0][0]
                print(f"   Non-monotonic at sample {idx}: off[{idx}]={off[idx]}, off[{idx + 1}]={off[idx + 1]}")
            if bad_start:
                print(f"   off[0]={off[0]} (should be 0)")
        else:
            print(f"✓ {f.name}")