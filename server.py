import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from pyroaring import BitMap
from flask import Flask, request, Response
import msgpack
import threading, time
from queue import Queue, Empty

from train import Net, GLOBAL_MAX, ACTIONS_MAX, VER_NUMBER, DECK_NAME
MODEL_DIR = f"models/{DECK_NAME}/ver{VER_NUMBER}"
IGNORE = MODEL_DIR + "/ignore.roar"
MODEL = MODEL_DIR + "/model.pt"

# Model + ignore list setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ignore bitmap (created during training)
with open(IGNORE, "rb") as f:
    IGNORE_BM = BitMap.deserialize(f.read())

# Load model checkpoint (policy logits + value)  # :contentReference[oaicite:3]{index=3}
server_model = Net(GLOBAL_MAX, ACTIONS_MAX).to(DEVICE).eval()
ckpt = torch.load(MODEL, map_location=DEVICE)
server_model.load_state_dict(ckpt["model_state_dict"])

# Optional CPU threading tuning
TORCH_THREADS = max(1, os.cpu_count() // 2)
torch.set_num_threads(TORCH_THREADS)

app = Flask(__name__)


req_counter = 0
req_counter_lock = threading.Lock()

class Pending:
    __slots__ = ("idx", "off", "evt", "out", "req_id", "pre_count", "post_count", "t_recv", "t_done", "num_bags")
    def __init__(self, req_id, indices, offsets):
        # Apply ignore filter server-side (same as test/train do before batching)  # :contentReference[oaicite:4]{index=4}
        self.req_id = req_id
        self.pre_count = len(indices)
        indices, offsets, num_bags = apply_ignore(indices, offsets, GLOBAL_MAX, IGNORE_BM)
        self.post_count = len(indices)
        self.num_bags = num_bags

        # Only now construct tensors (avoid surfacing earlier CUDA errors here)
        self.idx = torch.tensor(indices, dtype=torch.long)  # CPU
        self.off = torch.tensor(offsets, dtype=torch.long)  # CPU

        self.t_recv = time.perf_counter()
        self.evt = threading.Event()
        self.out = None
        self.t_done = 0.0

Q: "Queue[Pending]" = Queue(maxsize=4096)
def apply_ignore(indices: list[int], offsets: list[int] | None, global_max: int, ignore_bm: BitMap):
    if not offsets:
        offsets = [0]

    if offsets[0] != 0:
        offsets[0] = 0
    if any(o < 0 for o in offsets):
        raise ValueError("offsets contain negative values")
    if any(offsets[i] > offsets[i+1] for i in range(len(offsets)-1)):
        raise ValueError("offsets not non-decreasing")

    if len(offsets) == 1:
        # filter + bounds check
        filt = [i for i in indices if (i in ignore_bm) is False and 0 <= i < global_max]
        return filt, [0], 1

    new_indices = []
    new_offsets = [0]
    n = len(indices)
    for b in range(len(offsets)):
        start = offsets[b]
        end = offsets[b+1] if b+1 < len(offsets) else n
        if not (0 <= start <= end <= n):
            raise ValueError(f"bad bag bounds: {start}..{end} within 0..{n}")
        bag = indices[start:end]
        bag = [i for i in bag if (i in ignore_bm) is False and 0 <= i < global_max]
        new_indices.extend(bag)
        new_offsets.append(len(new_indices))
    new_offsets = new_offsets[:-1]

    return new_indices, new_offsets, len(new_offsets)

def worker_loop():
    print(f"[INIT] Device={DEVICE}, torch threads={TORCH_THREADS}, "
        f"model={MODEL}, ignore={IGNORE}")
    while True:
        p0 = Q.get()
        batch = [p0]
        t0 = time.time()


        p = batch[0]
        with torch.no_grad():
            idx = p.idx.to(DEVICE, non_blocking=True)
            off = p.off.to(DEVICE, non_blocking=True)
            pA, pB, tgt, bin2, val = server_model(idx, off)  # shapes: [B,A], [B,A], [B,A], [B,2], [B]

        # move to CPU
        pA = pA.detach().to("cpu")
        pB = pB.detach().to("cpu")
        tgt = tgt.detach().to("cpu")
        bin2 = bin2.detach().to("cpu")
        val = val.detach().to("cpu")

        if p.num_bags == 1:
            p.out = {
                "policy_player": pA[0].tolist(),
                "policy_opponent": pB[0].tolist(),
                "policy_target": tgt[0].tolist(),
                "policy_binary": bin2[0].tolist(),
                "value": float(val[0].item()),
            }
        else:
            # Multi-bag in a single HTTP request -> array of maps
            B = p.num_bags
            p.out = [
                {
                    "policy_player": pA[i].tolist(),
                    "policy_opponent": pB[i].tolist(),
                    "policy_target": tgt[i].tolist(),
                    "policy_binary": bin2[i].tolist(),
                    "value": float(val[i].item()),
                }
                for i in range(B)
            ]

        p.t_done = time.perf_counter()
        p.evt.set()


threading.Thread(target=worker_loop, daemon=True).start()

@app.post("/evaluate")
def evaluate():
    global req_counter
    """
    MessagePack request: {"indices": [int64], "offsets": [int64]}  (offsets optional → defaults to [0])
    Returns MessagePack: {"policy": [float], "value": float}
    """
    data = msgpack.unpackb(request.data, raw=False)
    with req_counter_lock:
        req_counter += 1

    indices = data.get("indices", [])
    offsets = data.get("offsets", [])
    pending = Pending(req_counter, indices, offsets)
    # Per-request log (received)
    print(f"[REQ {pending.req_id}] num_indices={pending.post_count}, num_ignored={pending.pre_count - pending.post_count}, batch_num={len(pending.off)} indices={pending.pre_count}")

    Q.put(pending)
    pending.evt.wait()

    # Per-request log (done)
    total_ms = (pending.t_done - pending.t_recv) * 1000.0
    print(f"[REQ {pending.req_id}] done: total_ms={total_ms:.3f}")

    payload = msgpack.packb(pending.out, use_bin_type=True)
    return Response(payload, mimetype="application/x-msgpack")

@app.get("/healthz")
def healthz():
    return "ok", 200

if __name__ == "__main__":
    # Dev run; for production use gunicorn:
    #gunicorn server:app --bind 127.0.0.1:50052 --workers 1 --threads 8 --keep-alive 120 --timeout 0
    app.run(host="127.0.0.1", port=50052, threaded=True)
