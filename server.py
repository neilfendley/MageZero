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

import train  # provides Net, GLOBAL_MAX, ACTIONS_MAX
from train import DECK_NAME, VER_NUMBER, GLOBAL_MAX, ACTIONS_MAX

MODEL_DIR = f"models/{DECK_NAME}/ver{VER_NUMBER}"
IGNORE = MODEL_DIR + "/ignore.roar"
MODEL = MODEL_DIR + "/model.pt"

# -------------------------
# Model + ignore list setup
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ignore bitmap (created during training)
# (train.py writes models/{DECK}/ver{VER}/ignore.roar)  # :contentReference[oaicite:2]{index=2}
with open(IGNORE, "rb") as f:
    IGNORE_BM = BitMap.deserialize(f.read())

# Load model checkpoint (policy logits + value)  # :contentReference[oaicite:3]{index=3}
model = train.Net(GLOBAL_MAX, ACTIONS_MAX).to(DEVICE).eval()
ckpt = torch.load(MODEL, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

# Optional CPU threading tuning

TORCH_THREADS = max(1, os.cpu_count() // 2)
torch.set_num_threads(TORCH_THREADS)

# -------------------------
# Flask app with (optional) micro-batching
# -------------------------
app = Flask(__name__)

BATCH_MS: float = 0.0   # 0 = no micro-batching now; set to 1–3 later if desired
BATCH_MAX: int  = 32

req_counter = 0
req_counter_lock = threading.Lock()

class Pending:
    __slots__ = ("idx", "off", "evt", "out", "req_id", "pre_count", "post_count", "t_recv", "t_done")
    def __init__(self, req_id, indices, offsets):
        # Apply ignore filter server-side (same as test/train do before batching)  # :contentReference[oaicite:4]{index=4}
        self.req_id = req_id
        self.pre_count = len(indices)
        if indices:
            indices = [i for i in indices if i not in IGNORE_BM]
        self.post_count = len(indices)
        self.idx = torch.tensor(indices, dtype=torch.long, device=DEVICE)
        # If caller didn't send offsets, default to a single bag starting at 0
        if not offsets:
            offsets = [0]
        self.t_recv = time.perf_counter()
        self.off = torch.tensor(offsets, dtype=torch.long, device=DEVICE)
        self.evt = threading.Event()
        self.out = None
        self.t_done = 0.0

Q: "Queue[Pending]" = Queue(maxsize=4096)

def worker_loop():
    print(f"[INIT] Device={DEVICE}, torch threads={TORCH_THREADS}, "
        f"model={MODEL}, ignore={IGNORE}, BATCH_MS={BATCH_MS}, BATCH_MAX={BATCH_MAX}")
    while True:
        p0 = Q.get()
        batch = [p0]
        t0 = time.time()
        if BATCH_MS > 0:
            while len(batch) < BATCH_MAX and (time.time() - t0) * 1000 < BATCH_MS:
                try:
                    batch.append(Q.get_nowait())
                except Empty:
                    break

        if len(batch) == 1:
            idx = batch[0].idx
            off = batch[0].off
            with torch.no_grad():
                pol, val = model(idx, off)   # Net forward returns logits + tanh value  # :contentReference[oaicite:5]{index=5}
            # Shapes: [1, A], [1]
            pol = pol.detach().to("cpu")
            val = val.detach().to("cpu")
            batch[0].out = {"policy": pol[0].tolist(), "value": float(val[0].item())}
            batch[0].t_done = time.perf_counter()
            batch[0].evt.set()
        else:
            # Concatenate batch along the bag dimension
            idx = torch.cat([p.idx for p in batch], dim=0)
            # Offsets are per-sample starts; stack and rely on EmbeddingBag semantics of concatenation
            off = torch.cat([p.off for p in batch], dim=0)
            with torch.no_grad():
                pol, val = model(idx, off)   # [B, A], [B]
            pol = pol.detach().to("cpu")
            val = val.detach().to("cpu")
            for i, p in enumerate(batch):
                p.out = {"policy": pol[i].tolist(), "value": float(val[i].item())}
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
