import os
import threading
import time
import gc
import subprocess
from queue import Queue, Empty

import torch
from pyroaring import BitMap
from flask import Flask, request, Response
from pathlib import Path
import msgpack
import signal


from .train import Net, GLOBAL_MAX, ACTIONS_MAX, VER_NUMBER, DECK_NAME, load_model, EMBEDDING_DIM


MODEL_DIR = f"models/{DECK_NAME}/ver{VER_NUMBER}"
IGNORE = MODEL_DIR + "/ignore.roar"
MODEL = MODEL_DIR + "/model.pt"

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Valid range bitmap for bounds checking (roaring compresses this to ~bytes)
VALID_RANGE = BitMap(range(GLOBAL_MAX))

model_lock = threading.RLock()
IGNORE_BM = BitMap()
server_model = None

# Threading config
TORCH_THREADS = os.environ.get("MAGEZERO_SERVER_THREADS", os.cpu_count() // 2)
torch.set_num_threads(int(TORCH_THREADS))

# Batching config
MAX_BATCH = 24
MAX_WAIT_MS = 0

app = Flask(__name__)

req_counter = 0
req_counter_lock = threading.Lock()


def _current_rss_mb():
    try:
        rss_kb = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        ).strip()
        return int(rss_kb) / 1024
    except Exception:
        return None


def _log_memory(prefix):
    rss_mb = _current_rss_mb()
    if rss_mb is None:
        print(prefix)
    else:
        print(f"{prefix} rss={rss_mb:.1f}MB")


def _build_model_from_state_dict(state_dict):
    emb_weight = state_dict["embedding_bag.weight"]
    num_embeddings, embedding_dim = emb_weight.shape
    policy_size_a = state_dict["player_priority_head.weight"].shape[0]

    if embedding_dim != EMBEDDING_DIM:
        raise ValueError(
            f"Checkpoint embedding_dim={embedding_dim} does not match "
            f"configured EMBEDDING_DIM={EMBEDDING_DIM}"
        )

    return Net(num_embeddings, policy_size_a).eval()


def _release_model(model):
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_server_artifacts(path=MODEL):

    with open(IGNORE, "rb") as f:
        ignore_bm = BitMap.deserialize(f.read())

    model = Net(GLOBAL_MAX, ACTIONS_MAX).eval()
    print(f'Model loaded, Must load weights to use model"')
    # ckpt = load_model(path, device='cpu')
    # model.load_state_dict(ckpt["model_state_dict"])
    # model = model.to(DEVICE)
    # print("Model weights loaded.")
    # print("Model artifacts loaded. )
    return model, ignore_bm


def reload_server_model():
    load_server_weights(MODEL)

def load_model_weights(path_to_weights):
    # path_to_weights = Path(path_to_weights)
    print("Loading server artifacts...")
    ckpt = load_model(path_to_weights, device='cpu')
    state_dict = ckpt["model_state_dict"]
    model = _build_model_from_state_dict(state_dict)

    ignore_path = Path(path_to_weights).parent / "ignore.roar"
    with open(ignore_path, "rb") as f:
        ignore_bm = BitMap.deserialize(f.read())    
    model.load_state_dict(state_dict)
    del state_dict
    del ckpt
    gc.collect()
    model = model.to(DEVICE)
    _log_memory("[LOAD] model ready")
    return model, ignore_bm

def load_server_weights(path_to_weights):
    global server_model, IGNORE_BM
    old_model = None
    old_ignore = None
    with model_lock:
        old_model = server_model
        old_ignore = IGNORE_BM
        server_model = None
        IGNORE_BM = BitMap()
    _release_model(old_model)
    del old_ignore
    gc.collect()
    _log_memory("[LOAD] previous model released")
    model, bm = load_model_weights(path_to_weights)
    with model_lock:
        server_model = model
        IGNORE_BM = bm
    print("Server model weights reloaded.")

# reload_server_model()


class Pending:
    __slots__ = ("idx", "off", "evt", "out", "req_id", "pre_count", "post_count", "t_recv", "t_done", "num_bags")

    def __init__(self, req_id, indices, offsets):
        self.req_id = req_id
        self.pre_count = len(indices)
        self.t_recv = time.perf_counter()
        self.evt = threading.Event()
        self.out = None
        self.t_done = 0.0

        indices, offsets, num_bags = apply_ignore(indices, offsets)
        self.post_count = len(indices)
        self.num_bags = num_bags

        self.idx = torch.tensor(indices, dtype=torch.long)
        self.off = torch.tensor(offsets, dtype=torch.long)


def apply_ignore(indices: list[int], offsets: list[int] | None):
    if not offsets:
        offsets = [0]

    if len(offsets) == 1:
        # Single bag - pure bitmap ops in C
        kept_bm = (BitMap(indices) - IGNORE_BM) & VALID_RANGE
        return list(kept_bm), [0], 1

    # Multi-bag
    n = len(indices)
    new_indices = []
    new_offsets = [0]

    for b in range(len(offsets)):
        start = offsets[b]
        end = offsets[b + 1] if b + 1 < len(offsets) else n

        kept_bm = (BitMap(indices[start:end]) - IGNORE_BM) & VALID_RANGE
        new_indices.extend(kept_bm)
        new_offsets.append(len(new_indices))

    new_offsets = new_offsets[:-1]
    return new_indices, new_offsets, len(new_offsets)


Q: "Queue[Pending]" = Queue(maxsize=4096)


def worker_loop():
    print(f"[INIT] Device={DEVICE}, torch threads={TORCH_THREADS}, "
          f"model={MODEL}, ignore={IGNORE}")

    while True:
        p0 = Q.get()
        batch = [p0]

        # Collect more requests up to MAX_BATCH or MAX_WAIT_MS
        deadline = time.perf_counter() + (MAX_WAIT_MS / 1000.0)
        while len(batch) < MAX_BATCH or not Q.empty():
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                remaining = 0
                if Q.empty():
                    break
            try:
                batch.append(Q.get(timeout=remaining))
            except Empty:
                break

        bag_counts = [p.num_bags for p in batch]

        # Fast path: single request
        if len(batch) == 1:
            idx = batch[0].idx.to(DEVICE, non_blocking=True)
            off = batch[0].off.to(DEVICE, non_blocking=True)
        else:
            # Concatenate indices
            idx = torch.cat([p.idx for p in batch]).to(DEVICE, non_blocking=True)

            # Vectorized offset adjustment
            all_off = torch.cat([p.off for p in batch])
            idx_lens = torch.tensor([len(p.idx) for p in batch])
            bag_counts_t = torch.tensor(bag_counts)
            adjustments = torch.repeat_interleave(
                torch.cat([torch.tensor([0]), idx_lens.cumsum(0)[:-1]]),
                bag_counts_t
            )
            off = (all_off + adjustments).to(DEVICE, non_blocking=True)

        # Single forward pass
        with model_lock:
            if server_model is None:
                for p in batch:
                    p.out = {"error": "model not loaded"}
                    p.t_done = time.perf_counter()
                    p.evt.set()
                continue
            with torch.no_grad():
                pA, pB, tgt, bin2, val = server_model(idx, off)

        # Move to CPU once
        pA = pA.cpu()
        pB = pB.cpu()
        tgt = tgt.cpu()
        bin2 = bin2.cpu()
        val = val.cpu()

        # Split results back to individual requests
        row = 0
        for p, num_bags in zip(batch, bag_counts):
            if num_bags == 1:
                p.out = {
                    "policy_player": pA[row].tolist(),
                    "policy_opponent": pB[row].tolist(),
                    "policy_target": tgt[row].tolist(),
                    "policy_binary": bin2[row].tolist(),
                    "value": float(val[row].item()),
                }
            else:
                p.out = [
                    {
                        "policy_player": pA[row + i].tolist(),
                        "policy_opponent": pB[row + i].tolist(),
                        "policy_target": tgt[row + i].tolist(),
                        "policy_binary": bin2[row + i].tolist(),
                        "value": float(val[row + i].item()),
                    }
                    for i in range(num_bags)
                ]
            row += num_bags
            p.t_done = time.perf_counter()
            p.evt.set()

        print(f"[BATCH] size={len(batch)}, total_bags={row}")


threading.Thread(target=worker_loop, daemon=True).start()


@app.post("/evaluate")
def evaluate():
    global req_counter

    data = msgpack.unpackb(request.data, raw=False)
    with req_counter_lock:
        req_counter += 1

    indices = data.get("indices", [])
    offsets = data.get("offsets", [])
    pending = Pending(req_counter, indices, offsets)

    print(f"[REQ {pending.req_id}] indices={pending.pre_count}, kept={pending.post_count}, bags={pending.num_bags}")

    Q.put(pending)
    pending.evt.wait()

    total_ms = (pending.t_done - pending.t_recv) * 1000.0
    print(f"[REQ {pending.req_id}] done: {total_ms:.1f}ms")

    return Response(msgpack.packb(pending.out, use_bin_type=True), mimetype="application/x-msgpack")


@app.get("/healthz")
def healthz():
    return "ok", 200

@app.post("/reload")
def reload_endpoint():
    reload_server_model()
    return {
        "status": "reloaded",
        "deck": DECK_NAME,
        "version": VER_NUMBER,
        "model": MODEL,
        "ignore": IGNORE,
    }, 200

@app.post("/load")
def load_endpoint():
    print('Received request to load new model weights')
    path = request.json.get("path")
    if not path:
        return {"error": "Path is required"}, 400
    load_server_weights(path)
    return {"status": "loaded"}, 200

if __name__ == "__main__":
    host = os.environ.get("MAGEZERO_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("MAGEZERO_SERVER_PORT", "50052"))
    app.run(host=host, port=port, threaded=True)
