"""extract_goodfire.py on a host whose disk cannot hold the 52 GB checkpoint: every decoder block is read over HTTP
range requests straight from the Hugging Face repo into CPU memory, one block at a time, with the next block fetched in
a background thread while the current one runs. Nothing but the small config and tokenizer files touches the disk.

The forward pass, batching, outputs and resume rule are extract_goodfire.py's; only extract.Weights is replaced.
--queue runs several traces files in one process: one line per job, "<traces.jsonl> <ids file or ->".

    HF_HUB_OFFLINE=0 python -m hermeneutics.goodfire_stream --model /root/work_goodfire/models/Qwen3.8-27B \\
        --queue /root/work_goodfire/plan/queue.txt -- --out /root/work_goodfire/acts --batch-tokens 1000000
"""
import argparse
import json
import os
import struct
import sys
import threading
import time

import torch

from . import extract_goodfire
from .extract import PREFIX

REPO_ID = "Qwen/Qwen3.8-27B"
REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
          "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
          "BOOL": torch.bool}
GAP = 64 << 20  # merge byte ranges of one shard whose gap is below this


class RemoteWeights:
    """extract.Weights over HTTP: same map, get() and layer_names(); each get() also starts fetching the next block."""

    def __init__(self, model_dir, repo_id=REPO_ID, revision=REVISION):
        import httpx
        from huggingface_hub import hf_hub_url
        self.dir = model_dir
        self.map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        self.url = {f: hf_hub_url(repo_id, f, revision=revision) for f in set(self.map.values())}
        self.client = httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120, connect=30))
        self.headers = {}
        self.pending = {}  # key -> (thread, box)
        self.lock = threading.Lock()
        self.fetched_bytes = 0
        self.fetch_seconds = 0.0

    def _range(self, f, a, b):
        """Bytes [a, b) of shard f, with retries."""
        for attempt in range(8):
            try:
                buf = bytearray()
                with self.client.stream("GET", self.url[f], headers={"Range": f"bytes={a}-{b - 1}"}) as r:
                    if r.status_code != 206:
                        raise IOError(f"{f} range {a}-{b}: HTTP {r.status_code}")
                    for c in r.iter_bytes(8 << 20):
                        buf += c
                if len(buf) != b - a:
                    raise IOError(f"{f} range {a}-{b}: got {len(buf)} bytes")
                return buf
            except Exception as e:  # network: back off and retry
                print(f"  fetch retry {attempt + 1}: {e}", flush=True)
                time.sleep(min(60, 2 ** attempt))
        raise IOError(f"{f} range {a}-{b}: failed after retries")

    def _header(self, f):
        with self.lock:
            if f in self.headers:
                return self.headers[f]
        n = struct.unpack("<Q", bytes(self._range(f, 0, 8)))[0]
        h = json.loads(bytes(self._range(f, 8, 8 + n)))
        h.pop("__metadata__", None)
        with self.lock:
            self.headers[f] = (8 + n, h)
        return self.headers[f]

    def _fetch(self, names):
        """{name: CPU tensor} for names, read by merged ranges."""
        t0 = time.time()
        out, by_file = {}, {}
        for n in names:
            by_file.setdefault(self.map[n], []).append(n)
        for f, ns in by_file.items():
            base, h = self._header(f)
            spans = sorted((h[n]["data_offsets"][0], h[n]["data_offsets"][1], n) for n in ns)
            groups, cur = [], [spans[0]]
            for s in spans[1:]:
                if s[0] - cur[-1][1] < GAP:
                    cur.append(s)
                else:
                    groups.append(cur)
                    cur = [s]
            groups.append(cur)
            for g in groups:
                a, b = g[0][0], max(s[1] for s in g)
                buf = self._range(f, base + a, base + b)
                self.fetched_bytes += b - a
                for s0, s1, n in g:
                    m = h[n]
                    dt = DTYPES[m["dtype"]]
                    t = torch.frombuffer(buf, dtype=dt, count=(s1 - s0) // dt.itemsize, offset=s0 - a) if s1 > s0 \
                        else torch.empty(0, dtype=dt)
                    out[n] = t.reshape(m["shape"])
        self.fetch_seconds += time.time() - t0
        return out

    def _start(self, key, names):
        box = {}

        def work():
            try:
                box["t"] = self._fetch(names)
            except Exception as e:
                box["e"] = e

        th = threading.Thread(target=work, daemon=True)
        th.start()
        self.pending[key] = (th, box)

    def _next_key(self, names):
        """The block after the one these names belong to (the embeddings are followed by block 0)."""
        p = PREFIX
        n = names[0]
        if n == f"{p}embed_tokens.weight":
            return 0
        if n.startswith(f"{p}layers."):
            return int(n[len(f"{p}layers."):].split(".")[0]) + 1
        return None

    def get(self, names, device):
        key = tuple(sorted(names))
        if key in self.pending:
            th, box = self.pending.pop(key)
            th.join()
            if "e" in box:
                raise box["e"]
            cpu = box["t"]
        else:
            cpu = self._fetch(names)
        nxt = self._next_key(list(key))
        if nxt is not None and nxt < self.n_blocks:
            nn = self.layer_names(nxt)
            k2 = tuple(sorted(nn))
            if nn and k2 not in self.pending and len(self.pending) < 2:
                self._start(k2, nn)
        return {n: t.to(device) for n, t in cpu.items()}

    @property
    def n_blocks(self):
        return self.stop_block if getattr(self, "stop_block", None) else 64

    def layer_names(self, i):
        p = f"{PREFIX}layers.{i}."
        return [n for n in self.map if n.startswith(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, help="lines '<traces.jsonl> <ids file or ->'")
    ap.add_argument("--model", required=True, help="directory with config, tokenizer and model.safetensors.index.json")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- then extract_goodfire.py arguments (--out, --batch-tokens ...)")
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    layers = [26, 38, 51]
    if "--layers" in rest:
        layers = [int(x) for x in rest[rest.index("--layers") + 1].split(",")]

    shared = {}

    def factory(mdir):
        if "w" not in shared:
            shared["w"] = RemoteWeights(mdir)
            shared["w"].stop_block = max(layers)
        return shared["w"]

    extract_goodfire.Weights = factory
    for line in open(a.queue):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        traces, ids = line.split()[:2]
        argv = ["extract_goodfire", "--traces", traces, "--model", a.model] + (["--ids-file", ids] if ids != "-" else []) + rest
        print(f"=== {time.strftime('%FT%TZ', time.gmtime())} job {traces} ids {ids}", flush=True)
        sys.argv = argv
        extract_goodfire.main()
        w = shared.get("w")
        if w:
            print(f"=== fetched {w.fetched_bytes / 1e9:.1f} GB in {w.fetch_seconds:.0f} s of fetch-thread time", flush=True)
    print("queue done", flush=True)


if __name__ == "__main__":
    main()
