"""Profile the real decode step (max_seq 512, pos 300; ENGINE=old|fused) under nsys, per kernel:
  nsys profile --cuda-graph-trace=node -o out/step -f true uv run python prof_engine.py
  nsys stats -r cuda_gpu_kern_sum --filter-nvtx=steps out/step.nsys-rep   (only the timed steps)
Without nsys it just prints ms/step (run in performance mode)."""
import os
import time

import torch
from transformers import AutoTokenizer

from rumik_fused import RumikFused
from rumik_int4 import WEIGHTS, RumikInt4

FUSED = os.environ.get("ENGINE", "fused") == "fused"     # ENGINE=old for the unfused step

tok = AutoTokenizer.from_pretrained(WEIGHTS / "hf", trust_remote_code=True)
eng = RumikFused(max_seq=512, max_new_tokens=400) if FUSED else RumikInt4(kv_buckets=(512,), max_seq=512, max_new_tokens=400)
eng.capture()
ids = tok('<text>Ira: <description="calm, steady pace"> Profiling the decode step today.<audio>').input_ids
with torch.inference_mode():
    eng.prefill(ids)
g = eng.graphs[512]
for _ in range(50):
    g.replay()
torch.cuda.synchronize()
N = 300
torch.cuda.nvtx.range_push("steps")
t = time.perf_counter()
for _ in range(N):
    eng.pos.fill_(300)            # keep the position fixed so the cache never overflows
    g.replay()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
print(f"{(time.perf_counter() - t) / N * 1e3:.2f} ms/step  ({N} steps, bucket 512)")
