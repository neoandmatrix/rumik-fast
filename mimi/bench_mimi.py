"""How fast is Rumik's Mimi decoder (8 codebooks, 12.5 Hz, 24 kHz) on this laptop's CPU vs its GPU?
Real time needs one 80 ms frame decoded in < 80 ms; streaming decodes one frame at a time.
  ../../04-gemma-stt-hindi/.venv/bin/python bench_mimi.py
"""
import inspect
import json
import time
from pathlib import Path

import torch
from transformers import MimiModel

CODEC = Path(__file__).parents[1] / "weights/hf/codec"
mimi = MimiModel.from_pretrained(CODEC).eval()
print("decode signature:", inspect.signature(mimi.decode))
torch.manual_seed(0)
res = {}


def run(dev, dtype, threads=None):
    if threads:
        torch.set_num_threads(threads)
    m = MimiModel.from_pretrained(CODEC).eval().to(dev, dtype)   # fresh: a codebook cache doesn't follow .to()
    codes1 = torch.randint(0, 2048, (1, 8, 1), device=dev)
    codes100 = torch.randint(0, 2048, (1, 8, 100), device=dev)
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    with torch.inference_mode():
        for _ in range(3):
            m.decode(codes1); m.decode(codes100)
        sync(); t = time.perf_counter()
        for _ in range(20):
            m.decode(codes1)
        sync(); one = (time.perf_counter() - t) / 20 * 1e3
        sync(); t = time.perf_counter()
        for _ in range(3):
            m.decode(codes100)
        sync(); hundred = (time.perf_counter() - t) / 3 / 100 * 1e3
    key = f"{dev}_{str(dtype).split('.')[-1]}" + (f"_{threads}t" if threads else "")
    res[key] = {"one_frame_ms": round(one, 1), "per_frame_in_batch_of_100_ms": round(hundred, 2)}
    print(key, res[key], flush=True)


for th in (6, 12):
    run("cpu", torch.float32, th)
run("cuda", torch.float32)
run("cuda", torch.float16)
(Path(__file__).parent / "results_mimi_3050.json").write_text(json.dumps(res, indent=1))
