"""Tune the GEMV configs inside the real decode step (not in isolation: the winners differ), and save them to
../kernels/gemv_configs.json. One shape at a time (coordinate descent), each candidate timed in the full CUDA graph.
  ../../04-gemma-stt-hindi/.venv/bin/python tune_step.py     (~3 min, performance mode)
"""
import json
import time

import torch

from rumik_fused import RumikFused  # (puts ../kernels on the path)
import int4_gemv  # noqa: E402

SHAPES = [(25088, 2048), (2048, 2048), (2048, 11008)]
CANDS = [dict(BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=2)
         for bn in (8, 16, 32) for bk in (256, 512) for w in (2, 4)]

eng = RumikFused(max_new_tokens=200)


def step_ms(reps=250):
    eng.graphs = {}
    eng.capture()
    g = eng.graphs[eng.max_seq]
    for _ in range(50):
        eng.pos.fill_(300); eng.step_i.fill_(100); g.replay()
    best = 1e9
    for _ in range(2):                                  # best of 2 rounds
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(reps):
            eng.pos.fill_(300); eng.step_i.fill_(100); g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / reps * 1e3)
    return best


cur = {s: dict(BLOCK_N=16, BLOCK_K=512, num_warps=2, num_stages=2) for s in SHAPES}
int4_gemv._PINNED.clear(); int4_gemv._PINNED.update(cur)
print(f"start: {step_ms():.2f} ms/step", flush=True)
for s in SHAPES:
    res = []
    for c in CANDS:
        int4_gemv._PINNED[s] = c
        res.append((step_ms(), c))
    res.sort(key=lambda r: r[0])
    cur[s] = res[0][1]
    int4_gemv._PINNED[s] = cur[s]
    print(f"{s}: best {res[0][0]:.2f} ms/step with {res[0][1]}; worst {res[-1][0]:.2f}", flush=True)
final = step_ms()
print(f"final: {final:.2f} ms/step", flush=True)
out = {"gpu": torch.cuda.get_device_name(), "tuned_in": "full decode step (tune_step.py)", "step_ms": round(final, 2)}
out.update({f"{n}x{k}": c for (n, k), c in cur.items()})
int4_gemv._CFG_FILE.write_text(json.dumps(out, indent=1))
