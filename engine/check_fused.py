"""Fused step vs the original step, from the same state (same prompt, same cache, same random seed):
compares the final hidden state, the K/V written into the cache, and the sampled token, over a few steps.
  ../../04-gemma-stt-hindi/.venv/bin/python check_fused.py
"""
import torch
from transformers import AutoTokenizer

from rumik_fused import RumikFused
from rumik_int4 import WEIGHTS, RumikInt4

tok = AutoTokenizer.from_pretrained(WEIGHTS / "hf", trust_remote_code=True)
eng = RumikFused(max_seq=512, max_new_tokens=64)
ids = tok('<text>Ira: <description="calm, steady pace"> Checking the fused kernels.<audio>').input_ids


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


with torch.inference_mode():
    eng.prefill(ids)
    for n in range(4):                                     # 4 steps; each compared from an identical start state
        state = [t.clone() for t in (eng.kc, eng.vc, eng.pos, eng.step_i, eng.tok, eng.out)]
        p = int(eng.pos)
        torch.manual_seed(n)
        RumikInt4._step(eng, 512)
        ref = (eng.hn.clone(), eng.kc[:, :, p].clone(), eng.vc[:, :, p].clone(), int(eng.tok))
        for dst, src in zip((eng.kc, eng.vc, eng.pos, eng.step_i, eng.tok, eng.out), state):
            dst.copy_(src)
        torch.manual_seed(n)
        eng._step()
        new = (eng.hn.clone(), eng.kc[:, :, p].clone(), eng.vc[:, :, p].clone(), int(eng.tok))
        print(f"step {n} pos {p}: hidden rel diff {rel(new[0], ref[0]):.4f}  K {rel(new[1], ref[1]):.4f}  "
              f"V {rel(new[2], ref[2]):.4f}  token {new[3]} vs {ref[3]} {'OK' if new[3] == ref[3] else 'DIFFERENT'}")
