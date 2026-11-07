"""Does a kernel change move the model's predictions? Over 40 decode steps, run the step with the Triton GEMV and
with the CUDA GEMV from the same state, and compare the sampling distributions (softmax(logits / 0.8) over the
top-30): KL divergence, and how often the most likely token agrees. A sampled token can differ on a near-tie even
when the distributions match, so this is the fair test.
  ../../04-gemma-stt-hindi/.venv/bin/python check_dist.py
"""
import torch
from transformers import AutoTokenizer

from rumik_fused import RumikFused
from rumik_int4 import WEIGHTS, gemv_int4

tok = AutoTokenizer.from_pretrained(WEIGHTS / "hf", trust_remote_code=True)
eng = RumikFused(max_seq=512, max_new_tokens=64)
cuda_gemv = eng.gemv
ids = tok('<text>Aisha: <description="calm, Hindi accent, steady pace"> Haan ji, aapka EMI ka payment mil gaya hai.<audio>').input_ids


def dist():
    logits = (eng.head_w @ eng.hn).float() / eng.temperature
    v, i = logits.topk(eng.top_k)
    return torch.zeros_like(logits).scatter_(0, i, torch.softmax(v, 0))


kls, agree = [], 0
with torch.inference_mode():
    eng.prefill(ids)
    for n in range(40):
        state = [t.clone() for t in (eng.kc, eng.vc, eng.pos, eng.step_i, eng.tok, eng.out)]
        eng.gemv = gemv_int4
        torch.manual_seed(n); eng._step()
        p = dist()
        for dst, src in zip((eng.kc, eng.vc, eng.pos, eng.step_i, eng.tok, eng.out), state):
            dst.copy_(src)
        eng.gemv = cuda_gemv
        torch.manual_seed(n); eng._step()                   # continue along the CUDA path
        r = dist()
        kls.append((p * ((p + 1e-12) / (r + 1e-12)).log()).sum().item())
        agree += int(p.argmax() == r.argmax())
kl = torch.tensor(kls)
print(f"40 steps: KL(triton || cuda) mean {kl.mean():.5f}, max {kl.max():.5f} nats; top-1 agrees {agree}/40")
