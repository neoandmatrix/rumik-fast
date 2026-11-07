"""Step 10 checks.
1. Fused sampler vs the PyTorch one on the same logits: the same top-30 *values* (=> the same sampling
   distribution; with bf16 logits, ties at equal values are common and the two sorts may order them differently),
   the same stop decision, and, when the top-30 has no ties, the same token for the same uniform random numbers.
2. int8 audio head (CUDA kernel) vs bf16 head along a real generation: KL of the sampling distributions, top-1 agreement.
  ../../04-gemma-stt-hindi/.venv/bin/python check_sample.py
"""
import torch
from transformers import AutoTokenizer

import sample
from rumik_fused import RumikFused
from rumik_int4 import WEIGHTS, RumikInt4

tok = AutoTokenizer.from_pretrained(WEIGHTS / "hf", trust_remote_code=True)
eng = RumikFused(max_seq=512, max_new_tokens=128, head="bf16")
ids = tok('<text>Aisha: <description="calm, Hindi accent, steady pace"> Haan ji, aapka EMI ka payment mil gaya hai.<audio>').input_ids

# ---- 1. sampler parity
ok_set = ok_tok = ok_stop = n_untied = 0
N = 40
with torch.inference_mode():
    eng.prefill(ids)
    for n in range(N):
        eng._step()                                           # advance with the fused path
        for step_val in (3, 20):                              # early (</audio> masked) and late
            eng.step_i.fill_(step_val)
            logits, stop = RumikInt4._logits_and_stop(eng)
            early = torch.tensor(step_val < eng.min_new, device="cuda")
            lg = torch.where(early & eng.is_end, float("-inf"), logits)
            tv, ti = lg.topk(eng.top_k)
            s = eng.sampler
            logits_b = s.logits_fn()
            sample.stop_hidden[(s.nz // 16,)](eng.hn, s.lnw, s.lnb, s.w1, s.b1, s.z1, 1e-5, H=eng.H, ROWS=16)
            sample.topk_partial[(s.nch,)](logits_b, eng.step_i, s.keys, s.nv, s.end_idx, eng.min_new, CH=sample.CH)
            s.u.uniform_()
            save = eng.out.clone(), eng.tok.clone()
            sample.pick[(1,)](s.keys, s.scratch, s.u, s.z1, s.w2, s.b2, eng.audio_ids, eng.step_i, eng.pos, eng.tok,
                              eng.out, 1.0 / eng.temperature, eng.end_id, eng.min_new, s.nch, CH=sample.CH,
                              KEEP=sample.KEEP, NC=s.nc, TOPK=eng.top_k, NZ=s.nz)
            fused_tok = int(eng.tok)
            eng.out.copy_(save[0]); eng.tok.copy_(save[1])
            top_idx = (s.scratch[:eng.top_k] & 0xFFFFFFFF).long()
            ok_set += int(torch.equal(lg[top_idx].sort().values, tv.sort().values))
            untied = tv.unique().numel() == eng.top_k and bool(lg[top_idx].min() > lg.sort(descending=True).values[eng.top_k])
            # the PyTorch sampler with the same uniforms (bf16 logits here, like the fused path)
            g = -torch.log(-torch.log(s.u[:eng.top_k].clamp(1e-10, 1.0)))
            ref_tok = int(eng.audio_ids[ti[torch.argmax(tv / eng.temperature + g)]])
            if bool(stop) and step_val >= eng.min_new:
                ref_tok = eng.end_id
            if untied:
                n_untied += 1
                ok_tok += int(fused_tok == ref_tok)
            zl = (s.z1 * s.w2).sum() + s.b2
            ok_stop += int(bool(zl > 0) == bool(stop))
        eng.step_i.fill_(n + 1)
print(f"sampler parity over {2 * N} cases: top-30 values equal {ok_set}, same stop {ok_stop}; "
      f"same token in {ok_tok}/{n_untied} cases without ties")

# ---- 2. int8 head (the CUDA kernel the engine uses) vs bf16 head
from cuda_gemv import gemv_int8_cuda, quantize_int8_rows  # noqa: E402
hq = quantize_int8_rows(eng.head_w)
lb = torch.empty(hq[0].shape[0], device="cuda", dtype=torch.bfloat16)


def dist(logits):
    v, i = logits.float().topk(eng.top_k)
    return torch.zeros(logits.shape[0], device="cuda").scatter_(0, i, torch.softmax(v / eng.temperature, 0))


kls, agree = [], 0
with torch.inference_mode():
    eng.prefill(ids)
    for n in range(80):
        eng._step()
        p = dist(torch.mv(eng.head_w, eng.hn))
        q = dist(gemv_int8_cuda(eng.hn, *hq, out=lb)[:eng.head_w.shape[0]])
        kls.append((p * ((p + 1e-12) / (q + 1e-12)).log()).sum().item())
        agree += int(p.argmax() == q.argmax())
kl = torch.tensor(kls)
print(f"int8 head vs bf16 head, 80 steps: KL mean {kl.mean():.4f} max {kl.max():.4f} nats; top-1 same {agree}/80 "
      f"(for scale: Triton vs CUDA GEMV in the layers was KL 0.007)")
