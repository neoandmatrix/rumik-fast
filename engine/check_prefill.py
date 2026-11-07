"""int4 prefill (step 8) vs the old dequantize-to-bf16 prefill: same prompt, compare the KV cache it writes, the
final hidden state, the sampling distribution of the first audio token, and the time.
  ../../04-gemma-stt-hindi/.venv/bin/python check_prefill.py
"""
import time

import torch
from transformers import AutoTokenizer

from rumik_fused import RumikFused
from rumik_int4 import WEIGHTS, RumikInt4

tok = AutoTokenizer.from_pretrained(WEIGHTS / "hf", trust_remote_code=True)
eng = RumikFused(max_seq=512, max_new_tokens=64)
for text in ["Your loan application has been approved.",
             "नमस्ते! आपकी किस्त की तारीख अगले सोमवार है, कृपया समय पर भुगतान कर दीजिए। बहुत समय पहले, एक छोटे से गाँव में रामू नाम का एक किसान रहता था।"]:
    ids = tok(f'<text>Ira: <description="calm, steady pace"> {text}<audio>').input_ids
    T = len(ids)
    out = {}
    for name, fn in (("old", lambda: RumikInt4.prefill(eng, ids)), ("int4", lambda: eng.prefill(ids))):
        fn(); torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / 3 * 1e3
        logits = (eng.head_w @ eng.hn).float() / eng.temperature
        v, i = logits.topk(eng.top_k)
        p = torch.zeros_like(logits).scatter_(0, i, torch.softmax(v, 0))
        out[name] = (ms, eng.kc[:, :, :T].clone(), eng.hn.clone(), p)
    (t0, k0, h0, p0), (t1, k1, h1, p1) = out["old"], out["int4"]
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
    kl = (p0 * ((p0 + 1e-12) / (p1 + 1e-12)).log()).sum().item()
    print(f"T={T:3d}: prefill {t0:6.1f} ms -> {t1:5.1f} ms | K cache diff {rel(k1, k0):.4f}, hidden {rel(h1, h0):.4f}, "
          f"KL first token {kl:.5f}, top-1 same: {bool(p0.argmax() == p1.argmax())}", flush=True)
