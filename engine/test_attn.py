"""Decode attention kernels vs plain fp32 attention, on a random KV cache at several positions; plus the time
per call (one layer). attn = one program per query head; attn_split + attn_combine = flash-decoding.
  uv run python test_attn.py
"""
import torch
import triton.testing as tt

import fused
from rumik_fused import BT, NSPLIT

NH, NKV, HD, S = 16, 4, 128, 2048
REP = NH // NKV
torch.manual_seed(0)
kc = torch.randn(NKV, S, HD, device="cuda").to(torch.bfloat16)
vc = torch.randn(NKV, S, HD, device="cuda").to(torch.bfloat16)
q = torch.randn(NH * HD, device="cuda").to(torch.bfloat16)
o1, o2 = (torch.empty(NH * HD, device="cuda", dtype=torch.bfloat16) for _ in range(2))
po = torch.empty(NH, NSPLIT, HD, device="cuda")
pm = torch.empty(NH, NSPLIT, device="cuda")
pl = torch.empty(NH, NSPLIT, device="cuda")
pos = torch.zeros(1, dtype=torch.long, device="cuda")
scale = HD ** -0.5


def ref():
    n = int(pos) + 1
    k = kc[:, :n].float().repeat_interleave(REP, 0)                     # [NH, n, HD]
    v = vc[:, :n].float().repeat_interleave(REP, 0)
    p = torch.softmax((k @ q.float().view(NH, HD, 1)).squeeze(-1) * scale, -1)
    return (p.unsqueeze(1) @ v).view(-1)


def one():
    fused.attn[(NH,)](q, kc, vc, pos, o1, S, scale, REP=REP, HD=HD, BT=64, num_warps=4)


def split(bf=False):
    fused.attn_split[(NKV, NSPLIT)](q, kc, vc, pos, po, pm, pl, S, scale, REP=REP, HD=HD, NSPLIT=NSPLIT, BT=BT,
                                    num_warps=2, PV_BF16=bf)
    fused.attn_combine[(NH,)](po, pm, pl, pos, o2, HD=HD, NSPLIT=NSPLIT, BT=BT, num_warps=4)


for p in (0, 31, 100, 300, 777, 1500, 2047):
    pos.fill_(p)
    r = ref()
    split(True)
    e3 = ((o2.float() - r).norm() / r.norm()).item()
    one(); split()
    e1 = ((o1.float() - r).norm() / r.norm()).item()
    e2 = ((o2.float() - r).norm() / r.norm()).item()
    t1 = tt.do_bench(one, warmup=20, rep=100) * 1e3
    t2 = tt.do_bench(split, warmup=20, rep=100) * 1e3
    print(f"pos {p:5d}: error one {e1:.4f} split {e2:.4f} | {t1:6.1f} us -> {t2:5.1f} us per layer "
          f"| split with bf16 p@v: error {e3:.4f} (x36 = {t1 * 36 / 1e3:.2f} -> {t2 * 36 / 1e3:.2f} ms/step)", flush=True)
