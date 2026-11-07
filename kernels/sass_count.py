"""Count the GPU instructions (SASS) Triton generates for the int4 GEMV kernels, to see the work per weight.
  ../../04-gemma-stt-hindi/.venv/bin/python sass_count.py
"""
import collections
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from int4_gemv import G, _gemv_int4, _gemv_int4_magic, quantize  # noqa: E402

n, k = 25088, 2048
q, s, m = quantize((torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16))
x = torch.randn(k, device="cuda", dtype=torch.bfloat16)
y = torch.empty(n, device="cuda", dtype=torch.bfloat16)
BN, BK, W = 32, 512, 2
kernels = {
    "v1": _gemv_int4.fn[(n // BN,)](x, q, s, m, y, n, k, GROUP=G, BLOCK_N=BN, BLOCK_K=BK, num_warps=W, num_stages=2),
    "v4": _gemv_int4_magic[(n // BN, 1)](x, q, s, m, y, n, k, GROUP=G, BLOCK_N=BN, BLOCK_K=BK, SPLIT_K=1,
                                          num_warps=W, num_stages=2),
}
weights_per_iter_per_thread = BN * BK / (W * 32)
for name, ck in kernels.items():
    sass = ck.asm["sass"]
    ops = [re.split(r"[ .;]", ln.split("*/")[-1].strip())[0] for ln in sass.splitlines()
           if re.match(r"\s*/\*[0-9a-f]{4}\*/", ln)]
    ops = [o for o in ops if o and o[0].isupper()]
    c = collections.Counter(ops)
    total = sum(c.values())
    print(f"{name}: {total} instructions in the kernel; {weights_per_iter_per_thread:.0f} weights per thread per loop "
          f"iteration (the loop body is most of the kernel)")
    print("   ", ", ".join(f"{o} {v}" for o, v in c.most_common(18)))
    (Path(__file__).parent / f"sass_{name}.txt").write_text(sass)
