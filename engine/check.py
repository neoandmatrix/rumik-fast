"""Is the engine's maths right? Compare one Cohere2 decoder layer and the prefill logits against the reference
implementation (transformers' Cohere2 code) running on the same dequantized int4 weights.
  ../../04-gemma-stt-hindi/.venv/bin/python check.py
"""
import torch
from transformers import Cohere2Config
from transformers.models.cohere2.modeling_cohere2 import Cohere2DecoderLayer, Cohere2RotaryEmbedding
from rumik_int4 import BF16, RumikInt4, cohere_ln, dequant, rope

eng = RumikInt4(kv_buckets=(256,))
cfg = Cohere2Config(**{k: v for k, v in eng.cfg.items() if k not in ("architectures", "auto_map", "model_type")})
cfg._attn_implementation = "eager"
rot = Cohere2RotaryEmbedding(cfg).to("cuda")
torch.manual_seed(0)
T = 12
x = torch.randn(1, T, eng.H, device="cuda", dtype=BF16)
pos_ids = torch.arange(T, device="cuda")[None]
pe = rot(x, pos_ids)
causal = torch.full((T, T), float("-inf"), device="cuda").triu(1)
for i in (0, 3):                                     # a sliding (rope) layer and a full (NoPE) layer
    layer = Cohere2DecoderLayer(cfg, i).to("cuda", BF16).eval()
    nq, nk = eng.nh * eng.hd, eng.nkv * eng.hd
    W = dequant(*eng.w_in[i])
    with torch.no_grad():
        sa, mlp = layer.self_attn, layer.mlp
        sa.q_proj.weight.copy_(W[:nq]); sa.k_proj.weight.copy_(W[nq:nq + nk]); sa.v_proj.weight.copy_(W[nq + nk:nq + 2 * nk])
        mlp.gate_proj.weight.copy_(W[nq + 2 * nk:nq + 2 * nk + eng.inter]); mlp.up_proj.weight.copy_(W[nq + 2 * nk + eng.inter:])
        sa.o_proj.weight.copy_(dequant(*eng.w_o[i])); mlp.down_proj.weight.copy_(dequant(*eng.w_down[i]))
        layer.input_layernorm.weight.copy_(eng.ln[i])
        ref = layer(x, position_embeddings=pe, attention_mask=causal[None, None].to(BF16))
        ref = ref[0] if isinstance(ref, tuple) else ref
        # ours (the prefill path, one layer)
        h = cohere_ln(x[0], eng.ln[i], eng.eps)
        y = h @ W.t()
        q = y[:, :nq].view(T, eng.nh, eng.hd).transpose(0, 1)
        k = y[:, nq:nq + nk].view(T, eng.nkv, eng.hd).transpose(0, 1)
        v = y[:, nq + nk:nq + 2 * nk].view(T, eng.nkv, eng.hd).transpose(0, 1)
        g, u = y[:, nq + 2 * nk:].split(eng.inter, -1)
        if eng.sliding[i]:
            q, k = rope(q, eng.cos[:T], eng.sin[:T]), rope(k, eng.cos[:T], eng.sin[:T])
        o = eng._attend(q, k, v, causal)
        ours = x[0] + o @ dequant(*eng.w_o[i]).t() + (torch.nn.functional.silu(g) * u) @ dequant(*eng.w_down[i]).t()
        d = ours.float() - ref[0].float()
        print(f"layer {i} ({'sliding' if eng.sliding[i] else 'full'}): rel diff {(d.norm() / (ref[0].float() - x[0].float()).norm()).item():.4f}")

# decode path vs prefill path: logits of token T computed by prefill(T+1 tokens) vs prefill(T) + one graph step
ids = list(range(1000, 1000 + T))
eng.prefill(ids + [eng.audio_ids[5].item()])
ref_logits = eng._logits_and_stop()[0].clone()
eng.prefill(ids)
eng.tok.fill_(eng.audio_ids[5].item())
eng._step(256)
step_logits = eng._logits_and_stop()[0]
print("decode step vs prefill logits: rel diff",
      round(((step_logits - ref_logits).norm() / ref_logits.norm()).item(), 4),
      "| top-1 same:", bool(step_logits.argmax() == ref_logits.argmax()))
