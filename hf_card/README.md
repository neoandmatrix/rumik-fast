---
license: cc-by-nc-4.0
license_link: https://cohere.com/cohere-labs-cc-by-nc-license
base_model: rumik-ai/rumik-oss-1
base_model_relation: quantized
pipeline_tag: text-to-speech
language:
- hi
- en
- te
- ta
- kn
- bn
- pa
tags:
- text-to-speech
- int4
- quantized
- rumik
- mimi
---

# rumik-oss-1-int4

A 4-bit weight-only quantization of [rumik-ai/rumik-oss-1](https://huggingface.co/rumik-ai/rumik-oss-1), a 3B
multilingual (Hindi, English, Indic) text-to-speech model with description-conditioned delivery and inline
`<laugh>` / `<chuckle>` / `<sigh>` tags. It is made for small GPUs: one 2.13 GB file that streams speech at
**~98 audio tokens/s on a 4 GB RTX 3050 Laptop GPU** (real time = 100 tok/s), with our own inference engine.

**Non-commercial use only.** This is a derivative of rumik-oss-1, which is CC-BY-NC 4.0 together with the
[Cohere Labs acceptable-use policy](https://docs.cohere.com/docs/cohere-labs-acceptable-use-policy) (it derives from
CohereLabs/tiny-aya-fire). The original model card says the licence does not cover commercial products, paid
synthesis services or self-hosted commercial deployments. `LICENSE` and `NOTICE` are kept from the original; `NOTICE`
records what was changed here.

## Files
| file | |
|---|---|
| `rumik-oss-1-int4.safetensors` | 2.13 GB: int4 layers + int8 embedding + bf16 audio head + norms + stop predictor |
| `export.json` | format, group size, and the relative quantization error of every matrix |
| `LICENSE`, `NOTICE` | from rumik-oss-1; `NOTICE` lists the modifications |

This is **not** a transformers checkpoint. It is a custom int4 layout for a dedicated engine (below). Tokenizer,
config and the Mimi codec come from the original repository.

## Format (`rumik-int4-g128-minmax-v1`)
- `model.layers.{i}.{self_attn.q/k/v/o_proj, mlp.gate/up/down_proj}.{qweight, scale, min}`: `qweight` is uint8
  `[N, K/2]` (two 4-bit codes per byte, low nibble = even column); `scale` and `min` are fp16 `[N, K/128]`.
  Dequantize: `w[n, k] = code[n, k] * scale[n, k // 128] + min[n, k // 128]`.
- `model.embed_tokens.{qweight (int8 [V, 2048]), scale (fp16 [V])}`.
- `head.audio_weight` (bf16 `[16385, 2048]`) and `head.audio_ids` (the token id of each row); the head is tied to
  the embedding in the original model.
- Layer norms (`model.layers.{i}.input_layernorm.weight`, `model.norm.weight`) and `stop_predictor.*`, unchanged.

## Quality
- Weights: mean relative error 10.5% (the worst are the late `v_proj`, ~15%); embedding 1.0%.
- On a cloud L4 with torchao's int4 kernel, the same recipe kept the CER of bf16 (Indic-Transcribe as the judge:
  English 0.000, Hindi 0.073 for both bf16 and int4).
- Our engine on the laptop: listening tests rated ~9/10 against the original. Known issues of the base model remain
  (for example, digits are read one by one unless the text spells them out).

## Running it
The engine: int4 CUDA/Triton kernels, CUDA graphs, flash-decoding, an int4 prefill matmul, exact streaming Mimi and a
local web UI, at [github.com/neoandmatrix/rumik-fast](https://github.com/neoandmatrix/rumik-fast) (private for now).
`python download_weights.py` fetches this file plus the original tokenizer/config/codec.

| RTX 3050 Laptop (4 GB) | |
|---|---|
| decode | 95–98 tok/s |
| prefill, 34-token prompt | ~50 ms |
| first audio frame | ~130–190 ms |
| VRAM | ~2.5 GB (with Mimi) |

## Credits
rumik-oss-1 by [Rumik AI](https://huggingface.co/rumik-ai), built on [tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire)
(Cohere Labs) and [Mimi](https://huggingface.co/kyutai/mimi) (Kyutai, CC-BY-4.0). Quantization and engine by
bringagi / neoandmatrix.
