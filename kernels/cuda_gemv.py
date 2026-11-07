"""Hand-written CUDA int4 GEMV (batch 1) for the layout of ../modal_export_int4.py, compiled on first use with
torch's load_inline (nvcc; cached in ~/.cache/torch_extensions).

Why CUDA and not Triton: the Triton kernels spend ~6 instructions per weight (SASS, sass_count.py): byte extracts,
int->float conversions, separate multiplies and adds. 16 SMs cannot issue that fast enough to keep up with
~150 GB/s (300 G weights/s), so v1 stalls at ~112 GB/s while a read-only kernel with the same loads reaches 153.
Here each 32-bit word of codes (8 weights) costs ~13 instructions, ~1.6 per weight:

  (w & 0x000F000F) | 0x64006400  -> two fp16 numbers 1024+c0, 1024+c4 in ONE instruction (LOP3). fp16 1024.0 is
                                    0x6400 and its last mantissa bit is worth 1, so OR-ing a 4-bit code in adds it.
  hsub2(that, 1024)              -> c0, c4 (exact)
  (w & 0x00F000F0) | 0x64006400  -> 1024 + 16*c1, 1024 + 16*c5 (no shift needed) ...
  hfma2(that, 1/16, -64)         -> c1, c5 (exact)
  w >> 8, same again              -> (c2, c6), (c3, c7)
  4 x hfma2(codes, x pairs, acc) -> the multiply-adds, two weights per instruction

So x must sit in shared memory in the same pair order: (x0,x4), (x1,x5), (x2,x6), (x3,x7) for every 8 columns.
Each lane's 32-weight fp16 partial sum (16 products per half, |sum| < ~2000 for Rumik's activations, far from
fp16's 65504) goes into an fp32 accumulator per row as  scale * dot + min * sum(x over those 32 columns).

Block: 8 warps x 2 rows = 16 rows. The x preparation (bf16 -> fp16 pairs + per-group sums) is done once per block.
"""
import torch
from torch.utils.cpp_extension import load_inline

_CUDA = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define WARPS 8
#define ROWS 2                 // rows per warp
#define GROUP 128

__device__ __forceinline__ half2 as_h2(uint32_t v) { return *reinterpret_cast<half2*>(&v); }

template <int WT, int RT>
__global__ void __launch_bounds__(WT * 32)
gemv_int4_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ q,
                 const half* __restrict__ sc, const half* __restrict__ mn,
                 __nv_bfloat16* __restrict__ y, int N, int K) {
    extern __shared__ uint8_t smem[];
    half2* xp = reinterpret_cast<half2*>(smem);                       // K/2 pairs, in (x0,x4),(x1,x5).. order
    float* xs = reinterpret_cast<float*>(smem + (size_t)K * 2);       // K/32 chunk sums (a lane's 32 columns)
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n8 = K / 8;

    // ---- x: bf16 -> fp16 pairs + per-chunk sums. Thread t takes 8-column blocks t, t+256, ...; 4 consecutive
    // blocks make one 32-column chunk, so 4 neighbouring lanes reduce one chunk's sum. (Per chunk, not per group:
    // each lane adds min * sum(x) for ITS 32 columns only; a group of 128 is split over 4 lanes.)
    for (int j0 = 0; j0 < n8; j0 += WT * 32) {
        int j = j0 + tid;
        float s = 0.f;
        if (j < n8) {
            uint4 raw = *reinterpret_cast<const uint4*>(x + (size_t)j * 8);
            const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
            float f[8];
            #pragma unroll
            for (int t = 0; t < 8; t++) { f[t] = __bfloat162float(b[t]); s += f[t]; }
            #pragma unroll
            for (int t = 0; t < 4; t++) xp[j * 4 + t] = __floats2half2_rn(f[t], f[t + 4]);
        }
        #pragma unroll
        for (int o = 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if ((lane & 3) == 0 && j < n8) xs[j / 4] = s;
    }
    __syncthreads();

    const int row0 = (blockIdx.x * WT + warp) * RT;
    const int nchunk = K / 32;                                        // 16-byte chunks per row
    const int ng = K / GROUP;
    const half2 h1024 = __float2half2_rn(1024.f), h16 = __float2half2_rn(1.f / 16.f),
                hm64 = __float2half2_rn(-64.f);
    float out[RT];
    #pragma unroll
    for (int r = 0; r < RT; r++) out[r] = 0.f;

    for (int c = lane; c < nchunk; c += 32) {
        uint4 w4[RT];
        #pragma unroll
        for (int r = 0; r < RT; r++)                                // issue all row loads first (in flight)
            w4[r] = __ldcs(reinterpret_cast<const uint4*>(q + (size_t)(row0 + r) * (K / 2)) + c);
        const half2* xc = xp + c * 16;                                // 32 columns = 16 pairs
        const int g = c >> 2;                                         // 4 chunks per group
        const float xsum = xs[c];
        #pragma unroll
        for (int r = 0; r < RT; r++) {
            const uint32_t* w = reinterpret_cast<const uint32_t*>(&w4[r]);
            half2 acc = __float2half2_rn(0.f);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint32_t a = w[i], b = w[i] >> 8;
                half2 c04 = __hsub2(as_h2((a & 0x000F000F) | 0x64006400), h1024);
                half2 c15 = __hfma2(as_h2((a & 0x00F000F0) | 0x64006400), h16, hm64);
                half2 c26 = __hsub2(as_h2((b & 0x000F000F) | 0x64006400), h1024);
                half2 c37 = __hfma2(as_h2((b & 0x00F000F0) | 0x64006400), h16, hm64);
                acc = __hfma2(c04, xc[i * 4 + 0], acc);
                acc = __hfma2(c15, xc[i * 4 + 1], acc);
                acc = __hfma2(c26, xc[i * 4 + 2], acc);
                acc = __hfma2(c37, xc[i * 4 + 3], acc);
            }
            float dot = __low2float(acc) + __high2float(acc);
            size_t gi = (size_t)(row0 + r) * ng + g;
            out[r] += __half2float(sc[gi]) * dot + __half2float(mn[gi]) * xsum;
        }
    }
    #pragma unroll
    for (int r = 0; r < RT; r++) {
        float v = out[r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0 && row0 + r < N) y[row0 + r] = __float2bfloat16(v);
    }
}

// block shape per call: WT warps per block, RT rows per warp (tuned per matrix shape, see bench_cuda.py)
template <int WT, int RT>
static void launch_int4(torch::Tensor x, torch::Tensor q, torch::Tensor s, torch::Tensor m, torch::Tensor y) {
    int N = q.size(0), K = q.size(1) * 2;
    TORCH_CHECK(N % (WT * RT) == 0 && K % GROUP == 0, "shape");
    size_t smem = (size_t)K * 2 + (K / 32) * 4;
    auto stream = at::cuda::getCurrentCUDAStream();
    gemv_int4_kernel<WT, RT><<<N / (WT * RT), WT * 32, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), q.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(s.data_ptr()), reinterpret_cast<const half*>(m.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), N, K);
}

void gemv_int4(torch::Tensor x, torch::Tensor q, torch::Tensor s, torch::Tensor m, torch::Tensor y,
               int64_t warps, int64_t rows) {
#define CASE(W, R) if (warps == W && rows == R) return launch_int4<W, R>(x, q, s, m, y);
    CASE(8, 2) CASE(8, 1) CASE(8, 4) CASE(4, 1) CASE(4, 2) CASE(4, 4) CASE(16, 1) CASE(16, 2)
#undef CASE
    TORCH_CHECK(false, "unsupported block shape");
}

// ---- int8 GEMV with one fp32 scale per row (the audio head). Weights stored as uint8 = int8 + 128.
// PRMT puts each byte b under 0x64 (fp16 1024.0, last mantissa bit = 1): half(1024 + b); minus 1152 = b - 128 =
// the signed weight, exactly. x pairs are natural here: (x0,x1), (x2,x3), ...
__global__ void __launch_bounds__(WARPS * 32)
gemv_int8_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ q, const float* __restrict__ sc,
                 __nv_bfloat16* __restrict__ y, int N, int K) {
    extern __shared__ uint8_t smem8[];
    half2* xp = reinterpret_cast<half2*>(smem8);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int j = tid; j < K / 2; j += WARPS * 32) {
        __nv_bfloat162 b = reinterpret_cast<const __nv_bfloat162*>(x)[j];
        xp[j] = __floats2half2_rn(__low2float(b), __high2float(b));
    }
    __syncthreads();
    const int row0 = (blockIdx.x * WARPS + warp) * ROWS;
    const int nchunk = K / 16;                                        // 16 weights per 16-byte load
    const half2 h1152 = __float2half2_rn(1152.f);
    float out[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) out[r] = 0.f;
    for (int c = lane; c < nchunk; c += 32) {
        uint4 w4[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            w4[r] = __ldcs(reinterpret_cast<const uint4*>(q + (size_t)(row0 + r) * K) + c);
        const half2* xc = xp + c * 8;
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const uint32_t* w = reinterpret_cast<const uint32_t*>(&w4[r]);
            half2 acc = __float2half2_rn(0.f);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint32_t lo, hi;
                asm("prmt.b32 %0, %1, %2, 0x4140;" : "=r"(lo) : "r"(w[i]), "r"(0x64646464u));
                asm("prmt.b32 %0, %1, %2, 0x4342;" : "=r"(hi) : "r"(w[i]), "r"(0x64646464u));
                acc = __hfma2(__hsub2(as_h2(lo), h1152), xc[i * 2 + 0], acc);
                acc = __hfma2(__hsub2(as_h2(hi), h1152), xc[i * 2 + 1], acc);
            }
            out[r] += __low2float(acc) + __high2float(acc);
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        float v = out[r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0 && row0 + r < N) y[row0 + r] = __float2bfloat16(v * sc[row0 + r]);
    }
}

void gemv_int8(torch::Tensor x, torch::Tensor q, torch::Tensor s, torch::Tensor y) {
    int N = q.size(0), K = q.size(1);
    TORCH_CHECK(N % (WARPS * ROWS) == 0 && K % 512 == 0, "shape");
    auto stream = at::cuda::getCurrentCUDAStream();
    gemv_int8_kernel<<<N / (WARPS * ROWS), WARPS * 32, (size_t)K * 2, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), q.data_ptr<uint8_t>(), s.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), N, K);
}
"""

_mod = None


def _arch_flags():
    """compile for the GPU in this machine (sm_86 = RTX 30xx / A10 / A40, sm_89 = RTX 40xx / L4, sm_80 = A100,
    sm_90 = H100). The kernel needs sm_80+ (bf16 conversions)."""
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError(f"GPU compute capability {major}.{minor}: the CUDA GEMV needs 8.0+ (Ampere or newer)")
    return [f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"], f"sm{major}{minor}"


def _load():
    global _mod
    if _mod is None:
        arch, tag = _arch_flags()
        _mod = load_inline(f"rumik_int4_gemv_{tag}", cpp_sources="void gemv_int4(torch::Tensor x, torch::Tensor q, "
                           "torch::Tensor s, torch::Tensor m, torch::Tensor y, int64_t warps, int64_t rows);\nvoid gemv_int8(torch::Tensor x, "
                           "torch::Tensor q, torch::Tensor s, torch::Tensor y);", cuda_sources=_CUDA,
                           functions=["gemv_int4", "gemv_int8"], extra_cuda_cflags=["-O3", "-lineinfo"] + arch,
                           verbose=False)
    return _mod


# block shape (warps per block, rows per warp) per (N, K): from bench_cuda.py --shapes on the 3050
SHAPE = {}


def gemv_int4_cuda(x, q, s, m, out=None, warps=None, rows=None):
    """same contract as int4_gemv.gemv_int4 (x bf16 [K] -> y bf16 [N])"""
    y = out if out is not None else torch.empty(q.shape[0], device=x.device, dtype=torch.bfloat16)
    w, r = SHAPE.get((q.shape[0], q.shape[1] * 2), (8, 2))
    _load().gemv_int4(x, q, s, m, y, warps or w, rows or r)
    return y


def quantize_int8_rows(w, pad_to=16):
    """bf16 [N, K] -> (uint8 [N', K] = int8 + 128, fp32 scale [N']); rows padded with zeros to a multiple of pad_to.
    Symmetric per row: w ~ (q - 128) * scale, scale = max|w| / 127."""
    n = w.shape[0]
    pad = (-n) % pad_to
    w = torch.cat([w, w.new_zeros(pad, w.shape[1])]) if pad else w
    s = (w.float().abs().amax(1) / 127).clamp(min=1e-8)
    q = ((w.float() / s[:, None]).round().clamp(-127, 127) + 128).to(torch.uint8).contiguous()
    return q, s.contiguous()


def gemv_int8_cuda(x, q, s, out=None):
    """x bf16 [K] -> y bf16 [N'] = ((q - 128) * s) @ x"""
    y = out if out is not None else torch.empty(q.shape[0], device=x.device, dtype=torch.bfloat16)
    _load().gemv_int8(x, q, s, y)
    return y
