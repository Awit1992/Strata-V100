// src/kernels/cuda/kv_tq.cu - see include/strata/kernels/kv_tq.hpp.  TurboQuant K/V: one FP16 norm per head,
// the head's 256 (already Hadamard-rotated) coordinates quantized with the Lloyd-Max codebook of Beta(128,128).
// 3/4/6/8-bit indices.  This cut has no host-streaming copy and no QJL residual yet: `--kv tq*` runs the
// non-streamed path (`--kv-resident` is refused for it in generate.cpp).
#include "strata/kernels/kv_tq.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_tq: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

#include "kv_tq_codebook.inc"

template <int B>
__device__ __forceinline__ const float* cent();
template <>
__device__ __forceinline__ const float* cent<3>() { return TQ_CENT_3; }
template <>
__device__ __forceinline__ const float* cent<4>() { return TQ_CENT_4; }
template <>
__device__ __forceinline__ const float* cent<6>() { return TQ_CENT_6; }
template <>
__device__ __forceinline__ const float* cent<8>() { return TQ_CENT_8; }

template <int B>
__device__ __forceinline__ uint8_t nearest(float v) {
    constexpr int n = 1 << B;
    const float* c = cent<B>();
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (v > c[mid]) lo = mid + 1;
        else hi = mid;
    }
    if (lo > 0 && fabsf(v - c[lo - 1]) <= fabsf(v - c[lo])) --lo;
    return (uint8_t) lo;
}

// The block layout (norm + dimension-ordered index groups) is kv_tq.hpp's TqBlock: the writers here and the
// inline readers in qsa_decode_attn.cu address the SAME struct, so a packing change cannot drift from a reader.
// One lane quantizes its own eight CONSECUTIVE coordinates (group `lane` = dimensions [8*lane, 8*lane + 8)), not
// a stride-32 slice - the stride-32 packing this file started with was a fixed permutation of the head's
// dimensions, so every inline reader (and the rotated-back output) saw a corrupted K/V.
// Eight indices (one lane's share of a head) packed little-endian into B bytes.
template <int B>
__device__ __forceinline__ void pack8(const uint8_t* c, uint8_t* out) {
    uint64_t acc = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) acc |= (uint64_t) c[i] << (i * B);
#pragma unroll
    for (int j = 0; j < B; ++j) out[j] = (uint8_t) (acc >> (8 * j));
}
template <int B>
__device__ __forceinline__ void quantize_head(const float* x, TqBlock<B>* blk, int lane) {
    constexpr int E = QK_TQ / 32;
    static_assert(E == 8, "an index group holds 8 dimensions (kv_tq.hpp): QK_TQ / 32 must be 8");
    const float* xl = x + lane * E;   // this lane's E consecutive coordinates (byte group `lane` of the block)
    float v[E];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < E; ++i) {
        v[i] = xl[i];
        ss += v[i] * v[i];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float nrm = sqrtf(ss);
    const float inv = nrm > 0.0f ? 1.0f / nrm : 0.0f;
    uint8_t c[E];
#pragma unroll
    for (int i = 0; i < E; ++i) c[i] = nearest<B>(v[i] * inv);
    uint8_t out[B];
    pack8<B>(c, out);
    if (lane == 0) blk->norm = f16_from_f32(nrm);
    uint8_t* grp = blk->qs + (kv_tq_group_bytes(B, lane) - 2);   // group `lane` = dimensions [8*lane, 8*lane + 8)
#pragma unroll
    for (int j = 0; j < B; ++j) grp[j] = out[j];
}

// `x` receives this lane's E consecutive coordinates; the caller places them at their dimension order.
template <int B>
__device__ __forceinline__ void dequantize_head(const TqBlock<B>* blk, float* x, int lane) {
    kv_tq_decode8<B>(*blk, lane, cent<B>(), x);   // group `lane` = dimensions [8*lane, 8*lane + 8)
}

template <int B>
__global__ void kv_append_tq_steps_kernel(uint8_t* __restrict__ k_tq, uint8_t* __restrict__ v_tq,
                                          const int32_t* __restrict__ table, const int32_t* __restrict__ steps,
                                          const float* __restrict__ kcur, const float* __restrict__ vcur,
                                          int heads, int head_dim, int page_size, int stride, int n_steps) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int plane = blockIdx.z;
    if (t >= n_steps) return;
    const bool is_v = plane != 0;
    const int32_t* step = steps + (size_t) t * stride;
    const long long pos = (long long) __ldg(step + kStepPos);
    const long long page = (long long) __ldg(table + pos / page_size);
    if (page < 0) return;   // streamed-out page (not used by this cut)
    const long long row = (page * heads + h) * page_size + (pos % page_size);
    const float* x = (is_v ? vcur : kcur) + (size_t) t * heads * head_dim + h * head_dim;
    quantize_head<B>(x, reinterpret_cast<TqBlock<B>*>((is_v ? v_tq : k_tq) + row * sizeof(TqBlock<B>)), threadIdx.x);
}

template <int B>
__global__ void kv_append_tq_prompt_kernel(uint8_t* __restrict__ k_tq, uint8_t* __restrict__ v_tq,
                                           const int32_t* __restrict__ table, int64_t pos0,
                                           const float* __restrict__ K, const float* __restrict__ V,
                                           int heads, int head_dim, int page_size) {
    const long long t = blockIdx.x;
    const int h = blockIdx.y;
    const bool is_v = blockIdx.z != 0;
    const long long pos = pos0 + t;
    const long long page = (long long) __ldg(table + pos / page_size);
    if (page < 0) return;
    const long long row = (page * heads + h) * page_size + (pos % page_size);
    const float* x = (is_v ? V : K) + t * heads * head_dim + h * head_dim;
    quantize_head<B>(x, reinterpret_cast<TqBlock<B>*>((is_v ? v_tq : k_tq) + row * sizeof(TqBlock<B>)), threadIdx.x);
}

template <int B>
__global__ void kv_gather_tq_kernel(const uint8_t* __restrict__ k_tq, const uint8_t* __restrict__ v_tq,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                    const int32_t* __restrict__ step, int heads, int head_dim, int page_size,
                                    uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int id = blockIdx.x;
    const int h = blockIdx.y;
    if (id >= n_ids) return;
    const int cell = __ldg(ids + id);
    const long long page = (long long) __ldg(table + cell / page_size);
    const long long row = (page * heads + h) * page_size + (cell % page_size);
    float kx[QK_TQ / 32], vx[QK_TQ / 32];
    dequantize_head<B>(reinterpret_cast<const TqBlock<B>*>(k_tq + row * sizeof(TqBlock<B>)), kx, threadIdx.x);
    dequantize_head<B>(reinterpret_cast<const TqBlock<B>*>(v_tq + row * sizeof(TqBlock<B>)), vx, threadIdx.x);
    // dimension order: lane L holds dimensions [8L, 8L + 8) of each side
    const long long dst = ((long long) id * heads + h) * head_dim + threadIdx.x * (QK_TQ / 32);
#pragma unroll
    for (int i = 0; i < QK_TQ / 32; ++i) {
        k_scratch[dst + i] = f16_from_f32(kx[i]);
        v_scratch[dst + i] = f16_from_f32(vx[i]);
    }
}

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != QK_TQ) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard/TurboQuant block)\n", what);
        std::exit(1);
    }
}

template <int B>
void append_steps(uint8_t* k_tq, uint8_t* v_tq, const int32_t* table, const int32_t* steps, int stride, int n_steps,
                  const float* kcur, const float* vcur, const QsaShapes& s, void* stream) {
    const dim3 grid((unsigned) n_steps, (unsigned) s.n_head_kv, 2);
    kv_append_tq_steps_kernel<B><<<grid, 32, 0, (cudaStream_t) stream>>>(
        k_tq, v_tq, table, steps, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, stride,
        n_steps);
    check("kv_append_tq steps launch");
}

template <int B>
void append_prompt(uint8_t* k_tq, uint8_t* v_tq, const int32_t* table, int64_t pos0, int64_t T, const float* K,
                   const float* V, const QsaShapes& s, void* stream) {
    const dim3 grid((unsigned) T, (unsigned) s.n_head_kv, 2);
    kv_append_tq_prompt_kernel<B><<<grid, 32, 0, (cudaStream_t) stream>>>(
        k_tq, v_tq, table, pos0, K, V, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size);
    check("kv_append_tq prompt launch");
}

template <int B>
void gather_step(const uint8_t* k_tq, const uint8_t* v_tq, const int32_t* table, const int32_t* ids,
                 const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                 uint16_t* v_scratch, void* stream) {
    const dim3 grid((unsigned) max_ids, (unsigned) s.n_head_kv);
    kv_gather_tq_kernel<B><<<grid, 32, 0, (cudaStream_t) stream>>>(
        k_tq, v_tq, table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,
        v_scratch);
    check("kv_gather_tq launch");
}

}  // namespace

void kv_append_tq_steps(int bits, uint8_t* k_tq, uint8_t* v_tq, const int32_t* page_table, const int32_t* step,
                        int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                        void* stream, const KvHostPools* host) {
    if (n_steps <= 0) return;
    need_256(s, "kv_append_tq_steps");
    (void) host;   // the first cut has no host-streaming copy
    const int stride = step_stride > 0 ? step_stride : kStepCount;   // 0: the records are a dense step buffer
    switch (bits) {
        case 3: append_steps<3>(k_tq, v_tq, page_table, step, stride, n_steps, kcur, vcur, s, stream); break;
        case 4: append_steps<4>(k_tq, v_tq, page_table, step, stride, n_steps, kcur, vcur, s, stream); break;
        case 6: append_steps<6>(k_tq, v_tq, page_table, step, stride, n_steps, kcur, vcur, s, stream); break;
        case 8: append_steps<8>(k_tq, v_tq, page_table, step, stride, n_steps, kcur, vcur, s, stream); break;
        default: std::fprintf(stderr, "kv_append_tq_steps: unsupported %d bits\n", bits); std::exit(1);
    }
}

void kv_append_tq_step(int bits, uint8_t* k_tq, uint8_t* v_tq, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    kv_append_tq_steps(bits, k_tq, v_tq, page_table, step, 0, 1, kcur, vcur, s, stream, host);
}

void kv_append_tq_batch(int bits, uint8_t* k_tq, uint8_t* v_tq, const int32_t* page_table, const int32_t* steps,
                        const float* kcur, const float* vcur, int64_t n_steps, const QsaShapes& s, void* stream,
                        const KvHostPools* host) {
    if (n_steps < 1) return;
    need_256(s, "kv_append_tq_batch");
    (void) host;
    switch (bits) {
        case 3: append_steps<3>(k_tq, v_tq, page_table, steps, kStepCount, (int) n_steps, kcur, vcur, s, stream); break;
        case 4: append_steps<4>(k_tq, v_tq, page_table, steps, kStepCount, (int) n_steps, kcur, vcur, s, stream); break;
        case 6: append_steps<6>(k_tq, v_tq, page_table, steps, kStepCount, (int) n_steps, kcur, vcur, s, stream); break;
        case 8: append_steps<8>(k_tq, v_tq, page_table, steps, kStepCount, (int) n_steps, kcur, vcur, s, stream); break;
        default: std::fprintf(stderr, "kv_append_tq_batch: unsupported %d bits\n", bits); std::exit(1);
    }
}

void kv_append_tq(int bits, uint8_t* k_tq, uint8_t* v_tq, const int32_t* page_table, int64_t pos0, int64_t T,
                  const float* K, const float* V, const QsaShapes& s, void* stream, const KvHostPools* host,
                  const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_tq");
    (void) host;
    (void) stage;
    switch (bits) {
        case 3: append_prompt<3>(k_tq, v_tq, page_table, pos0, T, K, V, s, stream); break;
        case 4: append_prompt<4>(k_tq, v_tq, page_table, pos0, T, K, V, s, stream); break;
        case 6: append_prompt<6>(k_tq, v_tq, page_table, pos0, T, K, V, s, stream); break;
        case 8: append_prompt<8>(k_tq, v_tq, page_table, pos0, T, K, V, s, stream); break;
        default: std::fprintf(stderr, "kv_append_tq: unsupported %d bits\n", bits); std::exit(1);
    }
}

void kv_gather_tq_step(int bits, const uint8_t* k_tq, const uint8_t* v_tq, const int32_t* page_table,
                       const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                       uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    need_256(s, "kv_gather_tq");
    switch (bits) {
        case 3: gather_step<3>(k_tq, v_tq, page_table, ids, step, max_ids, s, k_scratch, v_scratch, stream); break;
        case 4: gather_step<4>(k_tq, v_tq, page_table, ids, step, max_ids, s, k_scratch, v_scratch, stream); break;
        case 6: gather_step<6>(k_tq, v_tq, page_table, ids, step, max_ids, s, k_scratch, v_scratch, stream); break;
        case 8: gather_step<8>(k_tq, v_tq, page_table, ids, step, max_ids, s, k_scratch, v_scratch, stream); break;
        default: std::fprintf(stderr, "kv_gather_tq_step: unsupported %d bits\n", bits); std::exit(1);
    }
}

}  // namespace strata::kernels
