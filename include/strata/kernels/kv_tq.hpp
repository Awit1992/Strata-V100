// include/strata/kernels/kv_tq.hpp - TurboQuant K/V storage for the QSA layers (`--kv tq3|tq4|tq6|tq8`).
//
// TurboQuant (Zandieh et al., arXiv 2504.19874): rotate each head's K/V with the orthonormal 256-point
// Hadamard matrix, then quantize every coordinate with the Lloyd-Max scalar quantizer of the induced
// Beta(128,128) distribution (one FP16 norm per head, 3/4/6/8-bit centroid indices).  The codebooks are
// data-oblivious constants (tools/gen_tq_codebook.py).  The query is rotated the same way, so <Hq, Hk> is
// unchanged, and the attention output is rotated back with H.  This first cut stores the MSE-optimal
// centroids; the paper's 1-bit QJL residual for the unbiased inner product is a later addition.
#pragma once

#include <cstdint>
#if defined(STRATA_HIP_GFX906)
// gfx906: the compat layer, before f16_bits.hpp's device conversions - it supplies __forceinline__ and the
// device-side memcpy those use.  dp4a.hpp includes it for the same reason.
#include <cuda_runtime.h>
#endif

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/f16_bits.hpp"

namespace strata::kernels {

inline constexpr int QK_TQ = 256;

/// One head's TurboQuant block: the FP16 norm of the 256 rotated coordinates, then the Lloyd-Max indices packed
/// DIMENSION-ORDERED - index group j (B bytes) holds dimensions [8j, 8j + 8), the eight one warp lane owns in the
/// score phase.  A writer that packs any other order (kv_tq.cu first packed a stride-32 slice) and a reader that
/// assumes this one disagree silently: the head comes back permuted, which is neither an error nor a wrong size.
/// Both sides must address the groups through this layout (kv_tq_group_bytes below).
template <int B>
struct TqBlock {
    uint16_t norm;               ///< fp16 bits: the norm the coordinates were divided by
    uint8_t qs[QK_TQ * B / 8];   ///< 256 indices of B bits, 8 dimensions per group
};
static_assert(sizeof(TqBlock<3>) == 2 + QK_TQ * 3 / 8 && sizeof(TqBlock<4>) == 2 + QK_TQ * 4 / 8 &&
              sizeof(TqBlock<6>) == 2 + QK_TQ * 6 / 8 && sizeof(TqBlock<8>) == 2 + QK_TQ * 8 / 8,
              "a TqBlock must be packed exactly as kv_tq_bytes_per_head sizes it (no padding)");

/// Bytes per KV head: one FP16 norm + head_dim * bits / 8.
STRATA_HD inline constexpr uint64_t kv_tq_bytes_per_head(int bits, int head_dim) {
    return 2 + ((uint64_t) head_dim * (uint64_t) bits + 7) / 8;
}
/// Byte offset of index group `j` inside a block: the B bytes holding dimensions [8j, 8j + 8).  Host and device
/// both read blocks through this, so it is __host__ __device__ (f16_bits.hpp's STRATA_HD) rather than host-only.
STRATA_HD inline constexpr int kv_tq_group_bytes(int bits, int j) { return 2 + j * bits; }

#if defined(__CUDACC__) || defined(__HIPCC__)
/// Decode index group `j` - dimensions [8j, 8j + 8) - into `out` as the codebook's NORMALIZED centroids, without
/// the block's norm.  The readers that apply the norm through their own FP32 scale slots (the sm_70 prompt tensor
/// kernel's `ks`/`vs`) need it this way; every other reader wants kv_tq_decode8 below.  Both entry points share
/// this unpacking, so a packing change still cannot leave a reader behind.
template <int B>
__device__ __forceinline__ void kv_tq_decode8_raw(const TqBlock<B>& blk, int j, const float* cent, float* out) {
    static_assert(B == 3 || B == 4 || B == 6 || B == 8, "TurboQuant widths are 3, 4, 6 and 8 bits");
    const uint8_t* in = blk.qs + (kv_tq_group_bytes(B, j) - 2);
    uint64_t acc = 0;
#pragma unroll
    for (int b = 0; b < B; ++b) acc |= (uint64_t) in[b] << (8 * b);
    constexpr uint64_t mask = B == 8 ? 0xFFull : ((1ull << B) - 1ull);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = cent[(acc >> (i * B)) & mask];
}
/// Decode index group `j` - dimensions [8j, 8j + 8) - into `out`, scaled by the block's FP16 norm.  `cent` is the
/// codebook of the block's bit width (kv_tq_codebook.inc's TQ_CENT_B).  THE ONE DECODER: the writer's inverse in
/// kv_tq.cu and the attention readers in qsa_decode_attn.cu both go through it, so a packing change cannot leave a
/// reader behind, and a smoke test can hold the GPU's bytes to the same rule.
template <int B>
__device__ __forceinline__ void kv_tq_decode8(const TqBlock<B>& blk, int j, const float* cent, float* out) {
    kv_tq_decode8_raw<B>(blk, j, cent, out);
    const float nrm = f32_from_f16(blk.norm);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] *= nrm;
}
#endif
inline uint64_t kv_tq_bytes_per_cell(int bits, const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * kv_tq_bytes_per_head(bits, (int) s.head_dim) * 2;
}

// The `host` / `stage` arguments mirror kv_q4.hpp's family so a call site reads the same, but TurboQuant has NO
// host copy in this cut: every one of them is ignored (`qsa_kv_format` is kKvTq, and a TurboQuant state pins no
// host pools - layer.cpp), and --kv-resident is refused for it in generate.cpp.

/// Append one cell at step[kStepPos] (already rotated).
void kv_append_tq_step(int bits, uint8_t* k_tq, uint8_t* v_tq, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host = nullptr);

/// `n_steps` records, each at `step_stride` ints from the previous one (0: a dense kStepCount buffer).
void kv_append_tq_steps(int bits, uint8_t* k_tq, uint8_t* v_tq, const int32_t* page_table, const int32_t* step,
                        int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                        void* stream, const KvHostPools* host = nullptr);

/// The verify window's batch: `steps` is the first of the group's step records, positions come from them.
void kv_append_tq_batch(int bits, uint8_t* k_tq, uint8_t* v_tq, const int32_t* page_table, const int32_t* steps,
                        const float* kcur, const float* vcur, int64_t n_steps, const QsaShapes& s, void* stream,
                        const KvHostPools* host = nullptr);

/// The prompt path: T consecutive (rotated) cells from pos0, K/V [T, n_head_kv, 256].
void kv_append_tq(int bits, uint8_t* k_tq, uint8_t* v_tq, const int32_t* page_table, int64_t pos0, int64_t T,
                  const float* K, const float* V, const QsaShapes& s, void* stream,
                  const KvHostPools* host = nullptr, const KvHostPools* stage = nullptr);

/// Gather step[kStepWidth] cells into FP16 scratch [id][kv_head][head_dim] (still rotated).
void kv_gather_tq_step(int bits, const uint8_t* k_tq, const uint8_t* v_tq, const int32_t* page_table,
                       const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                       uint16_t* k_scratch, uint16_t* v_scratch, void* stream);

}  // namespace strata::kernels
