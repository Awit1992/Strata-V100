// include/strata/core/dflash2.hpp - native DFlash2 draft backend (block-parallel speculative decoding).
//
// The checkpoint (e.g. incoai/Qwen3.8-Flash-Next-DFlash2) is a DRAFT-ONLY model: five Qwen3-style
// decoder layers over one BLOCK of `block_size` positions, plus a candidate selector.  It owns no
// embedding table and no lm_head; the target's are used, and the target's completed layer outputs
// (taps) are the context the draft attends to.
//
// One step, at the anchor's global position `start_pos`:
//   * block rows occupy positions [start_pos, start_pos + block_size); row 0 is the committed anchor
//     token, rows 1.. are `mask_token_id`, later overwritten by the proposed tokens;
//   * the block's input embeddings come from the TARGET's embedding table (device fp32, given here);
//   * the context is the projected concatenation of the target's tap features over positions
//     [start_pos - ctx_len, start_pos) - appended here, with its per-layer K/V projected once;
//   * attention is BIDIRECTIONAL with a sliding window (`|q_pos - k_pos| < sliding_window`), and the
//     context K/V comes from the projected target features while the block K/V comes from the block;
//   * the final norm's row 0 (the anchor) is discarded; the remaining rows go through the TARGET's
//     lm_head (the `head` callback) and a top-k to get the candidates;
//   * the selector scores the candidate lattice over the two codebooks and walks one path, which
//     gives the proposed tokens and, per slot, the conditional distribution q over that slot's
//     candidates - the distribution the lossless verifier needs.
//
// Nothing here verifies or commits tokens: the caller runs the target over the block and applies the
// greedy / rejection-sampling rule (core/verify.hpp, core/spec_prob.hpp).
//
// Precision (the reference's own scheme): the checkpoint is bf16, so the weights are stored fp16
// (exact for every bf16 value in fp16's range - the loader checks that) and each projection is a
// cuBLAS fp16 matmul with fp32 accumulation and an FP32 OUTPUT.  Every activation, the residual
// stream, the context K/V and every elementwise step (norms, RoPE, conv, attention, softmax, silu,
// the selector) are fp32; only the operand that enters a matmul is rounded to fp16.  That matters:
// with the real weights the residual reaches RMS ~4e4 and the sublayer outputs far more, which an
// fp16 activation path turns into inf/NaN.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/// `block_size` may be 2..kDFlash2MaxSlots+1; the engine's verify window caps the useful value at 8
/// (7 proposed tokens), which is what the checkpoint was trained at.
namespace strata::core {

/// Proposal arrays are fixed size; a block of 8 (7 slots) and top-k 16 are this checkpoint's values.
inline constexpr int kDFlash2MaxSlots = 16;
inline constexpr int kDFlash2MaxTopK = 32;

struct DFlash2Config {
    /// Directory holding `model.safetensors` (bf16) of the draft checkpoint.
    std::string model_dir;
    int device = 0;
    int block_size = 8;               ///< checkpoint `dflash_config.block_size`
    int n_taps = 5;                   ///< checkpoint `dflash_config.target_layer_ids.size()`
    std::vector<int> tap_layers = {5, 15, 25, 35, 45};
    int target_hidden = 2560;         ///< target hidden_size (one hyper-connection stream)
    int hc_streams = 4;               ///< target hyper-connection streams, averaged per tap
    int draft_hidden = 2560;          ///< checkpoint hidden_size
    int64_t vocab = 248320;           ///< candidate ids are restricted to [0, vocab)
    int max_context = 2047;           ///< feature / context-KV ring capacity (>= sliding_window - 1)
    int sliding_window = 2048;        ///< checkpoint `sliding_window` (both sides)
    int top_k = 16;                   ///< checkpoint `dflash_config.selector_top_k`
    float rms_eps = 1e-6f;
    float rope_theta = 10000.0f;
};

/// Full-vocabulary logits of `n_rows` device fp32 rows (`[n_rows, cfg.draft_hidden]`) into `logits`,
/// a device fp32 `[n_rows, cfg.vocab]` buffer owned by this backend: the TARGET's own output
/// projection (dense or quantized).  Run it on `stream`.  Returns false and sets `err` on failure.
using DFlash2HeadFn = bool (*)(void* ctx, const float* rows, int n_rows, float* logits, void* stream,
                               std::string& err);

/// One block's proposal.  `tokens[e]` is the drafted token at block row e+1; `cand`/`q`/`unary` are
/// slot-major: slot e's candidates are `cand[e * n_cand + c]` with target-head logits `unary[...]` and
/// the selector's conditional distribution `q[...]` (sums to 1; a point mass on `tokens[e]` when the
/// request is greedy).  `q` is what the lossless sampler needs: scatter it into the full vocabulary
/// at `cand` to get the draft distribution of the slot.
struct DFlash2Proposal {
    int n_slots = 0;
    int n_cand = 0;
    int32_t tokens[kDFlash2MaxSlots] = {};
    int32_t cand[kDFlash2MaxSlots * kDFlash2MaxTopK] = {};
    float q[kDFlash2MaxSlots * kDFlash2MaxTopK] = {};
    float unary[kDFlash2MaxSlots * kDFlash2MaxTopK] = {};
};

class DFlash2Backend {
public:
    DFlash2Backend();
    ~DFlash2Backend();
    DFlash2Backend(const DFlash2Backend&) = delete;
    DFlash2Backend& operator=(const DFlash2Backend&) = delete;

    /// Reads the checkpoint's safetensors (bf16 -> fp16) and allocates the weights, the context ring
    /// and every activation buffer.  The caller's CUDA device must be `cfg.device`.  Everything stays
    /// allocated until destruction.
    bool load(const DFlash2Config& cfg, std::string& err);
    bool loaded() const;
    /// Drops the context ring (features and their K/V) and the step counters; the weights stay.
    void reset();

    int slots() const;                 ///< block_size - 1 (the number of proposed tokens)
    int block_size() const;
    int top_k() const;
    int n_draft_layers() const;
    uint64_t device_bytes() const;
    uint64_t steps() const;
    /// The checkpoint's `mask_token_id` (248077): the placeholder the block's rows 1.. carry.
    int32_t mask_token_id() const;

    /// Required before propose().  `ctx` is passed back unchanged.
    void set_target_head(DFlash2HeadFn fn, void* ctx);

    /// The target's features for `n_rows` consecutive positions ending at `base_pos + n_rows - 1`.
    /// `dev_tap_rows` is device fp32, tap-major: row `i` of tap `t` is at
    /// `(t * tap_stride_rows + i) * hc_streams * target_hidden` and holds tap `t`'s completed target
    /// layer output at global position `base_pos + i`.  `tap_stride_rows` is that per-tap row stride -
    /// pass the capture buffer's own capacity when it is handed over unchanged (no copy), or 0 when the
    /// taps are packed, i.e. the stride is `n_rows`.  The HC streams are averaged here, the taps
    /// concatenated in `tap_layers` order, projected by `fc` + `hidden_norm`, and each layer's context
    /// K/V is projected and written into the ring.
    /// `base_pos == next_pos()` once the ring holds rows; a fresh ring (ctx_len() == 0) may start at
    /// any position - a batched prefill or a cache resume hands its first window position.  More rows
    /// than the ring holds keep its newest `max_context` (a whole prompt can be appended in one call);
    /// the caller streams its own staging buffer.
    bool append_features(const float* dev_tap_rows, int64_t n_rows, int64_t tap_stride_rows,
                         int64_t base_pos, std::string& err);
    /// Drops every ring row at a position >= `keep_pos` (a partial accept that did not commit the
    /// speculative tail).  No-op when `keep_pos >= next_pos()`.
    void truncate(int64_t keep_pos);
    int64_t next_pos() const;          ///< the global position the next appended row must have
    int64_t ctx_len() const;           ///< ring rows currently held

    /// One draft step.  Synchronous: it joins its own stream before returning.
    ///   anchor_id    committed token at `start_pos` (block row 0)
    ///   start_pos    global position of the anchor; must be `next_pos()` once context exists
    ///   block_embeds device fp32 `[block_size, draft_hidden]`: the target's embedding of the block
    ///                ids (row 0 = anchor, rows 1.. = the mask token), already scaled by
    ///                `input_embedding_scale` (1.0 for this checkpoint)
    ///   uniforms     host `[slots]` in [0,1) used to sample the path; ignored (may be null) when
    ///                `greedy`
    bool propose(int32_t anchor_id, int64_t start_pos, const float* block_embeds, float temperature,
                 bool greedy, const float* uniforms, DFlash2Proposal& out, std::string& err);

    /// The last step's draft rows fed to the target head (`[slots, draft_hidden]`, fp32), copied to
    /// the host on demand - the parity harness's hook (the head callback already receives them, so a
    /// normal step needs neither this nor any other per-step transfer).
    bool copy_pred_hidden(float* out, std::string& err) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::core
