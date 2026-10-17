// CPU-only prefix trimming for a parked conversation image: promote the exact
// anchor checkpoint, drop what lies past it, and shrink every K/V section's
// extent while keeping its allocation.  Deliberately free of CUDA and of
// session state, so it can be reasoned about and tested without a device.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/layout.hpp"
#include "strata/kernels/qsa.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::core {

/// The runtime geometry identity a snapshot records in SavedConversation::geometry.
/// A restore, a stage image and a CPU-only trim all compare against this one key.
inline std::array<int64_t, 18> conversation_geometry_key(const ModelGeometry& g) {
    return {g.n_embd, g.n_layers, g.qsa_interval, g.ssm_state_size, g.ssm_k_heads,
            g.ssm_v_heads, g.ssm_d_conv, g.ssm_conv_channels, g.ssm_value_dim,
            g.n_head, g.n_head_kv, g.head_dim, g.idx_q_heads, g.idx_key_dim,
            g.hc, g.hc_lr, g.n_expert, g.n_ff};
}

/// Trims a CPU snapshot to its first `upto` tokens without copying the prefix: the
/// checkpoint whose ids are exactly `upto` becomes the live state, every checkpoint
/// past it is dropped, and each K/V section keeps its allocation and only shrinks
/// its extent.  A fork of a long parked conversation can then restore at a short
/// anchor while the source keeps its own long tail, and nothing is copied.
/// `includes_draft`: this image's kv.back() is the draft layer (the same flag its
/// validate/restore takes).  A layer split's stage image must be trimmed by its own
/// call with the flag that stage was parked with - this refuses an image that holds
/// stage images, a disk delta image (first_units != 0), a live checkpoint with stage
/// parts, and an image with no checkpoint at exactly `upto` (the caller replays from
/// an earlier anchor instead).  No allocation and no device access.
inline bool conversation_snapshot_prefix(SavedConversation& image, int64_t upto, const ModelGeometry& g,
                                         bool includes_draft, std::string& error) {
    auto fail = [&](const char* why) {
        error = std::string("conversation prefix trim: ") + why;
        return false;
    };
    if (upto <= 0 || upto > (int64_t) image.live.ids.size()) return fail("token extent outside the image");
    if (!image.stage_images.empty()) return fail("a layer split's image is trimmed stage by stage");
    if (!image.live.stage_parts.empty()) return fail("a live checkpoint with stage parts");
    if (image.geometry != conversation_geometry_key(g)) return fail("incompatible runtime geometry");
    if ((int64_t) image.live.ids.size() > upto) {
        size_t at = image.checkpoints.size();
        for (size_t i = 0; i < image.checkpoints.size(); ++i)
            if ((int64_t) image.checkpoints[i].ids.size() == upto) { at = i; break; }
        if (at == image.checkpoints.size()) return fail("no checkpoint at that token extent");
        if (!image.checkpoints[at].has_state()) return fail("the checkpoint at that extent has no running state");
    }
    // The layout's own boundaries at the new extent: page-capped cells for
    // k/v/k_scale/v_scale, completed rows plus the moving spare row for the
    // indexer, none for the draft layer.  Validate every layer before any change.
    const int64_t page = strata::kernels::qsa_real_shapes().page_size;
    const int64_t idx_block = strata::kernels::qsa_real_shapes().idx_block;
    const int64_t cells = ((upto + page - 1) / page) * page;
    const size_t layers = image.kv.size();
    for (size_t i = 0; i < layers; ++i) {
        const ConversationKv& layer = image.kv[i];
        const bool index = !includes_draft || i + 1 < layers;
        if (layer.first_units[0] || layer.first_units[1] || layer.first_units[2] ||
            layer.first_units[3] || layer.first_units[4]) return fail("a disk delta image");
        if (layer.heads != g.n_head_kv || layer.head_dim != g.head_dim || layer.idx_dim != g.idx_key_dim ||
            layer.page_size != page || layer.cells <= 0) return fail("K/V geometry differs from the model");
        const std::array<const ConversationBuffer*, 5> parts =
            {&layer.k, &layer.v, &layer.k_scale, &layer.v_scale, &layer.pooled};
        for (size_t e = 0; e < 4; ++e) {
            if (parts[e]->size() % (size_t) layer.cells != 0)
                return fail("K/V section has an incomplete cell");
            const size_t per_cell = parts[e]->size() / (size_t) layer.cells;
            if (per_cell && (size_t) cells > parts[e]->size() / per_cell)
                return fail("K/V section is shorter than the trimmed extent");
        }
        const size_t pooled = index ? (size_t) (upto / idx_block + 1) * (size_t) layer.idx_dim * sizeof(float) : 0;
        if (pooled > parts[4]->size()) return fail("indexer section is shorter than the trimmed extent");
    }
    if ((int64_t) image.live.ids.size() > upto) {
        size_t at = 0;
        while ((int64_t) image.checkpoints[at].ids.size() != upto) ++at;
        image.live = std::move(image.checkpoints[at]);
        image.checkpoints.erase(image.checkpoints.begin() + (std::ptrdiff_t) at);
    }
    size_t kept = 0;
    for (size_t i = 0; i < image.checkpoints.size(); ++i) {
        if ((int64_t) image.checkpoints[i].ids.size() >= upto) continue;
        if (kept != i) image.checkpoints[kept] = std::move(image.checkpoints[i]);
        ++kept;
    }
    image.checkpoints.resize(kept);
    image.unchanged_tokens = std::min<int64_t>(image.unchanged_tokens, upto);
    // A view trimmed to a prefix shorter than the input boundary this image
    // recorded is no longer the state that boundary precedes: its live state is
    // input again, so it must not claim a later reply.
    if (image.terminal_from > upto) image.terminal_from = 0;
    for (size_t i = 0; i < layers; ++i) {
        ConversationKv& layer = image.kv[i];
        const bool index = !includes_draft || i + 1 < layers;
        std::array<ConversationBuffer*, 5> parts = {&layer.k, &layer.v, &layer.k_scale, &layer.v_scale, &layer.pooled};
        for (size_t e = 0; e < 4; ++e)
            parts[e]->resize((parts[e]->size() / (size_t) layer.cells) * (size_t) cells);
        parts[4]->resize(index ? (size_t) (upto / idx_block + 1) * (size_t) layer.idx_dim * sizeof(float) : 0);
        layer.cells = cells;
        layer.pooled_rows = index ? upto / idx_block + 1 : 0;
    }
    return true;
}

} // namespace strata::core
