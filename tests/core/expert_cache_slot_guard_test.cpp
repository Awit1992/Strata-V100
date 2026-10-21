// STRATA_MMQ_DIRECT: the expert cache's slot guard (set_slot_guard), slot_room and slot_tail on the uniform cache
// (open: off_ empty, the stride is the largest blob) and on the sized one (open_sized: off_ holds each slot's start).
// No guard asked (the default) must give today's sized layout byte for byte.
// STRATA_PREFILL_PROMOTE: the rotation as its caller performs it - the D2D cold -> lent slot and the H2D hot -> head
// slot, one fence, then the checked three-way remap - and every copy or remap the checks decline leaving the
// destination bytes and the whole residency table exactly as they were.  Needs a CUDA device (exits 77 without).
#include "strata/core/expert_cache.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

// Tiny, deterministic blobs.  Distinct seeds differ at byte 0, so a copy in the wrong direction, a short copy or
// a copy that never ran is visible with verify_slot rather than only in a counter.
std::vector<uint8_t> bytes_like(uint8_t seed, size_t n) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = (uint8_t) (seed + (uint8_t) (i * 7));
    return v;
}

// What slot_of says for every (layer, expert): the remap's whole observable state, read through the public API.
std::vector<int32_t> residency_of(const strata::core::ExpertCache& c, int64_t layers, int64_t experts) {
    std::vector<int32_t> r;
    r.reserve((size_t) (layers * experts));
    for (int64_t l = 0; l < layers; ++l)
        for (int64_t e = 0; e < experts; ++e) r.push_back(c.slot_of(l, e));
    return r;
}

// A remap that does not meet every precondition must decline AND leave every mapping alone - no partial move of
// the old expert out, the cold expert over, or the hot expert in.
void remap_must_decline(strata::core::ExpertCache& c, int64_t layers, int64_t experts, int64_t layer,
                        int32_t old_expert, int32_t cold_expert, int32_t hot_expert, int32_t old_slot,
                        int32_t cold_slot, const char* what) {
    std::string err;
    const std::vector<int32_t> before = residency_of(c, layers, experts);
    check(!c.promote_remap(layer, old_expert, cold_expert, hot_expert, old_slot, cold_slot, err), what);
    const std::string kept = std::string(what) + " (a declined remap changed the table)";
    check(residency_of(c, layers, experts) == before, kept.c_str());
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    using strata::core::ExpertCache;
    std::string err;

    // slot_stride: the blob plus the guard, 256-byte aligned; guard 0 is the old (b + 255) / 256 * 256
    for (int64_t b : {1, 255, 256, 257, 1000, 2457600, 2949120, 3047424}) {
        check(ExpertCache::slot_stride(b) == (b + 255) / 256 * 256, "slot_stride without a guard is today's stride");
        check(ExpertCache::slot_stride(b, 256) == (b + 256 + 255) / 256 * 256, "slot_stride with a 256 B guard");
        check(ExpertCache::slot_stride(b, 256) - b >= 256, "a guarded slot has at least the guard after its blob");
    }

    // 1. uniform: off_ is empty; the guard is asked for but not applied (every slot is the largest blob)
    {
        ExpertCache c;
        c.set_slot_guard(256);
        if (!c.open(6, 3, 4, 1024, err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
        check(c.slot_guard() == 0, "uniform: no guard applied");
        for (int32_t s = 0; s < 6; ++s) {
            check(c.slot_room(s) == 1024, "uniform: slot_room is the blob size");
            check(c.slot_offset(s) == (uint64_t) s * 1024, "uniform: slot_offset");
        }
        // the last slot reads slot_offset(slots) - the end - without indexing an empty off_
        check(c.slot_tail(5, 1000, 24) == c.device_slot(5) + 1000, "uniform: the last slot's slack covers 24 B");
        check(c.slot_tail(5, 1000, 25) == nullptr, "uniform: the last slot has no room for 25 B past 1000");
        check(c.slot_tail(0, 1024, 0) == c.device_slot(0) + 1024, "uniform: need 0 at the slot's end");
        check(c.slot_tail(0, 1024, 1) == nullptr, "uniform: a full slot has no room");
        check(c.slot_tail(3, 512, 136) == c.device_slot(3) + 512, "uniform: a smaller blob's slack");
        check(c.slot_tail(-1, 10, 10) == nullptr && c.slot_tail(6, 10, 10) == nullptr, "uniform: a bad slot is null");
        c.close();
        check(c.slot_guard() == 0 && c.slot_tail(0, 10, 10) == nullptr, "uniform: closed");
    }

    // 2. sized without a guard (the default): today's layout, and slot_tail sees only the 256-byte rounding
    const std::vector<int64_t> blobs = {1000, 2457600, 700, 3047424, 256, 2949120};
    std::vector<uint64_t> plain(blobs.size() + 1, 0);
    for (size_t i = 0; i < blobs.size(); ++i) plain[i + 1] = plain[i] + ((uint64_t) blobs[i] + 255) / 256 * 256;
    {
        ExpertCache c;
        if (!c.open_sized(blobs, 3, 4, err)) { std::fprintf(stderr, "open_sized: %s\n", err.c_str()); return 1; }
        check(c.slot_guard() == 0, "sized, no guard: slot_guard 0");
        for (size_t s = 0; s <= blobs.size(); ++s) check(c.slot_offset((int64_t) s) == plain[s], "sized, no guard: today's offsets");
        check(c.slot_tail(0, 1000, 24) == c.device_slot(0) + 1000, "sized, no guard: the rounding's 24 B");
        check(c.slot_tail(0, 1000, 25) == nullptr, "sized, no guard: no more than the rounding");
        check(c.slot_tail(4, 256, 1) == nullptr, "sized, no guard: a blob of 256 has no room");
        check(c.slot_tail(5, 2949120, 1) == nullptr, "sized, no guard: the last slot (offset of the end)");
    }

    // 3. sized with the 256 B guard: every slot has >= 256 B after its blob; close() drops the applied guard and a
    //    reopen applies the kept request again
    {
        ExpertCache c;
        c.set_slot_guard(256);
        for (int round = 0; round < 2; ++round) {
            if (!c.open_sized(blobs, 3, 4, err)) { std::fprintf(stderr, "open_sized: %s\n", err.c_str()); return 1; }
            check(c.slot_guard() == 256, "sized, guard: slot_guard 256");
            uint64_t off = 0;
            for (size_t s = 0; s < blobs.size(); ++s) {
                const int32_t si = (int32_t) s;
                check(c.slot_offset(si) == off, "sized, guard: offsets");
                off += (uint64_t) ExpertCache::slot_stride(blobs[s], 256);
                check(c.slot_room(si) == (uint64_t) ExpertCache::slot_stride(blobs[s], 256), "sized, guard: slot_room");
                check(c.device_slot(si) == c.device_slot(0) + c.slot_offset(si), "sized, guard: device_slot");
                uint8_t* t = c.slot_tail(si, (uint64_t) blobs[s], 256);
                check(t == c.device_slot(si) + blobs[s], "sized, guard: slot_tail at the blob's end");
                check(c.slot_tail(si, (uint64_t) blobs[s], c.slot_room(si) - (uint64_t) blobs[s] + 1) == nullptr,
                      "sized, guard: no room past the slot");
            }
            check(c.slot_offset((int64_t) blobs.size()) == off, "sized, guard: the end");
            // the guard is writable device memory inside the arena (the prompt path zeroes it)
            uint8_t* t = c.slot_tail((int32_t) blobs.size() - 1, (uint64_t) blobs.back(), 256);
            check(t != nullptr && cudaMemset(t, 0, 256) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess,
                  "sized, guard: the last slot's guard can be zeroed");
            c.close();
            check(c.slot_guard() == 0 && c.slot_tail(0, 10, 10) == nullptr, "sized, guard: closed");
        }
        // asked for 0 again: today's layout
        c.set_slot_guard(0);
        if (!c.open_sized(blobs, 3, 4, err)) { std::fprintf(stderr, "open_sized: %s\n", err.c_str()); return 1; }
        check(c.slot_guard() == 0 && c.slot_offset((int64_t) blobs.size()) == plain.back(), "sized: guard reset to 0");
    }

    // 4. the rotation on a uniform cache: D2D cold -> the lent (old) slot, H2D hot -> the head (cold) slot, one
    //    fence, and only then the checked three-way remap.  The mappings are the old ones right up to the remap,
    //    and the bytes after the fence are the rotated ones, so an early remap, a reversed copy or a copy that
    //    never ran is visible from the public API alone.
    {
        ExpertCache c;
        if (!c.open(6, 3, 5, 1024, err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
        check(c.admit(0, 0) == 0 && c.admit(0, 1) == 1 && c.admit(1, 0) == 2, "rotation: the base admission");
        const std::vector<uint8_t> old_b = bytes_like(3, 1024);
        const std::vector<uint8_t> cold_b = bytes_like(41, 1024);
        const std::vector<uint8_t> hot_b = bytes_like(97, 1024);
        const std::vector<uint8_t> other_b = bytes_like(151, 1024);
        check(c.fill_slot_queued(0, old_b.data(), err), "rotation: the lent slot holds the old expert");
        check(c.fill_slot_queued(1, cold_b.data(), err), "rotation: the head slot holds the cold expert");
        check(c.fill_slot_queued(2, other_b.data(), err), "rotation: another layer's slot");
        check(c.sync_queued(err), "rotation: the fill fence");

        check(c.copy_slot_queued(0, 1, 1024, err), "rotation: cold is copied into the lent slot");
        check(c.fill_slot_queued(1, hot_b.data(), err), "rotation: hot is filled into the head slot");
        check(c.sync_queued(err), "rotation: the rotation fence");

        check(c.slot_of(0, 0) == 0 && c.slot_of(0, 1) == 1 && c.slot_of(0, 2) == strata::core::kNotResident,
              "rotation: no mapping is published before the remap");
        check(c.verify_slot(0, cold_b.data(), err, 1024), "rotation: the lent slot holds the cold bytes");
        check(c.verify_slot(1, hot_b.data(), err, 1024), "rotation: the head slot holds the hot bytes");

        check(c.promote_remap(0, 0, 1, 2, 0, 1, err), "rotation: the remap is accepted");
        check(c.slot_of(0, 0) == strata::core::kNotResident && c.slot_of(0, 1) == 0 && c.slot_of(0, 2) == 1,
              "rotation: old leaves, cold takes the lent slot, hot the head slot");
        check(c.slot_of(0, 3) == strata::core::kNotResident && c.slot_of(0, 4) == strata::core::kNotResident,
              "rotation: the untouched experts stay non-resident");
        check(c.slot_of(1, 0) == 2, "rotation: another layer's mapping is untouched");
        check(c.verify_slot(2, other_b.data(), err, 1024), "rotation: and so are its bytes");
    }

    // 5. the rotation on a sized cache: only the cold expert's own blob moves, so the lent slot's slack past those
    //    bytes keeps the old expert's tail.  Comparing the whole 1024-byte slot catches a copy of the wrong length
    //    and a copy in the wrong direction at once.
    {
        ExpertCache c;
        const std::vector<int64_t> rot = {1024, 512, 768, 256};
        if (!c.open_sized(rot, 2, 4, err)) { std::fprintf(stderr, "open_sized: %s\n", err.c_str()); return 1; }
        check(c.admit(0, 0) == 0 && c.admit(0, 1) == 1, "sized rotation: the base admission");
        const std::vector<uint8_t> old_b = bytes_like(5, 1024);
        const std::vector<uint8_t> cold_b = bytes_like(67, 512);
        const std::vector<uint8_t> hot_b = bytes_like(129, 512);
        check(c.fill_slot_queued(0, old_b.data(), err, 1024), "sized rotation: the lent slot holds the old expert");
        check(c.fill_slot_queued(1, cold_b.data(), err, 512), "sized rotation: the head slot holds the cold expert");
        check(c.sync_queued(err), "sized rotation: the fill fence");

        check(c.copy_slot_queued(0, 1, 512, err), "sized rotation: the cold blob is copied into the lent slot");
        check(c.fill_slot_queued(1, hot_b.data(), err, 512), "sized rotation: hot is filled into the head slot");
        check(c.sync_queued(err), "sized rotation: the rotation fence");

        check(c.slot_of(0, 0) == 0 && c.slot_of(0, 1) == 1 && c.slot_of(0, 2) == strata::core::kNotResident,
              "sized rotation: no mapping is published before the remap");
        std::vector<uint8_t> expected = old_b;
        for (size_t i = 0; i < cold_b.size(); ++i) expected[i] = cold_b[i];
        check(c.verify_slot(0, expected.data(), err, 1024),
              "sized rotation: the cold blob lands and the slack keeps the old bytes");
        check(c.verify_slot(1, hot_b.data(), err, 512), "sized rotation: the head slot holds the hot blob");

        check(c.promote_remap(0, 0, 1, 2, 0, 1, err), "sized rotation: the remap is accepted");
        check(c.slot_of(0, 0) == strata::core::kNotResident && c.slot_of(0, 1) == 0 && c.slot_of(0, 2) == 1,
              "sized rotation: old leaves, cold and hot take their slots");
        check(c.slot_of(0, 3) == strata::core::kNotResident, "sized rotation: the untouched expert stays non-resident");
    }

    // 6. a remap that misses any precondition declines and changes nothing: a stale mapping, an already-resident
    //    hot expert, an invalid or equal slot, and a second expert of the layer owning either slot.  Duplicates are
    //    built exactly as the counters produce them - normal admission, then per-layer admission, whose cursor
    //    starts at the layer's first slot and can hand out a slot the table already gave.
    {
        const int64_t L = 2, E = 6;
        {
            ExpertCache c;
            if (!c.open(8, L, E, 256, err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
            check(c.admit(0, 0) == 0 && c.admit(0, 1) == 1, "remap: the base admission");
            remap_must_decline(c, L, E, 0, 0, 1, 2, 2, 1, "remap: a stale lent slot declines");
            remap_must_decline(c, L, E, 0, 0, 1, 2, 0, 3, "remap: a stale head slot declines");
            remap_must_decline(c, L, E, 0, 0, 1, 2, 0, 0, "remap: equal slots decline");
            remap_must_decline(c, L, E, 0, 0, 1, 2, -1, 1, "remap: a negative slot declines");
            remap_must_decline(c, L, E, 0, 0, 1, 2, 0, 8, "remap: a slot past the arena declines");
            remap_must_decline(c, L, E, 0, E, 1, 2, 0, 1, "remap: an expert past the table declines");
            remap_must_decline(c, L, E, -1, 0, 1, 2, 0, 1, "remap: a bad layer declines");
            check(c.admit(0, 2) == 2, "remap: the hot expert is admitted");
            remap_must_decline(c, L, E, 0, 0, 1, 2, 0, 1, "remap: an already-resident hot expert declines");
            check(c.slot_of(0, 2) == 2, "remap: and the hot expert keeps its slot");
        }
        {   // duplicate ownership of the lent slot only
            ExpertCache c;
            if (!c.open(8, L, E, 256, err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
            check(c.admit(0, 0) == 0 && c.admit(0, 1) == 1, "remap: duplicate fixture, old expert first");
            c.set_per_layer_admission(true);
            check(c.admit(0, 5) == 0, "remap: a second expert is given the lent slot");
            check(c.slot_of(0, 0) == 0 && c.slot_of(0, 5) == 0, "remap: two experts own the lent slot");
            remap_must_decline(c, L, E, 0, 0, 1, 2, 0, 1,
                               "remap: duplicate ownership of the lent slot declines");
        }
        {   // duplicate ownership of the head slot only: cold takes the first slot, old the second
            ExpertCache c;
            if (!c.open(8, L, E, 256, err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
            check(c.admit(0, 1) == 0 && c.admit(0, 0) == 1, "remap: duplicate fixture, cold expert first");
            c.set_per_layer_admission(true);
            check(c.admit(0, 5) == 0, "remap: a second expert is given the head slot");
            check(c.slot_of(0, 1) == 0 && c.slot_of(0, 5) == 0, "remap: two experts own the head slot");
            remap_must_decline(c, L, E, 0, 0, 1, 2, 1, 0,
                               "remap: duplicate ownership of the head slot declines");
        }
    }

    // 7. copy_slot_queued's own bounds: a non-positive or oversized copy, and a sized slot too small on either
    //    side, is refused before anything is queued, so the destination keeps every byte it had.
    {
        ExpertCache c;
        if (!c.open(2, 1, 2, 1024, err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
        const std::vector<uint8_t> dst_b = bytes_like(13, 1024);
        const std::vector<uint8_t> src_b = bytes_like(201, 1024);
        check(c.fill_slot_queued(0, dst_b.data(), err), "d2d: the destination starts filled");
        check(c.fill_slot_queued(1, src_b.data(), err), "d2d: the source starts filled");
        check(c.sync_queued(err), "d2d: the fill fence");
        check(!c.copy_slot_queued(0, 1, 0, err), "d2d: zero bytes are refused");
        check(!c.copy_slot_queued(0, 1, -4, err), "d2d: negative bytes are refused");
        check(!c.copy_slot_queued(0, 1, 1025, err), "d2d: a copy larger than a uniform slot is refused");
        check(c.sync_queued(err), "d2d: the refused copies queue nothing");
        check(c.verify_slot(0, dst_b.data(), err, 1024), "d2d: a refused uniform copy leaves the destination");
    }
    {
        ExpertCache c;
        const std::vector<int64_t> small = {256, 1024};
        if (!c.open_sized(small, 1, 2, err)) { std::fprintf(stderr, "open_sized: %s\n", err.c_str()); return 1; }
        const std::vector<uint8_t> dst_b = bytes_like(23, 256);
        const std::vector<uint8_t> src_b = bytes_like(211, 1024);
        check(c.fill_slot_queued(0, dst_b.data(), err, 256), "d2d sized: the destination starts filled");
        check(c.fill_slot_queued(1, src_b.data(), err, 1024), "d2d sized: the source starts filled");
        check(c.sync_queued(err), "d2d sized: the fill fence");
        // 257 fits the source's room but not the 256-byte destination's
        check(!c.copy_slot_queued(0, 1, 257, err), "d2d sized: a copy larger than the destination's room is refused");
        check(c.sync_queued(err), "d2d sized: the refused copy queues nothing");
        check(c.verify_slot(0, dst_b.data(), err, 256), "d2d sized: the destination keeps its bytes");
        // 257 fits the destination's room but not the 256-byte source's
        check(!c.copy_slot_queued(1, 0, 257, err), "d2d sized: a copy larger than the source's room is refused");
        check(c.sync_queued(err), "d2d sized: the second refusal queues nothing");
        check(c.verify_slot(1, src_b.data(), err, 1024), "d2d sized: the other destination keeps its bytes");
    }

    if (g_fail != 0) {
        std::fprintf(stderr, "expert cache slot guard: %d checks failed\n", g_fail);
        return 1;
    }
    std::puts("expert cache slot guard: uniform and sized caches, with and without the guard, and the rotation, passed");
    return 0;
}
