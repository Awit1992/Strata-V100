// strata/core/adapt_budget.hpp - the implicit adaptive tier's swap count (STRATA_ADAPT_OVERLAP, single GPU).
//
// The adaptive tier's copies land while the next decode window runs, so their time is hidden only as far as one
// window's work is longer than the copies take.  With --adapt-swaps the operator fixes the count; without it the
// count follows the measured window. Counts above the legacy minimum must fit kWindowShare of the recent verify
// window's own time at the measured host->device bandwidth. The gate's wait is excluded so a slow copy round
// cannot raise the budget that caused it. The legacy minimum remains when that budget is smaller or a reading is
// missing. kMaxSwaps bounds the round however fast the link or long the window.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace strata::core {

/// The implicit adaptive swap count of an overlapped round.  Header-only and free of GPU state so a host test can
/// check the bounds and the fallback (src/core/adapt_budget_test.cpp).
struct AdaptCopyBudget {
    static constexpr int64_t kMinSwaps = 96;        ///< the fixed default, and the floor of the measured count
    static constexpr int64_t kMaxSwaps = 384;       ///< the ceiling of the measured count
    static constexpr double kWindowShare = 0.5;     ///< counts above the legacy minimum must fit half of one window
    static constexpr double kNewWindowWeight = 0.25;  ///< the newest window's share of the moving average

    bool on = false;          ///< the mode is eligible and the operator did not fix --adapt-swaps
    double link_gbps = 0.0;   ///< the measured host->device bandwidth (0: not measured)
    double win_ms = 0.0;      ///< the recent verify window's time, the overlap gate's wait taken out
    bool have_win = false;    ///< one window has been seen

    /// One decode window's verify time in ms, with the overlap gate's wait already subtracted.  A measurement that
    /// is not finite or not positive leaves the average unchanged.
    void observe(double verify_ms) {
        if (!on || !(verify_ms > 0.0) || !std::isfinite(verify_ms)) return;
        win_ms = have_win ? win_ms * (1.0 - kNewWindowWeight) + verify_ms * kNewWindowWeight : verify_ms;
        have_win = true;
    }

    /// The allowed swap count for `candidates` swaps in gain order; `blob_bytes_of(i)` gives the bytes of the i-th
    /// one.  The result is the longest prefix whose copies fit kWindowShare of `win_ms` at `link_gbps`, clamped to
    /// [kMinSwaps, kMaxSwaps] and to `candidates`.  A missing reading, a zero size, or the mode being off returns
    /// the fixed default (kMinSwaps); 384 is never exceeded.
    template <class BlobBytesOf>
    int64_t count(int64_t candidates, BlobBytesOf blob_bytes_of) const {
        if (candidates <= 0) return 0;
        if (!on || !(win_ms > 0.0) || !(link_gbps > 0.0) ||
            !std::isfinite(win_ms) || !std::isfinite(link_gbps))
            return std::min<int64_t>(candidates, kMinSwaps);
        if (candidates <= kMinSwaps) return candidates;
        // ms * GB/s = 1e-3 s * 1e9 bytes/s = 1e6 bytes
        const double allowed = win_ms * link_gbps * 1e6 * kWindowShare;
        double used = 0.0;
        int64_t k = 0;
        for (int64_t i = 0; i < std::min(candidates, kMaxSwaps); ++i) {
            const double bytes = (double) blob_bytes_of(i);
            if (!(bytes > 0.0)) return kMinSwaps;   // an unknown copy size: the fixed default
            if (used + bytes > allowed) break;
            used += bytes;
            ++k;
        }
        return std::max(k, kMinSwaps);
    }
};

}  // namespace strata::core
