// src/core/adapt_budget_test.cpp - the implicit adaptive swap count (header-only, no GPU needed).
#include "strata/core/adapt_budget.hpp"

#include <cstdio>
#include <limits>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    using strata::core::AdaptCopyBudget;
    constexpr int64_t kBlob = 3000000;   // an expert blob of 3 MB, near the IQ3_S pack's

    AdaptCopyBudget b;
    const double invalid_readings[] = {
        0.0, -3.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()
    };

    // no candidates: nothing to copy
    b.on = true;
    CHECK(b.count(0, [](int64_t) { return (uint64_t) kBlob; }) == 0);
    CHECK(b.count(-4, [](int64_t) { return (uint64_t) kBlob; }) == 0);

    // no measurement yet: the fixed default, and never more candidates than there are
    CHECK(b.count(500, [](int64_t) { return (uint64_t) kBlob; }) == AdaptCopyBudget::kMinSwaps);
    CHECK(b.count(40, [](int64_t) { return (uint64_t) kBlob; }) == 40);
    CHECK(b.count(96, [](int64_t) { return (uint64_t) kBlob; }) == 96);

    // a measured window that only affords a few swaps still gets the floor
    b.win_ms = 1.0;
    b.link_gbps = 25.0;   // 1 ms * 25 GB/s * 0.5 = 12.5 MB -> 4 blobs
    CHECK(b.count(300, [](int64_t) { return (uint64_t) kBlob; }) == AdaptCopyBudget::kMinSwaps);

    // a moderate window: 30 ms * 25 GB/s * 0.5 = 375 MB -> 125 blobs
    b.win_ms = 30.0;
    b.link_gbps = 25.0;
    CHECK(b.count(300, [](int64_t) { return (uint64_t) kBlob; }) == 125);

    // the ceiling holds however fast the link or long the window
    b.win_ms = 10000.0;
    b.link_gbps = 100.0;
    CHECK(b.count(5000, [](int64_t i) {
        return i < AdaptCopyBudget::kMaxSwaps ? (uint64_t) kBlob : (uint64_t) 0;
    }) == AdaptCopyBudget::kMaxSwaps);

    // An invalid link reading keeps the legacy count.
    for (double reading : invalid_readings) {
        b.link_gbps = reading;
        CHECK(b.count(300, [](int64_t) { return (uint64_t) kBlob; }) == AdaptCopyBudget::kMinSwaps);
    }

    // an unknown blob size falls back to the fixed default
    b.win_ms = 10000.0;
    b.link_gbps = 100.0;
    CHECK(b.count(300, [](int64_t i) { return i == 200 ? (uint64_t) 0 : (uint64_t) kBlob; }) ==
          AdaptCopyBudget::kMinSwaps);

    // the prefix is taken in gain order and does not skip a candidate that does not fit
    b.win_ms = 30.0;   // 375 MB allowed
    b.link_gbps = 25.0;
    CHECK(b.count(300, [](int64_t i) { return i < 100 ? (uint64_t) kBlob : (uint64_t) (1ull << 40); }) == 100);

    // the mode being off ignores every reading
    b.on = false;
    b.win_ms = 10000.0;
    b.link_gbps = 100.0;
    CHECK(b.count(300, [](int64_t) { return (uint64_t) kBlob; }) == AdaptCopyBudget::kMinSwaps);

    // Invalid windows must not poison later measurements or increase the copy count.
    AdaptCopyBudget w;
    w.link_gbps = 25.0;
    w.observe(30.0);   // off: ignore the sample
    CHECK(w.count(1000, [](int64_t) { return (uint64_t) kBlob; }) == AdaptCopyBudget::kMinSwaps);
    w.on = true;
    for (double reading : invalid_readings) w.observe(reading);
    CHECK(w.count(1000, [](int64_t) { return (uint64_t) kBlob; }) == AdaptCopyBudget::kMinSwaps);
    w.observe(40.0);   // 500 MB, enough for 166 blobs
    CHECK(w.count(1000, [](int64_t) { return (uint64_t) kBlob; }) == 166);
    for (double reading : invalid_readings) w.observe(reading);
    CHECK(w.count(1000, [](int64_t) { return (uint64_t) kBlob; }) == 166);
    w.observe(80.0);   // the recent average is 50 ms: 625 MB, enough for 208 blobs
    CHECK(w.count(1000, [](int64_t) { return (uint64_t) kBlob; }) == 208);

    if (fails == 0) std::printf("adapt_budget_test: PASS\n");
    return fails == 0 ? 0 : 1;
}
