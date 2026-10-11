// The decode-default rules of generate.cpp (0.1.43): the route tail skip's residency gate (#1884) and the canonical
// pack's PCIe calibration opt-in.  No GPU, no model.
#include <cstdio>

#include "strata/program/decode_defaults.hpp"

using namespace strata::program::decode_defaults;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {

    // The gate fires at OR ABOVE the threshold (>=), and never for an empty table
    CHECK(nearly_all_cached(90, 100, 0.90));       // exactly at the line: the gate fires
    CHECK(!nearly_all_cached(89, 100, 0.90));      // one below: the skip stays the default
    CHECK(nearly_all_cached(100, 100, 0.90));
    CHECK(!nearly_all_cached(0, 0, 0.90));         // no table: nothing to judge
    CHECK(nearly_all_cached(0, 100, 0.0));         // 0%: any non-empty table counts (the skip is then never the default)
    CHECK(!nearly_all_cached(100, 100, 1.01));     // 101%: never fires (STRATA_TAIL_SKIP_MAX_CACHED=101)
    // the running model's numbers: 24576 experts, 90% = 22118.4
    CHECK(!nearly_all_cached(22118, 24576, 0.90));
    CHECK(nearly_all_cached(22119, 24576, 0.90));
    CHECK(!nearly_all_cached(14005, 24576, 0.90)); // the Q2_0 service's 57%: the gate is far from firing

    // tail_skip_default_ok: on by default, and off for each reason
    CHECK(tail_skip_default_ok(false, false, 9044, 24576, 0.90, false, false));   // the IQ3_S service (36.8% cached)
    CHECK(!tail_skip_default_ok(true, false, 9044, 24576, 0.90, false, false));   // STRATA_ROUTE_TAIL_SKIP was set
    CHECK(!tail_skip_default_ok(false, true, 24576, 24576, 0.90, false, false));  // every expert resident in VRAM
    CHECK(!tail_skip_default_ok(false, false, 22119, 24576, 0.90, false, false)); // nearly every expert cached (#1884)
    CHECK(!tail_skip_default_ok(false, false, 9044, 24576, 0.90, true, false));   // --batch: a window mixes requests
    CHECK(!tail_skip_default_ok(false, false, 9044, 24576, 0.90, false, true));   // a peer/helper GPU
    CHECK(tail_skip_default_ok(false, false, 0, 0, 0.90, false, false));         // no residency table: no reason to refuse

    // canon_pcie_auto_ok: Linux CUDA refines the canonical share unless STRATA_PCIE_FRAC_DEFAULT=old
    CHECK(canon_pcie_auto_ok(false, false, false, false));
    CHECK(!canon_pcie_auto_ok(true, false, false, false));
    // Windows CUDA: opt-in (STRATA_PCIE_AUTO_CANON=1)
    CHECK(!canon_pcie_auto_ok(false, true, false, false));
    CHECK(canon_pcie_auto_ok(false, true, true, false));
    CHECK(!canon_pcie_auto_ok(true, true, true, false));
    // HIP: never
    CHECK(!canon_pcie_auto_ok(false, false, false, true));

    if (failed == 0) std::printf("decode_defaults_test: ok\n");
    return failed ? 1 : 0;
}
