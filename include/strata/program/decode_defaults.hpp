// include/strata/program/decode_defaults.hpp - the pure rules behind two 0.1.43 decode defaults, checkable without a
// GPU or a model (the same shape as split_rules.hpp).  generate.cpp calls them; tests/program/decode_defaults_test.cpp
// pins their boundaries.
#pragma once

#include <cstddef>
#include <cstdlib>

namespace strata::program::decode_defaults {

/// STRATA_TAIL_SKIP_MAX_CACHED is a percent (101 = never).  Null or empty: `default_frac` (0.90 on CUDA, 0.60 on a
/// HIP layer split).
inline double max_cached_fraction(const char* percent_env, double default_frac) {
    return percent_env != nullptr && percent_env[0] != 0 ? std::atof(percent_env) / 100.0 : default_frac;
}

/// #1884: is nearly every expert in the GPU cache (at least `max_cached` of the total)?  An empty table is not: there
/// is nothing to judge.  At `max_cached` 0 every non-empty table counts (the skip is then never the default).
inline bool nearly_all_cached(size_t cached, size_t total, double max_cached) {
    return total != 0 && (double) cached >= max_cached * (double) total;
}

/// The route tail skip's default: on unless the user asked (STRATA_ROUTE_TAIL_SKIP set), every expert is resident,
/// nearly every expert is cached, or a batch/peer layout mixes requests (one request's skip would then depend on
/// another's tokens).  Callers AND in the architecture check (decode_defaults_arch_ok).
inline bool tail_skip_default_ok(bool env_set, bool all_resident, size_t cached, size_t total, double max_cached,
                                 bool batch, bool peer) {
    return !env_set && !all_resident && !batch && !peer && !nearly_all_cached(cached, total, max_cached);
}

/// 0.1.43: may the canonical pack (no --native) probe the link and let PcieAuto refine the share?  CUDA only; Windows
/// CUDA needs the opt-in (STRATA_PCIE_AUTO_CANON=1); STRATA_PCIE_FRAC_DEFAULT=old keeps 0.1.42.  `windows`/`hip` are
/// the build's platform, passed as constants by generate.cpp.
inline bool canon_pcie_auto_ok(bool pf_old, bool windows, bool windows_opt_in, bool hip) {
    return !hip && !pf_old && (!windows || windows_opt_in);
}

}  // namespace strata::program::decode_defaults
