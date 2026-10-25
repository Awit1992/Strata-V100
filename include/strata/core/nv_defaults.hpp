// strata/core/nv_defaults.hpp - per-architecture defaults for NVIDIA sm_86, sm_70 and sm_60.
//
// 148-nv measured every switch of the 0.1.45 bit-exact ON set, one at a time and in the stacked set, on an RTX 3060, an RTX A4000
// (sm_86) and a Tesla P100 (sm_60): a switch is in a table only when its token ids equal the default's and it was faster on that
// architecture (5+ interleaved pairs of fresh engines, noise arm, equal expert-cache slots; notes-148-nv.md).  The identity gate
// (10 prompts + exact_pp 4K/20K, fixed flags) is IDENTICAL with that original table on. Switches that changed bits,
// hung or lost speed were rejected from that measured set.
//
// Two sm_70 entries extend that set with a different contract:
//   * STRATA_ADAPT_OVERLAP=1 on Linux CUDA moves the adaptive copies under the next window. The implicit copy budget
//     can change cache residency; it does not guarantee identical token IDs. An explicit --adapt-swaps N fixes the count.
//   * STRATA_PF_WMMA2=1 selects the second prompt-expert tile inside the opt-in STRATA_PF_WMMA=1 path. It keeps
//     FP16 operands and FP32 sums; a run without WMMA is unchanged.
//
// The other eleven sm_70 entries are a combined stack ported from upstream PR #3411 (koshikawa-masato):
// dual V100 PCIe 32 GB IQ3_S/MTP4 measured +5.8/+7.6 % decode for the first six and +4.7/+4.8 % for the
// seven additions, with ten decoded texts matching. That is combined-stack evidence, not the token-id gate
// above. This fork's 2xV100 16 GB A/B/A/B/A/B measured 83.1 vs 74.6 tok/s median (+11.4 %).
//
//   STRATA_SM86_PERF=0   turns the sm_86 table off (cards of compute capability 8.6: RTX 3050-3090, RTX A2000-A6000)
//   STRATA_SM70_PERF=0   turns the sm_70 table off (Volta: Tesla V100, Titan V)
//   STRATA_SM60_PERF=0   turns the sm_60 table off (Pascal GP100: Tesla P100)
//
// The sm_70 table on Linux CUDA also turns on adaptive overlap (STRATA_ADAPT_OVERLAP=1): the copies land while the next
// verify window runs. The serve log names the state and the reason when it cannot run. A layer
// split, an unusable buffer, a resident RAM source without STRATA_ADAPT_OVERLAP_RESIDENT=1, or the user's own
// STRATA_ADAPT_OVERLAP setting (0 included) keeps the previous behaviour.  Without --adapt-swaps the round's count
// follows the window (include/strata/core/adapt_budget.hpp); an explicit --adapt-swaps N stays fixed.
//
// A switch the user set (to anything, including 0) is never overridden.  Applied once at start, before any engine code reads its
// switches; the table is used only when every visible CUDA device has the same compute capability.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The environment settings that are the defaults of a card with compute capability major.minor (raw, as the driver reports it);
/// empty for every other card and when its off switch is 0.
std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor);

/// Sets `nv_default_env(major, minor)` in this process's environment where the user has not set the variable, prints one line to
/// stderr when it set anything, and returns the names it set.
std::vector<std::string> apply_nv_defaults(int major, int minor);

}  // namespace strata::core
