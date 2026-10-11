# Release v0.1.43: selected V100 performance changes

## Source and scope

Source: [upstream release v0.1.43](https://github.com/Niko1221/Strata/releases/tag/v0.1.43),
commit `7695859aa13afb248bdb4d36f9b2e95707db079d`.
The fork baseline is `29c4fd78621dacec19caa549cfb7f90889692bcd`.
This is a selective port. It is not a full release sync.
Keep the fork's Volta kernels, prompt planner, layer-split path, and MTP work.
The engine version remains 0.1.42.

## Selected changes

- Use one warp per token in the resident-expert route swap. The routing option
  remains off by default. Set `STRATA_ROUTE_RESIDENT=1` to use it. The parallel
  parity check compares IDs and counters exactly on finite logits. It permits
  a weight difference of eight ULP. This is not a NaN-handling guarantee.
- Overlap adaptive expert copies with the next decode window on Linux CUDA.
  Keep the upstream exclusions for layer splits and resident RAM sources.
  Set `STRATA_ADAPT_OVERLAP=0` to use the previous copy schedule.
  Windows requires `STRATA_ADAPT_OVERLAP=1`.
- Use the measured PCIe share for the canonical Q2_0 pack on Linux CUDA.
  Windows requires `STRATA_PCIE_AUTO_CANON=1`.
- Keep the automatic route tail skip off when at least 90% of experts are in
  the GPU cache. An explicit `STRATA_ROUTE_TAIL_SKIP` takes precedence.
  `STRATA_TAIL_SKIP_MAX_CACHED` changes the percentage limit.
- Upload prompt step records and token IDs from pinned buffers. Keep each
  buffer until its asynchronous copy is complete. This preserves the fork's
  prompt planner. No separate prompt-speed gain is claimed.

## Changes not selected

- The Pascal SM60 kernel table does not apply to Volta SM70.
- The Ampere Q3_K interleaved table is measured for SM86, not SM70.
  Keep the fork's measured Volta dense-matvec table.
- AMD and Intel kernels and pipeline defaults do not apply to CUDA V100.
- Windows file mapping changes do not apply to this Linux measurement.
- Multi-socket NUMA changes are host-specific. The measurement host has one
  socket. No multi-socket performance claim is made.
- The NaN recovery feature and API timing fixes are outside this performance
  port. This change does not claim a complete v0.1.43 bug-fix sync.

## Verification

See the [V100 verification report](../benchmarks/v100-upstream-v0.1.43-2026-10-11.md)
for the measured results and operating limits.
