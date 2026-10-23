# IQ3_S prefill and decode study, 2026-10-23

## Scope

All performance and branch-validation model tests use the regular IQ3_S assets.
They do not use the daily abliterated model.
The pack, tokenizer, native shard, and PLE shard match.
The tests keep the model quantization unchanged.
They keep INT8 KV and a 524,288-cell logical context limit.
KV growth starts with 16,384 physical cells.
The tests do not reduce the context limit to obtain a speed gain.

The prefill change connects the existing second WMMA tile to grouped experts.
It supports IQ2_S gate/up and IQ4_NL down projections.
It accepts resident and streamed expert pointers and gathered input rows.
It also bounds the final raw-weight load for odd IQ2_S super-block counts.
Other shapes keep the existing tile.
`STRATA_PF_WMMA2=1` remains opt-in.
It also requires `STRATA_PF_WMMA=1` and the experimental CUDA build.

## Prefill result

Four fresh engine runs use the order control, W2, W2, control.
Each run reads two fixed prompts at each size.
Each request reads its full prompt and generates one token.
No request reuses a conversation prefix.
Each table row contains four measurements per arm.

The table reports the median engine prompt time.
The throughput is the prompt length divided by that median time.
The speed gain is the control time divided by the W2 time, minus one.
It is not the median of the individual throughput values.

| Prompt tokens | Control time, ms | W2 time, ms | Control tok/s | W2 tok/s | Speed gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 2117.65 | 2067.85 | 967.11 | 990.40 | 2.41% |
| 8,192 | 5735.85 | 5685.25 | 1428.21 | 1440.92 | 0.89% |
| 32,768 | 16121.45 | 15988.15 | 2032.57 | 2049.52 | 0.83% |

The prompt hashes match across all four runs.
The generated token also matches for every corresponding request.
This is not a general model-quality evaluation.
The small gains do not establish a universal speedup on other hardware.

Raw records:
[control A](../bench/results/2026-10-23-iq3-prefill-decode/prefill-control-a.json),
[W2 A](../bench/results/2026-10-23-iq3-prefill-decode/prefill-w2-a.json),
[W2 B](../bench/results/2026-10-23-iq3-prefill-decode/prefill-w2-b.json),
[control B](../bench/results/2026-10-23-iq3-prefill-decode/prefill-control-b.json).

### Prefill controls

- Two Tesla V100-PCIE-16GB GPUs.
- GPU 0 uses PCIe Gen3 x4 and layers 0-15.
- GPU 1 uses PCIe Gen3 x16 and layers 16-47.
- Both GPU power limits are 175 W.
- CPU: Ryzen 5 3600, six cores and twelve threads.
- Usable system RAM: 94.2 GiB. Model storage: a local ext4 volume.
- Operating system: Linux 7.0.0-38-generic.
- NVIDIA driver: 580.178.04. CUDA compiler: 12.8. Release SM70 build.
- Layer split: `16`. Pipeline windows: `2`. Expert ring: `192`.
- Maximum context: `524288`. KV format: `int8`. VRAM reserve: `750` MiB.
- CPU prefill share: `auto`. Stream threshold: `3072` tokens.
- The WMMA, skipped-unrouted, SM70-table, and wide-dequant paths stay enabled.
- KV growth and embedding reuse accounting stay enabled.
- The checkpoint pool and checkpoint worker stay enabled.
- Expert-cache promotion stays at `2048` per stage.
- Resident-aware routing stays enabled. Mixer overlap stays at `0`.
- The seed is `20261003`. The prompt pool contains 69,773 tokens.

All four arms use executable SHA-256:
`6eab03bc2b9caac384b13ba168f7310e78b6c19afcfb83b711c6ca9bfa1de358`.
Only `STRATA_PF_WMMA2` differs between the two arms.
The executable contains the W2 wiring and the bounded-load correction.

The test does not set clock locks.
The maximum sampled temperatures are 54 C and 72 C.
No sample reports thermal slowdown.
Some samples report the software power-cap flag.
The runs have no cooldown delay.
The expert arena is in host memory before request timing starts.
The test does not clear the operating system file cache between runs.
Cold and later requests remain in the measurements.

The harness has an older strict preset with a 700 MiB reserve.
These runs use `--no-settings-strict` because both arms retain 750 MiB.
The mismatch is not a change between arms.
The raw records contain the effective arguments and environment settings.

### Prefill reproduction

Build the experimental CUDA targets as described in
[the V100 guide](../docs/NVIDIA_V100.md).
Use a private configuration with the regular IQ3_S assets and the controls above.
Do not publish API keys or private configurations.

Run `bench/run_v100_prefill.py` with these common arguments:

```sh
python bench/run_v100_prefill.py \
  --config REGULAR_IQ3_CONFIG.json --exe build-v100/strata \
  --targets 2048,8192,32768 --repeats 2 --max-new 1 \
  --seed 20261003 --no-settings-strict --no-cooldown \
  --env STRATA_PF_WMMA=1 --env STRATA_SIDE_MIXER=0 \
  --env STRATA_PF_WMMA2=0 --arm-label control --out bench/results/control
```

Keep the remaining environment settings from the raw records.
For W2, change only `STRATA_PF_WMMA2=0` to `STRATA_PF_WMMA2=1`.
Use a new output directory and arm label for each run.
Stop other inference engines on the test GPUs before each run.
Repeat in the order control, W2, W2, control.

## Matched-work decode result

This comparison uses the corrected mixer overlap on physical GPU 1 only.
Both arms use the same 2,048-token prompt and forced continuation.
Each run emits 128 tokens in 128 two-row verify rounds.
MTP, suffix drafts, and lookup chains are off.
The first native-pack verify window has one row.

The logical context limit stays at 524,288 cells with INT8 KV.
Both arms start with 16,384 physical KV cells.
The expert-cache budget argument is `4000`.
Both engines report 4,831 actual slots and 9.20 GiB.
The PCIe share stays at `0.35`.
Cache adaptation, cache promotion, resident-biased routing, and tail skipping
stay off to keep execution placement fixed.
W2 stays off to isolate mixer overlap.
No timing diagnostics or window-hash collection run in these timing arms.

Four fresh processes use the order inline, overlap, overlap, inline.
Each arm has two measurements.
The table uses the median of each pair.
Throughput is 128 tokens divided by the median total decode time.

| Metric | Inline | Corrected overlap | Change |
| --- | ---: | ---: | ---: |
| Total decode time, ms | 8479.50 | 8340.25 | 1.64% less time |
| Derived decode tok/s | 15.10 | 15.35 | 1.67% more throughput |
| GPU-reach wait, ms/round | 14.8535 | 14.2185 | 4.28% less wait |

GPU-reach wait is the host's per-layer wait for the GPU.
It is not a pure kernel-time profile.
The CPU code is unchanged. CPU pool times also vary between arms.
Treat this small total gain as specific to this host and fixed workload.
Do not interpret it as normal speculative-generation throughput.
The next section reports the separate pipelined MTP result.

All four timing arms use executable SHA-256:
`68a139bda79240c930c69c1612f73e597c12f64493ffeb01c99d9eefa354d919`.
Raw records:
[inline A](../bench/results/2026-10-23-iq3-prefill-decode/release-control-a.json),
[overlap A](../bench/results/2026-10-23-iq3-prefill-decode/release-overlap-a.json),
[overlap B](../bench/results/2026-10-23-iq3-prefill-decode/release-overlap-b.json),
[inline B](../bench/results/2026-10-23-iq3-prefill-decode/release-control-b.json).

## Normal pipelined MTP result

A separate four-run comparison tests the corrected mixer overlap.
It uses the order inline, overlap, overlap, inline.
Each process receives three prompts at 2K and three at 8K.
The output cap is 256 tokens per request.
Pipeline windows stay at 2. MTP and the existing cache policies stay enabled.
W2 stays off in both arms to isolate mixer overlap.

The inputs match, but the generated continuations and draft acceptance differ.
The table divides total generated tokens by total engine decode time.
It does not establish a matched-work speed gain.

| Prompt size | Inline tokens | Inline decode, ms | Inline tok/s | Overlap tokens | Overlap decode, ms | Overlap tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 1289 | 14046.6 | 91.77 | 1365 | 15283.4 | 89.31 |
| 8K | 1119 | 11916.9 | 93.90 | 731 | 8258.9 | 88.51 |

These runs do not show a reliable normal-MTP throughput gain.
Do not quote the fixed-work result as a gain for this workload.
The final patch therefore leaves the Volta architecture defaults unchanged.
Mixer overlap remains an explicit `STRATA_SIDE_MIXER=2` option.

The trial executable selected mode 2 automatically in the overlap arms.
Its SHA-256 is `ff66647439ade627c5692460c027916ca21db30a44f68e2171007a00efecff17`.
That default-table change is removed from the final patch.
An explicit mode 2 selects the same corrected graph path.

Raw records:
[inline A](../bench/results/2026-10-23-iq3-prefill-decode/decode-control-a.json),
[overlap A](../bench/results/2026-10-23-iq3-prefill-decode/decode-overlap-a.json),
[overlap B](../bench/results/2026-10-23-iq3-prefill-decode/decode-overlap-b.json),
[inline B](../bench/results/2026-10-23-iq3-prefill-decode/decode-control-b.json).

## Indexer snapshot order during mixer overlap

The study found a state error in both mixer-overlap modes.
The indexer branch takes a tail snapshot before it appends speculative rows.
The main stream also takes a snapshot into the same destination.
That second copy has no dependency on the branch.
It can overwrite the correct snapshot after the speculative append.

The commit graph restores this snapshot and appends only the accepted rows.
An incorrect snapshot can retain a rejected row's key.
When an accepted row completes a pooled indexer block, that key changes the block.
The error can remain hidden while attention selects the full history.
It becomes observable when sparse selection uses the affected scores.

The fixed-work regression starts with a 2,048-token prompt.
It uses two-row verify windows and accepts one row per window.
Before the fix, both overlap modes first differ from control at window 33.
That window starts at position 2080.
An identical control repeat matches all 64 windows.
A fixed PCIe share does not remove the failure.
The position is an observed property of this input, not a special case in code.

The fix removes the duplicate main-stream snapshot.
The branch keeps its snapshot before its append.
The inline path uses the same existing snapshot helper.
No projection arithmetic, model quantization, KV format, or KV capacity changes.

The new model-free QSA case checks the rollback invariant at a block boundary.
It compares the restored state with the canonical accepted-prefix state.
It also proves that a post-window snapshot corrupts the pooled row.
The full window-graph regression requires the regular model assets.

After the fix, modes 1 and 2 match the control for all 64 windows.
The corrected overlap path also matches the control for 128 two-row windows
and 64 eight-row windows.
The comparison includes the final residual hash and all window argmaxes.
See the [before-and-after identity record](../bench/results/2026-10-23-iq3-prefill-decode/indexer-identity.json).
Labels ending in `default` refer to a trial build that selected mode 2 at startup.
The final patch does not change architecture defaults.
Set `STRATA_SIDE_MIXER=2` to select the same corrected path.
The timings in the identity records include hash collection.
Do not use them as performance measurements.

### Fixed-work regression reproduction

Set `PACK`, `NATIVE`, and `PLE` to the matching regular IQ3_S assets.
Set `EXPERT_PROFILE` to an existing expert profile for that pack.
A different profile can change placement, hashes, and performance.
Use the same profile for both arms.
The saved prompt and continuation contain synthetic benchmark data.

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
STRATA_ARENA_LOCK=0 STRATA_WATCHDOG_S=300 STRATA_SPLIT_RING=192 \
STRATA_EXPERT_PAIR=1 STRATA_SM70_TABLE=1 STRATA_DEQUANT_WIDE=1 \
STRATA_KV_GROW=1 STRATA_PF_WMMA=1 STRATA_PF_SKIP_UNROUTED=1 \
STRATA_PF_WMMA2=0 \
STRATA_ROUTE_RESIDENT=0 STRATA_ROUTE_TAIL_SKIP=0 \
STRATA_PREFILL_PROMOTE=0 STRATA_SIDE_MIXER=0 \
build-v100/strata \
  --pack "$PACK" --native "$NATIVE" --ple-gguf "$PLE" \
  --expert-profile "$EXPERT_PROFILE" --expert-cache 4000 --prefill auto \
  --spec 2 --spec-min-p 0 --suffix-draft 0 --lookup-chain 0 \
  --adapt-every 0 --pcie-frac 0.35 --max-context 524288 --kv int8 \
  --rope-scaling yarn --rope-scale 2 --vision --vram-reserve-mib 750 \
  --conversation-cache-mib 0 --prompt-cache 0 \
  --prompt-cache-every 0 --prompt-cache-root 0 \
  --tokens-file bench/results/2026-10-23-iq3-prefill-decode/fixed-prompt.ids \
  --spec-follow bench/results/2026-10-23-iq3-prefill-decode/fixed-follow.ids \
  --max-new 128 --window-hashes /tmp/control.hash
```

Repeat with `STRATA_SIDE_MIXER=2` and `/tmp/overlap.hash`.
Compare the two files with `cmp`.
For the eight-row check, use `--spec 8 --max-new 64` in both arms.
Do not pass `--mtp` in this check.
Each round must emit one forced token and use the same window size.
The first native-pack window has one row.
The continuation is forced. These checks do not measure normal MTP throughput.


## Correctness evidence for W2

The generic and opted-in numerical parity checks pass.
They compare GPU outputs with a double-precision reference.
They check finiteness and untouched output rows and columns.
The cases include contiguous weights, reversed expert pointer tables,
gathered input rows, empty experts, partial tiles, and K=3840 IQ2_S rows.
The weight allocation has no added guard padding in these cases.

CUDA Compute Sanitizer reports zero memory errors with W2 enabled.
The numerical check reports zero failures in the same run.
The check includes the final two-byte word of an odd IQ2_S row.

A separate Nsight Systems capture observes the actual W2 kernels during
regular IQ3_S HTTP inference.
It records 957 IQ4_NL down launches and 444 IQ2_S gate/up launches.
That run passes exact READY, arithmetic, JSON, and long-prompt READY checks.
It is an execution check, not a timing arm.
The [profiler records](../bench/results/2026-10-23-iq3-prefill-decode/profiled-kernels.json)
identify both instantiated kernels.

W2 uses the same stored quantized weights as the first WMMA tile.
It dequantizes operands to FP16 and accumulates in FP32.
Do not assume bitwise agreement with the CPU or integer MMQ path.

## Final verification

The final Release SM70 build passes these five checks:
`verify_parity`, `native_multi_parity`, `fusions_parity`, `qsa_parity`,
and `nv_defaults_test`.
The focused `pf_wmma2_parity` check also passes.
The W2 numerical cases pass under CUDA Compute Sanitizer with zero memory errors.
The benchmark harness passes all 46 unit tests.

Authenticated HTTP tests exercise both optimization options together.
Four text cases pass on one GPU and on the two-GPU layer split.
They require exact READY, arithmetic result 42, a specified JSON object,
and READY after a long prompt.
All eight final checks pass.
See the [single-GPU results](../bench/results/2026-10-23-iq3-prefill-decode/quality-release-single.json)
and [two-GPU results](../bench/results/2026-10-23-iq3-prefill-decode/quality-release-dual.json).

The [verification summary](../bench/results/2026-10-23-iq3-prefill-decode/validation.json)
records the build hash and check outcomes.
These checks do not establish general model quality.
They do not test image inference or other GPU architectures.
No private configuration, credential, or installed engine binary is included.

## Research and rejected candidates

A web research subagent examined primary sources before implementation.
The useful directions were grouped tensor-core work and CPU/GPU scheduling.
The paper results below are not performance claims for this engine.

- [NVIDIA CUTLASS: Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
  explains tile shape, data reuse, occupancy, and software pipelining.
  This supports testing the existing smaller WMMA tile on grouped experts.
  Hopper-specific TMA and warp specialization do not apply to V100.
- [Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models](https://arxiv.org/abs/2402.07033)
  motivates measured placement when GPU memory cannot hold all experts.
  The local placement experiment did not establish a repeatable decode gain.

The study rejects these candidates:

1. A per-stage measured PCIe/CPU share policy.
   A reverse-order comparison regresses aggregate decode throughput.
   The implementation is removed.
2. Two scalar AVX2 IQ3_S index-assembly variants.
   Both pass numerical parity.
   Both make the one-token gate/up microbenchmark slower.
   Both implementations are removed.
3. Forced gathered IQ3_S decode on this Zen 2 CPU.
   The one-token engine gate/up time rises from about 0.33 ms to 0.61 ms.
   The existing CPU selection rule is unchanged.
4. A combined decode-fusion setting.
   Initial MTP results vary too much to justify a default change.
   Fused plan waits alone show no useful gain in the CPU-bound fixed-work run.

No rejected implementation or speculative tuning is installed in the daily service.
