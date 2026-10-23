# IQ3_S prefill and decode study, 2026-10-23

## Scope

All performance and branch-validation model tests use the regular IQ3_S assets.
They do not use the daily abliterated model.
The pack, tokenizer, native shard, and PLE shard match.
The tests keep the model quantization unchanged.
They keep INT8 KV and a 524,288-cell logical context limit.
KV growth starts with 16,384 physical cells.
The tests do not reduce the context limit to obtain a speed gain.

The study has two phases.
Stage 1 added grouped W2 and corrected the mixer snapshot order.
Round 2 broadens the expert prefill tile set and changes the measured cache-overlap defaults.

This report leads with the final round 2 repeated comparisons.
These runs compare the PR 74 baseline binary with the round 2 final binary.
Two fresh process runs per arm measure each prefill cell.
Two fresh process runs per arm and per seed measure the normal MTP route.
The stage 1 measurements use stage 1 binaries and stage 1 harness settings.
They are not measurements of the current defaults.
[Stage 1 historical evidence](#stage-1-historical-evidence) keeps that evidence with its original links.
The earlier pilot runs are superseded.
The section [Superseded pilot observations](#superseded-pilot-observations-historical) keeps them for history only.

## Round 2 results

These are the final repeated comparisons for round 2.
They replace the pilot numbers in this report.
The data set has 20 complete fresh-process runs.
All 20 runs report status ok and no errors.

The top-line observations are:

- Dual-GPU prefill gains are +2.83% to +3.85% with an equal prefill cap.
- Single-GPU prefill gains are +5.18% at 8K and +9.37% at 32K.
  The single-GPU prefill cap also changes from 8192 to 16384.
- Single-GPU 2K prefill is unchanged at -0.10%.
- The all-run normal MTP aggregate change is +5.306%, descriptive only.

### Builds and measured defaults

Both frozen binaries report checkout HEAD `49be5b18` (`v0.2.0-beta-1346-g49be5b18`).
The candidate includes the round 2 source changes, which were not committed when the binary was frozen.
The executable SHA-256 identifies each measured build.
The executable is the only difference between the arms in each cohort.
The harness records the same arguments and environment for both arms.
The expert cache is also equal in both arms of each cohort.
Single-GPU prefill and normal runs report 4,278 slots and 8,357 MiB.
Dual-GPU prefill runs report 11,945 slots and 22,537 MiB.
The automatic PCIe share is equal in both arms:
0.36 for the single-GPU prefill cohort and 0.08 for the dual-GPU prefill cohort.

| Item | Baseline | Candidate |
| --- | --- | --- |
| Executable SHA-256 | `68a139bda79240c930c69c1612f73e597c12f64493ffeb01c99d9eefa354d919` | `add55d48a3ace40ed206843f09fed856b4ba6ea9edde10a2cfef9737c6167e2b` |
| SM70 W2 default | off (opt-in) | on |
| W2 orientation | default only | alternate at `maxRows >= 128` |
| Linux SM70 adaptive copy overlap default | 0 | 1 |
| Single-GPU prefill cap at startup | 8192 | 16384 |
| Dual-GPU prefill cap at startup | 8192 | 8192 |

The model settings do not change between the arms.
Both arms use the regular IQ3_S assets and the matching pack, tokenizer, native shard, and PLE shard.
Both arms use INT8 KV, a logical limit of 524,288 cells, and 16,384 initial physical KV cells.
The single-GPU prefill cap and the dual-GPU prefill cap are engine startup values.
The normal single-GPU runs also report 8192 versus 16384; their 8K prompts fit in one chunk in both arms.

### Prefill

Each cell is the recorded median of two fresh process runs.
For two values, the median is their midpoint.
The four runs follow the ABBA order: baseline, candidate, candidate, baseline.
Each request reads its full prompt and generates one token.
The speed gain is the baseline median time divided by the candidate median time, minus one.
It is not the median of the individual throughput values.

| GPUs | Prompt tokens | Baseline median, ms | Candidate median, ms | Speed gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2,048 | 2245.35 | 2247.60 | -0.10% |
| 1 | 8,192 | 4996.60 | 4750.50 | +5.18% |
| 1 | 32,768 | 19412.05 | 17748.55 | +9.37% |
| 2 | 2,048 | 2229.70 | 2167.10 | +2.89% |
| 2 | 8,192 | 5752.10 | 5593.65 | +2.83% |
| 2 | 32,768 | 16142.75 | 15543.70 | +3.85% |

Single-GPU 2K is unchanged. The difference is -0.10%.
This value is inside the run-to-run spread of the two arms.
Do not report it as a gain.

The single-GPU 8K and 32K comparisons include the prefill cap difference.
The baseline selects 8192 and the candidate selects 16384.
The dual-GPU arms select 8192 in both arms.
The dual-GPU comparison therefore has no prefill-cap difference between the arms.

The prompt hash matches for every corresponding request in all six comparisons.
The one generated token also matches in all six comparisons.
This is a determinism check, not a model-quality evaluation.

### Normal MTP, full route

Each seed has four fresh process runs in ABBA order: baseline, candidate, candidate, baseline.
Each arm therefore has two runs per seed.
Each request reads an 8,192-token prompt.
MTP spec is 8. The minimum p is 0.70. The new-token cap is 512.
The table aggregates the two runs of each arm, per seed.
The rate is the total generated tokens divided by the total decode time.

| Seed | Baseline tokens | Baseline decode, ms | Baseline tok/s | Candidate tokens | Candidate decode, ms | Candidate tok/s | Rate change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261003 | 1,024 | 19756.0 | 51.832 | 791 | 14899.3 | 53.090 | +2.43% |
| 20261004 | 598 | 12220.9 | 48.933 | 888 | 15846.4 | 56.038 | +14.52% |
| 20261005 | 1,024 | 18319.5 | 55.897 | 1,024 | 18045.2 | 56.746 | +1.52% |

All 12 runs together give these descriptive totals:

| Arm | Generated tokens | Decode, ms | tok/s |
| --- | ---: | ---: | ---: |
| Baseline | 2,646 | 50296.4 | 52.608 |
| Candidate | 2,703 | 48790.9 | 55.400 |

The aggregate rate change is +5.306%.
This value is descriptive. It is not a fixed-trace speed gain.
The token counts are not equal.

The outputs differ between the arms.
The accepted draft counts also differ.
Some runs stop at end of sequence:

- Seed 20261003, candidate run b2: 279 tokens.
- Seed 20261004, baseline runs a1 and a2: 299 tokens.
- Seed 20261004, candidate run b1: 376 tokens.

All other runs reach the 512-token cap.
Do not median the raw times of unequal-length runs.
The sample is small: two runs per arm and per seed.
The per-seed percentages vary from +1.52% to +14.52%.
Do not report a universal 20% gain.
Do not attribute the change to one kernel.
The changed defaults include W2, the alternate orientation, the adaptive copy overlap default,
and the single-GPU prefill cap.

### Public records

The [comparison summary](../bench/results/2026-10-23-iq3-prefill-decode/round2-comparison-summary.json)
contains the exact hashes, all recorded times, the per-run lengths, the acceptance counts, and the caveats.
The 20 public raw records follow.
All records are sanitized.

Prefill, single GPU:
[a1](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-single-a1.json),
[b1](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-single-b1.json),
[b2](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-single-b2.json),
[a2](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-single-a2.json).

Prefill, dual GPU:
[a1](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-dual-a1.json),
[b1](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-dual-b1.json),
[b2](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-dual-b2.json),
[a2](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-dual-a2.json).

Normal MTP, seed 20261003:
[a1](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-normal-a1.json),
[b1](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-normal-b1.json),
[b2](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-normal-b2.json),
[a2](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-normal-a2.json).

Normal MTP, seed 20261004:
[a1](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-normal-s20261004-a1.json),
[b1](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-normal-s20261004-b1.json),
[b2](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-normal-s20261004-b2.json),
[a2](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-normal-s20261004-a2.json).

Normal MTP, seed 20261005:
[a1](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-normal-s20261005-a1.json),
[b1](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-normal-s20261005-b1.json),
[b2](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-normal-s20261005-b2.json),
[a2](../bench/results/2026-10-23-iq3-prefill-decode/round2-repeat-normal-s20261005-a2.json).

## Round 2 implementation

### Expert prefill tiling

The prefill change connects the existing second WMMA tile to grouped experts.
The tile now covers four quantized projection types:
IQ3_XXS (type 18), IQ3_S (type 21), and IQ2_S (type 22) for gate/up,
and IQ4_NL (type 20) for down.
The raw-weight prefetch stays safe for an odd super-block count, including K=3840 rows.
The dispatch uses the existing table and ids.
It accepts resident and streamed expert pointers and gathered input rows.
It bounds the final raw-weight load for odd IQ2_S super-block counts.
Other shapes keep the existing tile.

The kernel selects one of two operand orientations.
At `maxRows >= 128`, it uses the alternate orientation: 64 output rows and 128 token rows.
Short groups use the default orientation: 128 output rows and 64 token rows.
Both orientations dequantize operands to FP16.
Both keep the FP32 K order.
Set `STRATA_PF_WMMA2_N64=0` to select the old orientation.
Set `STRATA_PF_WMMA2_N64=1` to force the alternate orientation.
W2 is enabled by default on SM70 only.
It also requires `STRATA_PF_WMMA=1` and the experimental CUDA build.
The CMake configuration registers permanent focused tests for the explicit 0 and 1 settings.

### Adaptive copy overlap

On Linux SM70, `ADAPT_OVERLAP` now defaults to 1.
Windows keeps the opt-in default.
The implicit adaptive copy budget uses an EWMA with factor 0.25.
It measures verified work time and excludes gate wait.
The byte budget fits the bytes that half of the recent window can transfer
at the measured host-to-device rate.
The legacy floor of 96 can exceed that budget.
A hard cap of 384 applies.
Candidate blobs follow the actual gain order, with the prefix first.
Explicit swap counts, including 0, are preserved.
An invalid reading falls back to the legacy value of 96.
The change is inactive in pipeline, split, batch, peer, helper, and complement modes.
Bare `auto` selects 16K only on a single SM70 GPU with native assets, a full anonymous RAM arena, and enough RAM and VRAM.
The RAM check retains an additional 6 GiB margin.
The rule excludes mapped, shared, and resident complements.
Explicit N, `auto:N`, and the environment cap are unchanged.
Two GPUs stay at 8K.
`SM70_CHUNK=0` also keeps 8K.

### Unchanged configuration

The KV configuration is unchanged: logical 524,288 cells, INT8, and 16,384 initial physical cells.
The assets are the same regular IQ3_S assets.
The snapshot fix from stage 1 remains in the tree.

## Indexer snapshot order during mixer overlap

The study found a state error in both mixer-overlap modes.
The fix is retained in round 2.
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
Labels ending in `default` refer to a stage 1 trial build that selected mode 2 at startup.
The stage 1 final patch did not change architecture defaults.
Round 2 adds the Linux SM70 adaptive-overlap default.
Set `STRATA_SIDE_MIXER=2` to select the corrected path explicitly.
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
This command was used for the stage 1 records.
It does not set the round 2 adaptive-copy default.

## Round 2 measurement method

The round 2 comparisons use the same two-GPU host as stage 1.
The host has two Tesla V100-PCIE-16GB GPUs and a Ryzen 5 3600 CPU.
GPU 0 uses PCIe Gen3 x4 and layers 0-15.
GPU 1 uses PCIe Gen3 x16 and layers 16-47.
Both GPU power limits are 175 W.
The usable system RAM is 94.2 GiB. Model storage is a local ext4 volume.
The operating system is Linux 7.0.0-38-generic.
The NVIDIA driver is 580.178.04. The CUDA compiler is 12.8.
The build is a Release SM70 build.

Single-GPU arms use one Tesla V100-PCIE-16GB.
Dual-GPU arms use both GPUs.

All arms use the regular IQ3_S assets and matching pack, tokenizer, native, and PLE files.
The logical context is 524,288 cells. KV is INT8.
KV growth starts at 16,384 physical cells.
The VRAM reserve is 750 MiB.
Dual-GPU arms use layer split 16 and two pipeline windows. The expert ring is 192.
The CPU prefill share is `auto`. The prefill PCIe share is automatic.
The stream threshold is 3,072 tokens.
The WMMA, skip-unrouted, SM70-table, and wide-dequant paths are enabled.
KV growth and embedding-reuse accounting are enabled.
The checkpoint pool and checkpoint worker are enabled.
Promotion is 2,048 per stage in prefill.
Resident-aware routing is enabled in the prefill arms.

Prefill runs use seed 20261003 and prompt pool 45,033.
The targets are 2K, 8K, and 32K.
Each request reads its full prompt and generates one token.
No request reuses a conversation prefix.
Both modes use the harness request mode `geni`.
The single-GPU baseline selects prefill cap 8192 at startup.
The single-GPU candidate selects prefill cap 16384.
Both dual-GPU arms select prefill cap 8192.
Each comparison has four fresh processes in ABBA order: baseline, candidate, candidate, baseline.
Each arm therefore has two measurements per target.

Normal decode runs use 8K prompts only and one GPU.
The prompt pool depends on the seed: 17,647 for 20261003, 18,096 for 20261004, and 17,496 for 20261005.
MTP spec is 8. The minimum p is 0.70.
The new-token cap is 512.
The arms set `STRATA_ROUTE_RESIDENT=0`, `STRATA_ROUTE_TAIL_SKIP=0`, and `STRATA_PCIE_BALANCE=0`.
The PCIe share is fixed at 0.35.
Each seed has four fresh processes in ABBA order: baseline, candidate, candidate, baseline.
Each arm therefore has two runs per seed.
The two runs of an arm are aggregated before the rates are computed.

The harness removes all five timing diagnostic variables from inherited and configured environments.
Explicit `--env` overrides are applied after filtering.
Timing arms do not collect window hashes.

The stage 1 prefill used prompt pool 69,773 and two repeats per target.
The round 2 pool is 45,033.
The two prefill sets are not comparable.

## Round 2 reproduction

These commands use the existing harness
[`bench/run_v100_prefill.py`](../bench/run_v100_prefill.py).
The harness SHA-256 in the records is
`86fd56ec9352084e63b2ebe31a2fb2efb1ccd68cc79f3ec7b73512552ece625a`.

Set `CONFIG` to the private configuration path and `EXPERIMENT` to the directory that holds the binaries.
Run the commands from the repository root.
Build the baseline from `49be5b18` and the candidate from this PR's current source.
Use the build procedure in the V100 guide.
The commands below require the recorded frozen executables.
After a rebuild, replace the expected hashes with the hashes of your own binaries.

The engine settings must be in the configuration `args` list.
`--xargs` applies only to chain mode.
It does not apply to the prefill mode or the normal mode used here.
The harness adds `--serve` to the executable.
It also disables prompt reuse, removes a conversation disk tier and a profile save,
and filters the five timing diagnostic variables.
`--env` values are applied after that filter.
Do not pass a timing diagnostic in `--env`.

Use a private JSON configuration with the regular IQ3_S assets.
Replace the `$MODEL_ROOT` and `$EXPERT_PROFILE` placeholders with local paths in that private file.
Use the same existing expert profile for both arms.
Keep the file private.

```json
{
  "gpu": [1],
  "args": [
    "--pack", "$MODEL_ROOT/packs/iq3_s",
    "--native", "$MODEL_ROOT/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "--ple-gguf", "$MODEL_ROOT/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
    "--expert-profile", "$EXPERT_PROFILE",
    "--expert-cache", "auto", "--prefill", "auto",
    "--spec", "8", "--mtp", "$MODEL_ROOT/mtp/rt",
    "--max-context", "524288", "--rope-scaling", "yarn", "--rope-scale", "2",
    "--kv", "int8", "--spec-min-p", "0.70", "--vision",
    "--vram-reserve-mib", "750",
    "--conversation-cache-mib", "25600", "--conversation-cache-slots", "32",
    "--stats"
  ],
  "env": {
    "STRATA_EXPERT_PAIR": "1",
    "STRATA_SPLIT_RING": "192",
    "STRATA_PREFILL_CPU_SHARE": "auto",
    "STRATA_PREFILL_STREAM_MIN": "3072",
    "STRATA_EMB_REUSE_ACCOUNT": "1",
    "STRATA_SM70_TABLE": "1",
    "STRATA_PF_WMMA": "1",
    "STRATA_PF_SKIP_UNROUTED": "1",
    "STRATA_KV_GROW": "1",
    "STRATA_CKPT_POOL": "1",
    "STRATA_CKPT_THREADS": "1",
    "STRATA_DEQUANT_WIDE": "1",
    "STRATA_PREFILL_PROMOTE": "2048",
    "STRATA_ROUTE_RESIDENT": "1",
    "STRATA_SIDE_MIXER": "0"
  }
}
```

Change the configuration for each cohort:

- The tested single-GPU configuration uses physical GPU 1 (Gen3 x16). The `gpu` list can select another device.
- Dual-GPU prefill: set `"gpu": [0, 1]` and add `"--pipeline-windows", "2"` and `"--layer-split", "16"` to `args`.
- Normal MTP: add `"--pcie-frac", "0.35"` to `args`.
  Set `STRATA_ROUTE_RESIDENT` to `0`.
  Add `STRATA_ROUTE_TAIL_SKIP` with value `0`.
  Add `STRATA_PCIE_BALANCE` with value `0`.
  Keep one GPU in `gpu`.

Run the single-GPU prefill baseline first.
Use a new output directory for each run.
The arm labels are a1, b1, b2, and a2.

```sh
python bench/run_v100_prefill.py \
  --config "$CONFIG" --exe "$EXPERIMENT/strata-pr74-baseline" \
  --arm-label repeat-single-a1 \
  --targets 2048,8192,32768 --repeats 1 --max-new 1 --seed 20261003 \
  --no-cooldown --no-settings-strict \
  --expect-sha256 68a139bda79240c930c69c1612f73e597c12f64493ffeb01c99d9eefa354d919 \
  --out "$EXPERIMENT/repeat-single-a1"
```

Then run the same command again with these changes:

- Set `--exe` to `$EXPERIMENT/strata-round2-final`.
- Set `--expect-sha256` to `add55d48a3ace40ed206843f09fed856b4ba6ea9edde10a2cfef9737c6167e2b`.
- Set the label and output directory to `repeat-single-b1`.
- Run it a second time with `b2`.
- Run the baseline command again as `repeat-single-a2` last.

For the dual-GPU arms, use the dual configuration and the same four steps.
Change only `--arm-label` and `--out` for each step.
Use the labels and directories `repeat-dual-a1`, `repeat-dual-b1`,
`repeat-dual-b2`, and `repeat-dual-a2`.

For the normal full route, use the normal configuration.
Each seed has four runs in the same ABBA order.

```sh
python bench/run_v100_prefill.py \
  --config "$CONFIG" --exe "$EXPERIMENT/strata-pr74-baseline" \
  --arm-label repeat-normal-a1 \
  --targets 8192 --repeats 1 --max-new 512 --seed 20261003 \
  --no-cooldown --no-settings-strict \
  --expect-sha256 68a139bda79240c930c69c1612f73e597c12f64493ffeb01c99d9eefa354d919 \
  --out "$EXPERIMENT/repeat-normal-a1"
```

Run the normal command for the candidate as `b1` and `b2`, then run it for the baseline as `a2`.
Change only `--exe`, `--expect-sha256`, `--arm-label`, and `--out` for those runs.
Repeat the four runs for seeds 20261004 and 20261005.
Name the output directories and arm labels after the round 2 records,
for example `repeat-normal-s20261004-a1`.

`--no-settings-strict` is necessary.
The shared strict preset requires a 700 MiB VRAM reserve.
These arms keep a 750 MiB reserve.
For a single GPU, the legacy audit also reports a missing `--layer-split auto`; the engine correctly starts without a split.
The same mismatch applies to both arms.
Stop other inference engines on the test GPUs before each run.

## Output and routing caveats

Normal MTP outputs differ between the arms, even with identical prompts.
The round 2 normal runs show different generated token counts, different continuations,
and different draft acceptance.
The stage 1 normal-MTP table below also shows different generated token counts.
Do not divide unequal token totals and call the result a matched-work speed gain.
Do not median unequal-length raw times.
Do not say that the normal outputs are identical.

The round 2 end-of-sequence runs above have unequal lengths.
The baseline and candidate outputs share a 7-token prefix for seed 20261003, 23 tokens for 20261004, and 18 tokens for 20261005.
The draft acceptance counts also differ.
The candidate runs do not repeat each other's outputs on any seed.
The baseline runs repeat on the first two seeds only.
The two arms therefore did different verification work.

The R1 profile is a logit-margin profile with threshold 1.
It is not a boolean gate.
It is not the engine default.
A profile change can change expert IDs and softmax weights, depending on the cache contents.
Tail skip also drops non-resident rank 7 and above through the engine CUDA default.
The full-route control sets resident selection and tail skip to 0.
This change enables neither behavior.

The timing-based cache policy can change CPU/GPU placement and output, even with resident routing disabled.
No quality claim or universal speed claim follows.

## Superseded pilot observations (historical)

These are stage 1 pilot runs. They are not round 2 comparison results.
The round 2 repeats above replace them.
Do not use the 20.322% pilot value as a final result.

A reviewed resource binary (`5707f912…`) ran the full route at 512 tokens.
The baseline took 10,195.5 ms. The auto arm took 8,473.5 ms.
The auto arm gives 20.322% more throughput and 16.89% less time.
Do not report 20.3% less time. The time reduction is 16.89%.
The outputs differ.
Draft acceptance differs, so verification work is not held fixed.
The run does not establish causality for the auto path.

A normal private R1 profile pilot used ABBA order:
512 tokens in 7,384.9 ms, 512 tokens in 6,562.5 ms,
264 tokens in 3,889 ms, and 512 tokens in 6,995.2 ms.
The 264-token run reached end of sequence.
The lengths are unequal.
Do not median the raw times and claim 37%.
The aggregate rate changes from 71.21 to 74.25 tok/s.
That is +4.27%, descriptive only.
Do not use it as a matched-work gain.

## Stage 1 historical evidence

Stage 1 used opt-in W2 with the old orientation only.
It kept mixer overlap as an explicit mode.
The current build enables W2 on SM70 and selects the orientation automatically.
The current Linux SM70 default for adaptive copy overlap is 1.
The tables below keep the stage 1 numbers and raw links.
They do not describe the current defaults.

### Stage 1 grouped W2 prefill result

Stage 1 W2 was opt-in with `STRATA_PF_WMMA2=1`.
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
These gains apply only to the stage 1 opt-in binary.

Raw records:
[control A](../bench/results/2026-10-23-iq3-prefill-decode/prefill-control-a.json),
[W2 A](../bench/results/2026-10-23-iq3-prefill-decode/prefill-w2-a.json),
[W2 B](../bench/results/2026-10-23-iq3-prefill-decode/prefill-w2-b.json),
[control B](../bench/results/2026-10-23-iq3-prefill-decode/prefill-control-b.json).

### Stage 1 prefill controls

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
These controls describe the stage 1 binary.

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

### Stage 1 prefill reproduction

The command below was used for the stage 1 records.
The stage 1 harness used prompt pool 69,773.
The current harness uses pool 45,033.
The command predates the round 2 defaults: W2 default, alternate orientation, and Linux adaptive overlap.
It is not the round 2 reproduction.

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

### Stage 1 matched-work decode result

Stage 1 kept mixer overlap off by default.
This comparison used explicit modes on physical GPU 1 only.
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

### Stage 1 normal pipelined MTP result

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

The arms produced unequal token counts.
Do not median the raw times.
Do not quote the fixed-work result as a gain for this workload.

The final stage 1 patch left the Volta architecture defaults unchanged.
Mixer overlap remained an explicit `STRATA_SIDE_MIXER=2` option.
Round 2 changes the Linux SM70 adaptive-overlap default.

The trial executable selected mode 2 automatically in the overlap arms.
Its SHA-256 is `ff66647439ade627c5692460c027916ca21db30a44f68e2171007a00efecff17`.
The stage 1 final patch removed that default-table change.
An explicit mode 2 selects the same corrected graph path.

Raw records:
[inline A](../bench/results/2026-10-23-iq3-prefill-decode/decode-control-a.json),
[overlap A](../bench/results/2026-10-23-iq3-prefill-decode/decode-overlap-a.json),
[overlap B](../bench/results/2026-10-23-iq3-prefill-decode/decode-overlap-b.json),
[inline B](../bench/results/2026-10-23-iq3-prefill-decode/decode-control-b.json).

### Stage 1 W2 profiling

A stage 1 Nsight Systems capture observed the old W2 orientation during
regular IQ3_S HTTP inference.
It records 957 IQ4_NL down launches and 444 IQ2_S gate/up launches.
That run passes exact READY, arithmetic, JSON, and long-prompt READY checks.
It is an execution check, not a timing arm.
The [profiler records](../bench/results/2026-10-23-iq3-prefill-decode/profiled-kernels.json)
identify both instantiated kernels.

### Stage 1 verification records

The stage 1 Release SM70 build passed these five checks:
`verify_parity`, `native_multi_parity`, `fusions_parity`, `qsa_parity`,
and `nv_defaults_test`.
The focused `pf_wmma2_parity` check also passed.
The stage 1 W2 numerical cases passed under CUDA Compute Sanitizer with zero memory errors.
The benchmark harness passed all 46 unit tests.
The stage 1 numerical cases included contiguous weights, reversed expert pointer tables,
gathered input rows, empty experts, partial tiles, and K=3840 IQ2_S rows.
The weight allocation had no added guard padding.
The sanitizer run also included the final two-byte word of an odd IQ2_S row.

Authenticated HTTP tests exercised both stage 1 optimization options together.
Four text cases passed on one GPU and on the two-GPU layer split.
They require exact READY, arithmetic result 42, a specified JSON object,
and READY after a long prompt.
All eight stage 1 HTTP checks passed.
See the [single-GPU results](../bench/results/2026-10-23-iq3-prefill-decode/quality-release-single.json)
and [two-GPU results](../bench/results/2026-10-23-iq3-prefill-decode/quality-release-dual.json).

The [verification summary](../bench/results/2026-10-23-iq3-prefill-decode/validation.json)
records the stage 1 build hash and check outcomes.
These checks apply to the stage 1 binary.
They do not establish general model quality.
They do not test image inference or other GPU architectures.
No private configuration, credential, or installed engine binary is included.

## Correctness and verification evidence

Both permanent focused CTest cases pass on the final build.
Each explicit orientation (`STRATA_PF_WMMA2_N64=0` and `=1`) passes 20 numerical and stray-write checks.
Both orientations also pass 20 CUDA Compute Sanitizer cases with zero memory errors.
The alternate orientation passes the 16 kernel-bench numerical checks.
The other passing checks are `verify_parity`, `native_multi_parity`, `fusions_parity`,
`qsa_parity`, `nv_defaults_test`, and `adapt_budget_test`.
The benchmark harness passes all 46 unit tests.
See the [final validation summary](../bench/results/2026-10-23-iq3-prefill-decode/round2-validation.json).

Three actual CLI checks verify override and low-memory behavior.
`--prefill auto:8192 --adapt-swaps 96` keeps an 8192 cap and disables automatic copy sizing.
`--prefill 4096 --adapt-swaps 0` keeps a 4096 cap and makes zero adaptive swaps.
A small resident-complement configuration selects a 4864-token chunk under the existing memory-fit rules.
It does not activate the new 16K rule or automatic copy sizing.
These are functional checks, not performance samples.
See the [mode-check records](../bench/results/2026-10-23-iq3-prefill-decode/round2-mode-checks.json).

Eight authenticated HTTP checks pass on the final candidate: four on one GPU and four on the layer split.
They check exact READY, arithmetic result 42, a specified JSON object, and an explanation of boiling.
Both configurations report a loaded model and the full context.
See the [single-GPU answers](../bench/results/2026-10-23-iq3-prefill-decode/round2-http-single.json)
and [dual-GPU answers](../bench/results/2026-10-23-iq3-prefill-decode/round2-http-dual.json).

A clean development Nsight capture has `gpu_stamp=0`.
Kernel types 18, 20, 21, and 22 record 490, 1,106, 283, and 559 launches.
The [profiler record](../bench/results/2026-10-23-iq3-prefill-decode/round2-profiled-kernels.json)
confirms execution of all four formats before the final orientation tuning.
It is not a throughput comparison.
The listed checks do not test image inference.
No other GPU architecture was tested.

W2 uses the same stored quantized weights as the first WMMA tile.
It dequantizes operands to FP16 and accumulates in FP32.
Do not assume bitwise agreement with the CPU or integer MMQ path.

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
  The global cache-placement scoring experiment did not establish a repeatable decode gain.
- [HybriMoE](https://arxiv.org/abs/2504.05897)
  examines hybrid CPU-GPU expert scheduling and cache management.
  It supports measuring placement and overlap on a limited-memory host.
- [MoE-Infinity](https://arxiv.org/abs/2401.14361)
  examines expert offloading and cache management for MoE serving.
  It supports measuring cache placement and reuse.
- [ProMoE](https://arxiv.org/abs/2410.22134)
  examines proactive expert caching.
  It supports measuring cache-hit behavior.

The study rejects these candidates:

1. A per-stage measured PCIe/CPU share policy.
   A reverse-order comparison regresses aggregate decode throughput.
   The implementation is removed.
2. A two-output-row CPU interleave for IQ3_S.
   It is bit-exact but raises total expert time from 0.444 ms to 0.468 ms.
   The implementation is removed.
3. Two scalar AVX2 IQ3_S index-assembly variants.
   Both pass numerical parity.
   Both make the one-token gate/up microbenchmark slower.
   Both implementations are removed.
4. Forced gathered IQ3_S decode on this Zen 2 CPU.
   The one-token engine gate/up time rises from about 0.33 ms to 0.61 ms.
   The existing CPU selection rule is unchanged.
5. A cold HC weight-once path.
   It raises time from 44.8 us to 63.4 us at T3.
   A CTA interleave gives 7.6% at T4, but its whole-window benefit is below 1%, so it is discarded.
   Both are rejected.
6. A forced small generic WMMA tile.
   It gives about 2% on one format and regresses other formats.
   It is rejected.
7. A combined decode-fusion setting.
   Initial MTP results vary too much to justify a default change.
   Fused plan waits alone show no useful gain in the CPU-bound fixed-work run.
8. MTP with minimum p 0.5 or 0.
   The 0.5 setting shows no promising gain. The 0 setting is slower.
   Neither setting is used.
9. Generic dual 16K and 32K prefill.
   It takes 16,675.6 ms against 16,069.8 ms and is slower.
   It is excluded.
10. A cache-cost selector and a history prefetch.
    The selector changes work by -0.79%, which is insufficient.
    The history prefetch reaches about 12% precision and 1.89% useful for all non-resident experts.
    The study did not refactor for Foresight.

No rejected implementation or speculative tuning is installed in the daily service.

## Limitations and observations

- Other GPU architectures are untested. Do not make a universal speed claim.
- The host has asymmetric PCIe: Gen3 x4 on GPU 0 and Gen3 x16 on GPU 1.
- The round 2 sample is small: two fresh runs per arm for each prefill cell.
  Each normal seed has two fresh runs per arm.
- The round 2 normal outputs differ between the arms.
  The generated lengths, continuations, and acceptance counts are not equal.
  The aggregate rate change is descriptive.
- The round 2 candidate changes several defaults together.
  It does not isolate one kernel. Do not attribute the full gain to W2.
- The single-GPU prefill arms use different prefill caps: 8192 and 16384.
  The dual-GPU arms use 8192 in both arms.
- The normal route uses a fixed PCIe share of 0.35, resident routing off, and tail skip off.
  The full-route arms do not use the prefill resident-routing default.
- Stage 1 did not set clock locks. Stage 1 samples include a software power-cap flag.
- Stage 1 did not clear the operating system file cache between runs.
  Cold and warm requests remain in the stage 1 measurements.
- Stage 1 prompt pool 69,773 and round 2 pool 45,033 are not comparable.
- In the measured traces, expert compute is about 46-48% of prefill GPU time.
  Copy wait is about 0.8%. Weight DMA is hidden.
  The normal dual cache has a 94-97% hit rate.
  The trace is not a forced CPU-dominated trace.
- The CPU already uses transparent huge pages for 46.8 GiB.
  The measured 18 GB/s is near the calibration ceiling.
  Do not claim missing huge pages.
- This report contains no API key, username, home path, hostname, private IP address, private configuration, or UUID.

## Sources

- [Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models](https://arxiv.org/abs/2402.07033)
- [HybriMoE](https://arxiv.org/abs/2504.05897)
- [MoE-Infinity](https://arxiv.org/abs/2401.14361)
- [ProMoE](https://arxiv.org/abs/2410.22134)
- [NVIDIA CUTLASS: Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)

Volta has no `cp.async` or TMA path in this build.
The study did not implement those features.
The paper percentages are not local measurements.
