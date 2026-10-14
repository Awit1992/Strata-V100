# V100 mixed-format agentic prefill

## Purpose

Reduce agent-turn latency on the two V100 GPUs. Compare the new runtime with both the installed engine and the first-pass engine from PR 61. Keep the production engine and settings unchanged.

The [first-pass report](v100-agentic-prefill-2026-10-13.md) measured activation-transfer and launch changes. This experiment addresses the remaining grouped expert kernels and expert-cache refill.

## Actual pack formats

The pack name is not a complete description of its expert formats. The pack metadata identifies these gate/up formats across 48 layers:

| Gate/up format | Layers |
| --- | ---: |
| IQ3_XXS | 17 |
| IQ2_S | 20 |
| IQ3_S | 10 |
| IQ4_XS | 1 |

Down projections use IQ4_NL in 39 layers and Q2_0 in nine layers. With the tested split at layer 16, GPU 0 has four IQ3_XXS gate/up layers and twelve IQ2_S layers. It has no IQ3_S gate/up layer. An IQ3_S-only optimization therefore misses all gate/up work on GPU 0.

The per-layer format measurements remain local. Raw benchmark JSON is not tracked in this repository.

## Short-row kernels

Groups with at most 128 rows in their largest routed expert use smaller tiles for IQ3_XXS, IQ3_S, IQ2_S, and IQ4_NL. Larger groups retain the original tiles. Q2_0 and the other formats retain their existing kernels.

The new gate/up kernels decode a 128-value half of each 256-value quantization block. They preserve the decoded half values and increasing-K tensor-core accumulation order. Each block handles two 64-row tiles instead of four.

| Short kernel | Dynamic shared bytes | Static shared bytes | Total shared bytes |
| --- | ---: | ---: | ---: |
| IQ3_XXS gate/up | 44,032 | 1,024 | 45,056 |
| IQ3_S gate/up | 44,032 | 2,048 | 46,080 |
| IQ2_S gate/up | 39,936 | 8,192 | 48,128 |
| IQ4_NL down | 44,032 | 16 | 44,048 |

The shared-memory budgets permit two blocks per Volta SM. The two-block launch bound also limits register use. NVIDIA specifies 96 KB of shared memory and 64K 32-bit registers per Volta SM; see the [Volta occupancy guide](https://docs.nvidia.com/cuda/volta-tuning-guide/#occupancy). These are resource budgets, not an end-to-end speed claim.

An unconditional small-tile experiment made the 256-row IQ4_NL microbenchmark approximately 16 percent slower. The row-count guard rejects that design. A separate vector-store experiment was also rejected. Neither experiment is the final selected implementation.

The parity harness covers short and wide row boundaries, all existing formats, gathered activation IDs, and indirect gate/up and down weight tables. Its `--bench` mode validates outputs against a reference before reporting CUDA-event timings. Each timed graph has eight launches. Each shape reports the median of seven windows.

The final build uses 128 registers per thread in the selected short kernels. The compiler resource report records `LOCAL:0`. Stack-frame sizes are eight or sixteen bytes; they are not reported as zero.

## Validated kernel measurements

The ABBA sequence ran the first-pass kernel library, the final library twice, and the first-pass library again. Both libraries used the same expanded harness. Every run validated all sixteen shapes and reported zero failures. The timed launch includes its quantized-weight decode. Reference dequantization and allocation are outside the timed windows.

| Format, 64 rows per expert | First-pass median ms | Final median ms | Latency reduction |
| --- | ---: | ---: | ---: |
| IQ3_XXS gate/up | 1.4055 | 1.0045 | 28.53% |
| IQ3_S gate/up | 1.3995 | 0.9895 | 29.30% |
| IQ2_S gate/up | 1.3225 | 0.9655 | 26.99% |
| IQ4_NL down | 0.6080 | 0.5520 | 9.21% |

These shapes use 64 experts. They are kernel measurements, not whole-model gains. The 256-row shapes use the original kernels. One final IQ3_S wide sample was 5.3 percent slower; its second sample was close to baseline. The other wide shapes stayed within 0.3 percent.

## Complete-output correctness control

The first-pass and final engines completed twenty matching requests each. Both used CPU share 0.25 and disabled route-tail skipping. The output cap was sixteen tokens. All twenty complete output-ID sequences matched. Prompt hashes and actual read counts also matched. Median decode throughput was 68.63 versus 68.28 tokens per second, a difference of approximately 0.5 percent.

This control covers growing and shrinking prefix reads. It is stronger than the first-pass one-token smoke. It is not a general model-quality evaluation. Automatic CPU sharing can still change output sequences.

All four focused C++ checks passed on the final build: grouped WMMA parity, DMA batch parity, CPU quantized-row parity, and FP16 tensor-core prefill. Trace output also recorded eleven early refills before their batched reads completed. The trace requests used the normal checkpoint and short-window transitions.


## Earlier cache refill

The runtime admits the native Volta format pairs used by the actual mixed pack to the existing early-refill path. The explicit off switch and the serial, blocking, and last-stage guards remain in place.

A stage can return its borrowed expert-cache space while the next GPU finishes the batched segment. Refill can start at prompt completion or before a short-read window, which must return every loan before it reads. This includes checkpoint cuts before a short assistant header. Refill cannot start before another batched segment reuses the same scratch.

The existing streamed-KV copy fence now runs before the stage-complete callback. This prevents refill from writing borrowed memory while a KV prefetch still uses it. No extra staging allocation is introduced.

## Measurement rules

Use the same model, split, environment, prompt construction, output cap, and harness for corresponding arms. Record actual prompt reads. Separate the initial cold prefix from reused-prefix turns. Include whole HTTP request time and decode throughput, not only prefill latency.

Boundary-pinned chains measure exact prompt differences. Their final one-token window now permits early refill after the preceding batched segment, because the window returns the loan before it reads. Very short differences use only windows and do not launch the grouped prefill kernels. HTTP tool requests exercise the real agent API with its normal checkpoint cuts.

The initial HTTP trial had one arm with unusually repetitive output, complete expert-cache hits, and complete acceptance of suffix drafts. Its large apparent gain is not evidence of a kernel improvement. The final HTTP comparison disables route-tail skipping equally for all arms and retains private responses for diagnostic checks. Response text, private engine logs, and credentials must not be published.

Intermediate IQ3_S-only results remain local as controls. They must not be labeled as results from the final mixed-format binary. Raw measurement files are excluded from the pull request.

## Final agent API comparison

The final HTTP experiment completed 180 requests: three engines, four rounds, and fifteen requests per round. Arm order was reversed on alternate rounds. Each round used a fresh server and isolated cache state. The initial prefix read 8,828 tokens. Fourteen later turns added synthetic tool results.

Route-tail skipping was disabled equally in all arms. All corresponding requests read the same number of new tokens. All requests generated 64 tokens and ended at the output cap. The private response check found no duplicate complete responses and no repeated eight-word sequence within a response. This check rejects the observed repetition artifact; it is not a model-quality evaluation.

Across the fourteen short turns, the median reduction in each turn's median prefill latency was 15.52 percent against the installed engine. The equivalent whole-HTTP latency reduction was 8.18 percent. Against the first-pass engine, the median whole-HTTP reduction was 6.95 percent.

| Actual new tokens | Installed median HTTP ms | Final median HTTP ms | HTTP latency reduction |
| ---: | ---: | ---: | ---: |
| 65 | 1,764.0 | 1,782.2 | -1.03% |
| 129 | 2,035.2 | 1,824.6 | 10.35% |
| 190 | 2,005.6 | 1,883.6 | 6.08% |
| 320 | 2,167.8 | 2,017.7 | 6.92% |
| 569 | 2,536.8 | 2,309.5 | 8.96% |
| 1,089 | 3,074.0 | 2,912.7 | 5.25% |
| 2,110 | 3,966.3 | 3,516.5 | 11.34% |
| 72 | 1,851.9 | 1,691.3 | 8.67% |
| 121 | 1,995.2 | 1,732.9 | 13.15% |
| 193 | 2,054.0 | 1,891.2 | 7.92% |
| 318 | 2,286.6 | 2,093.9 | 8.43% |
| 576 | 2,482.6 | 2,302.1 | 7.27% |
| 1,081 | 2,946.0 | 2,742.9 | 6.89% |
| 2,113 | 3,839.1 | 3,357.2 | 12.55% |

Each row is the median of four corresponding requests. The last two large turns are slightly above 2K because the chat template adds tokens. Thirteen of fourteen short turns improved. The 65-token turn did not improve.

Overall HTTP decode medians were 66.40 tokens per second for the installed engine, 66.95 for the first pass, and 66.10 for the final engine. Individual decode buckets varied; the 1,089-token turn was slower. Generated sequences are not identical under automatic sharing. The total-request measurements include this cost rather than hiding it.

The HTTP experiment used reversed arm order, but did not apply the direct harness's per-request thermal gate. It is not a precise thermal-equivalence test. Four requests per turn do not establish a universal speedup.

## Matched exact-difference chain

Both engines completed thirty requests, with three rounds and a 64-token output cap. Route-tail skipping was disabled in both arms. Corresponding prompt hashes and actual read counts matched. The initial cold prefix and later warm resets are excluded from this table.

| Requested difference | Actual read | First-pass median ms | Final median ms | Prefill latency reduction |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 55.4 | 55.7 | -0.54% |
| 1 | 3 | 153.3 | 153.1 | 0.13% |
| 32 | 34 | 362.5 | 360.3 | 0.61% |
| 64 | 66 | 611.5 | 610.9 | 0.10% |
| 128 | 130 | 858.8 | 698.8 | 18.63% |
| 256 | 258 | 1,053.1 | 878.4 | 16.59% |
| 512 | 514 | 1,265.1 | 1,096.3 | 13.34% |
| 1,024 | 1,026 | 1,716.6 | 1,512.3 | 11.90% |
| 2,048 | 2,050 | 2,440.5 | 2,165.9 | 11.25% |

The smallest differences use short windows and remain approximately unchanged. The larger differences exercise the new kernels and refill schedule. All sixty cooldown gates passed. Both arms reached a maximum sampled temperature of 53 degrees C. No sampled thermal slowdown was reported. Four baseline GPU/request pairs had no temperature summary; the thermal maximum does not describe those missing samples.

## Cold prefill through 32K

Each engine completed three requests per size with a 64-token output cap. Prompt hashes matched. All actual reuse counts were zero. The same configuration and cold-run environment were used in both arms.

| Prompt tokens | First-pass median ms | Final median ms | Prefill latency reduction |
| ---: | ---: | ---: | ---: |
| 2,048 | 2,260.4 | 2,028.0 | 10.28% |
| 4,096 | 4,028.0 | 3,531.0 | 12.34% |
| 8,192 | 6,278.4 | 5,498.6 | 12.42% |
| 16,384 | 9,633.3 | 8,852.8 | 8.10% |
| 32,768 | 16,467.7 | 15,655.2 | 4.93% |

All thirty cooldown gates passed. The maximum sampled temperature was 63 degrees C in each arm. No sampled thermal slowdown was reported. These comparisons are against the first pass, not the older installed binary.

## Streamed-KV lifetime smoke

The final engine completed 16K and 32K requests with early refill enabled by default, then repeated them with early refill disabled. Both arms used CPU share 0.25 and a sixteen-token cap. All complete output-ID sequences matched between the two arms.

This smoke used int8 KV, 32,768 resident cells, a 524,288-cell maximum context, disabled KV growth, enabled KV prefetch, disabled prompt cache, and disabled short-read segmentation. The engine reported streamed KV and borrowed expert-cache space on both GPUs. Trace output recorded nonzero prefetch counts, early refill before batched-read completion, and zero cache slots left lent after refill.

This exercises the moved copy fence with the default mixed-format guard. It is not a scored speed benchmark: there was one request per size and trace was enabled. Diagnostic logs remain local.

## Provenance and restored service

The experiment used two 16-GiB V100 GPUs, CUDA 12.8, SM70, the split at layer 16, and the same private model configuration. Binary hashes, final source hashes, and full measurements remain local. Build checkout commit fields describe the collection context; they do not identify all final source changes. This report retains the measured summaries and reproduction commands.

The installed engine and first-pass binary were preserved. No candidate was installed. After all GPU runs, the original service was started with its unchanged engine and settings. Health reported the model loaded. An authenticated chat request returned `READY` with HTTP status 200 and finish reason `stop`. The installed engine hash still matched the baseline. The restoration timer was stopped after these checks.

The final benchmark CPU suites passed sixty tests with the repository's `unittest` runner. Raw JSON, private response captures, credentials, and complete engine logs are not committed.

## Reproduce

Use CUDA 12.8 and target SM70. Build the candidate in a separate directory. Preserve both reference binaries. Keep the model configuration private. Do not run these commands while the production service uses the GPUs.

Set `CONFIG` to the tested private configuration. Set `ENGINE` to the selected reference or candidate binary. Set `ARM` to a distinct result label. Use the same configuration for corresponding arms.

```bash
export STRATA_ARENA_LOCK=0 STRATA_WATCHDOG_S=300
export STRATA_PREFILL_CPU_SHARE=auto STRATA_PREFILL_STREAM_MIN=3072
export STRATA_EMB_REUSE_ACCOUNT=1 STRATA_SM70_TABLE=1
export STRATA_PF_WMMA=1 STRATA_PF_SKIP_UNROUTED=1
export STRATA_KV_GROW=1 STRATA_ROUTE_RESIDENT=1
export STRATA_SPLIT_RING=192 STRATA_EXPERT_PAIR=1

python bench/run_v100_prefill.py --chain --config "$CONFIG" --exe "$ENGINE" \
  --arm-label "$ARM" --chain-prefix 8192 \
  --chain-diffs 0,1,32,64,128,256,512,1024,2048 --chain-rounds 3 \
  --seed 20261013 --max-new 64 --chain-no-disk --no-settings-strict \
  --env STRATA_ROUTE_TAIL_SKIP=0 \
  --cooldown-c 55 --cooldown-stable-s 10 --out "bench/results/local-$ARM-chain"

python bench/run_v100_prefill.py --config "$CONFIG" --exe "$ENGINE" \
  --arm-label "$ARM" --targets 2048,4096,8192,16384,32768 --repeats 3 \
  --seed 20261013 --max-new 64 --no-settings-strict \
  --cooldown-c 55 --cooldown-stable-s 10 --out "bench/results/local-$ARM-cold"
```

The historical strict profile expects a 700-MiB reserve. The tested configuration uses 750 MiB in every arm. This is the reported strict-profile mismatch. The actual reserve remains constant; `--no-settings-strict` does not change the engine settings.

For the complete-output control, set CPU share 0.25, use two chain rounds, and cap output at sixteen tokens. Enable trace only for diagnostic requests. Keep timing diagnostics off in scored runs.

For the HTTP comparison, set `INSTALLED`, `FIRSTPASS`, and `FINAL` to the corresponding binaries:

```bash
STRATA_ROUTE_TAIL_SKIP=0 python tools/agent_turns_bench.py "$CONFIG" \
  "installed:exe=$INSTALLED" "firstpass:exe=$FIRSTPASS" "mixed:exe=$FINAL" \
  --workload agentic --rounds 4 --prefix-tokens 8192 \
  --additions 8,64,128,256,512,1024,2048 --turns 14 --max-new 64 --seed 1234 \
  --port 8097 --log bench/results/local-http.jsonl \
  --workdir bench/results/local-http
```

This harness reverses arm order on alternate rounds. Each arm has its own configuration, process group, conversation store, and expert profile.
