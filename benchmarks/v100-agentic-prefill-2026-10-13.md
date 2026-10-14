# V100 agentic prefill: runtime and kernel changes

## Scope

This experiment uses two V100 PCIe 16 GB GPUs and the native IQ3_S model. The operator authorized a service test window. The production service was stopped. Each benchmark owned its engine process. No candidate was installed over the production engine.

The source changes are on a separate branch. They do not select new runtime defaults.

## Source changes

1. Select CPU experts before allocating or copying activation rows. Deduplicate the token rows those experts need. Gather the same FP32 rows on the device. Download the compact rows on a separate stream. A completion event protects CPU access. Join the CPU work before the next layer can reuse the activation tensor.
2. Size the activation, output, gather, and index buffers for the selected share. Track capacity across requests. Grow the index buffer when a later chunk is larger. If allocation fails, undo CPU selection before constructing the expert-row layout.
3. Pass grouped Volta expert-weight pointers in the GEMM launch parameters. Remove the device pointer-table setup kernel and its API. Keep the expert-ring release order.

The CPU quantizer receives the same FP32 values. Automatic CPU sharing still depends on measured timings. Identical model answers are not guaranteed.

## Baseline phase evidence

A separate diagnostic 2K request reported the following costs:

| Stage | Host grouping | Host staging | Wait copy | CPU expert evaluations |
| --- | ---: | ---: | ---: | ---: |
| GPU 0, layers 0–15 | 172 ms | 8 ms | 12 ms | 350 |
| GPU 1, layers 16–47 | 114 ms | 45 ms | 267 ms | 3,561 |

Diagnostic timings locate costs. They are not scored speed measurements. They support work on serialized activation downloads. They do not establish that the host staging thread is the largest cost.

The candidate diagnostic reduced host grouping to 40 ms on GPU 0 and 58 ms on GPU 1. The total fell from 286 to 98 ms. GPU 0 still evaluated 350 experts on the CPU. GPU 1 evaluated 3,576 instead of 3,561, with automatic share 0.53 instead of 0.51. Some cost moved into later waits. This phase reduction is not a 66-percent whole-prefill speed claim.

## Cold prefill through 32K

Both arms used exact token IDs, seed 20261013, three requests per size, and a 64-token output cap. Prefix and conversation reuse were disabled. Prompt hashes matched between arms. All 30 requests completed without an error. The cooldown target was 55 degrees C, with a ten-second stable interval.

The maximum sampled temperature was 63 degrees C in each arm. All cooldown gates passed. No sampled thermal slowdown was reported.

| Prompt tokens | Baseline median ms | Candidate median ms | Latency reduction |
| ---: | ---: | ---: | ---: |
| 2,048 | 2,415.1 | 2,265.3 | 6.20% |
| 4,096 | 4,033.8 | 4,023.3 | 0.26% |
| 8,192 | 6,299.9 | 6,277.7 | 0.35% |
| 16,384 | 9,691.1 | 9,621.7 | 0.72% |
| 32,768 | 16,527.2 | 16,465.9 | 0.37% |

The 2K result supports a short-prefill improvement. Changes below one percent in the larger controls are not treated as meaningful gains. This was a baseline-then-candidate sweep, not a randomized statistical study.

Complete generated sequences matched on two of the fifteen prompt pairs. The runtime uses timing-dependent CPU sharing and cache adaptation. These runs do not prove output determinism or model-quality equivalence.

## Configuration trials

A separate HTTP trial ran 90 requests. It compared the baseline, 512-token stage-owned buffers with a 2K eligibility limit, and pipelined short reads with a threshold of 128. Arm order was reversed in the second round. Every arm read the same number of new tokens on each corresponding turn.

Stage-owned buffers reduced median prefill latency by approximately 16–20 percent for additions that produced 120–320 new tokens. They increased latency by approximately 13 percent near 1K new tokens. This setting is not selected as a general 0–2K solution. The pipelined arm mainly helped the smallest additions. Neither trial changes production settings.

## Exact-difference measurement

The direct harness now has a chained mode. It builds nested prompts with requested differences of 0, 1, 32, 64, 128, 256, 512, 1,024, and 2,048 tokens. It records requested additions, actual prompt reads, and reused tokens separately.

By default, each request pins a checkpoint at prompt length minus two. The engine ignores a pin at prompt length minus one. The next turn can therefore read up to the requested addition plus two tokens. A larger read rejects the pinned-turn measurement. The explicit `--no-chain-pin-tail` option measures checkpoint gaps instead.

The checkpoint-gap control showed why nominal additions can be misleading. A requested 2,048-token addition read 4,066 tokens when it resumed from the older 8,191-token checkpoint. This is observed extra work, not a kernel throughput limit.

The pinned comparison used an 8,192-token prefix, three passes of the same ladder, and a 64-token output cap. All 60 requests completed. Prompt hashes and actual read counts matched between arms.

| Requested difference | Actual read tokens | Baseline median ms | Candidate median ms | Latency reduction |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 54.4 | 55.1 | -1.29% |
| 1 | 3 | 152.0 | 148.4 | 2.37% |
| 32 | 34 | 362.3 | 361.7 | 0.17% |
| 64 | 66 | 619.3 | 604.7 | 2.36% |
| 128 | 130 | 868.7 | 874.5 | -0.67% |
| 256 | 258 | 1,047.4 | 1,013.6 | 3.23% |
| 512 | 514 | 1,305.6 | 1,268.0 | 2.88% |
| 1,024 | 1,026 | 1,762.7 | 1,703.8 | 3.34% |
| 2,048 | 2,050 | 2,540.1 | 2,437.9 | 4.02% |

The source changes provide a modest prefill improvement in the 256–2,048 range. They do not remove short-turn checkpoint and transfer costs. The smallest differences show no consistent improvement.

Decode is not a universal win. At the 128-token difference, the measured decode median fell from 80.42 to 63.21 tokens per second. At the 512-token difference it rose from 80.95 to 82.50. Generated sequences differ between arms under automatic sharing. Do not interpret this as a matched-output decode comparison, or deploy it as a guaranteed whole-request speedup.

## Actual agent API workload

The candidate also completed 30 HTTP chat requests with tools. They used the same workload as the 30 baseline requests in the configuration trial. The actual new-token counts matched on every corresponding turn.

| Actual new tokens | Baseline median prefill ms | Candidate median prefill ms | Latency reduction |
| ---: | ---: | ---: | ---: |
| 65–72 | 718.9 | 705.2 | 1.90% |
| 121–129 | 876.5 | 852.0 | 2.81% |
| 190–193 | 989.0 | 963.0 | 2.62% |
| 318–320 | 1,187.3 | 1,166.2 | 1.77% |
| 569–576 | 1,471.3 | 1,410.8 | 4.11% |
| 1,081–1,089 | 1,967.3 | 1,857.5 | 5.58% |
| 2,110–2,113 | 2,827.3 | 2,636.3 | 6.76% |

The last bucket is slightly above 2K because the chat template adds tokens. These HTTP runs were separate sessions without the direct harness's thermal gate. The smaller changes are not strong statistical evidence.

Whole-request time did not improve in every bucket. It rose slightly in the three smallest buckets. At approximately 2K new tokens, median HTTP time fell from 3,910.1 to 3,765.0 ms. This includes decode and request setup.

## Correctness checks

The candidate built for `sm_70` with CUDA 12.8. The following focused checks passed:

- Grouped Volta WMMA parity, including both weight-table halves.
- DMA batch parity.
- CPU activation quantization parity, including compact feeds.
- FP16 tensor-core prefill parity.

Two fused-expert checks skipped on V100. They require a newer architecture. They are not counted as passed runtime checks. The benchmark CPU suites passed 60 tests.

A separate fixed-share control used CPU share 0.25 and a one-token output cap. All ten baseline/candidate next-token outputs matched. The ladder exercised growing and shrinking prompt reads. This is a next-token correctness smoke, not a long-output quality check or decode benchmark.

## Provenance and limits

Baseline engine SHA-256: `873a123114c8457e7f1a97196a458a15ffb16615319ba324e23ea95ce89ea870`.

Candidate engine SHA-256: `f9a59dc86dcaf8add771f1cdc87d381556f7a54b0d9bb9e36727058dcd4f3b1f`.

Candidate source commit: `a64694e3`. The binary was built from these source changes before that commit was created. A result's checkout commit describes the harness checkout at collection time; it is not a substitute for the engine source provenance.

The fixed settings were layer split 16, ring 192, automatic prefill chunks, INT8 KV, 524,288-token context capacity, and a 750 MiB reserve. The scored environment matched the service profile: CPU sharing `auto`, stream threshold 3,072, embedding reuse accounting, Volta tables, WMMA, unrouted-expert skipping, KV growth, and resident routing enabled.

The legacy cold harness requires a 700 MiB reserve in its strict check. These arms intentionally used 750 MiB and passed `--no-settings-strict`. Its old first-occurrence check also reported conversation cache 15,360 MiB, although the appended effective value was zero. Startup and every request confirmed no reuse. The updated harness reads the effective last value.

Do not publish private engine logs, credentials, or production request text. All prompts in these controlled runs are synthetic.

## Reproduce

Use the tested private model configuration. Set `CONFIG` to that file. Set `ENGINE` to the selected baseline or candidate binary. Set `ARM` to a distinct result label. Run each arm from the candidate checkout. Stop the production service before starting a benchmark. Restore it after all benchmark engines exit.

```bash
export STRATA_ARENA_LOCK=0 STRATA_WATCHDOG_S=300
export STRATA_PREFILL_CPU_SHARE=auto STRATA_PREFILL_STREAM_MIN=3072
export STRATA_EMB_REUSE_ACCOUNT=1 STRATA_SM70_TABLE=1
export STRATA_PF_WMMA=1 STRATA_PF_SKIP_UNROUTED=1
export STRATA_KV_GROW=1 STRATA_ROUTE_RESIDENT=1

python bench/run_v100_prefill.py --chain --config "$CONFIG" --exe "$ENGINE" \
  --arm-label "$ARM" --chain-prefix 8192 \
  --chain-diffs 0,1,32,64,128,256,512,1024,2048 --chain-rounds 3 \
  --seed 20261013 --max-new 64 --chain-no-disk --no-settings-strict \
  --cooldown-c 55 --cooldown-stable-s 10 --out "bench/results/local-$ARM"

python bench/run_v100_prefill.py --config "$CONFIG" --exe "$ENGINE" \
  --arm-label "$ARM" --targets 2048,4096,8192,16384,32768 --repeats 3 \
  --seed 20261013 --max-new 64 --no-settings-strict \
  --cooldown-c 55 --cooldown-stable-s 10 --out "bench/results/local-$ARM-cold"
```

The original cold results used the unchanged legacy harness. The updated cold mode uses the same request construction, but corrects effective-option reporting. Do not include diagnostic timing in scored runs. For the fixed-share correctness smoke, use share 0.25, one chain round, and `--max-new 1`.

## Local evidence and service restoration

Cold, chained, fixed-share, and HTTP measurements remain local, with the environment and binary hashes. Raw benchmark JSON is not tracked in this repository. This report retains the measured summaries and reproduction commands. Engine and protocol logs remain private.

After testing, the original service was started with its unchanged engine and settings. Health reported `loaded: true`. An authenticated chat request returned `READY` with finish reason `stop`. The installed engine hash still matched the baseline. The restoration safety timer was stopped after the health and chat checks.
