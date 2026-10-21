# V100 prefill cache admission, 2026-10-21

## Result

This study compares four speed arms on one dual-V100 host.
The warm expert-cache admission policy gives the lowest nine-turn
time-to-first-output (TTFT) sum and the lowest nine-turn TTFT median of the
four arms.
The median falls from 1176.9 ms to 987.3 ms. The sum falls by 13.6%.
The gain is largest at small and medium reads. It falls as the read grows.
The policy also adds a full-request cost on the smallest turn and a small TTFT
cost on the cold prefix. The cold-prefix cost is relative to the `work` arm
(8002.2 ms to 8084.2 ms). The `hot` arm is still 1.7% faster than `installed`
on that turn. Do not read this result as a universal speedup.

The cold prefix gain is small. This is not a cold-prefill throughput result.
Four reads stay above one second in the admission arm: 573 tokens at
1240.3 ms, 1082 tokens at 1679.3 ms, 2108 tokens at 2384.8 ms, and 3053 tokens
at 3282.1 ms.

The policy selects the expert-cache victim. It does not change the prompt
tokens or the reused prefix. It does not change model quantization, KV
quantization, or KV capacity.
The two wide-kernel arms use the same frozen executable. Only the
`STRATA_PREFILL_PROMOTE` value differs between them.

This report links to the earlier incremental-TTFT study
[V100 incremental TTFT, 2026-10-21](v100-incremental-ttft-2026-10-21.md).
That study covers the first change in this series.

## Test system and fixed conditions

- Two Tesla V100-PCIE-16GB GPUs, 16 GiB each.
- GPU 0: PCIe Gen3 x4, layers 0-15. GPU 1: Gen3 x16, layers 16-47.
- Ryzen 5 3600, six cores and twelve threads, approximately 94.2 GiB usable RAM.
- Linux 7.0.0-38-generic, NVIDIA driver 580.178.04, CUDA 12.8, Release SM70 build.
- The GPU power limits stayed at 175 W each.
- Model: Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S. The pack, tokenizer, native shard,
  and PLE shard match. These are the regular assets, not the abliterated assets.
  The daily abliterated runtime was not a test candidate.
- GPU order `0,1`, layer split `16`, and `--pipeline-windows 2` in all arms.
- Maximum context 524,288 cells, INT8 KV, and 750 MiB VRAM reserve in all arms.
- CPU prefill share `auto`, stream threshold 3072, embedding reuse accounting on.
- The SM70 table, WMMA, skipped-unrouted, KV-growth, and resident-routing
  settings stayed enabled. Route-tail skipping stayed off.
- The expert ring was 192 for both speed arms. The short-turn ring is 32.
- The native expert arena is 46.84 GiB. It has 24,576 slots, that is 48 groups of
  512 experts. The largest expert blob is 2.66 MB.
- The output cap was 16 tokens for each request. Each request used seed 20261021
  and the same sampling settings.

The two rounds used fresh servers, configurations, and caches.
The byte probe was removed from the production path before these rounds.

## Arms

The four arms ran in one measured set. Round 0 used the order below. Round 1
used the reverse order. Each arm and round had a new server process, a new
expert profile, and an empty conversation store.

| Arm | Executable SHA-256 | Checkpoint pool | Dequant kernel | Admission policy |
| --- | --- | --- | --- | --- |
| installed | `f740c535277fa66d825d2d443937d895406c69095c3224f48410dd8761b920c4` | off, 0 threads | narrow | off |
| checkpoint | `090ba4faa5c95c171c9f68e8a29295a812ff03c5bd1840c2ea63a9556c8aac7e` | on, 1 thread | narrow | off |
| work | `6a8f1629f80277bce738c75d0affd912375ed9217a76e97793c801a2bacc8b78` | on, 1 thread | wide, SM70 default | off |
| hot | `6a8f1629f80277bce738c75d0affd912375ed9217a76e97793c801a2bacc8b78` | on, 1 thread | wide, SM70 default | `STRATA_PREFILL_PROMOTE=2048` |

The `installed` arm is the original daily-profile binary. The `checkpoint` arm
is the earlier incremental-TTFT candidate. The `work` and `hot` arms share one
frozen binary. The startup log of the `hot` arm reports
`STRATA_PREFILL_PROMOTE=2048: up to 2048 stable-head rotations per part per loan`.
The `work` arm starts the same binary without that line.

## Workload and measured read counts

The workload is the agentic workload of `tools/agent_turns_bench.py`.
It sends one shared prefix, then nine tool-result turns.
The prefix target is 8192 tokens. The tool-result targets are 0, 32, 64, 128,
256, 512, 1024, 2048, and 3000 tokens. A target is not a token count.
The rendered prompt is 8828 tokens at the prefix turn. Message and tool syntax
add tokens even when the tool result is empty.

All four arms read and reused the same token counts on all ten turns in both
rounds. The read count is the engine's actual new-token count. The reused count
is the prefix length taken from the conversation cache.

| Tool-result target | Prompt tokens | Tokens read | Tokens reused |
| ---: | ---: | ---: | ---: |
| prefix | 8828 | 8828 | 0 |
| 0 | 8879 | 52 | 8827 |
| 32 | 8971 | 93 | 8878 |
| 64 | 9092 | 122 | 8970 |
| 128 | 9279 | 188 | 9091 |
| 256 | 9587 | 309 | 9278 |
| 512 | 10159 | 573 | 9586 |
| 1024 | 11240 | 1082 | 10158 |
| 2048 | 13347 | 2108 | 11239 |
| 3000 | 16399 | 3053 | 13346 |

## Measurement definition

TTFT starts before the HTTP connection. It ends at the first generated SSE
delta. A generated content delta, a reasoning delta, or a tool name or argument
delta all count. Role-only and control-only events do not count. A turn with no
generated output has no TTFT value.
TTFT includes client parser buffering. It is not raw engine-token latency.
Full request time covers all 16 output tokens and request completion.

Each value below is the median of the two rounds. Each round value is one
request on a fresh server.

## Per-turn TTFT

| Tool-result target | Read | installed ms | checkpoint ms | work ms | hot ms | hot vs installed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix | 8828 | 8223.6 | 8068.6 | 8002.2 | 8084.2 | -1.7% |
| 0 | 52 | 611.0 | 518.1 | 524.6 | 559.5 | -8.4% |
| 32 | 93 | 861.6 | 752.6 | 784.0 | 654.7 | -24.0% |
| 64 | 122 | 969.0 | 1017.1 | 942.8 | 793.8 | -18.1% |
| 128 | 188 | 1009.9 | 928.2 | 815.4 | 795.4 | -21.2% |
| 256 | 309 | 1176.9 | 1089.8 | 1014.7 | 987.3 | -16.1% |
| 512 | 573 | 1448.8 | 1357.0 | 1288.5 | 1240.3 | -14.4% |
| 1024 | 1082 | 1899.2 | 1802.0 | 1713.7 | 1679.3 | -11.6% |
| 2048 | 2108 | 2683.8 | 2556.1 | 2492.6 | 2384.8 | -11.1% |
| 3000 | 3053 | 3665.6 | 3558.8 | 3471.6 | 3282.1 | -10.5% |

The two-round spread is small on the large reads and larger on the small reads.
Examples: `hot` at 3053 read is 3260.0 to 3304.2 ms. `work` at 93 read is
657.9 to 910.2 ms. `checkpoint` at 188 read is 927.2 to 929.3 ms.

The admission arm is slower than the `work` arm on two turns.
At 52 read it costs 6.7% more TTFT (559.5 ms against 524.6 ms).
On the cold prefix it costs 1.0% more TTFT (8084.2 ms against 8002.2 ms).
Both round values agree on these two turns.

## Aggregates

| Arm | Nine incremental TTFT sum ms | Nine incremental TTFT median ms |
| --- | ---: | ---: |
| installed | 14,325.9 | 1176.9 |
| checkpoint | 13,579.8 | 1119.0 |
| work | 13,047.9 | 1038.7 |
| hot | 12,377.2 | 987.3 |

Against `installed`, the nine-turn TTFT sum falls 5.2% for `checkpoint`, 8.9%
for `work`, and 13.6% for `hot`.
The ten-turn TTFT sum falls 4.0%, 6.7%, and 9.3% in the same order.

## Full request time and the decode tradeoff

TTFT is the first-output metric. It is not the cost of the complete turn.
The table gives full request time for the two outer arms. The admission arm
reduces full request time on eight of the nine incremental turns.

| Tool-result target | Read | installed ms | hot ms | hot vs installed |
| ---: | ---: | ---: | ---: | ---: |
| prefix | 8828 | 8483.5 | 8488.5 | +0.1% |
| 0 | 52 | 853.1 | 922.7 | +8.2% |
| 32 | 93 | 1147.5 | 999.3 | -12.9% |
| 64 | 122 | 1239.8 | 1168.0 | -5.8% |
| 128 | 188 | 1299.3 | 1145.9 | -11.8% |
| 256 | 309 | 1389.2 | 1341.9 | -3.4% |
| 512 | 573 | 1714.8 | 1585.6 | -7.5% |
| 1024 | 1082 | 2141.8 | 2009.3 | -6.2% |
| 2048 | 2108 | 2930.6 | 2715.3 | -7.3% |
| 3000 | 3053 | 3923.9 | 3631.7 | -7.4% |

The 52-token read is a real regression. The turn takes 69.6 ms longer in the
`hot` arm than in the `installed` arm. It takes 93.5 ms longer than in the
`work` arm, and the `checkpoint` arm is faster still at 518.1 ms TTFT. Both
rounds show the same direction, so this is not single-sample noise.
The nine-turn full-request sum falls 6.7% (33,279.9 ms to 31,039.6 ms).
The ten-turn full-request sum falls 4.4%.

The engine `decode_ms` field explains most of the difference on the small turn.
It is the time from first output to the last output.

| Tool-result target | Read | installed decode ms | work decode ms | hot decode ms |
| ---: | ---: | ---: | ---: | ---: |
| prefix | 8828 | 280.4 | 389.9 | 429.1 |
| 0 | 52 | 261.9 | 324.9 | 384.5 |
| 32 | 93 | 305.1 | 339.2 | 365.2 |
| 64 | 122 | 289.6 | 367.4 | 390.8 |
| 128 | 188 | 308.6 | 349.4 | 367.4 |
| 256 | 309 | 231.1 | 316.3 | 371.1 |
| 512 | 573 | 284.9 | 337.0 | 349.1 |
| 1024 | 1082 | 261.5 | 399.0 | 332.8 |
| 2048 | 2108 | 265.1 | 352.3 | 333.1 |
| 3000 | 3053 | 277.0 | 306.4 | 346.9 |

The output length is 16 tokens in every cell, so this is a per-token change and
not a token-count change.
Three effects can contribute. The earlier checkpoint change moves a host copy
into the decode interval. The wide kernel changes the expert path. The
admission policy changes the expert placement. This study does not separate
them.
Draft acceptance does not explain the result on its own. The total accepted
counts are 132 of 216 for `installed`, 128 of 232 for `checkpoint`, 135 of 226
for `work`, and 127 of 225 for `hot`. The `checkpoint` arm has the lowest rate
but not the slowest decode.
Do not use these numbers as steady-state token throughput.

## Mechanism

1. The wide dequant kernel is on by default for SM70 in the frozen binary. An
   explicit override still wins. `SM70_PERF=0` turns it off.
2. The pinned CPU output buffer tests the actual need, `rows_cpu * N`. It
   doubles the allocation only when the actual need exceeds the capacity. This
   avoids a false host allocation or free.
3. The admission policy is opt-in. `STRATA_PREFILL_PROMOTE` defaults to 0 and
   gives a per-part, per-loan rotation budget. The budget counts stable-head
   rotations.
4. The policy uses the full-RAM expert arena only. It declines the RAM
   complement, a static peer, a helper, a remote part, a batch, and a head
   alias. It also declines to start when the helper path is enabled, and it
   logs the reason. The two features do not run together.
5. An incoming nonresident expert is hot when its hot count exceeds both the
   old victim count and the cold head count. The layer must also have enough
   slot room.
6. The policy prices the head across every positive chunk from 1 to the
   original cap. It reprices for each device and geometry. It does not allocate
   extra cache memory.
7. A refill copies the cold head from its slot to the lent slot on the device.
   It then copies the hot expert from host memory to the freed slot. The same
   one-host-to-device blob budget applies to each loan entry.
8. The cache map and the host map are published together after a fence.
9. A retry phase does not repeat a device copy after the source slot is
   overwritten. Counters reset for each loan.
10. KV growth skips a stale or inconsistent active entry. It never repairs one.

Expert placement can change the routing of an existing resident expert, and it
can change CPU versus GPU rounding. Do not claim bit-identical output against
an arm with different expert placement.

## Prior Nsight evidence and its limits

An earlier Nsight Compute capture measured the same narrow and wide dequant
kernel set on the same prompt shape. The capture used a 8192-token prefix,
diffs of 128, 512, 1024, 2048, and 3000, and a 16-token output cap. Q6 dtype 14
was the measured kernel.
GPU 0 ran 222 calls and fell from 129.721 ms to 20.466 ms. GPU 1 ran 546 calls
and fell from 334.964 ms to 51.753 ms.

These are captured kernel-time sums. They are not one 512-token request and
they are not a six-times end-to-end claim.
A separate older trace of a 512-token read showed about 9.6 GB of host-to-device
traffic. GPU 0 moved 1.4 GB in about 0.49 s. GPU 1 moved 8.2 GB in about
0.638 s. Both rates are near the PCIe limit of their link.
No matched traffic profile exists for this four-arm set. This report gives no
bytes-saved value.

Telemetry came from 1 Hz `nvidia-smi` samples. There were 1264 samples, 632 per
GPU, with no query errors. The maximum temperature was 48 C on GPU 0 and 58 C
on GPU 1. No sample reported a software thermal flag, a hardware thermal flag,
or a hardware slowdown flag. The power-cap flag was active in 5 samples on
GPU 0 and 14 samples on GPU 1. SM clocks ranged from 135 MHz to 1380 MHz and
include idle and startup time. Do not describe the clocks as locked, and do not
describe the run as free of all throttling.

## Correctness checks

- The exact dequant parity check passed in 16.96 s on CUDA.
- The default-settings test passed.
- The conversation-cache suite passed 4928 checks.
- The Python suites passed 67 tests.
- The expert cache slot guard test now covers a real GPU byte-copy rotation with
  uniform and sized payloads. It checks atomic remap errors, a stale entry, an
  already resident entry, a duplicate alias, and D2D room errors. It passed.
- A temporary byte probe ran five requests before release. The prefix was 8192
  tokens and the diffs were 8192, 128, 16384, and 256. The output cap was 16
  tokens. KV growth moved cells 16384 to 24576 to 40960. The probe made 4470
  rotations and 8940 slot memory compares. All compares passed.
- The byte probe was removed from the production path before the measured
  rounds. Its latencies are not scored.
- A consumer smoke test ran on the shared wide binary with the policy off and
  with the policy at 2048. Each arm did a warm 8K turn and two tool turns and
  returned the exact `READY` text. It checked 17 plus 25 equals 42, the JSON
  answer 42, and a forced lookup for `alpha`. All 8 of 8 checks passed. The
  admission arm made 3063 rotations.

These are limited checks. They are not a general model-quality evaluation and
they are not a universal bit-wise proof.

## Rejected experiments

| Trial | Evidence | Decision |
| --- | --- | --- |
| CPU batch 16 or 32 with a higher global cap | ABBA at 1K read: 1682 to 1828 ms. At 3K read batch 16 gave 3469 to 3391 ms, but the median ratio was 1.00. | Reject. Fully reverted. |
| Row-normalized CPU cost | Mixed results. | Reject. Fully reverted. |
| Full-capacity KV staging workspace | The existing KV-growth mode 0 inactivates the idea. | Reject. Fully reverted, zero effect. |
| Larger streamed ring for short turns | The ring is already 32 when the turn is below 3072 tokens. | Reject. No change. |
| Global CPU-share calibration | The CPU `auto` EMA is already per stage of each prefill implementation and uses its own GPU events. | Reject. The premise was wrong. |
| Helper refill path with fixed fraction 0.2 and 0.4, stream threshold 1024, and refill overlap off | Both helper arms stopped at the first prefix with `prefill: the PLE rows' upload failed: an illegal memory access was encountered`. The two direct controls completed 9 of 9 turns. At max-new 1, the stream-threshold-1024 control took 2366.5 ms at 1K read and 4117.7 ms at 3K read. The default-threshold control took 1648.1 ms and 3509.8 ms. | Reject. Not shipped. This report gives no root cause and does not reproduce the unsafe option. |

## Research

The admission policy is local to this runtime. It is not a port of a published
cache algorithm.

- [HybriMoE](https://arxiv.org/html/2504.05897), April 2025, caches high-row
  experts on the GPU and runs low-row experts on the CPU, with a dynamic
  cache. This study does not implement its MRS algorithm.
- The NVIDIA post
  [How to optimize data transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
  gives the rules used here: move fewer bytes, use pinned memory, batch the
  transfers, and overlap the work. It also warns that overlap does not remove
  the bandwidth cost.
- [llama.cpp `llama-moe-cache.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-moe-cache.cpp)
  shows a sorted-miss and SLRU cache. It is background for the victim choice.
- [ktransformers](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
  targets SM80 and later with AVX-512 or AMX. This host is a V100 with AVX-2.
  Do not copy its settings or its performance numbers to this system.

## Reproduce

Use a regular IQ3_S pack, tokenizer, native shard, and PLE shard. Stop the daily
service while these commands use the GPUs. Set `CONFIG` to a private
configuration file. Set `INSTALLED`, `CHECKPOINT`, and `WIDE` to the three
executables. Keep the output directory private.

```sh
export STRATA_ARENA_LOCK=0 STRATA_WATCHDOG_S=300
export STRATA_EXPERT_PAIR=1 STRATA_SPLIT_RING=192
export STRATA_PREFILL_CPU_SHARE=auto STRATA_PREFILL_STREAM_MIN=3072
export STRATA_EMB_REUSE_ACCOUNT=1 STRATA_SM70_TABLE=1 STRATA_PF_WMMA=1
export STRATA_PF_SKIP_UNROUTED=1 STRATA_KV_GROW=1 STRATA_ROUTE_RESIDENT=1
export STRATA_ROUTE_TAIL_SKIP=0

python tools/agent_turns_bench.py "$CONFIG" \
  "installed:exe=$INSTALLED" \
  "checkpoint:exe=$CHECKPOINT" \
  "work:exe=$WIDE,STRATA_PREFILL_PROMOTE=0" \
  "hot:exe=$WIDE,STRATA_PREFILL_PROMOTE=2048" \
  --workload agentic --rounds 2 --port 8097 --prefix-tokens 8192 \
  --additions 0,32,64,128,256,512,1024,2048,3000 --turns 9 \
  --max-new 16 --seed 20261021 --log "$RESULTS/results.jsonl" \
  --workdir "$RESULTS/runs"
```

The checkpoint pool and the one save thread are shipped profile values of the
last three executables. The `installed` executable ships them off with a narrow
kernel. The runner reverses the arm order on the second round and gives each
arm a new server and an empty conversation store.

To record telemetry, run this query in a 1 Hz loop during the measurement:

```sh
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw,power.limit,clocks_throttle_reasons.active \
  --format=csv,noheader
```

## Limits

- One host, one workload shape, one seed, and two rounds per arm. The medians
  are robust on the large turns and weak on the small turns.
- The policy has a cost at the smallest read and on the cold prefix. The study
  does not instrument per-request rotation counts, so the split between decision
  cost and refill cost is an interpretation, not a measured value.
- The study does not isolate the per-token decode cost. The checkpoint copy, the
  wide kernel, and the admission policy all changed together in that interval.
- A later build (`d0eb45c4107ce3d6e44486c229d677d5b6fb91ea7076b5fcac315b97b41b29f7`)
  adds a helper opt-in refusal and active-slot bounds. It is not part of this
  study, and this report gives no performance value for it.
- The configured VRAM reserve was 750 MiB in all arms. A separate direct
  harness check reported a settings mismatch only because its canonical value
  is 700 MiB. That check used an explicit settings-strictness waiver, and the
  runtime argument stayed at 750 MiB.
- During the HTTP rounds the frozen wide binary logged 456 MiB free on a
  request, below its 512 MiB warning threshold. The message suggests an 832 MiB
  reserve. That value is the suggestion, not the configured setting. The run did
  not prove paging.
- No matched Nsight traffic profile exists for these arms. No transfer-volume
  saving is claimed.
- Do not reuse the old generic 1.38 MB expert blob figure. This model's largest
  blob is 2.66 MB.
- Raw prompts, response captures, credentials, private configuration, and full
  logs stay local. This report contains aggregates only.
