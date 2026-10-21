# V100 incremental TTFT, 2026-10-21

## Result

The revised runtime reduces client-observed time to first output (TTFT).
It does not achieve sub-second TTFT for all 0–3000-token increments.
The isolated A/B confirmation reduced TTFT by 67–134 ms across nine tool turns.
All nine per-turn medians improved. Four small turns remained below one second.
A 1082-token read took 1.815 seconds. A 3053-token read took 3.592 seconds.

This is primarily a scheduling improvement. The checkpoint copy still occurs
during the request. Do not interpret the TTFT reduction as an equal reduction
in full response time or complete tool-call latency.

The [cache admission study](v100-prefill-cache-admission-2026-10-21.md)
adds a four-arm comparison of checkpoint scheduling, native dequantization,
and expert-cache placement. The measurements below remain the original
checkpoint results. Do not combine timings from the two studies.

## Test system and fixed settings

- Two Tesla V100-PCIE-16GB GPUs, 16 GiB each.
- GPU 0: PCIe Gen3 x4, layers 0–15. GPU 1: Gen3 x16, layers 16–47.
- Ryzen 5 3600, six cores and twelve threads, approximately 94.2 GiB usable RAM.
- Linux 7.0.0-38-generic, NVIDIA driver 580.178.04, CUDA 12.8, Release SM70 build.
- The GPU power limits remained at 175 W each.
- Regular Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S assets, not the daily abliterated assets.
- GPU order `0,1`, layer split `16`, and `--pipeline-windows 2` in both speed arms.
- Maximum context 524,288 cells, INT8 KV, and 750 MiB VRAM reserve in both arms.
- Model quantization, KV quantization, and KV capacity remained unchanged.
- CPU prefill share `auto`, stream threshold 3072, and expert ring 192.
- The existing WMMA, skipped-unrouted, KV-growth, Volta-table, and resident-routing settings remained enabled.
- Route-tail skipping was disabled in both speed arms.
- The output cap was sixteen tokens. Each request used the same seed and sampling settings.

The baseline executable was preserved before any change. Its SHA-256 was
`f740c535277fa66d825d2d443937d895406c69095c3224f48410dd8761b920c4`.
The final candidate SHA-256 was
`090ba4faa5c95c171c9f68e8a29295a812ff03c5bd1840c2ea63a9556c8aac7e`.
The source checkout started at `5ec4b106`. The hashes identify the measured
executables; the checkout identifier alone does not identify the candidate.

## Bottleneck evidence

A diagnostic run reused the complete live prefix on each incremental turn.
Full conversation parking was not on that path. At a 128-token increment:

- The batched read took 881.7 ms.
- Two checkpoint saves took 153.6 ms. Device synchronization took about 0.1 ms each.
- Expert-cache refill took 37.3 ms.
- The final short window took 29.2 ms.

The 512-token read spent 1496.9 ms in the batched path. GPU 1 spent 341 ms
waiting for expert work/copies in its 931 ms GPU timeline. At 3000 tokens,
GPU 0 and GPU 1 reported 530 ms and 783 ms of copy wait respectively.
These diagnostic timings include instrumentation. They are not scored TTFT.

The checkpoint host copy was a removable first-output dependency. The expert
processing and transfer work was not removed by the selected change.

## Selected changes

1. Capture the input anchor at `n-1` on its owning GPU. Copy it to host memory
   after the first output window is flushed. Insert it into the checkpoint
   chain before the request ends, when decode is quiescent.
2. Use a request-owned snapshot for pipelined decode. Reuse the existing
   first-window snapshot in serial decode. The pipeline uses about 113 MiB
   of temporary GPU memory across both stages. Release it after the host copy.
3. Keep synchronous capture for prefill-only requests, merged first windows,
   the explicit opt-out, and GPU snapshot allocation failure. Never resume
   from an incomplete checkpoint or suppress a checkpoint-save failure.
4. Avoid a duplicate response-end copy when that checkpoint is the input anchor.
5. Use overwrite-aware storage for checkpoint payloads. Do not copy shared
   bytes that the device copy will replace. Preserve the archived owner's bytes.
6. Reset the segment plan when a checkpoint buffer returns to its pool.
7. Enable pooled buffers and parallel stage saves in the shipped V100 profile.
   These settings added only a small gain in the initial study. They do not
   justify a large checkpoint-throughput claim on their own.
8. Measure HTTP TTFT from streamed output. Ignore role-only, keep-alive, and
   control-only events. Accept generated content, reasoning, or tool function
   names/arguments. Leave TTFT absent if there is no generated output.
9. Parse the engine protocol's named DONE extensions without treating them as
   positional integers. The old parser failed on `chunk=0` before collection.

`STRATA_INPUT_ANCHOR_DEFER=0` selects the synchronous control on the same
candidate. It does not remove the response-end deduplication change.
The shipped profile adds `STRATA_CKPT_POOL=1` and `STRATA_CKPT_THREADS=1`.
The installed daily profile was not changed during this experiment.

## Streaming HTTP comparison

The workload used a shared prefix target of 8192 and tool-result targets of
0, 32, 64, 128, 256, 512, 1024, 2048, and 3000. Targets are not token counts.
The rendered cold prompt had 8828 tokens. Message and tool syntax add tokens
even when the tool result is empty. Both arms had the same actual read counts
and reused counts on all ten turns in every round.

Each arm and round used a new server, expert profile, and conversation store.
Round order alternated A/B and B/A. There were four initial rounds and a
separate two-round ABBA confirmation. The table uses the isolated confirmation.
No build, test, or other GPU workload ran during that confirmation.

TTFT starts before the HTTP connection and ends at the first generated SSE
output. It includes parser buffering. It is not raw engine-token latency.
Full request time includes all sixteen output tokens and request completion.

| Tool-result target | Actual tokens read | Baseline TTFT ms | Candidate TTFT ms | Latency reduction |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 52 | 624.9 | 535.8 | 14.27% |
| 32 | 93 | 866.7 | 732.6 | 15.47% |
| 64 | 122 | 987.0 | 908.0 | 8.00% |
| 128 | 188 | 974.0 | 899.0 | 7.70% |
| 256 | 309 | 1199.0 | 1125.7 | 6.11% |
| 512 | 573 | 1443.3 | 1369.0 | 5.15% |
| 1024 | 1082 | 1881.7 | 1814.8 | 3.56% |
| 2048 | 2108 | 2642.6 | 2557.9 | 3.20% |
| 3000 | 3053 | 3685.9 | 3591.8 | 2.55% |

The cold turn improved from 8230.0 to 8091.4 ms. This is not a cold-prefill
throughput breakthrough.

The initial four-round comparison also improved all nine incremental medians.
Its baseline/candidate medians were 623.8/531.0 ms at 52 tokens,
1912.9/1784.3 ms at 1082 tokens, and 3669.6/3568.2 ms at 3053 tokens.
A 10.3-second CPU regression-target build occurred during that collection.
The independent confirmation avoids that possible CPU-load confound.

### Full response time and decode accounting

The host copy moves from the prompt interval into the decode interval.
The engine's reported decode tokens/second therefore decreases. It is not a
steady-state token-throughput comparison. The change can also introduce a
pause after the first output window.

There was no consistent full-request speedup. In the isolated confirmation:

| Actual read | Baseline full request ms | Candidate full request ms |
| ---: | ---: | ---: |
| 52 | 890.1 | 888.4 |
| 573 | 1710.5 | 1709.4 |
| 1082 | 2137.5 | 2179.8 |
| 3053 | 3968.2 | 3915.7 |

## Rejected changes

These were exploratory single-process chain runs, not the final HTTP score.
They used the same exact prompt stream and a fresh first round.

| Trial | Evidence | Decision |
| --- | --- | --- |
| Fixed CPU share 0.25 instead of auto | 129-token read: 1018.6 to 1461.1 ms. 513-token read: 1487.2 to 2021.8 ms. | Reject for speed. Use only as a correctness control. |
| Full nonresident CPU share | 129-token read: 1200.7 ms. 513-token read: 1872.7 ms. | Reject. |
| Stream all experts from 1024 tokens | 2049-token read: 2819.1 to 3433.5 ms. The 3001-token read improved to 3732.5 ms. | Reject the broad crossover change. |
| `--prefill-pipe 512` | Cold 8192 tokens: 6254.5 to 11656.1 ms. 3001-token read: 4028.2 to 5968.6 ms. | Reject. |

Repeated direct-chain rounds can reuse archived copies of the same prompts.
In the second baseline round, several large requested increments read only
one token. Those values were not treated as fresh incremental measurements.
The final HTTP comparison uses fresh per-round processes and stores.

## Correctness checks and limits

- The benchmark Python suites passed 67 tests.
- The CPU conversation-cache suite passed 4928 checks.
- A temporary verification build compared the deferred host image with a
  synchronous reference from the same request. All four comparisons matched
  byte-for-byte across both stages. The comparisons covered serial and
  pipelined cancellation scenarios. Fields were `ids`, `gdn`, `ple`, `tails`,
  `dead`, and `block_pos`.
- Serial decode passed exact token and first-window-logit comparisons with
  resident-biased routing disabled and `--adapt-every 0`. The scenarios covered
  prefill-only input, cancellation after prompt completion, cancellation after
  the first token, continuation, and repeated identical continuation.
- Full token parity was not established for pipelined cancellation comparisons.
  Token counts after STOP are asynchronous. Some continuation outputs differed
  even with the adaptive tier and resident-biased routing disabled. Do not
  describe these pipeline runs as a token-for-token pass.
- The unchanged baseline also produced different internal state fingerprints
  across independent pipeline processes under the pinned-placement control.
  Its repeated outputs within each process matched. The production routing and
  adaptive tier also changed repeated outputs in same-process controls.
- The direct byte comparison proves checkpoint-image equivalence for the tested
  boundaries. It is not a general model-quality or universal determinism claim.
- No diagnostic hook remains in the source. Raw prompts, response captures,
  credentials, private configurations, and complete logs remain local.

## Research

The implementation reuses this runtime's existing checkpoint and snapshot
mechanisms. It does not import hardware-specific settings from other projects.

- [SGLang Unified Radix Cache](https://www.sglang.io/blog/unified-radix-cache)
  explains why hybrid recurrent models need an exact state checkpoint at the
  reuse boundary. A matching token prefix alone is not sufficient.
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
  separates reusable-prefix work from new-token work. This informed the actual
  read-count checks and the fresh-store A/B design.
- [llama.cpp CUDA optimization discussion](https://github.com/ggml-org/llama.cpp/discussions/17621)
  discusses launch and synchronization costs. Such costs must be measured on
  this split; single-GPU gains do not establish a dual-V100 gain.
- [NVIDIA Volta tuning guide](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html)
  describes transfer and occupancy limits. The existing short-row kernel study
  already addresses the principal Volta tile/occupancy changes in this fork.

The remaining large-increment cost is batched expert work and transfer.
This report does not establish sub-second TTFT for 512–3000 new tokens.

## Reproduce

Use a private regular-IQ3_S test configuration. Stop the daily service while
these commands use its GPUs. Preserve the baseline executable first. Build the
candidate in a separate directory with CUDA 12.8 and SM70.

Set `CONFIG`, `BASELINE`, and `CANDIDATE` to the private configuration and the
two executables. Keep the output directory private.

```sh
export STRATA_ARENA_LOCK=0 STRATA_WATCHDOG_S=300
export STRATA_SPLIT_RING=192 STRATA_PREFILL_CPU_SHARE=auto
export STRATA_PREFILL_STREAM_MIN=3072 STRATA_EMB_REUSE_ACCOUNT=1
export STRATA_SM70_TABLE=1 STRATA_PF_WMMA=1 STRATA_PF_SKIP_UNROUTED=1
export STRATA_KV_GROW=1 STRATA_ROUTE_RESIDENT=1 STRATA_ROUTE_TAIL_SKIP=0

python tools/agent_turns_bench.py "$CONFIG" \
  "baseline:exe=$BASELINE,STRATA_CKPT_POOL=0,STRATA_CKPT_THREADS=0" \
  "candidate:exe=$CANDIDATE,STRATA_CKPT_POOL=1,STRATA_CKPT_THREADS=1" \
  --workload agentic --rounds 2 --port 8097 --prefix-tokens 8192 \
  --additions 0,32,64,128,256,512,1024,2048,3000 --turns 9 \
  --max-new 16 --seed 20261021 --log "$RESULTS/turns.jsonl" \
  --workdir "$RESULTS/arms"
```

Restore the daily service with its original executable and matching abliterated
pack, tokenizer, native shard, and PLE shard after the test. Wait for model
loading. Verify authenticated inference before removing the restoration guard.

## Daily runtime restoration

The original installed executable still matched the baseline hash after tests.
The daily service was restored with its unchanged private configuration.
The running command selected the matching abliterated shards and pack.
It retained layer split `16` and pipeline windows `2`.
Health reported a loaded model and context capacity 524,288.
The models endpoint listed the served model. An authenticated chat request
returned `READY` with finish reason `stop`.
The service was active with zero restarts. The three auxiliary speech services
remained inactive and disabled. The restoration timer was stopped only after
these checks. No candidate executable or revised drop-in was installed.
