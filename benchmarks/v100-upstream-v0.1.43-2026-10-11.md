# V100: selected upstream v0.1.43 performance ports

## Scope and hardware

Date: 2026-10-11. Source: upstream release v0.1.43, commit
`7695859aa13afb248bdb4d36f9b2e95707db079d`.
Fork source baseline: `29c4fd78621dacec19caa549cfb7f90889692bcd`.
See the [selection record](../improvements/RELEASE-v0.1.43-V100.md).
This is not a full upstream sync. The engine version remains 0.1.42.

Hardware: two Tesla V100-PCIE-16GB cards, SM70, 80 SMs per card.
Host: Ryzen 5 3600, 94 GiB RAM, Linux. Driver: 580.178.04.
Build: Release, CUDA 12.8, GCC 13, `sm_70`, native experts, AVX2 host code.
The user service was stopped for the GPU measurements. Its installed engine
and private configuration were not replaced.

## Resident route-swap kernel

Run `router_chain_bench T 20` with T equal to 1, 2, and 8.
The resident-swap section compares the pre-port serial kernel with the new
warp kernel on identical finite logits. Each captured graph has 48 layers.
Restore the input IDs before each pass. Warm both graphs, alternate the order,
and report the median of 20 passes. The synthetic table has 40% resident
experts and a margin of 4. This measures the kernel, not model decode speed.

| Tokens per window | Serial, us/layer | Warp, us/layer | Kernel speed ratio |
| ---: | ---: | ---: | ---: |
| 1 | 114.69 | 16.17 | 7.09x |
| 2 | 202.37 | 21.38 | 9.47x |
| 8 | 308.20 | 28.63 | 10.77x |

The counter probe recorded 4, 11, and 46 swaps for these cases.
`route_resident_parity` passed 240 cases. It exercised 205 cases with swaps,
with 2,004 swaps in total. IDs and counters had zero mismatches. Weights were
within eight ULP. The parity claim applies to finite logits, not NaN inputs.
The fork also replaces redundant same-address weight stores with one store
per lane for the ten selected experts.

## Real-model comparison

Model: Qwen3.8-Flash-Next IQ3_S native GGUF, two-GPU layer split at layer 16,
INT8 KV, speculative window 8, real-time MTP. Both arms set
`STRATA_ROUTE_RESIDENT=1`. Keep the private service's Volta tuning.
Disable RAM and disk conversation caches for this comparison.

Use `tools/ab_engine.py` for three rounds, alternating arm order.
Use its fixed three chat prompts and seeded 2,000-word prompt. Use greedy
sampling and an output limit of 128 tokens. All prompts were read in full.
The baseline is the pre-existing local 0.1.42 build, saved before compilation.
Its SHA256 is `c69620515a895acc4b9c8fe00d98196ea35ecb89234ba163268a93b6e54f6e08`.
The binary's exact source provenance is not established by this comparison.
Thus the table does not isolate each source change. The kernel comparison
above does isolate the route-swap implementation.

| Request | Prompt tokens | Baseline decode tok/s | Candidate decode tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| Merge sorted lists | 34 | 64.5 | 67.2 | +4.2% |
| Explain refrigerator heat transfer | 27 | 50.2 | 54.3 | +8.2% |
| Language-learning tips | 29 | 52.3 | 56.3 | +7.6% |
| Seeded word summary | 2,146 | 50.9 | 54.5 | +7.1% |

The values are medians, not a general V100 speed guarantee.
The chat requests produced 128 output tokens. The long request produced 111
in five runs and 106 in the last candidate run. Output hashes matched in all
three sorted-list pairs and in two of three pairs for each other request.
Do not infer byte-identical model output from this comparison.
Long-prompt speed was 416.4 versus 416.2 tok/s. No long-prompt gain is claimed.
The candidate log confirms that adaptive overlap stays off on a layer split.
The new canonical-pack calibration does not apply to these native packs.
The 90% tail-skip limit does not apply at this configuration's lower residency.

Raw numeric results and hashes:
[`model-ab.jsonl`](../bench/results/2026-10-11-v100-v0143/model-ab.jsonl) and
[`router-swap.json`](../bench/results/2026-10-11-v100-v0143/router-swap.json).
No private paths, credentials, prompt text, or answer text are included.

## Adaptive-copy and residency smoke

A single-V100 IQ3_S run used an 8K context, `--adapt-every 1`,
`--adapt-swaps 16`, `STRATA_ADAPT_OVERLAP=1`,
`STRATA_ADAPT_OVERLAP_RESIDENT=1`, and
`STRATA_ADAPT_OVERLAP_SLOW_US=2000`.
It completed three chat requests and an 887-token prompt, with 64 output
tokens per request. The log recorded 117 windows with copies still in flight
and 536.38 ms of accumulated per-layer gate waits. This is a scheduling smoke,
not a speed measurement. The resident-RAM override remains opt-in.

The same run set `STRATA_TAIL_SKIP_MAX_CACHED=0`. At 19.7% cache residency,
the log confirmed that automatic tail skipping was off. The normal dual-GPU
run retained the automatic rank-7 skip. The policy test checks the exact 90%
boundary, explicit-setting precedence, and canonical-PCIe platform controls.
No canonical Q2_0 expert pack is available on this machine. Calibration is
covered by policy tests, not a canonical-pack throughput measurement.

`file_expert_source_test` passed. Its GPU soft-commit and rotation checks also
passed on a V100. These checks cover staged expert aliases before and after
commit, and refusal for pageable or rotating exchange buffers.

## Build and regression checks

The integrated Release build completed with CUDA 12.8 for SM70.
Run `ctest --test-dir build-v100-prefill-opt -E '^ple_parity$'
--output-on-failure -j 1`: 135 tests were selected, 132 passed, and three
were skipped. No selected test failed.
The skips were `prefill_fused_moe_test`, `prefill_fused_iq_test`, and
`expert_multi_test`.

`ple_parity` was excluded because the previous sync report already records
its missing Q2_0 fixture. This is not a full-suite pass claim.
The final router parity check also passed after the unique-lane weight-store
and captured-graph benchmark corrections.

Review found an unsafe callback-failure fallback in the upstream overlap
code. The port corrects it: synchronize the copies already queued, then
publish only the completed layer's gate tick. Do not open gates for layers
whose copies are not yet queued. A synchronization error stops the refill.

The final engine also completed the single-GPU smoke with an `LD_PRELOAD`
injector that rejects every fifth host-function registration. The log recorded
643 injected failures. All four requests completed with 64 output tokens each.
Their output hashes matched the previous delayed-overlap smoke in all four
cases. The layer gates remained active. This exercises the fallback; it is
not a performance result. The temporary injector is not part of the port.

The original user service was restarted after verification. The PR does not
install the candidate engine.
