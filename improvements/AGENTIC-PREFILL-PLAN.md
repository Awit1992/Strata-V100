# Agentic prefill optimization plan

## Scope

Optimize repeated agent turns on two V100 16 GB GPUs. Do not change the production service during source work. Keep all candidate changes on a separate branch. Use a stopped-service test window for GPU measurements. Restore the original service after the window. Do not change the installed engine or private configuration.

## Request evidence

A read-only sample from the runtime log contains 71 completed requests. In this sample, 48 requests read fewer than 300 new prompt tokens. These requests read approximately 5,930 tokens in 39,862 ms. They used approximately 41% of the sample's prompt time. Prefix reuse is already high. These figures describe this sample only. They are not controlled benchmark results. Do not publish request text or private runtime logs.

The short-read code in `src/program/generate.cpp` describes a fixed batched-prefill cost of approximately 300 ms and a later expert-slot refill cost of approximately 180 ms. This makes short additions a different optimization target from large cold prompts.

## Measurement contract

Use the existing agent-turn benchmark. Record new prompt tokens, reused tokens, prompt milliseconds, prompt tokens per second, decode tokens per second, and HTTP elapsed time. Compare the same turn across arms. Separate the first cold turn from later reused turns. Do not calculate effective prefill throughput from total context tokens when most tokens are reused.

The direct harness [`bench/run_v100_prefill.py`](../bench/run_v100_prefill.py) has the same shape with `--chain`: one prefix turn, then one turn per exact diff, with the prompt cache on and a private cache store. It reports the engine's own `reused` and `prompt_read` counts for each turn. Use it when the HTTP layer is not under test.

Use fresh conversation storage for each arm. Keep the model, layer split, ring size, generation settings, and request order fixed. Alternate arm order across rounds. Do not use production conversation storage. Keep diagnostic timing off in scored runs.

## Experiment order

1. Establish the current engine baseline with short shared-prefix tool turns. Include 64, 128, 256, and 512-token targets. Report actual token counts; text size is not token count.
2. Test `--pipeline-windows 1` with the existing short-read threshold. Then test thresholds of 128 and 256. This separates pipeline overlap from threshold changes. Reject an arm if short-turn latency or decode performance regresses.
3. Confirm CPU-share settings from the complete service unit, including drop-ins. The split path accepts explicit `STRATA_PREFILL_CPU_SHARE=auto`; only default attachment excludes a split. The earlier split-CPU experiment reports lower 256- and 600-token latency. Do not treat that experiment as a result for this branch.
4. Test explicit layer splits of 20 and 24 against 16. Measure both short-turn latency and long-prefill throughput. Record expert residency and decode performance. Do not select a split from layer count alone.
5. Keep a large cold-prompt control. A short-turn improvement must not silently replace a long-prefill regression.

Only measured winning settings can become recommended runtime settings. A benchmark candidate is not a production recommendation. Pipeline and CPU/GPU arithmetic can change numerical results; identical output is not assumed.

## Later kernel and cache work

- Inspect short-chunk expert transfers before a kernel rewrite. The weak host-to-device link and fixed slot-loan cost can dominate arithmetic.
- Profile conversation parking separately from prefill. The sampled park operations take approximately one second. Incremental or asynchronous capture needs explicit ownership and cancellation rules before implementation.
- For Volta kernels, inspect coalesced memory access, transfer overlap, occupancy, and FP16 tensor-core use with FP32 accumulation. Use an isolated benchmark for each change. Do not copy Ampere performance claims to V100.
- Do not replace attention with official FlashAttention-2 on V100. Its documented NVIDIA GPU support is Ampere, Ada, and Hopper. A Volta-specific implementation requires separate compatibility and correctness evidence.

## Research sources

- [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/): shared-prefix reuse avoids repeated prefill. This runtime already has substantial reuse.
- [vLLM prefill tuning](https://docs.vllm.ai/en/v0.21.0/configuration/optimization/): token budgets trade latency against throughput. Its scheduler settings are not Strata settings.
- [NVIDIA Volta tuning guide](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html): architecture-specific memory, occupancy, and tensor-core guidance.
- [Official FlashAttention support](https://github.com/Dao-AILab/flash-attention): GPU compatibility limits.
- [Prior local split CPU-share experiment](../benchmarks/v100-multi-gpu-cpu-prefill-2026-10-07.md): prior measurements and numerical limitations. These are not measurements of this branch.

## Release gate

Run CPU checks and the benchmark command-line smoke first. Run GPU comparisons only in an authorized service test window. Record observed results and restore service health. Keep private paths, credentials, prompts, and logs out of commits. Open a pull request only to the user's fork. The operator merges the pull request.

## Branch verification

The CPU benchmark tests passed: 15 tests. The server regression suite passed: 319 tests. The focused prompt-progress checks passed: six tests.

A two-arm HTTP smoke used the mock engine, one prefix, and four additions per arm. Both arms completed. The report separated cold and short turns. It recorded HTTP elapsed time and measured additions. Engine prefill timings remained unavailable, as expected for the mock engine. This smoke proves benchmark execution, not GPU speed.

A direct lazy-engine smoke checked a configured 256-token short-read tail. Progress at the tail boundary was omitted. Progress one token outside the boundary was retained. No model process was started.

The initial verification did not stop or restart the production service. The operator then authorized a GPU test window. The branch now includes compact CPU activation downloads and a grouped Volta launch change. The engine was built and tested in the isolated worktree. It was not installed over the production engine.

The [GPU report](../benchmarks/v100-agentic-prefill-2026-10-13.md) records the 0–2K prompt-difference comparison and the cold sweep through 32K. The candidate reduced median 2K cold-prefill latency by 6.2 percent. In the pinned chained test, differences of 256–2,048 tokens improved by approximately 3–4 percent. The smallest turns did not improve consistently. Automatic sharing changed generated sequences. One decode bucket regressed materially, so no guaranteed whole-request speedup is claimed.

The 512-token stage-owned buffer trial improved some short turns, but its 2K eligibility limit regressed the 1K case. This setting was rejected as a general recommendation. No production defaults changed.

The candidate completed the real HTTP agent workload. Median prefill latency improved by approximately 4.1 percent near 570 new tokens, 5.6 percent near 1K, and 6.8 percent near 2K. The original production service was restored. Health reported the model loaded. An authenticated chat request returned `READY`. The installed binary and production settings remain unchanged.

## Second-pass implementation

The actual expert pack is mixed. Most gate/up layers use IQ3_XXS or IQ2_S, not IQ3_S. GPU 0 has no IQ3_S gate/up layer. The second pass therefore adds short-row kernels for IQ3_XXS and IQ2_S as well as IQ3_S and IQ4_NL. Larger groups retain the original tiles.

Early refill now supports the actual mixed Volta format pairs. A batched segment can queue refill before a short header window returns its scratch loan. It cannot queue refill before another batched segment reuses that scratch. Pending KV-prefetch copies finish before the stage-complete callback.

The final isolated build passed all four focused C++ checks. All twenty fixed-share requests matched the first-pass engine's complete sixteen-token outputs. The fixed-share decode median differed by approximately 0.5 percent. The [second-pass report](../benchmarks/v100-agentic-occupancy-2026-10-14.md) records the implementation, selected and rejected kernel shapes, and matched measurement controls.

The final 180-request HTTP comparison used the installed, first-pass, and final engines. Route-tail skipping was disabled equally. Corresponding actual reads matched, and every request generated 64 tokens. Across fourteen short turns, the median per-turn prefill latency reduction was 15.52 percent against the installed engine. The whole-HTTP reduction was 8.18 percent against the installed engine and 6.95 percent against the first pass. Thirteen short turns improved; the 65-token turn did not.

The matched pinned chain reduced prefill latency by 11–19 percent at differences of 128–2,048 tokens against the first pass. Smaller differences remained approximately unchanged. The cold sweep reduced median prefill latency by 10.28 percent at 2K and 4.93 percent at 32K. All direct-run cooldown gates passed. The streamed-KV on/off smoke matched complete outputs and exercised nonzero prefetch transfers with early refill.

The original service was restored with its unchanged installed binary and settings. Health reported the model loaded. An authenticated chat request returned `READY`. The candidate was not installed. These measured gains do not establish identical outputs under automatic sharing or a universal decode improvement.
