# CUDA V100 upstream sync: 19 October 2026

## Source and hardware

- Fork baseline: `ebd2f7ad`.
- Upstream source: `3d1fb480`.
- GPU: two Tesla V100 PCIe cards, 16 GiB each.
- Build: CUDA 12.8, compute capability 70, Release, Ninja.
- Keep the measured GPU order `0,1`, layer split `16`, and two pipeline windows.
- Use the regular IQ3_S pack and both regular model shards for model checks.
  Do not use the daily abliterated assets for PR comparisons.

## Build

Use a separate worktree and build directory. Do not replace the installed engine.

```sh
cmake -S . -B build-v100 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON \
  -DSTRATA_PREFILL_MMQ=OFF \
  -DSTRATA_MMQ_KQUANTS=OFF \
  -DSTRATA_Q6K_EXPERTS=OFF
cmake --build build-v100 -j 6
```

`STRATA_PREFILL_MMQ` controls the optional HIP build path. Native CUDA
experts still build the MMQ support library. This build also compiles the
architecture-guarded FP4 unit. Its probe rejects FP4 MMA on V100.
The selected configuration does not enable K-quant MMQ or the Q6_K experts.

## Integration repairs

- Keep the fork's shared conversation payloads. Copy their ownership handles,
  not their complete payloads, when upstream progress-aware checkpoint copying
  copies a checkpoint.
- Restore the fork's archive append helpers and explicit-token checkpoint path.
- Pass the original held size to the conversation-buffer reserve limit.
- Apply the alignment check before either eight-value int8 K/V gather path.
  The four-value path remains valid for pools that are not eight-byte aligned.
- Avoid a second BF16-to-FP16 conversion when an activation writer already
  produces FP16 bits. A numeric GPU regression covers all four writers and
  both activation-image modes.
- Select the original pipelined GDN kernel explicitly in its bitwise control.
  Keep the installed fork's normal Volta chain-split dispatch unchanged.
- Reuse one PLE prefetch slot for repeated token keys. A duplicate request
  does not require one distinct prefetch slot per token.
- Include the guarded trellis CPU translation units in the ISA audit.
  The audit still rejects unguarded wide instructions and static initializers.
- Make installer hash checks independent of remote release availability.
  Count only ranged piece requests in the parallel-download check.

## Python checks

The following checks completed:

| Check | Result |
| --- | --- |
| Server suite | 910 tests; passed, 11 skipped |
| Tools suite with the GGUF Python path and a temporary pytest dependency | 1,009 tests; passed, 2 skipped |
| NVFP4 local-pack module in a temporary pytest environment | 9 passed, 2 skipped because torch is absent |
| Retained terminal modules with optional monitor dependencies | 714 tests; passed, 14 skipped |
| Tuning suite | 269 tests; passed |
| Time-skip suite | 35 tests; passed, 1 skipped |
| CLI and server help commands | Passed |
| CLI doctor | Passed; reported that monitor dependencies are optional |
| Minimal server transport smoke in an isolated server-only environment | 39 tests; passed, 3 Windows-only checks skipped; `rich` and `textual` absent |

The daily Python environment was not changed. The monitor checks used a
separate dependency overlay with `rich` and `textual`, a normal `TERM`, and
no `NO_COLOR`. Tests for excluded desktop and distribution packages are not
part of this sync.

The minimal-server transport smoke passed before scope reduction. The temporary
development smoke scripts are not retained in this PR. The underlying server
transport correctness tests remain.

## Native and GPU suite

Before scope reduction, the build completed and CTest ran 168 checks in 504.55 seconds:
158 passed, 10 skipped, and none failed. The new FP16 activation-image
regression, K/V gather parity, GDN chunk parity, PLE reader self-test,
conversation cache tests, and ISA audit passed.

```sh
ctest --test-dir build-v100 --output-on-failure -j 1 -E '^ple_parity$'
```

The skipped checks were `qsa_slots_parity`, `iq_parity_fixtures`,
`iq_parity`, `gr_upmix_parity`, `prefill_fused_moe_test`,
`prefill_fused_iq_test`, `expert_multi_test`, `native_hadamard_test`,
`tq_cpu_parity`, and `tq_cpu_parity_avx2`. Do not count these as passes.

## GPU test limits

The standalone PLE parity test needs a fixture that is not installed on this
host. It is excluded explicitly with `-E '^ple_parity$'`.
The pinned upstream GGML has no CPU trellis oracle. Those comparisons report
skip status, not pass status. The available Hadamard and Q8_0 checks still run
before the missing oracle is reported. No trellis-model quality claim is made.

## Regular-model inference comparison

The installed baseline and candidate each loaded the regular IQ3_S model on
both V100 cards. Both completed cold prompts of exactly 2,048 and 8,192 tokens,
with 32 generated tokens per request and no request errors.

| Cold prompt | Baseline prompt ms | Candidate prompt ms | Baseline tokens/s | Candidate tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 2,048 | 2,202.5 | 2,319.4 | 929.85 | 882.99 |
| 8,192 | 5,541.6 | 5,769.8 | 1,478.27 | 1,419.81 |

This is a single-sample smoke comparison, not a performance qualification.
The candidate was approximately 5.3% and 4.1% slower by prompt time in these
two samples. Do not claim a speed gain from this sync. The 2,048-token output
started with a different token; this check does not prove output parity.

Both arms used the same regular assets, int8 K/V, split `16`, two pipeline
windows, reserve 750 MiB, speculative depth 8, and the measured Volta switches.
Prompt caching and conversation caching were disabled for these cold reads.
Cooldown was disabled. The harness settings check expects a 700 MiB reserve;
the 750 MiB setting was intentional and accepted with `--no-settings-strict`.

The candidate executable SHA-256 was
`7ae2041adeef1b5496bdccfd898ef0c6afd699618bb9c41b942eae7e72bee223`.
The baseline executable SHA-256 was
`7286eeeb74edfa069c14f7a3889041c55a670fa13a26b4058f68c1e1041b48d6`.
The candidate was built before the merge commit. Its report's Git identifier
therefore names the fork baseline, not the merged source tree.

## Multi-turn HTTP smoke

The candidate server loaded the regular model and completed four real HTTP
requests. Each request produced 32 tokens. The first prompt read 2,684 tokens.
The next three turns read 117, 189, and 314 new tokens and reused 2,683, 2,799,
and 2,987 tokens. Their measured reuse shares were 95.8%, 93.7%, and 90.5%.
Decode rates were 59.7, 67.8, 70.4, and 67.4 tokens/s.

The first launch used a port that was already occupied. The smoke then used
a separate loopback port. No process on the occupied port was stopped.
The harness stopped its candidate server and engine at the end.

## Daily runtime restoration

The installed engine was not replaced. After the checks, the daily user service
started with the matching abliterated SC117 IQ3_S shards, pack, tokenizer, and
vision model. Its running engine retained GPU order `0,1`, split `16`, and two
pipeline windows. The service was active with zero restarts.

The authenticated `/v1/models` response reported a loaded model. An
authenticated chat request returned `2 + 2 equals 4.`. The three speech services
remained inactive and disabled. No private configuration or credential is part
of this PR.

## Benchmark scope correction

At the user's request, remove the new standalone timing targets, community
benchmark clients, quantization corpora, profile-mixing tools, model-development
harnesses, and development smoke scripts. Remove the new CLI benchmark command
and its benchmark-only management actions and tests. Remove new AMD tuning data
and restore the unrelated HIP packaging scripts to the fork baseline.
Remove benchmark result tables and instructions for the excluded tools.

Keep existing fork benchmark tools and results. Keep the numeric correctness
checks, including `bench/micro/fusions_parity.cu`: that file tests kernel parity,
not timing. Keep calibration used by model setup.

After this reduction, the CUDA build completed and all 20 selected CPU/cache/ISA
checks passed. The engine help command ran successfully. The reduced tools suite
ran 984 tests and passed with two skips in a temporary pytest dependency overlay.
The retained terminal suite ran 687 tests and passed with 14 skips. The ordinary
parallel-request server suite passed all 39 tests. CLI help no longer lists the
removed benchmark command. The terminal suite emitted resource warnings and a
Textual event-loop-close exception from a background teardown thread; unittest
reported no failures. This correction does not change that UI lifecycle code.
No daily runtime configuration or installed binary changed during this correction.
