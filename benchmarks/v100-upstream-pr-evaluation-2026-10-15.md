# Recent upstream PR performance evaluation

## Selection

The review covers 374 open upstream PRs updated in the 72-hour window that starts on 2026-10-12 at 23:53:35 UTC. Of these, 266 were created in the window. See the [complete inventory](v100-upstream-pr-inventory-2026-10-15.md). The inventory separates triage from executed measurements. Not every patch was built.

The selected improvement is the existing `--pipeline-windows 2` option reported in [upstream PR #2395](https://github.com/Niko1221/Strata/pull/2395). This PR changes the service profile, not the inference kernels. It does not import an upstream branch wholesale.

The local hardware has two 16-GiB V100 PCIe GPUs at 175 W and a Ryzen 5 3600. GPU 0 has an x4 link. GPU 1 has an x16 link. All arms keep GPU order `0,1` and layer split `16`. The test does not assume equal transfer bandwidth.

The model is the installed native IQ3_S mixed-format pack. The arms retain int8 KV, the 524,288-token configured capacity, vision reservation, MTP spec 8, expert ring 192, CPU prefill share `auto`, stream threshold 3072, KV growth, and resident routing. Requests use at most 32,768 prompt tokens. Each cold request has a 16-token output cap; HTTP requests have a 256-token cap and stay below 13K total tokens. No larger prompt was benchmarked.

## Clean HTTP comparison

Each arm starts a separate server and uses an isolated conversation store and expert profile. Four rounds reverse arm order on alternate rounds. Each round sends one 8,828-token cold prefix and six continuation requests. The seed is 1234. The table gives the median of four requests per arm and turn.

| New prompt tokens | Baseline decode tokens/s | Pipeline decode tokens/s | Decode change | Baseline HTTP ms | Pipeline HTTP ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 121 | 70.20 | 79.45 | +13.18% | 2735.60 | 2563.90 |
| 321 | 70.60 | 82.05 | +16.22% | 3671.75 | 3146.55 |
| 1086 | 76.60 | 79.00 | +3.13% | 4538.75 | 3852.20 |
| 128 | 73.00 | 79.90 | +9.45% | 2965.80 | 3066.65 |
| 313 | 74.35 | 86.55 | +16.41% | 3595.75 | 2931.05 |
| 1089 | 74.15 | 80.75 | +8.90% | 4378.15 | 4108.05 |

All six short-turn decode medians improve. The median of the six throughput ratios is +11.31%. Cold-prefix decode is approximately unchanged: 70.20 to 70.45 tokens/s. This is not an 11% whole-request speed guarantee. One short-turn HTTP median regresses. Generated lengths and MTP work differ under automatic sharing and resident routing. These are production-profile observations, not matched-output or fixed-work decode measurements. There is no per-request thermal gate in the HTTP harness.

## Cold prefill

Each size has three exact-token requests. Corresponding prompt hashes match and every reuse count is zero. The table gives median prompt time.

| Prompt tokens | Baseline ms | Pipeline ms |
| ---: | ---: | ---: |
| 2,048 | 2,088.5 | 2,090.3 |
| 8,192 | 5,486.8 | 5,493.7 |
| 16,384 | 8,827.9 | 8,857.3 |
| 32,768 | 15,667.7 | 15,677.2 |

Cold prefill remains approximately unchanged. All 24 cooldown gates pass. The strict harness profile expects a 700-MiB reserve; both arms use the installed 750-MiB reserve. `--no-settings-strict` permits this known mismatch and does not change the reserve. Five of twelve complete output-ID sequences match under the production profile. Do not claim bit-identical production responses.

## Rejected local trials

- [#2357](https://github.com/Niko1221/Strata/pull/2357): built the SM70 multi-block greedy opt-in. Four HTTP rounds give a median short-turn decode ratio near 0.98. Removed the change.
- [#2358](https://github.com/Niko1221/Strata/pull/2358): ported the missing register split-norm kernel and CUDA opt-in. `gr_parity --selftest` passes on each GPU with `STRATA_HC_SPLIT=2` and the opt-in enabled. Four HTTP rounds with the CPU trial give a median short-turn decode ratio near 0.96 without pipelining. This does not establish an isolated norm gain. Removed the change. The existing `fused_gr_bench` reports zero differences, but does not isolate the new norm; its timing is not evidence for that norm.
- [#1949](https://github.com/Niko1221/Strata/pull/1949): AVX2 router parity passes. One pinned-core router measurement is 0.222 ms before and 0.224 ms after. No local improvement is established. Removed the change. No claim is made for Q5_1 or Q8_0 expert speed; this pack does not use those down formats.
- [#2356](https://github.com/Niko1221/Strata/pull/2356): inspected the port conflict. The upstream entry point does not support the fork's indirect resident weight table and row IDs. The fork also has short-group tiles. A direct cherry-pick is not safe. No runtime gain is claimed.
- [#2355](https://github.com/Niko1221/Strata/pull/2355): the fork already uses per-device capability handling for its prompt tiles.
- [#2689](https://github.com/Niko1221/Strata/pull/2689) and [#2690](https://github.com/Niko1221/Strata/pull/2690): excluded because CPU deferral changes model output.

Two exploratory three-arm HTTP sweeps each complete 84 requests. The clean two-arm sweep completes 56 requests. The cold sweep completes 24 requests. Only the clean sweep is the selected profile comparison.

## Reproduction

Stop the production service before GPU benchmarks. Build with CUDA 12.8 and SM70. Preserve the installed engine as `$BASELINE`. Set `$PIPELINE` to the rebuilt branch engine and `$CONFIG` to the existing private configuration. Use the installed service environment, including the Volta and prefill settings listed above.

```sh
python tools/agent_turns_bench.py "$CONFIG" \
  "baseline:exe=$BASELINE" \
  "pipeline:exe=$PIPELINE,xargs=--pipeline-windows;2" \
  --workload agentic --rounds 4 --prefix-tokens 8192 \
  --additions 64,256,1024 --turns 6 --max-new 256 --seed 1234 \
  --port 8097 --log /tmp/v100-pipeline-http.jsonl \
  --workdir /tmp/v100-pipeline-http

python bench/run_v100_prefill.py --config "$CONFIG" --exe "$BASELINE" \
  --arm-label baseline --targets 2048,8192,16384,32768 --repeats 3 \
  --max-new 16 --no-settings-strict --out /tmp/v100-cold-baseline

python bench/run_v100_prefill.py --config "$CONFIG" --exe "$PIPELINE" \
  --arm-label pipeline --targets 2048,8192,16384,32768 --repeats 3 \
  --max-new 16 --pipeline-windows 2 --no-settings-strict \
  --out /tmp/v100-cold-pipeline
```

The source parent is `c79f8e1f836aafc313dfdad6179a8468b049907e`. The installed reference SHA256 is `4b3adcad84838a72ed0117cb550bae851b6b10bb27f9fbc71fe5032d2c5dbbba`. The clean rebuilt engine SHA256 is `b064b86fda8a65668f04fa946df27e15587a3a644ba5e88d1d03ee9d69486e6a`. There are no retained inference-source changes. These results compare the installed binary with the clean fork build and the selected pipeline setting; they do not isolate compiler or build-provenance effects.

Raw measurements, private configurations, logs, and rollback copies remain on the test machine. They are not committed.

## Output control and installed service

A separate control uses fixed CPU share 0.25, resident routing off, and route
tail skipping off. At 2,048, 8,192, and 32,768 prompt tokens, all three prompt
hashes and all three complete 16-token output sequences match between the
reference and pipeline arms. This is a short correctness control, not a
quality benchmark or a proof for every production request.

The selected clean build is installed as the service engine. The service
uses the separate private pipeline configuration. The previous configuration,
engine, unit, and drop-ins are preserved for rollback. The unit passes
`systemd-analyze --user verify`. The running engine arguments include
`--pipeline-windows 2 --layer-split 16`. Both GPUs retain their measured link
widths and 175-W limits.

After installation, the user service is active. `/health` reports loaded,
524,288 context cells, and image support. An authenticated chat request returns
HTTP 200, text `READY`, and finish reason `stop`. The installed binary hash
matches the clean benchmark engine. The pipeline version remains running.
