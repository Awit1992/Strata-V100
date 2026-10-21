# The measured dual-V100 service profile

These four drop-ins define the fork's dual-V100 service profile. Install them under
`~/.config/systemd/user/strata-v100.service.d/` only after you install the matching engine.
They contain no credentials or private paths. Keep the model paths, API key, and
conversation cache in the private configuration and `.strata-service.env`.

| File | What it selects |
| --- | --- |
| `90-cpu-prefill.conf` | the CPU-assisted prompt staging (`STRATA_PREFILL_CPU_SHARE=auto`) |
| `95-v0141-profile.conf` | the v0.1.41 profile: the CPU prefill share, the staged-chunk limit, the embedding-buffer accounting and the Volta master switch (`STRATA_SM70_TABLE=1`) |
| `96-prefill-performance.conf` | FP16 tensor-core experts, skipped unrouted experts, elastic KV growth, pooled checkpoint buffers, parallel stage checkpoint saves, exact wide dequantization, and up to 2048 expert-cache admissions per stage and input loan |
| `97-resident-routing.conf` | the residency-biased routing (`STRATA_ROUTE_RESIDENT=1`); it can change the model's answers |

Install them with:

```sh
install -d ~/.config/systemd/user/strata-v100.service.d
install -m 644 service/strata-v100.service.d/*.conf ~/.config/systemd/user/strata-v100.service.d/
systemctl --user daemon-reload
systemctl --user restart strata-v100.service
```

`STRATA_SM70_TABLE=1` is the engine's own default on a compute capability 7.0 card since the Volta defaults
table (`src/core/nv_defaults.cpp`); the line stays here so the profile is explicit. `STRATA_SM70_PERF=0`
turns the table off.

The measurements behind these switches, and the ones that did **not** help on this rig, are in
[the Volta report](../bench/results/2026-10-20-v100-sm70-speed-stack/README.md).

The checkpoint settings were measured with the deferred input-anchor engine.
The engine copies the input anchor to host memory after the first output window.
This reduces time to first output. It does not remove the copy from the request.
See the [incremental TTFT report](../../benchmarks/v100-incremental-ttft-2026-10-21.md)
for the A/B results and limits. The test did not install this revised profile in
the daily service.

The revised profile also selects `STRATA_DEQUANT_WIDE=1` and
`STRATA_PREFILL_PROMOTE=2048`. Wide dequantization preserves the narrow
kernel output bit for bit. Cache admission changes expert placement.
Resident-aware routing and CPU/GPU rounding can then change an answer.
Set `STRATA_PREFILL_PROMOTE=0` to keep the static cache refill policy.

Cache admission requires a full host arena. It is off with batch slots,
a RAM complement, a peer or helper tier, `STRATA_PREFILL_HELP`, or a
prefill head alias. The engine logs the reason for an explicit refusal.
Do not combine these paths or install this profile with an older engine.
The cache size, KV capacity, KV format, and model quantization stay unchanged.

See the [cache admission report](../../benchmarks/v100-prefill-cache-admission-2026-10-21.md)
for the four-arm comparison and full-request limits.
