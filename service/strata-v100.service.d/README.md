# The measured dual-V100 service profile

These four drop-ins define the fork's dual-V100 service profile. Install them under
`~/.config/systemd/user/strata-v100.service.d/` only after you install the matching engine.
They contain no credentials or private paths. Keep the model paths, API key, and
conversation cache in the private configuration and `.strata-service.env`.

| File | What it selects |
| --- | --- |
| `90-cpu-prefill.conf` | the CPU-assisted prompt staging (`STRATA_PREFILL_CPU_SHARE=auto`) |
| `95-v0141-profile.conf` | the v0.1.41 profile: the CPU prefill share, the staged-chunk limit, the embedding-buffer accounting and the Volta master switch (`STRATA_SM70_TABLE=1`) |
| `96-prefill-performance.conf` | FP16 tensor-core experts, skipped unrouted experts, elastic KV growth, pooled checkpoint buffers, and parallel stage checkpoint saves |
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
