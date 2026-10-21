# The measured dual-V100 service profile

These are the four drop-ins the fork's daily service runs with, as they are installed under
`~/.config/systemd/user/strata-v100.service.d/`. They hold no credentials and no private paths: the
model paths, the API key and the conversation cache live in the private configuration file
(`strata-iq3_s-pipeline.json`) and in `.strata-service.env`.

| File | What it selects |
| --- | --- |
| `90-cpu-prefill.conf` | the CPU-assisted prompt staging (`STRATA_PREFILL_CPU_SHARE=auto`) |
| `95-v0141-profile.conf` | the v0.1.41 profile: the CPU prefill share, the staged-chunk limit, the embedding-buffer accounting and the Volta master switch (`STRATA_SM70_TABLE=1`) |
| `96-prefill-performance.conf` | the prompt path's performance switches: the FP16 tensor-core experts (`STRATA_PF_WMMA=1`), the skipped unrouted experts and the elastic KV growth |
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
