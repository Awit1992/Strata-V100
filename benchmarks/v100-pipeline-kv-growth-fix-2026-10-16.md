# Pipeline decode: grow the mapped KV cache before the next windows

## Observed failure

The installed two-V100 service failed twice during long generation. The service journal and engine log show these failures:

| Request prompt tokens | Output tokens reported before failure | Mapped KV cells on each GPU | Error |
| ---: | ---: | ---: | --- |
| 55,086 | 2,256 | 57,344 | `verify: layer 3 never rang (an illegal memory access was encountered)`; engine exit -11 |
| 56,398 | 947 | 57,344 | `mtp: staging the window's residual rows failed`; engine exit 1 |

The first request's reported prompt-plus-output count is 57,342. Speculative windows and the drafter can access positions beyond the committed output frontier. The second count is 57,345. Both failures are at the mapped boundary. The CUDA error is primary evidence; the server's generic RAM advice after exit -11 does not establish a RAM out-of-memory kill.

The logs already demonstrate the failure. The experiment did not replay private prompts or deliberately crash the old production engine again.

## Root cause and change

With `STRATA_KV_GROW=1`, the engine reserves a large virtual address range but maps physical memory only for the cells in use. At request start, it ensures room for the prompt plus a small margin. The serial decode loop checks again before each window. The pipelined decode loop did not make that check. A long response could therefore make a verifier or the MTP drafter access unmapped memory.

The repair adds a growth boundary to pipelined decode:

1. After a verified verdict, calculate the cells needed for two verifier windows and the drafter's lookahead. Bound the target by the configured maximum context.
2. If growth is needed, do not promote the speculative next window and do not start a new drafter chain.
3. Serve any speculative work to completion and use the existing rollback path to restore the accepted state.
4. Drain GPU work and settle adaptive expert copies before moving expert slots or changing their memory mappings.
5. Grow the KV mapping on all active stages. Resume the drafter from the accepted window's residual rows, then continue normal pipelining.

The change retains pipelining, elastic KV, the context capacity, and the model settings. It does not allocate memory for the entire requested output up front. The extra drain occurs only when the next bounded lookahead needs more physical cells. The diagnostic pipeline trace records this boundary as `KG`.

## Executed runtime checks

The checks use the real model and both installed V100-PCIE-16GB GPUs: GPU 0 on PCIe x4, GPU 1 on x16, split at layer 16. The configured context remains 524,288 cells with int8 KV, spec 8, automatic CPU prefill sharing, resident routing, the existing Volta settings, and elastic KV enabled.

Prompts contain deterministic synthetic text, not user conversations. All prompt-plus-output sizes are below 32,768 tokens. An out-of-vocabulary EOS stop ID, `2147483647`, prevents an early stop so the boundary is exercised. This changes only the diagnostic stop condition; it is not installed in the service. Conversation caches and profile writes are disabled for these isolated checks.

| Runtime arm | Prompt tokens | Generated tokens | Result |
| --- | ---: | ---: | --- |
| Normal pipeline | 16,000 | 1,200 | Complete; both GPUs grow from 16,384 to 24,576 cells during decode |
| Same engine, next request | 24,000 | 1,500 | Complete; both GPUs grow to 32,768 cells during decode |
| Same engine, long generation | 2,048 | 15,000 | Complete; mapping remains at 32,768 cells |
| Same engine, follow-up | 256 | 32 | Complete |
| Forced speculative misses | 16,000 | 1,200 | Complete; both GPUs grow to 24,576 cells |
| Forced-miss follow-up | 256 | 32 | Complete |
| Requested asynchronous adaptation | 16,000 | 1,200 | Complete; runtime uses its documented blocking fallback; both GPUs grow to 24,576 cells |
| Same fallback engine, follow-up | 256 | 32 | Complete |

All eight requests return their full requested output count. All three engine processes exit cleanly after `QUIT`. No run logs an illegal memory access, missing layer doorbell, or failed residual staging.

The normal trace contains two growth barriers, 816 speculative promotions, and 876 rollbacks. At its second growth boundary, a rollback `U` precedes `KG`, and the fresh drafter chain follows it. The forced-miss trace contains one growth barrier and 54 rollbacks. The third trace contains one growth barrier and 129 rollbacks.

The asynchronous option is not claimed as active coverage: the engine explicitly declines it because this service does not use resident-RAM mode. The current eligibility check also excludes elastic KV in resident-RAM mode. The installed service keeps its original blocking adaptation setting.

The first smoke driver expected four normal growth events and reported a bookkeeping assertion after all four requests and a clean engine exit. Inspection showed two actual growth events: the later small prompts retain the 32K mapping. The final acceptance checks use the observed mapping lifetime and verify both actual growth barriers and the subsequent requests. No inference failure occurred in that run.

## Reproduce the boundary with the existing harness

Stop the production service before using its GPUs. Set `CONFIG` to the private pipeline configuration and `ENGINE` to the repaired binary. Use the installed service environment, including `STRATA_KV_GROW=1`, the Volta settings, automatic CPU sharing, and resident routing. Do not publish the private configuration or raw logs.

The following integration command uses an allowed harness prompt size. Its 16,384-token prompt plus 9,000 generated tokens stays below 32K and crosses the initial rounded mapping during generation:

```sh
python bench/run_v100_prefill.py --config "$CONFIG" --exe "$ENGINE" \
  --arm-label pipeline-kv-boundary --targets 16384 --repeats 2 \
  --max-new 9000 --pipeline-windows 2 --no-settings-strict \
  --xargs='--eos-ids;2147483647' \
  --out /tmp/strata-pipeline-kv-boundary
```

For speculative rollback coverage, add `--env STRATA_PIPELINE_DEBUG=1`, `--env STRATA_PIPELINE_FORCE_MISS=1`, and `--env STRATA_PIPELINE_THETA=0`. A private `STRATA_PIPELINE_TRACE` path records `KG`, `U`, and window events. This command is a reproduction recipe; the executed cases are the exact sizes in the table above.

The historical harness profile expects a 700-MiB vision reserve. This service uses 750 MiB; `--no-settings-strict` permits that known difference without changing the reserve. These are model integration checks, not a model-free CI test or a new speed benchmark. They do not establish output parity or quality for every conversation. The earlier pipeline speed measurements capped outputs at 16 or 256 tokens and did not validate growth during a long decode.

## Installed runtime

The repaired engine is installed at the service's existing executable path. The service keeps its persistent pipeline configuration, `--pipeline-windows 2`, split 16, int8 KV, image support, and 524,288-cell capacity. The old engine and local logs are preserved for rollback and diagnosis.

After installation, health reports the model loaded. An authenticated chat request returns HTTP 200, `READY`, and finish reason `stop`.

The tested and installed engine SHA256 is `0755ca0a9b4d173d505b6f2de31218dee72c4d3bdec4a9dfa73db2e618d104db`. The changed `src/program/generate.cpp` SHA256 is `9e0286c0c8afd9d62f75d3db0d1669780615f8165c4a653e777b5cdb5d9c3e1d`. The source parent is the merged pipeline profile, `62c0d27b`.
