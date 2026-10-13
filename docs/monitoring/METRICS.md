# Asynchronous Prometheus metrics

## Read the endpoint

Send `GET /metrics` with `Accept: text/plain`, or use `/metrics?format=prometheus`.
The response uses Prometheus text format 0.0.4. The content type is
`text/plain; version=0.0.4; charset=utf-8`.
Use the same bearer API key as the inference API, if a key is set.
Use `docs/monitoring/prometheus.yml` for the scrape configuration.

A request without a text format selection keeps the existing JSON response.
The web Monitor uses this response. `/metrics/prometheus` keeps the existing
llama.cpp-compatible response. It does not use the new snapshot sampler.

## Asynchronous collection

One background thread collects and encodes the Prometheus snapshot.
It waits one second between collections. The existing HTTP server sends the
cached UTF-8 bytes from a separate request thread. Scrapes do not call the
engine, read GPU devices, poll log files, or rebuild metric families.
Concurrent scrapes share one immutable response. A slow reader does not hold
the service status lock. Collection takes short status locks to copy state;
this design does not promise zero CPU cost or zero lock contention.
The Python interpreter and HTTP request threads still share host resources.

Before the first successful collection, the text endpoint returns HTTP 503.
If collection fails, the endpoint keeps the last successful response and logs
the error. Check `strata:metrics_snapshot_timestamp_seconds` to detect stale
values. `strata:metrics_collection_errors_total` reports failed collections
on the next successful snapshot. Server close signals the sampler to stop.
A collection already in progress can finish after that signal.

The snapshot can be more than one second old if collection takes time.
A request that has just finished can appear in the next snapshot, not in the
current response. Use a normal scrape interval, such as 15 seconds.
Do not use this endpoint as an inference admission or control interface.

## Data coverage

The exporter keeps the existing `vllm:*` and `strata:*` families and adds all
numeric and Boolean fields from the following snapshot sections. Fields that
are not available are omitted. String values, prompts, output text, request
identifiers, file paths, and history samples are not exported.

| Source | Families and data |
| --- | --- |
| `engine` | `strata:engine_*`: context size, image support, and numeric engine configuration and capacity facts |
| `live` | `strata:live_*`: state, token counts, elapsed time, generation and prefill rates, queue, admission, and slot counts |
| `totals` | `strata:totals_*_total`: finished requests, prompt and output tokens, prefix reuse, draft tokens, and accumulated timing |
| Latest finished request | `strata:last_*`: numeric request timing, token counts, throughput, and expert-cache hit ratio |
| `conversation_cache` | `strata:conversation_cache_*`: enable state, budget, slots, parked state, reuse, and recent request statistics |
| `l3` | `strata:l3_*`: disk-cache enable state, budget, records, bytes, hits, misses, evictions, corruption, recovery, and numeric event totals |
| Hardware | GPU utilization, memory, temperature, power, power limit, energy, PCIe generation, width, and transfer rates; CPU utilization, RAM, host disk rates, and engine disk rates |
| Static hardware | `strata:hardware_*`: GPU count, physical cores, logical threads, and availability of psutil |
| Slots | `strata:slot_*`: numeric slot state, with a bounded `slot` label |
| Replicas | `strata:replica_*`: alive state, active requests, routed requests, and batch capacity, with a bounded `replica` label |
| Exporter | Last successful collection timestamp and failed collection count |

Model metrics use `model_name`. GPU metrics use the configured `gpu` index.
State labels use a fixed set. No per-request label is added.

New numeric field names keep the source key, with these unit conversions:
`_ms` becomes `_seconds`; `_mib` and `_gib` become `_bytes`; hardware `_mb`
rate fields become `_bytes_per_second`; `_wh` becomes `_joules`.
The source hardware rates use MiB/s, so conversion uses 1,048,576 bytes.
New CPU and GPU utilization ratios use 0 to 1. Existing utilization percentage
families remain available for existing dashboards. Existing GPU power values
use watts. PCIe generation and width are dimensionless.

Service totals are counters. Engine, latest-request, slot, replica, hardware,
and cache snapshots are gauges because they can fall or reset independently
of the HTTP service. Recent cache request counts cover the retained request
window, not service lifetime. GPU energy can reset with the sampler.
`strata:start_time_seconds` identifies the request-accounting start time.

## Compatibility limits

The existing vLLM names are a compatibility mapping, not an identical engine.
KV usage estimates the active request's token share of context, not physical
KV allocation. Inter-token latency uses the mean gap from each request's
decode time; it is not a measurement of each individual token gap.
Finished-request counters include the requests recorded by the service;
they are not a separate HTTP success/error classification.
No new measurement is invented for unavailable engine or hardware data.
The endpoint sends Prometheus text even if the Accept header mentions
OpenMetrics. It does not claim OpenMetrics format or add exemplars.

## Research references

- [Prometheus exposition formats](https://prometheus.io/docs/instrumenting/exposition_formats/): text syntax, label escaping, terminal newline, and content type.
- [Prometheus exporter guidance](https://prometheus.io/docs/instrumenting/writing_exporters/): base units, counter semantics, cardinality, and collection cost.

## Deployment

This change does not restart or reconfigure a running service.
The running process keeps its current code until the operator deploys the PR
and restarts the service at an approved time.
