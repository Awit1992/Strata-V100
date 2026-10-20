#!/usr/bin/env python3
"""DFlash2 benchmark runner for the Strata V100 fork.

Two suites measure a service that serves the native Qwen3.8-Flash-Next target with the
DFlash2 draft model (docs/DFLASH2.md):

  short       Three chat workloads x three repeats. This is the published comparison
              table's workload, so a DFlash2 arm and a baseline arm pair request for
              request.
  acceptance  The DFlash2 acceptance-length protocol over the model card's 128-prompt
              cohort per dataset (bench/prepare_dflash2_prompts.py). Each request's
              engine-generated tokens divided by its verification windows, averaged per
              dataset. Each dataset's warmup record runs first with a 64-token budget;
              its output and metrics are discarded.

PREREQUISITES
  * Send one request at a time. The DFlash2 draft path has no batch slots (docs/DFLASH2.md),
    so a second concurrent request invalidates every measurement.
  * Start the engine with STRATA_DECODE_TIMING=1. Each request then writes one
    "strata decode timing: N windows, ..." line to the engine's stderr. The acceptance
    suite reads those lines from --engine-log and divides the request's engine-generated
    tokens by its verification windows.
  * For the acceptance suite, set answer_room, reasoning_close_retry,
    reasoning_loop_recovery, and tool_call_recovery to false in the service configuration.
    Set repeat_stop_tokens to 0. These settings prevent server-side changes to the output.
    A continuation starts a second engine generation. The runner rejects such a request.
  * Serve the target with --spec 4 --dflash2-block 4 and the other settings in
    docs/DFLASH2.md for a matched comparison.

Every row records only the engine's own numeric metrics (an allowlist). The runner never
writes response text, request ids, timestamps, paths, or credentials. Multi-turn assistant
replies stay in memory for the next turn and are never persisted.

Usage:
  python bench/prepare_dflash2_prompts.py --out PROMPTS_JSON
  python bench/run_dflash2_bench.py --suite short --label dflash2 \\
      --config SERVICE_JSON --out SHORT_ROWS_JSON
  python bench/run_dflash2_bench.py --suite acceptance --label dflash2 \\
      --config SERVICE_JSON --prompts PROMPTS_JSON --engine-log ENGINE_LOG \\
      --out ACCEPTANCE_ROWS_JSON
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path

import requests

# The repository's live-service default (bench/run_v100_bench.py). A DFlash2 test service
# runs its own port; pass it with --base.
BASE = "http://127.0.0.1:8088"

# short suite: the published table's workloads, three ordered repeats.
SHORT_REPEATS = 3
SHORT_SEED = 4107
SHORT_MAX_TOKENS = 256
SHORT_TEMPERATURE = 0.0
SHORT_THINKING = False
SHORT_TIMEOUT = 900

WORKLOADS = (
    ("code",
     "Write a complete Python implementation of an LRU cache with get and put methods, "
     "using a dictionary and a doubly linked list. Explain the time complexity and edge cases."),
    ("math",
     "Solve this problem step by step. A train travels 180 km at 60 km/h, then 240 km at 80 km/h, "
     "and stops for 30 minutes. Find its average speed for the entire trip and explain why the "
     "arithmetic mean of its speeds is wrong."),
    ("prose",
     "Explain how a database write-ahead log and checkpoints work together after a power failure. "
     "Include an example with two transactions and explain the order of durable writes."),
)

# acceptance suite: the DFlash2 model card's protocol (docs/DFLASH2.md).
ACCEPT_LIMIT = 128
WARMUP_MAX_TOKENS = 64
ACCEPT_SEED_BASE = 4200
ACCEPT_MAX_TOKENS = 4096
ACCEPT_TIMEOUT = 1800
ACCEPT_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
ACCEPT_TEMPLATE_KWARGS = {"enable_thinking": True, "reasoning_effort": "xhigh",
                          "reasoning_strength": "xhigh"}

# One engine generation writes exactly one of these lines (STRATA_DECODE_TIMING=1).
DECODE_TIMING = re.compile(
    r"strata decode timing: (\d+) windows, avg T ([0-9.]+), ([0-9.]+) tokens/window")

# The engine /metrics history-row fields recorded in the raw results. Numeric only, and an
# allowlist: no request ids, timestamps, or free text ever enter a row.
METRIC_FIELDS = (
    "prompt_tokens", "prompt_total", "prompt_read", "reused",
    "output_tokens", "engine_generated",
    "prompt_ms", "decode_ms", "decode_tok_s",
    "first_engine_token_s", "ttft_ms", "duration_s",
    "hit_rate", "pcie_share",
    "drafts_offered", "drafts_accepted",
    "reasoning_recoveries", "reasoning_repeat_coverage",
    "ram_blobs", "file_blobs", "file_mb",
)


def positive_int(value: str) -> int:
    """argparse type: a positive integer (--repeats, --limit)."""
    v = int(value)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return v


def api_key(args: argparse.Namespace) -> str:
    """The bearer key: --config's api_key only, or the STRATA_API_KEY environment variable."""
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        key = cfg.get("api_key") if isinstance(cfg, dict) else None
        if not key:
            raise SystemExit("--config: no api_key")
        return key
    key = os.environ.get("STRATA_API_KEY")
    if not key:
        raise SystemExit("pass --config PRIVATE_JSON (api_key) or set STRATA_API_KEY")
    return key


def model_id(session: requests.Session, base_url: str) -> str:
    r = session.get(f"{base_url}/v1/models", timeout=30)
    r.raise_for_status()
    data = r.json().get("data") or []
    if not data:
        raise SystemExit("/v1/models: no model")
    return data[0]["id"]


def latest_after(session: requests.Session, since: float, base_url: str) -> dict | None:
    """The newest /metrics request row that started at or after `since` (the request's own)."""
    r = session.get(f"{base_url}/metrics", timeout=30)
    r.raise_for_status()
    cands = [e for e in r.json().get("requests", []) if e.get("time", 0) >= since]
    return max(cands, key=lambda e: e["time"]) if cands else None


def scalar_metrics(entry: dict) -> dict:
    """The allowlisted numeric engine metrics: no ids, timestamps, or free text."""
    return {k: entry[k] for k in METRIC_FIELDS
            if isinstance(entry.get(k), (int, float)) and not isinstance(entry.get(k), bool)}


def scalar_usage(usage: object) -> dict:
    """Record token counts, not arbitrary server extensions."""
    if not isinstance(usage, dict):
        return {}
    return {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            if isinstance(usage.get(k), int) and not isinstance(usage.get(k), bool)}


def completion(answer: object) -> dict:
    """The response's first choice, or a clear failure on a malformed response."""
    if not isinstance(answer, dict):
        raise SystemExit("invalid response: not a JSON object")
    choices = answer.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SystemExit(f"invalid response: no choices ({sorted(answer)})")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") not in ("stop", "length"):
        raise SystemExit("benchmark response did not finish normally")
    return choice


def write_rows(out: Path, rows: list) -> None:
    """Rewrite the raw-rows file. Called after every row, so a later failure keeps earlier rows."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def run_short(session: requests.Session, base_url: str, model: str, label: str,
              out: Path, repeats: int) -> list:
    rows = []
    for rep in range(repeats):
        for name, prompt in WORKLOADS:
            # Unique prefix: identical for a given (repeat, workload) in every arm, and never
            # shared with an earlier request, so the engine's cache carries no prefix over.
            content = f"[paired-evaluation-{rep}-{name}]\n" + prompt
            body = {"model": model, "messages": [{"role": "user", "content": content}],
                    "temperature": SHORT_TEMPERATURE, "max_tokens": SHORT_MAX_TOKENS,
                    "seed": SHORT_SEED, "chat_template_kwargs": {"enable_thinking": SHORT_THINKING}}
            start = time.time()
            r = session.post(f"{base_url}/v1/chat/completions", json=body, timeout=SHORT_TIMEOUT)
            wall = time.time() - start
            r.raise_for_status()
            answer = r.json()
            choice = completion(answer)
            entry = latest_after(session, start, base_url)
            if entry is None:
                raise SystemExit(f"{name}#{rep}: no /metrics entry for the request")
            if entry.get("reused", 0):
                raise SystemExit(f"{name}#{rep}: cached prompt tokens invalidate the short suite; "
                                 "restart the engine before each arm")
            row = {"label": label, "workload": name, "repeat": rep, "wall_s": wall,
                   "usage": scalar_usage(answer.get("usage")), "metrics": scalar_metrics(entry),
                   "finish_reason": choice["finish_reason"]}
            rows.append(row)
            write_rows(out, rows)
            m = row["metrics"]
            print(f"{name}#{rep} wall {wall:.1f}s finish {row['finish_reason']} "
                  f"decode {m.get('decode_ms')} ms {m.get('decode_tok_s')} tok/s", flush=True)
    for name, _ in WORKLOADS:
        ms = [r["metrics"]["decode_ms"] for r in rows
              if r["workload"] == name and r["metrics"].get("decode_ms") is not None]
        tps = [r["metrics"]["decode_tok_s"] for r in rows
               if r["workload"] == name and r["metrics"].get("decode_tok_s") is not None]
        if ms:
            print(f"{name}: {len(ms)} requests, median decode {statistics.median(ms):.1f} ms "
                  f"({statistics.median(tps):.1f} tok/s)", flush=True)
    return rows


def accept_body(model: str, messages: list, row_number: int, max_tokens: int) -> dict:
    """The acceptance request body: the model card's sampling, seeded per record row."""
    return {"model": model, "messages": messages, "max_tokens": max_tokens,
            "seed": ACCEPT_SEED_BASE + row_number,
            "chat_template_kwargs": ACCEPT_TEMPLATE_KWARGS,
            "reasoning_effort": "xhigh", **ACCEPT_SAMPLING}


def run_acceptance(session: requests.Session, base_url: str, model: str, label: str, out: Path,
                   prompts_path: Path, log_path: Path, limit: int) -> list:
    prompts = json.loads(Path(prompts_path).read_text(encoding="utf-8"))
    for item in prompts:
        if not isinstance(item, dict) or not isinstance(item.get("turns"), list) \
                or not {"dataset", "row"} <= item.keys():
            raise SystemExit("--prompts: every record needs dataset, row, and a turns list")
    log = Path(log_path)
    if not log.is_file():
        raise SystemExit(f"engine log not found: {log_path}")

    rows = []
    for item in prompts:
        # Prepared cohorts put one warmup immediately before each dataset's measured rows.
        # Its output and metrics do not enter the result.
        if item.get("warmup"):
            turns = item["turns"]
            if not turns:
                raise SystemExit(f"{item['dataset']}:{item['row']}: warmup record has no turns")
            body = accept_body(model, [{"role": "user", "content": turns[0]}], item["row"],
                               min(WARMUP_MAX_TOKENS, ACCEPT_MAX_TOKENS))
            r = session.post(f"{base_url}/v1/chat/completions", json=body, timeout=ACCEPT_TIMEOUT)
            r.raise_for_status()
            completion(r.json())
            print(f"{item['dataset']}:{item['row']} warmup done", flush=True)
            continue
        if item["row"] >= limit:
            continue
        dataset, row_number, turns = item["dataset"], item["row"], item["turns"]
        messages = []
        for turn, prompt in enumerate(turns):
            messages.append({"role": "user", "content": prompt})
            before = log.stat().st_size
            start = time.time()
            body = accept_body(model, messages, row_number, ACCEPT_MAX_TOKENS)
            r = session.post(f"{base_url}/v1/chat/completions", json=body, timeout=ACCEPT_TIMEOUT)
            wall = time.time() - start
            r.raise_for_status()
            choice = completion(r.json())
            reply = choice.get("message")
            if not isinstance(reply, dict):
                raise SystemExit(f"{dataset}:{row_number}:{turn}: invalid response: no message")
            # The assistant reply stays in memory for the next turn only; it is never written out.
            messages.append(reply)
            entry = latest_after(session, start, base_url)
            if entry is None:
                raise SystemExit(f"{dataset}:{row_number}:{turn}: no /metrics entry for the request")
            generated = entry.get("engine_generated")
            if not isinstance(generated, int) or isinstance(generated, bool) or generated <= 0:
                raise SystemExit(f"{dataset}:{row_number}:{turn}: engine_generated={generated!r}")
            with log.open("rb") as f:
                f.seek(before)
                text = f.read().decode(errors="replace")
            steps = DECODE_TIMING.findall(text)
            if len(steps) != 1:
                raise SystemExit(f"{dataset}:{row_number}:{turn}: expected exactly one engine "
                                 f"generation (one decode-timing line), found {len(steps)}")
            windows = int(steps[0][0])
            if windows <= 0:
                raise SystemExit(f"{dataset}:{row_number}:{turn}: verification windows={windows}")
            row = {"label": label, "dataset": dataset, "row": row_number, "turn": turn,
                   "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                   "completion_tokens": generated, "verification_steps": windows,
                   "acceptance_length": generated / windows, "wall_s": wall,
                   "metrics": scalar_metrics(entry), "finish_reason": choice["finish_reason"]}
            source_row = item.get("source_row")
            if isinstance(source_row, int) and not isinstance(source_row, bool):
                row["source_row"] = source_row
            rows.append(row)
            write_rows(out, rows)
            print(f"{dataset}:{row_number}:{turn} acceptance {row['acceptance_length']:.3f} "
                  f"({generated} tokens / {windows} windows) wall {wall:.1f}s "
                  f"finish {row['finish_reason']}", flush=True)
    for dataset in sorted({r["dataset"] for r in rows}):
        values = [r["acceptance_length"] for r in rows if r["dataset"] == dataset]
        print(f"{dataset}: {len(values)} requests, mean acceptance "
              f"{statistics.mean(values):.3f}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=("short", "acceptance"), default="short",
                    help="short: the three-workload comparison table; acceptance: the "
                         "acceptance-length protocol (default: short)")
    ap.add_argument("--base", default=BASE, help=f"server base URL (default: {BASE})")
    ap.add_argument("--config", type=Path,
                    help="private JSON holding api_key (alternative: STRATA_API_KEY)")
    ap.add_argument("--label", required=True, help="arm label recorded in every row")
    ap.add_argument("--out", type=Path, required=True,
                    help="raw-rows JSON file (rewritten after every row)")
    ap.add_argument("--repeats", type=positive_int, default=SHORT_REPEATS,
                    help=f"short suite: repeats per workload (default: {SHORT_REPEATS}; "
                         "use 1 for a smoke run)")
    ap.add_argument("--prompts", type=Path,
                    help="acceptance suite: JSON records {dataset, row, turns}")
    ap.add_argument("--engine-log", type=Path,
                    help="acceptance suite: the engine's stderr log (STRATA_DECODE_TIMING=1)")
    ap.add_argument("--limit", type=positive_int, default=ACCEPT_LIMIT,
                    help=f"acceptance suite: highest row index to run (default: {ACCEPT_LIMIT})")
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key(args)}",
                            "Content-Type": "application/json"})
    base_url = args.base.rstrip("/")
    model = model_id(session, base_url)
    print(f"model {model}", flush=True)

    if args.suite == "short":
        run_short(session, base_url, model, args.label, args.out, args.repeats)
    else:
        if not args.prompts or not args.engine_log:
            raise SystemExit("--suite acceptance needs --prompts and --engine-log")
        run_acceptance(session, base_url, model, args.label, args.out,
                       args.prompts, args.engine_log, args.limit)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
