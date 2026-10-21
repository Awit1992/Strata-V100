#!/usr/bin/env python3
"""The measurement script of this report.

It starts one Strata engine with the given configuration, speaks the engine's own
stdin/stdout serve protocol (src/program/generate.cpp, the --serve block) and reports the
engine's own numbers: the prompt read (prefill) and the decode time.

  python3 measure.py --label base --config <config.json> --targets 32768 \
      --decode-prompt 512 --decode-max-new 256 --decode-repeats 3

The configuration is a private file (it names the model's paths); the arms in this directory
are its output with the private paths replaced.  The engine's own timing lines
(STRATA_DECODE_TIMING=1) are the stable numbers: a single request's decode rate moves by
more than 10 % between identical runs, because the speculative drafter's window structure
changes with the timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

WORDS = (
    "attention cache layer kernel tensor token sequence prefill decode batch stride block "
    "memory bandwidth latency throughput shard expert router projection residual norm rotary "
    "context window prompt sample greedy verify draft acceptance queue pipeline stream slot "
    "index table hash map reduce scan prefix suffix segment scratch barrier launch grid warp "
    "the model reads a prompt in chunks and writes the key value state into the cache while "
    "each layer attends over the positions it has already seen and then projects the result "
    "a deterministic transform maps an integer through a modulo and returns the remainder"
).split()


class EngineIO:
    def __init__(self, proc):
        self.proc = proc
        self.q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            for line in self.proc.stdout:
                self.q.put(line.rstrip("\n"))
        except Exception:
            pass
        finally:
            self.q.put(None)

    def next_line(self, timeout):
        try:
            item = self.q.get(timeout=max(0.1, timeout))
        except queue.Empty:
            raise TimeoutError("no protocol line within the timeout")
        if item is None:
            raise EOFError("the engine closed its stdout")
        return item


def build_prompts(tokenizer_dir: Path, targets):
    from strata_tokenizer import Tokenizer
    vocab = json.loads((tokenizer_dir / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for tok, num in vocab.items():
        tokens[num] = tok
    merges = (tokenizer_dir / "merges.txt").read_text().splitlines()
    tok = Tokenizer(tokens, merges)
    need = max(targets) + 64
    ids = []
    block = tok.encode(" ".join(WORDS))
    while len(ids) < need:
        ids.extend(block)
    out = {}
    for n in targets:
        out[n] = ids[:n]
    return out


def run_request(io, request, timeout, on_line):
    io.proc.stdin.write(request + "\n")
    io.proc.stdin.flush()
    ev = {"pp": [], "tokens": [], "done": None, "err": None}
    deadline = time.monotonic() + timeout
    while True:
        line = io.next_line(deadline - time.monotonic())
        on_line(line)
        if line.startswith("T "):
            ev["tokens"].append(int(line[2:]))
        elif line.startswith("DONE "):
            ev["done"] = line.split()[1:]
            return ev
        elif line.startswith("ERR "):
            ev["err"] = line[4:]
            return ev
        elif line.startswith("PP "):
            ev["pp"].append(line.split()[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "strata-iq3_s-test.json"))
    ap.add_argument("--exe", default=None)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--targets", default="2048,8192,32768")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--prefill-max-new", type=int, default=1)
    ap.add_argument("--decode-prompt", type=int, default=512)
    ap.add_argument("--decode-max-new", type=int, default=128)
    ap.add_argument("--decode-repeats", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra-args", action="append", default=[])
    ap.add_argument("--replace", action="append", default=[],
                    help="replace a config flag's value, e.g. --replace prefill=16384")
    ap.add_argument("--startup-timeout-s", type=float, default=2400.0)
    ap.add_argument("--nsys", default=None, help="run the engine under nsys with this output prefix")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    exe = args.exe or cfg["exe"]
    targets = [int(t) for t in args.targets.split(",") if t.strip()]

    env = dict(os.environ)
    env.update(cfg.get("env") or {})
    env.update({"STRATA_ARENA_LOCK": "0", "STRATA_WATCHDOG_S": "300"})
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    engine_args = list(cfg["args"])
    if len(cfg.get("gpu") or []) > 1 and "--layer-split" not in engine_args:
        engine_args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
    # no conversation reuse: every request is a cold prompt
    engine_args += ["--prompt-cache", "0", "--prompt-cache-every", "0", "--prompt-cache-root", "0",
                    "--conversation-cache-mib", "0"]
    for a in args.extra_args:
        engine_args += a.split()
    for rep in args.replace:
        flag, _, val = rep.partition("=")
        if not flag.startswith("-"):
            flag = "--" + flag
        if flag not in engine_args:
            raise SystemExit(f"--replace: {flag} is not in the config args")
        i = engine_args.index(flag)
        if i + 1 < len(engine_args) and not engine_args[i + 1].startswith("-"):
            engine_args[i + 1] = val
        else:
            engine_args.insert(i + 1, val)

    argv = [exe, "--serve", *engine_args]
    if args.nsys:
        argv = ["nsys", "profile", "-o", args.nsys, "-f", "true", "-t", "cuda,nvtx,osrt",
                "--cuda-memory-usage=false", "--cpuctxsw=none", *argv]
    print("ARGV:", " ".join(argv), flush=True)
    print("ENV overrides:", {k: v for k, v in env.items() if k.startswith("STRATA_")}, flush=True)

    log_path = Path("/tmp") / f"strata_ab_{args.label}.log"
    logf = log_path.open("a", encoding="utf-8")
    t0 = time.monotonic()
    proc = subprocess.Popen(argv, cwd=cfg.get("cwd") or str(REPO), env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=logf, text=True, encoding="utf-8", bufsize=1)
    io = EngineIO(proc)
    info = {}
    try:
        while True:
            line = io.next_line(args.startup_timeout_s - (time.monotonic() - t0))
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    info[k] = v
            elif line.startswith("READY"):
                break
            elif line.startswith("ERR "):
                raise RuntimeError("engine startup error: " + line[4:])
        print(f"READY after {time.monotonic() - t0:.1f}s: {info}", flush=True)

        prompts = build_prompts(Path(cfg["tokenizer"]), targets + [args.decode_prompt])
        emb = Path("/tmp/geni-empty.sve")
        emb.write_bytes(b"")
        results = []
        for n in targets:
            ids = prompts[n]
            for rep in range(args.repeats):
                req = f"GENI {args.prefill_max_new} temperature=0 {emb} " + ",".join(map(str, ids))
                t1 = time.monotonic()
                ev = run_request(io, req, 3600.0, lambda l: None)
                wall = time.monotonic() - t1
                if ev["err"]:
                    raise RuntimeError("request error: " + ev["err"])
                d = ev["done"]
                gen, ptok, pms, dms = int(d[0]), int(d[1]), float(d[2]), float(d[3])
                row = {
                    "kind": "prefill", "target": n, "rep": rep, "prompt_tokens": ptok, "prompt_ms": pms,
                    "prefill_tok_s": (ptok / pms * 1000.0) if pms > 0 else 0.0,
                    "wall_s": round(wall, 3), "pp": ev["pp"],
                }
                results.append(row)
                print(json.dumps(row), flush=True)
        # decode arm: a short prompt, many generated tokens
        ids = prompts[args.decode_prompt]
        if args.warmup:
            req = f"GENI {args.decode_max_new} temperature=0 {emb} " + ",".join(map(str, ids))
            run_request(io, req, 3600.0, lambda l: None)
        for rep in range(args.decode_repeats):
            req = f"GENI {args.decode_max_new} temperature=0 {emb} " + ",".join(map(str, ids))
            t1 = time.monotonic()
            ev = run_request(io, req, 3600.0, lambda l: None)
            wall = time.monotonic() - t1
            if ev["err"]:
                raise RuntimeError("request error: " + ev["err"])
            d = ev["done"]
            gen, ptok, pms, dms = int(d[0]), int(d[1]), float(d[2]), float(d[3])
            acc = int(d[5]) if len(d) > 5 else 0
            off = int(d[14]) if len(d) > 14 else 0
            row = {
                "kind": "decode", "target": args.decode_prompt, "rep": rep, "prompt_tokens": ptok,
                "prompt_ms": pms, "generated": gen, "decode_ms": dms,
                "decode_tok_s": (gen / dms * 1000.0) if dms > 0 and gen > 1 else None,
                "draft_acc": acc, "offloaded": off, "wall_s": round(wall, 3),
                "tok_sha": hashlib.sha256(",".join(map(str, ev["tokens"])).encode()).hexdigest()[:16],
                "n_tok": len(ev["tokens"]),
            }
            results.append(row)
            print(json.dumps(row), flush=True)
        out = {"label": args.label, "env": {k: v for k, v in env.items() if k.startswith("STRATA_")},
               "argv": argv, "results": results}
        Path(args.out or f"/tmp/strata_ab_{args.label}.json").write_text(json.dumps(out, indent=1))
        # summary
        for n in targets:
            rows = [r for r in results if r["kind"] == "prefill" and r["target"] == n]
            if not rows:
                continue
            pf = statistics.median([r["prefill_tok_s"] for r in rows])
            print(f"SUMMARY {args.label} prefill target={n} prefill_tok_s={pf:.1f}", flush=True)
        dec = [r["decode_tok_s"] for r in results if r["kind"] == "decode" and r["decode_tok_s"]]
        if dec:
            tot_tok = sum(r["generated"] for r in results if r["kind"] == "decode")
            tot_ms = sum(r["decode_ms"] for r in results if r["kind"] == "decode")
            print(f"SUMMARY {args.label} decode prompt={args.decode_prompt} "
                  f"median={statistics.median(dec):.2f} mean={statistics.mean(dec):.2f} "
                  f"pooled={tot_tok / tot_ms * 1000.0:.2f} "
                  f"runs={[round(d, 2) for d in dec]}", flush=True)
    finally:
        try:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=20)
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except Exception:
                pass
        logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
