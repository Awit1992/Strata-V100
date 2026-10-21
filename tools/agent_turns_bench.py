"""A/B prefill latency, new tokens and reuse on coding-agent turns: per-turn numbers from the server's own clock.

    python tools/agent_turns_bench.py strata-iq3_s-arena.json base: cand:STRATA_HC_Q8=2 [--rounds 2]
    python tools/agent_turns_bench.py strata-iq3_s-arena.json base: cand:STRATA_X=1 --workload agentic
    python tools/agent_turns_bench.py strata-iq3_s-arena.json A: B: --engine mock   # a smoke run, no GPU

Two workloads.  Both send the same requests to every arm and neither downloads anything:

  repo (the default)  five simulated coding-agent sessions over this repository's own files (README, serve/,
      setup.py, the expert cache, the hyper-connection kernels, the tokenizer): a system prompt, four tools, a
      task (three in English, two in French), then tool calls whose results are the real files, at most 2,000
      lines per read as agents' read tools return them.  Every point right after a tool result is one turn the
      model continues - 23 turns from ~2.5K to ~100K tokens, consecutive turns sharing their prefix as in a
      real session.

  agentic  the short-turn shape of an agent at work: one long shared prefix (the system prompt, the task and
      one large tool result, --prefix-tokens) and then --turns short turns, each adding one assistant line and
      one tool result of about 64, 128, 256 or 512 tokens (--additions).  A fixed cost per request shows up
      here as milliseconds per new token: the additions are small and the conversation cache holds the rest of
      the prompt.

Every turn's numbers come from the server's own `timings` (serve/server.py request_timings): `prompt_n` the
new tokens read, `cache_n` the tokens the conversation cache already held, `prompt_ms` the time to read them,
`prompt_per_token_ms` and `prompt_per_second` the prefill rate, `predicted_*` the decode.  The report gives,
per turn and per arm, the tokens the turn added to its conversation, the new tokens the engine read, the reuse
share, the time to first generated output (`ttft ms`: this tool's clock from just before the request is sent
to the first output the client sees - content, reasoning, or a tool call's function name or arguments; the
role-only opening delta, keep-alive comments and a call's id/type/index announcement (even with an empty
function object) are not output, and a turn with none of them reports n/a rather than the whole call's time;
the client's and the parser's buffering are included, so it is not the engine's raw first-token time), the
prefill latency and its milliseconds per new token, the decode rate, and the whole HTTP call's milliseconds
(`req ms`, measured by this tool; parking a conversation and restoring one are inside it, and the engine's
own time is not).  It then gives the per-turn factor of every other arm against the first arm for TTFT,
prefill latency and decode (>1 is better for all three).

Every request is streamed (`"stream": true`) and the stream is rebuilt into the same response object a
non-streamed call would return: joined content and reasoning, tool calls merged by index, the final
`finish_reason`, `usage` and the engine's `timings`.  A non-streamed answer is never accepted in its place:
a body that is not SSE, an error event in the stream, and a stream that ends without `[DONE]` all raise with
the server's own words.

A cold turn (the first turn of a session, `agentic.prefix` in the agentic workload: the whole prompt is new)
is listed with the others but kept out of the medians of the short turns: one long cold read is not the
per-turn cost of an agent at work.

`--additions` and `--prefix-tokens` are TARGETS, not measurements.  The text is common English words and one
word is about one token, and a turn also adds the assistant's line and the tool-call envelope (measured: about
55 tokens with these four tools and this chat template).  The measured sizes are the report's own numbers:
`added` is the whole turn's new tokens, and `new` and `reuse` are the engine's `prompt_n` and `cache_n`.

Arms are given as ab_engine.py takes them (NAME:key=value,...; `exe` replaces the engine, `xargs` appends
';'-separated engine arguments, any other key goes to the engine's environment).  The rounds mirror the arm
order (A B, B A, ...).  Each arm gets its own server and its own private directory for the engine log, the
conversation-cache disk store and the expert profile, one directory per arm and per round: the disk store
survives a restart and would otherwise restore another arm's (or an earlier round's) parked conversation,
which is the state an A/B must not share.  Every round starts a fresh server process, so a repeated prompt
can reuse a conversation only inside the round that created it.  The generated config
binds 127.0.0.1 and carries no API key, so the operator's secret is never copied out of the config file.  The
server and the engine it starts are stopped after every arm (SIGTERM, then SIGKILL of the arm's own process
group), and the private directory is removed at the end unless --workdir names one to keep.

Every arm inherits this process's environment and adds its own key=value pairs to it.  Export the settings the
live service uses (its EnvironmentFile, or the STRATA_* variables) before a measured run: an arm that misses
one of them measures another configuration.

Sampling is the agent's (temperature 1, top-p 0.95, top-k 20, a fixed seed), 256 new tokens per turn.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_engine import wait_ready  # noqa: E402

SYSTEM = ("You are an autonomous coding agent working in the user's repository. You can read files, search the code, "
          "edit files and run commands with the tools below. Work step by step: read the code you need before changing "
          "it, make small and correct edits, keep the existing style, and explain briefly what you did. Never invent "
          "file contents: read them. When the task is done, give a short summary of the changes and how to test them.")

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file of the repository (UTF-8 text).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "path from the repository root"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Search the repository for a regular expression; returns matching lines.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace an exact string in a file with a new one.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                                                        "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {
        "name": "run", "description": "Run a shell command in the repository and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]

# (name, task, [(tool, argument, what the assistant says before the call)])
SESSIONS = [
    ("server-option",
     "Add a `--max-queue N` option to the HTTP server that rejects new chat requests with HTTP 503 when N requests are "
     "already waiting, document it in the README's settings section, and add a test.",
     [("read_file", "README.md", "I'll start with the README to see how settings are documented."),
      ("grep", "add_argument", "Now let me find where the server parses its options."),
      ("read_file", "serve/server.py", "The options are in serve/server.py; reading it."),
      ("read_file", "serve/frontend.py", "Requests go through the frontend; let me read it too."),
      ("read_file", "docs/DETAILS.md", "Checking DETAILS.md for the existing API documentation style.")]),
    ("expert-cache",
     "Explain precisely how the expert cache decides which experts stay in VRAM during decode, then propose and "
     "implement a small improvement to its eviction policy with a measurement plan.",
     [("read_file", "docs/HOW_IT_WORKS.md", "Let me read the design overview first."),
      ("read_file", "include/strata/core/expert_cache.hpp", "Now the cache's interface."),
      ("read_file", "src/core/expert_cache.cpp", "And its implementation."),
      ("grep", "adapt_every", "Searching for the adaptive swap logic."),
      ("read_file", "src/core/expert_source.cpp", "The dispatch that consults the cache is in expert_source.cpp.")]),
    ("installer-fr",
     "Dans l'installateur, la détection de la RAM disponible se trompe parfois sous Windows quand un autre programme "
     "réserve beaucoup de mémoire. Trouve où elle est faite, corrige-la, ajoute un test et explique la correction en "
     "français.",
     [("read_file", "setup.py", "Je commence par lire l'installateur."),
      ("grep", "def .*ram", "Je cherche les fonctions qui mesurent la RAM."),
      ("read_file", "docs/INSTALL.md", "Je vérifie ce que la documentation promet sur la RAM."),
      ("read_file", "docs/MODELS.md", "Le choix du modèle dépend de la RAM ; je lis cette page aussi.")]),
    ("cuda-kernel",
     "The hyper-connection read kernels are latency bound on small batches. Read them, explain where the time goes, "
     "and write a faster variant for 1-4 tokens that keeps the same results bit for bit.",
     [("read_file", "include/strata/kernels/fused_gr.hpp", "First the kernel interface."),
      ("read_file", "src/kernels/cuda/fused_gr.cu", "Now the kernels themselves."),
      ("grep", "fused_gr_read_multi", "Where are they called from?"),
      ("read_file", "src/core/verify.cpp", "The verify window calls them; reading it.")]),
    ("tokenizer-fr",
     "Écris une suite de tests pytest pour le tokenizer Python (encodage, décodage, tokens spéciaux, texte multilingue, "
     "cas limites), puis corrige les bugs que les tests révèlent. Réponds en français.",
     [("read_file", "tools/strata_tokenizer.py", "Je lis d'abord le tokenizer."),
      ("grep", "strata_tokenizer", "Je cherche qui l'utilise."),
      ("read_file", "serve/frontend.py", "Le frontend l'appelle ; je le lis pour voir les usages réels."),
      ("read_file", "serve/chat_template.jinja", "Je regarde aussi le modèle de conversation."),
      ("read_file", "chat.py", "Et le client en ligne de commande.")]),
]

# ---------------------------------------------------------------- the agentic workload (synthetic, short turns)
AGENTIC_SYSTEM = ("You are an autonomous coding agent working in the user's repository. Read the code before you change "
                  "it, make small and correct edits, and keep the existing style. Never invent file contents.")

AGENTIC_TASK = ("Find every place that reads the expert cache during decode, explain what each one needs from it, and "
                "propose a smaller interface. Work step by step and report what you changed.")

# Common English words.  One word is about one token, so a target of N words is about N tokens; the measured size is
# the server's own prompt_n, which every report prints.
AGENTIC_WORDS = ("the of and to in a is that for it with as was on be at by this from or an are not but they you all we "
                 "can her has there their if one do into your which so then them would about will these when make like "
                 "time just him know take people year good some could see other than now look only come its over think "
                 "also back after use two how our work first well way even new want because any give day most us file "
                 "line code call read write test build server engine cache token prompt context memory layer weight "
                 "kernel batch queue thread lock buffer").split()

# (what the assistant says, tool, argument).  The tool result is synthetic, so the argument is only prompt text.
AGENTIC_STEPS = (
    ("Let me read the first part of the search result.", "read_file", {"path": "src/core/expert_cache.cpp"}),
    ("Now the callers of that function.", "grep", {"pattern": "expert_cache"}),
    ("The declaration should say what the cache keeps.", "read_file", {"path": "include/strata/core/expert_cache.hpp"}),
    ("Let me see how the tests drive it.", "read_file", {"path": "tools/conversation_cache_disk.py"}),
)

# What the report reads out of every row.
METRIC_KEYS = ("new_tokens", "reused", "reuse_ratio", "ttft_ms", "prefill_ms", "ms_per_new_token", "prefill_tok_s",
               "decode_tok_s", "request_ms")


def tool_result(tool: str, arg: str) -> str:
    if tool == "read_file":
        lines = (ROOT / arg).read_text(encoding="utf-8", errors="replace").splitlines()
        text = "\n".join(lines[:2000])
        return text + (f"\n... (truncated: lines 1-2000 of {len(lines)})" if len(lines) > 2000 else "")
    if tool == "grep":
        r = subprocess.run(["git", "-C", str(ROOT), "grep", "-n", "-I", "-E", arg], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        lines = r.stdout.splitlines()
        return "\n".join(lines[:200]) + (f"\n... ({len(lines) - 200} more lines)" if len(lines) > 200 else "")
    raise ValueError(tool)


def turns() -> list[tuple[str, list[dict]]]:
    """[(tag, messages)]: one entry per point right after a tool result."""
    out = []
    for name, task, steps in SESSIONS:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
        for i, (tool, arg, before) in enumerate(steps):
            cid = f"call_{name}_{i}"
            args = {"path": arg} if tool == "read_file" else {"pattern": arg}
            msgs.append({"role": "assistant", "content": before,
                         "tool_calls": [{"id": cid, "type": "function",
                                         "function": {"name": tool, "arguments": json.dumps(args)}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "content": tool_result(tool, arg)})
            out.append((f"{name}.{i}", [dict(m) for m in msgs]))
    return out


def synthetic_result(tokens: int, rng: random.Random) -> str:
    """A synthetic tool result of about `tokens` tokens: common English words, one about one token each.

    The size is an approximation.  The report prints the server's own `prompt_n` and `prompt_tokens` for every
    turn, and those are the measured sizes."""
    return " ".join(rng.choice(AGENTIC_WORDS) for _ in range(max(0, int(tokens))))


def agentic_turns(prefix_tokens: int, additions: list[int], turns_n: int, seed: int) -> list[tuple[str, list[dict]]]:
    """[(tag, messages)] for the short-turn workload: one long shared prefix, then `turns_n` short turns.

    The prefix (system, task, one assistant call, one large tool result) is the same text in every turn, so the
    conversation cache can hold it; each later turn adds one assistant line and one tool result of the next size
    in `additions`, in order.  `seed` fixes the text, so two arms send byte-identical requests."""
    rng = random.Random(seed)
    text, tool, args = AGENTIC_STEPS[0]
    msgs = [{"role": "system", "content": AGENTIC_SYSTEM}, {"role": "user", "content": AGENTIC_TASK},
            {"role": "assistant", "content": text,
             "tool_calls": [{"id": "call_agentic_prefix", "type": "function",
                             "function": {"name": tool, "arguments": json.dumps(args)}}]},
            {"role": "tool", "tool_call_id": "call_agentic_prefix", "content": synthetic_result(prefix_tokens, rng)}]
    out = [("agentic.prefix", [dict(m) for m in msgs])]
    for i in range(turns_n):
        text, tool, args = AGENTIC_STEPS[(i + 1) % len(AGENTIC_STEPS)]
        cid = f"call_agentic_{i}"
        msgs.append({"role": "assistant", "content": text,
                     "tool_calls": [{"id": cid, "type": "function",
                                     "function": {"name": tool, "arguments": json.dumps(args)}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": synthetic_result(additions[i % len(additions)], rng)})
        out.append((f"agentic.{i}", [dict(m) for m in msgs]))
    return out


# ---------------------------------------------------------------- metrics (the server's own fields, plus the ratios it does not report)
def turn_metrics(timings: dict | None, usage: dict | None, request_ms: float | None = None,
                 ttft_ms: float | None = None) -> dict:
    """One turn's numbers, from the server's own fields: `timings` is llama.cpp's (serve/server.py
    request_timings), `usage` is the OpenAI one, `request_ms` is the caller's own wall clock around the
    whole HTTP call, and `ttft_ms` is the caller's clock from just before the request is sent to the first
    client-observed generated output: content or reasoning, or a tool call's function name or arguments.

    new_tokens   what this request read (timings.prompt_n)
    reused       what the conversation cache already held (timings.cache_n)
    reuse_ratio  reused over the whole prompt (usage.prompt_tokens); None when the prompt is empty
    ttft_ms      the first generated output; None when the stream carried only controls (role, keep-alive
                 comments, a call's id/type/index) before it ended (never the whole call's time or the
                 completion's).  Parser buffering is inside it: it is the client's observation, not the
                 engine's raw first-token time.
    prefill_ms   the time to read the new tokens; ms_per_new_token that time per new token (None when nothing
                 was read: a fully reused turn has no prefill rate)
    prefill_tok_s  new tokens per second; decode_ms, out_tokens, decode_tok_s  the generated part
    total_ms     prefill plus decode, the engine's own time for the request
    request_ms   the whole call, from the client: parking a conversation, restoring one, and the setup around
                 the engine's own time are all in it, so a difference between the arms that total_ms does not
                 show is a difference outside the engine

    Every number the server does not give stays None (an engine without a clock, a request it cancelled)."""
    t = timings if isinstance(timings, dict) else {}
    u = usage if isinstance(usage, dict) else {}
    prompt_tokens = u.get("prompt_tokens")
    new_tokens, reused = t.get("prompt_n"), t.get("cache_n")
    if reused is None and isinstance(u.get("prompt_tokens_details"), dict):
        reused = u["prompt_tokens_details"].get("cached_tokens")
    prefill_ms, decode_ms = t.get("prompt_ms"), t.get("predicted_ms")
    out_tokens = t.get("predicted_n")
    if out_tokens is None:
        out_tokens = u.get("completion_tokens")
    return {
        "prompt_tokens": prompt_tokens, "new_tokens": new_tokens, "reused": reused,
        "reuse_ratio": (reused / prompt_tokens) if reused is not None and prompt_tokens else None,
        "ttft_ms": ttft_ms,
        "prefill_ms": prefill_ms, "ms_per_new_token": t.get("prompt_per_token_ms"),
        "prefill_tok_s": t.get("prompt_per_second"),
        "decode_ms": decode_ms, "out_tokens": out_tokens, "decode_tok_s": t.get("predicted_per_second"),
        "total_ms": None if prefill_ms is None or decode_ms is None else round(prefill_ms + decode_ms, 1),
        "request_ms": request_ms,
        "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
    }


def median_of(values) -> float | None:
    """The median of the numbers that are not None (None when there are none)."""
    v = [x for x in values if x is not None]
    return statistics.median(v) if v else None


def improvement(arm: float | None, first: float | None, lower_is_better: bool = False) -> float | None:
    """The arm's factor against the first arm: above 1 is better, whichever way the metric runs.  For a latency
    (`lower_is_better`) the factor is first/arm, so the shorter time wins.  None when either number is missing
    or zero."""
    if not arm or not first:
        return None
    return (first / arm) if lower_is_better else (arm / first)


def _num(v) -> str:
    return "n/a" if v is None else str(v)


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _factor(v) -> str:
    return "n/a" if v is None else f"x{v:.2f}"


def session_starts(work: list[tuple[str, list[dict]]]) -> list[bool]:
    """True for each turn that starts a session: the first one, and any turn whose predecessor's messages are
    not a prefix of its own.  Such a turn has nothing before it, so the engine reads its whole prompt: it is the
    cold turn of its session, not a turn of the agent's short loop."""
    return [i == 0 or work[i - 1][1] != msgs[:len(work[i - 1][1])] for i, (_, msgs) in enumerate(work)]


def session_additions(name: str, work: list[tuple[str, list[dict]]], res: dict) -> dict:
    """{tag: the tokens this turn added to its conversation}, the whole prompt less the turn before it in the
    same session (median over rounds).  None for a session's first turn, and for a turn whose predecessor has no
    row in the same round: there is nothing to subtract."""
    starts = session_starts(work)
    out: dict[str, float | None] = {}
    for i, (tag, _) in enumerate(work):
        out[tag] = None
        if starts[i]:
            continue                                     # a session's first turn: the whole prompt is new
        before = {r["round"]: r["prompt_tokens"] for r in res[name].get(work[i - 1][0], [])}
        now = {r["round"]: r["prompt_tokens"] for r in res[name].get(tag, [])}
        out[tag] = median_of([now[k] - before[k] for k in now
                              if k in before and now[k] is not None and before[k] is not None])
    return out


# ---------------------------------------------------------------- one arm's own state and process
def arm_paths(home: Path, name: str, rnd: int) -> dict:
    """This arm's own paths under `home`, one directory per arm per round.

    The conversation-cache disk store and the expert profile survive a restart, so two arms must never share
    them: a later arm would restore or start from the state of an earlier one."""
    d = home / f"{name}_r{rnd}"
    return {"dir": d, "log": d / "engine.log", "disk": d / "conversations", "profile": d / "expert-profile.bin"}


def _redirect_arg(args: list[str], flag: str, path: str) -> None:
    """Point every `flag PATH` in `args` at `path`, in place.

    The engine takes the last occurrence of an option, and an arm's own `xargs` may add another one after the
    config's, so every occurrence is rewritten.  The engine's parser reads the flag and its value as two
    arguments (src/program/generate.cpp), so there is no `flag=PATH` form to rewrite."""
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            args[i + 1] = path


def isolated_config(base: dict, name: str, rnd: int, home: Path, port: int) -> dict:
    """The config one arm's server gets: `base` with this arm's own log, conversation-cache disk store and expert
    profile, bound to the loopback address.

    The API key is left out, so the operator's secret is never copied into a generated file.  The server then
    answers only on 127.0.0.1, which needs no key (serve/server.py: an IP address passes the Host check).  Every
    `--conversation-cache-disk` and `--expert-profile-save` in the config's own arguments, and in an arm's
    `xargs`, is redirected as well.  `base` is not modified."""
    cfg = dict(base)
    cfg.pop("api_key", None)
    cfg["host"] = "127.0.0.1"
    cfg["port"] = port
    p = arm_paths(home, name, rnd)
    cfg["log"] = str(p["log"])
    disk = str(p["disk"])
    args = list(cfg.get("args") or [])
    _redirect_arg(args, "--conversation-cache-disk", disk)
    _redirect_arg(args, "--expert-profile-save", str(p["profile"]))
    if args != (base.get("args") or []):
        cfg["args"] = args
    if cfg.get("conversation_cache_disk") is not None:
        cfg["conversation_cache_disk"] = disk
    if isinstance(cfg.get("expert_profile_save"), str) and cfg["expert_profile_save"].strip():
        cfg["expert_profile_save"] = str(p["profile"])
    return cfg


def start_arm(cfg_path: Path, port: int, server_log: Path, env: dict, engine: str = "strata") -> subprocess.Popen:
    """Start one arm's server: its own process group (so its engine can be ended with it), its own stdout and
    stderr in the arm's directory, and no terminal.  `engine` is the server's own choice: "strata" for the model,
    "mock" for the scripted stand-in that needs no GPU (a smoke run, with no engine timings)."""
    out = open(server_log, "ab")
    try:
        return subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", engine, "--config",
                                 str(cfg_path), "--port", str(port)], cwd=str(ROOT), env=env, stdout=out,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    finally:
        out.close()


def stop_arm(proc: subprocess.Popen) -> None:
    """Stop one arm's server and the engine it started: SIGTERM first (the server sends QUIT to the engine and
    waits for it), then SIGKILL of the arm's whole process group if anything is still there.  The group is the
    arm's own (start_arm's new session), so this can never reach the benchmark itself.  Its id is the server's
    pid, which still names the group when the server itself is gone and its engine is not."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(120)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        pass


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def post(port: int, messages: list[dict], max_tokens: int, seed: int) -> tuple[dict, float | None]:
    """One streamed turn's request, rebuilt into the response a non-streamed call would return.

    Returns `(response, ttft_ms)`.  `ttft_ms` is this tool's clock from immediately before the request is
    sent to the first client-observed generated output: a content or reasoning delta, or a tool-call piece
    that carries a function name or arguments.  The role-only opening delta, SSE keep-alive comments and a
    call's id/type/index announcement (even with an empty function object) are not output.  It is None when
    the stream ended without any such output; the whole call's time is reported separately as `request_ms`
    and is never substituted for it.
    The client's and the parser's buffering are inside it: it is not the engine's raw first-token time.

    A refusal (the context is too small for the prompt, a bad key) is raised with the server's own words, and
    so is an error event inside the stream or a stream that ends without `[DONE]`.  A whole (non-streamed)
    completion is refused rather than read as a stream: TTFT cannot be recovered from it."""
    body = {"messages": messages, "tools": TOOLS, "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95,
            "top_k": 20, "seed": seed, "reasoning_effort": "high", "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    content: list[str] = []
    reasoning: list[str] = []
    by_index: dict[int, dict] = {}
    usage = timings = None
    meta: dict = {}
    finish = None
    ttft_ms = None
    done = False
    t_send = time.perf_counter()          # before the connection is made: the client's own first-output clock
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype and ctype != "text/event-stream":
                raise RuntimeError(f"the server answered {ctype}, not an SSE stream")
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line or line.startswith(":"):      # an empty separator or an SSE keep-alive comment
                    continue
                if not line.startswith("data:"):
                    if line.startswith("{"):
                        raise RuntimeError("the server answered with a whole completion, not a stream")
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                try:
                    chunk = json.loads(data)
                except ValueError as e:
                    raise RuntimeError(f"the stream carried a line that is not JSON: {data[:300]!r}") from e
                if not isinstance(chunk, dict):
                    raise RuntimeError(f"the stream carried a non-object event: {data[:300]!r}")
                if chunk.get("error"):
                    err = chunk["error"]
                    detail = err.get("message") if isinstance(err, dict) and err.get("message") else str(err)
                    raise RuntimeError(f"the stream ended with an error: {detail}")
                for key in ("id", "created", "model"):
                    if chunk.get(key) is not None:
                        meta[key] = chunk[key]
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                if chunk.get("timings") is not None:
                    timings = chunk["timings"]
                choice = (chunk.get("choices") or [{}])[0]
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]      # a later usage-only chunk has no choices to overwrite it
                delta = choice.get("delta") or {}
                generated = bool(delta.get("content") or delta.get("reasoning_content"))
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                pieces = delta.get("tool_calls") or []
                for tc in pieces:                            # streamed calls come in pieces: merge by index
                    cur = by_index.setdefault(tc.get("index", len(by_index)),
                                              {"id": None, "type": "function",
                                               "function": {"name": "", "arguments": ""}})
                    cur["id"] = tc.get("id") or cur["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name") or fn.get("arguments"):     # a name or its arguments, not id/type/index
                        generated = True
                    cur["function"]["name"] += fn.get("name") or ""
                    cur["function"]["arguments"] += fn.get("arguments") or ""
                if ttft_ms is None and generated:
                    ttft_ms = round((time.perf_counter() - t_send) * 1000, 1)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"the server answered {e.code}: {e.read().decode('utf-8', 'replace')[:600]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"the request failed: {e.reason}") from e
    except (OSError, http.client.HTTPException) as e:
        raise RuntimeError(f"the stream broke after {len(content) + len(reasoning)} deltas: {e}") from e
    if not done:
        raise RuntimeError(f"the stream ended before its [DONE] marker ({len(content) + len(reasoning)} "
                           "content/reasoning deltas, no whole answer to report)")
    calls = [by_index[i] for i in sorted(by_index)]
    if finish != "tool_calls":                    # as the server's own collector: a call the output ended inside
        calls = [c for c in calls if _is_json(c["function"]["arguments"])]
    message = {"role": "assistant", "content": "".join(content) or None}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if calls:
        message["tool_calls"] = calls
    out = {"id": meta.get("id"), "object": "chat.completion",
           "created": meta.get("created"), "model": meta.get("model"),
           "choices": [{"index": 0, "message": message, "finish_reason": finish}],
           "usage": usage}
    if timings is not None:
        out["timings"] = timings
    return out, ttft_ms


def turn_line(tag: str, m: dict) -> str:
    """One measured turn, as the run prints it."""
    return (f"{tag:16s} new {_num(m['new_tokens']):>6s}  reused {_num(m['reused']):>7s} ({_pct(m['reuse_ratio']):>6s})"
            f"  ttft {_num(m['ttft_ms']):>8s} ms"
            f"  prefill {_num(m['prefill_ms']):>9s} ms ({_num(m['ms_per_new_token']):>6s} ms/tok)"
            f"  decode {_num(m['decode_tok_s']):>6s} tok/s  n {_num(m['out_tokens']):>4s}"
            f"  req {_num(m['request_ms']):>9s} ms")


def summarize(arms: list[tuple[str, dict]], work: list[tuple[str, list[dict]]], res: dict, targets: dict) -> None:
    """Print the per-turn medians per arm, then the factor of every other arm against the first arm.

    A cold turn (a session's first: the whole prompt is new) is listed like the others and kept out of the
    medians of the short turns: one long cold read is not the per-turn cost of an agent at work."""
    names = [name for name, _ in arms]
    first = names[0]
    starts = session_starts(work)
    kind = {tag: ("cold" if start else "short") for (tag, _), start in zip(work, starts)}
    short = [tag for (tag, _), start in zip(work, starts) if not start]
    cold = [tag for (tag, _), start in zip(work, starts) if start]
    med = {(name, tag): {k: median_of([row[k] for row in res[name].get(tag, [])]) for k in METRIC_KEYS}
           for name in names for tag, _ in work}
    added = {name: session_additions(name, work, res) for name in names}
    print("\nper-turn medians over rounds ('target' is the requested size of the turn's tool result; 'added', "
          "'new', 'reuse', 'ttft' and 'req ms' are measured; 'req ms' is the whole HTTP call, 'ttft' the client's "
          "time to the first generated delta, 'prefill ms' the engine's own time):")
    print(f"  {'turn':16s} {'kind':>5s} {'target':>6s} {'added':>7s} {'new':>7s} {'reuse':>7s} {'ttft ms':>8s} "
          f"{'prefill ms':>10s} {'ms/tok':>7s} {'req ms':>9s} {'decode tok/s':>12s}  arm")
    for tag, _ in work:
        for name in names:
            m = med[(name, tag)]
            print(f"  {tag:16s} {kind[tag]:>5s} {_num(targets.get(tag)):>6s} {_num(added[name].get(tag)):>7s} "
                  f"{_num(m['new_tokens']):>7s} {_pct(m['reuse_ratio']):>7s} {_num(m['ttft_ms']):>8s} "
                  f"{_num(m['prefill_ms']):>10s} {_num(m['ms_per_new_token']):>7s} {_num(m['request_ms']):>9s} "
                  f"{_num(m['decode_tok_s']):>12s}  {name}")
    if len(names) < 2:
        return
    print(f"\nper turn against {first}: ttft and prefill are its milliseconds over the arm's, decode is the "
          "arm's tokens per second over its; above 1 is better for all three.")
    print(f"  {'turn':16s} {'kind':>5s} {'new':>7s} {'ttft':>8s} {'prefill':>8s} {'decode':>8s}  arm")
    for tag, _ in work:
        for name in names[1:]:
            t = improvement(med[(name, tag)]["ttft_ms"], med[(first, tag)]["ttft_ms"], lower_is_better=True)
            p = improvement(med[(name, tag)]["prefill_ms"], med[(first, tag)]["prefill_ms"], lower_is_better=True)
            d = improvement(med[(name, tag)]["decode_tok_s"], med[(first, tag)]["decode_tok_s"])
            new_first, new_arm = med[(first, tag)]["new_tokens"], med[(name, tag)]["new_tokens"]
            differs = "" if new_first == new_arm else \
                f"  <- different work: {first} new {_num(new_first)}, {name} new {_num(new_arm)}"
            print(f"  {tag:16s} {kind[tag]:>5s} {_num(new_arm):>7s} {_factor(t):>8s} "
                  f"{_factor(p):>8s} {_factor(d):>8s}  {name}{differs}")
    for name in names[1:]:
        for label, tags in (("short", short), ("cold", cold)):
            if not tags:
                continue
            t = [improvement(med[(name, tag)]["ttft_ms"], med[(first, tag)]["ttft_ms"], lower_is_better=True)
                 for tag in tags]
            p = [improvement(med[(name, tag)]["prefill_ms"], med[(first, tag)]["prefill_ms"], lower_is_better=True)
                 for tag in tags]
            d = [improvement(med[(name, tag)]["decode_tok_s"], med[(first, tag)]["decode_tok_s"]) for tag in tags]
            no_ttft = sum(1 for tag in tags
                          if any(row.get("ttft_ms") is None
                                 for n in (first, name) for row in res[n].get(tag, [])))
            print(f"  {name}: {len(tags)} {label} turn{'' if len(tags) == 1 else 's'}: ttft "
                  f"{_factor(median_of(t))} median ({sum(1 for x in t if x and x > 1)}/{len(tags)} faster), "
                  f"prefill {_factor(median_of(p))} median ({sum(1 for x in p if x and x > 1)}/{len(tags)} faster), "
                  f"decode {_factor(median_of(d))} median ({sum(1 for x in d if x and x > 1)}/{len(tags)} faster)"
                  + (f"; ttft n/a on {no_ttft}/{len(tags)} turns" if no_ttft else ""))
    measured = [tag for tag, _ in work if med[(first, tag)]["new_tokens"] is not None]
    if not measured:
        print("  no engine timings in this run: there is no new-token count to check between the arms")
        return
    same = sum(1 for tag in measured if len({med[(name, tag)]["new_tokens"] for name in names}) == 1)
    print(f"  the same new-token count in every arm on {same}/{len(measured)} measured turns"
          + ("" if same == len(measured) else "  <- the arms did different work: do not compare those turns"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--workload", choices=("repo", "agentic"), default="repo")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-turns", type=int, default=0, help="only the first N turns (0: all)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log", default="agent_turns_bench.jsonl")
    ap.add_argument("--prefix-tokens", type=int, default=8192, help="agentic: the shared prefix's target size")
    ap.add_argument("--additions", default="64,128,256,512", help="agentic: each turn's target tool-result tokens")
    ap.add_argument("--turns", type=int, default=16, help="agentic: how many short turns")
    ap.add_argument("--engine", choices=("strata", "mock"), default="strata",
                    help="the server's engine: mock needs no GPU and gives no engine timings (a smoke run)")
    ap.add_argument("--workdir", default=None, help="where the arms keep their logs and caches (default: a fresh "
                                                    "temporary directory, removed at the end)")
    a = ap.parse_args()
    base = json.loads(Path(a.config).read_text(encoding="utf-8"))
    arms = []
    for s in a.arms:
        name, _, kv = s.partition(":")
        arms.append((name, dict(x.split("=", 1) for x in kv.split(",") if x)))
    targets: dict[str, int] = {}
    additions: list[int] = []
    if a.workload == "agentic":
        additions = [int(x) for x in a.additions.replace(" ", "").split(",") if x]
        if not additions:
            raise SystemExit("--additions needs at least one size")
        work = agentic_turns(a.prefix_tokens, additions, a.turns, a.seed)
        targets = {"agentic.prefix": a.prefix_tokens, **{f"agentic.{i}": additions[i % len(additions)]
                                                         for i in range(a.turns)}}
    else:
        work = turns()
    if a.max_turns:
        work = work[:a.max_turns]
    owned = a.workdir is None
    home = Path(tempfile.mkdtemp(prefix="agent-turns-")) if owned else Path(a.workdir)
    home.mkdir(parents=True, exist_ok=True)
    print(f"{a.workload}: {len(work)} turns, {a.rounds} rounds, arms {', '.join(n for n, _ in arms)}")
    if a.workload == "agentic":
        print(f"  prefix target {a.prefix_tokens} tokens, additions {additions} (targets: the report prints the "
              "measured sizes)")
    print(f"  workdir {home}" + (" (removed at the end; pass --workdir to keep it)" if owned else ""), flush=True)
    out = open(a.log, "a", encoding="utf-8")
    res: dict = {n: {} for n, _ in arms}
    try:
        for rnd in range(a.rounds):
            for name, env in (arms if rnd % 2 == 0 else arms[::-1]):
                cfg = dict(base)
                e = dict(env)
                if "exe" in e:
                    cfg["exe"] = e.pop("exe")
                if "xargs" in e:
                    cfg["args"] = list(base.get("args") or []) + [x for x in e.pop("xargs").split(";") if x]
                cfg = isolated_config(cfg, name, rnd, home, a.port)
                paths = arm_paths(home, name, rnd)
                paths["dir"].mkdir(parents=True, exist_ok=True)
                paths["disk"].mkdir(parents=True, exist_ok=True)
                cpath = paths["dir"] / "config.json"
                cpath.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
                proc = start_arm(cpath, a.port, paths["dir"] / "server.log", dict(os.environ, **e), a.engine)
                try:
                    t0 = time.time()
                    wait_ready(a.port, proc)
                    print(f"round {rnd} {name}: ready in {time.time() - t0:.0f} s", flush=True)
                    for tag, msgs in work:
                        t1 = time.perf_counter()
                        r, ttft_ms = post(a.port, msgs, a.max_new, a.seed)
                        request_ms = round((time.perf_counter() - t1) * 1000, 1)
                        m = turn_metrics(r.get("timings"), r.get("usage"), request_ms, ttft_ms)
                        row = {"round": rnd, "arm": name, "turn": tag, "workload": a.workload,
                               "target": targets.get(tag), **m}
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        res[name].setdefault(tag, []).append(row)
                        print("  " + turn_line(tag, m), flush=True)
                finally:
                    stop_arm(proc)
                    time.sleep(5)
    except BaseException:
        print(f"\nworkdir kept for diagnosis: {home}", flush=True)
        raise
    finally:
        out.close()
    if owned:
        shutil.rmtree(home, ignore_errors=True)
    summarize(arms, work, res, targets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
