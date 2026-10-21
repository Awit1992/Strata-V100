"""Tests for tools/agent_turns_bench.py without a GPU, a model or a server: the metric arithmetic and its
boundaries, the streamed turn's first output and reconstruction, the agentic workload's shape, and the per-arm
isolation.

    python -m unittest tools.test_agent_turns_bench
"""
from __future__ import annotations

import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import agent_turns_bench as B  # noqa: E402

HOME = Path("/home/agent-turns-home")


class Metrics(unittest.TestCase):
    """What the tool derives from the server's fields, and what it does when the server gives nothing."""

    USAGE = {"prompt_tokens": 8064, "completion_tokens": 256}

    def test_the_reuse_share_is_the_cached_part_of_the_prompt(self):
        self.assertAlmostEqual(B.turn_metrics({"prompt_n": 64, "cache_n": 8000}, self.USAGE)["reuse_ratio"],
                               8000 / 8064)
        self.assertEqual(B.turn_metrics({"prompt_n": 8064, "cache_n": 0}, self.USAGE)["reuse_ratio"], 0.0)
        # a fully reused turn: the engine read nothing, so the whole prompt came from the cache
        self.assertEqual(B.turn_metrics({"prompt_n": 0, "cache_n": 8064}, self.USAGE)["reuse_ratio"], 1.0)

    def test_the_engines_own_time_is_the_prefill_plus_the_decode(self):
        m = B.turn_metrics({"prompt_ms": 120.0, "predicted_ms": 6000.0}, self.USAGE)
        self.assertEqual(m["total_ms"], 6120.0)
        self.assertIsNone(B.turn_metrics({"prompt_ms": 120.0}, self.USAGE)["total_ms"])   # no decode time

    def test_an_empty_or_missing_prompt_has_no_reuse_share(self):
        self.assertIsNone(B.turn_metrics({}, {})["reuse_ratio"])
        self.assertIsNone(B.turn_metrics({"cache_n": 0}, {"prompt_tokens": 0})["reuse_ratio"])

    def test_an_engine_without_a_clock_leaves_every_time_none(self):
        """MockEngine keeps no clock: the HTTP time still exists, the engine's times do not."""
        m = B.turn_metrics(None, {"prompt_tokens": 100, "completion_tokens": 7,
                                  "prompt_tokens_details": {"cached_tokens": 40}}, request_ms=1234.5)
        self.assertAlmostEqual(m["reuse_ratio"], 0.4)          # the fallback: usage carries the cached count
        self.assertEqual(m["request_ms"], 1234.5)
        for key in ("new_tokens", "prefill_ms", "ms_per_new_token", "prefill_tok_s", "decode_ms", "decode_tok_s",
                    "total_ms", "draft_n", "draft_n_accepted"):
            self.assertIsNone(m[key], key)

    def test_median_and_factor_boundaries(self):
        self.assertIsNone(B.median_of([]))
        self.assertIsNone(B.median_of([None, None]))
        self.assertEqual(B.median_of([None, 2.0, 4.0]), 3.0)
        self.assertEqual(B.median_of([1.0, 3.0]), 2.0)                  # even count: the middle two
        self.assertIsNone(B.improvement(None, 5.0))
        self.assertIsNone(B.improvement(5.0, None))
        self.assertIsNone(B.improvement(0.0, 5.0))
        self.assertIsNone(B.improvement(5.0, 0.0))
        self.assertEqual(B.improvement(50.0, 40.0), 1.25)               # more is better
        self.assertAlmostEqual(B.improvement(40.0, 50.0, lower_is_better=True), 1.25)
        self.assertAlmostEqual(B.improvement(50.0, 40.0, lower_is_better=True), 0.8)


class Workloads(unittest.TestCase):
    def test_the_agentic_workload_shares_one_prefix_and_grows(self):
        work = B.agentic_turns(512, [64, 128, 256, 512], 8, 7)
        self.assertEqual([tag for tag, _ in work], ["agentic.prefix"] + [f"agentic.{i}" for i in range(8)])
        prefix = work[0][1]
        for i, (tag, msgs) in enumerate(work[1:]):
            self.assertEqual(msgs[:len(prefix)], prefix, tag)           # the shared prefix, message for message
            self.assertEqual(len(msgs), len(prefix) + 2 * (i + 1), tag)
            self.assertEqual(msgs[-1]["role"], "tool", tag)
            self.assertEqual(msgs[-2]["role"], "assistant", tag)
        self.assertTrue(all("tool_calls" in msgs[-2] for _, msgs in work[1:]))

    def test_each_agentic_turn_adds_the_requested_size(self):
        additions = [64, 128, 256, 512]
        work = B.agentic_turns(256, additions, 8, 7)
        for i, (tag, msgs) in enumerate(work[1:]):
            self.assertEqual(len(msgs[-1]["content"].split()), additions[i % len(additions)], tag)
        self.assertEqual(len(work[0][1][-1]["content"].split()), 256)

    def test_the_agentic_workload_is_reproducible(self):
        self.assertEqual(B.agentic_turns(256, [64, 128], 4, 11), B.agentic_turns(256, [64, 128], 4, 11))
        self.assertNotEqual(B.agentic_turns(256, [64, 128], 4, 11), B.agentic_turns(256, [64, 128], 4, 12))

    def test_a_turn_that_does_not_extend_the_one_before_it_starts_a_session(self):
        first = [{"role": "user", "content": "a"}]
        work = [("s.0", first), ("s.1", first + [{"role": "tool", "content": "b"}]),
                ("t.0", [{"role": "user", "content": "z"}])]
        self.assertEqual(B.session_starts(work), [True, False, True])
        # the agentic workload is one session: only its prefix is cold, and the report keeps it out of the
        # medians of the short turns
        self.assertEqual(B.session_starts(B.agentic_turns(64, [64], 3, 7)), [True, False, False, False])


class Report(unittest.TestCase):
    """The measured per-turn addition: the whole prompt less the turn before it in the same session."""

    WORK = [("s.0", [{"role": "system", "content": "a"}]),
            ("s.1", [{"role": "system", "content": "a"}, {"role": "tool", "content": "b"}]),
            ("t.0", [{"role": "system", "content": "z"}])]

    def test_the_added_tokens_are_measured_from_the_turn_before(self):
        res = {"A": {"s.0": [{"round": 0, "prompt_tokens": 100}, {"round": 1, "prompt_tokens": 100}],
                     "s.1": [{"round": 1, "prompt_tokens": 164}],
                     "t.0": [{"round": 0, "prompt_tokens": 40}]}}
        added = B.session_additions("A", self.WORK, res)
        self.assertIsNone(added["s.0"])                 # a session's first turn has nothing before it
        self.assertIsNone(added["t.0"])                 # another session's first turn, too
        self.assertEqual(added["s.1"], 64)              # only round 1 has both rows to subtract

    def test_a_missing_prompt_count_leaves_the_addition_unknown(self):
        res = {"A": {"s.0": [{"round": 0, "prompt_tokens": None}], "s.1": [{"round": 0, "prompt_tokens": 164}]}}
        self.assertIsNone(B.session_additions("A", self.WORK, res)["s.1"])


class _Clock:
    """A stand-in for `time.perf_counter` that moves only when the scripted stream hands over a line."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def tick(self, ms: float) -> None:
        self.t += ms / 1000.0


class _Stream:
    """The parts of an HTTP response `post` uses: headers, a context manager, and lines of bytes.

    Each line's delay is applied to the clock just before it is handed over, so the value `post` reads at the
    first output says which line that output came from.
    """

    def __init__(self, lines, clock: _Clock, content_type: str = "text/event-stream"):
        self._lines = lines
        self._clock = clock
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for delay_ms, line in self._lines:
            self._clock.tick(delay_ms)
            yield line if isinstance(line, bytes) else line.encode()


def _sse(payload: dict, delay_ms: float = 10.0) -> tuple[float, str]:
    """One SSE data line and how long the clock runs before the client reads it."""
    return delay_ms, "data: " + json.dumps(payload) + "\n"


def _delta(**delta) -> dict:
    """One streamed choice with a delta and no finish reason."""
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


class StreamedTurn(unittest.TestCase):
    """`post` on a scripted stream with a clock the test moves line by line: what the client sees, and when."""

    MESSAGES = [{"role": "user", "content": "go"}]
    USAGE = {"prompt_tokens": 32, "completion_tokens": 4}
    TIMINGS = {"prompt_n": 32, "cache_n": 0, "prompt_ms": 5.0, "predicted_ms": 6.0}

    def setUp(self):
        self.clock = _Clock()

    def post(self, lines, content_type: str = "text/event-stream"):
        """Run `post` over `lines` on `self.clock`; the transport is a response fake, not a socket."""
        stream = _Stream(lines, self.clock, content_type)
        with mock.patch.object(B.time, "perf_counter", self.clock), \
                mock.patch.object(B.urllib.request, "urlopen", lambda req, timeout=None: stream):
            return B.post(9999, self.MESSAGES, 16, 7)

    def test_the_first_output_is_content_reasoning_or_a_calls_name_or_arguments(self):
        """Role, keep-alive and id/type/index (an empty function included) are controls, not output."""
        outputs = {
            "content": {"content": "x"},
            "reasoning": {"reasoning_content": "x"},
            "name": {"tool_calls": [{"index": 0, "function": {"name": "read_file"}}]},
            "arguments": {"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]},
        }
        for label, output in outputs.items():
            with self.subTest(label):
                self.clock = _Clock()
                lines = [
                    _sse(_delta(role="assistant")),                                              # 10 ms
                    (10.0, ": keep-alive\n"),                                                    # 20 ms
                    _sse(_delta(tool_calls=[{"index": 0, "id": "call_1", "type": "function"}])),  # 30 ms
                    _sse(_delta(tool_calls=[{"index": 0, "function": {}}])),                      # 40 ms
                    _sse(_delta(**output)),                                                      # 50 ms: output
                    _sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                    _sse({"choices": [], "usage": self.USAGE}),
                    (10.0, "data: [DONE]\n"),
                ]
                out, ttft = self.post(lines)
                self.assertEqual(ttft, 50.0)
                self.assertEqual(out["choices"][0]["finish_reason"], "stop")

    def test_a_stream_with_only_controls_has_no_ttft(self):
        """None, not the whole call's time: the client saw no generated output."""
        lines = [
            _sse(_delta(role="assistant")),
            (10.0, ": keep-alive\n"),
            _sse(_delta(tool_calls=[{"index": 0, "id": "call_1", "type": "function"}])),
            _sse(_delta(tool_calls=[{"index": 0, "function": {"name": "", "arguments": ""}}])),
            _sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            _sse({"choices": [], "usage": self.USAGE}),
            (10.0, "data: [DONE]\n"),
        ]
        out, ttft = self.post(lines)
        self.assertIsNone(ttft)
        message = out["choices"][0]["message"]
        self.assertIsNone(message["content"])
        self.assertNotIn("tool_calls", message)

    def test_interleaved_tool_pieces_and_usage_are_rebuilt(self):
        self.clock = _Clock()
        lines = [
            _sse(_delta(reasoning_content="think ")),
            _sse(_delta(content="Look")),
            _sse(_delta(tool_calls=[{"index": 0, "id": "call_a", "type": "function",
                                     "function": {"name": "read_", "arguments": "{\"path\":"}}])),
            _sse(_delta(tool_calls=[{"index": 1, "id": "call_b", "type": "function",
                                     "function": {"name": "grep", "arguments": "{\"pattern\":"}}])),
            _sse(_delta(reasoning_content="ing")),
            _sse(_delta(content="ing")),
            _sse(_delta(tool_calls=[{"index": 0, "function": {"arguments": "\"a.txt\"}"}}])),
            _sse(_delta(tool_calls=[{"index": 1, "function": {"arguments": "\"cache\"}"}}])),
            _sse({"id": "chatcmpl-1", "created": 123, "model": "m",
                  "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
            _sse({"choices": [], "usage": self.USAGE, "timings": self.TIMINGS}),
            (10.0, "data: [DONE]\n"),
        ]
        out, ttft = self.post(lines)
        self.assertEqual(ttft, 10.0)                        # the first reasoning piece
        message = out["choices"][0]["message"]
        self.assertEqual(message["content"], "Looking")
        self.assertEqual(message["reasoning_content"], "think ing")
        calls = message["tool_calls"]
        self.assertEqual([c["id"] for c in calls], ["call_a", "call_b"])
        self.assertEqual([c["function"]["name"] for c in calls], ["read_", "grep"])
        self.assertEqual([c["function"]["arguments"] for c in calls],
                         ["{\"path\":\"a.txt\"}", "{\"pattern\":\"cache\"}"])
        self.assertEqual(out["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(out["usage"], self.USAGE)
        self.assertEqual(out["timings"], self.TIMINGS)
        self.assertEqual((out["id"], out["created"], out["model"]), ("chatcmpl-1", 123, "m"))

    def test_a_stream_without_done_is_rejected(self):
        lines = [
            _sse(_delta(content="half")),
            _sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        ]
        with self.assertRaisesRegex(RuntimeError, r"\[DONE\]"):
            self.post(lines)

    def test_an_error_event_is_rejected(self):
        lines = [
            _sse(_delta(content="half")),
            _sse({"error": {"message": "the context is too small"}}),
        ]
        with self.assertRaisesRegex(RuntimeError, "the context is too small"):
            self.post(lines)

    def test_an_http_error_keeps_the_servers_words(self):
        failure = urllib.error.HTTPError(
            "http://127.0.0.1:9999/v1/chat/completions", 400, "Bad Request", None,
            io.BytesIO(b'{"error": {"message": "no such model"}}'))
        with mock.patch.object(B.time, "perf_counter", self.clock), \
                mock.patch.object(B.urllib.request, "urlopen", mock.Mock(side_effect=failure)):
            with self.assertRaisesRegex(RuntimeError, "400: .*no such model"):
                B.post(9999, self.MESSAGES, 16, 7)


class Isolation(unittest.TestCase):
    """Every arm runs on its own state, and the operator's API key never leaves the config file."""

    BASE = {"exe": "/engine/strata", "host": "0.0.0.0", "api_key": "SECRET-KEY", "port": 8088,
            "log": "/prod/strata.log", "conversation_cache_disk": "/prod/conversations",
            "conversation_cache_disk_gib": 50, "expert_profile_save": "/prod/profile.bin",
            "args": ["--pack", "/mnt/pack", "--conversation-cache-disk", "/prod/other",
                     "--expert-profile-save", "/prod/args-profile.bin", "--conversation-cache-disk", "/prod/last"],
            "cwd": "/prod"}

    def test_each_arm_round_gets_its_own_state_and_no_key(self):
        a = B.isolated_config(self.BASE, "base", 0, HOME, 8090)
        b = B.isolated_config(self.BASE, "cand", 0, HOME, 8090)
        again = B.isolated_config(self.BASE, "base", 1, HOME, 8090)
        for cfg in (a, b, again):
            self.assertNotIn("api_key", cfg)
            self.assertNotIn("SECRET-KEY", json.dumps(cfg))
            self.assertEqual(cfg["host"], "127.0.0.1")           # only the loopback address, so no key is needed
            self.assertEqual(cfg["port"], 8090)
            self.assertEqual(cfg["exe"], "/engine/strata")
            self.assertEqual(cfg["cwd"], "/prod")                # the engine's own folder is not this tool's
            for key in ("log", "conversation_cache_disk", "expert_profile_save"):
                self.assertTrue(str(cfg[key]).startswith(str(HOME)), (key, cfg[key]))
        for key in ("log", "conversation_cache_disk", "expert_profile_save"):
            self.assertNotEqual(a[key], b[key], key)
            self.assertNotEqual(a[key], again[key], key)

    def test_every_stateful_argument_is_redirected(self):
        """The engine takes the last occurrence of an option, so all of them must point at this arm."""
        cfg = B.isolated_config(self.BASE, "base", 0, HOME, 8090)
        args = cfg["args"]
        self.assertEqual(args[:2], ["--pack", "/mnt/pack"])      # every other argument is untouched
        self.assertEqual(args[args.index("--conversation-cache-disk") + 1], cfg["conversation_cache_disk"])
        self.assertEqual(args[-1], cfg["conversation_cache_disk"])            # the last occurrence as well
        self.assertEqual(args[args.index("--expert-profile-save") + 1], cfg["expert_profile_save"])

    def test_a_path_only_in_the_config_key_is_redirected_and_one_that_is_absent_stays_absent(self):
        kept = B.isolated_config({"args": ["--pack", "p"], "conversation_cache_disk": "/prod/c"},
                                 "base", 0, HOME, 8090)
        self.assertEqual(kept["args"], ["--pack", "p"])                       # the arguments are untouched
        self.assertTrue(str(kept["conversation_cache_disk"]).startswith(str(HOME)))
        self.assertNotIn("conversation_cache_disk",
                         B.isolated_config({"args": ["--pack", "p"]}, "base", 0, HOME, 8090))

    def test_the_base_config_is_not_modified(self):
        before = json.dumps(self.BASE, sort_keys=True)
        B.isolated_config(self.BASE, "base", 0, HOME, 8090)
        self.assertEqual(json.dumps(self.BASE, sort_keys=True), before)


if __name__ == "__main__":
    unittest.main()
