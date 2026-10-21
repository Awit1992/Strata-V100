"""Tests for bench/run_v100_prefill.py without a GPU, a model or an engine: the chained plan and its
prompt nesting, the prompt-cache argv and its isolation, the settings checks, and the DONE accounting.

    python -m unittest bench.test_run_v100_prefill
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
import run_v100_prefill as P  # noqa: E402


class ChainPlan(unittest.TestCase):
    """The schedule and the prompts: exact sizes, one nested stream, and the diff-0 repeat."""

    DIFFS = [0, 1, 32, 64, 128, 256, 512, 1024, 2048]

    def test_the_plan_starts_with_the_prefix_then_extends_by_each_diff(self):
        plan = P.chain_plan(8192, self.DIFFS)
        self.assertEqual(len(plan), 10)
        self.assertEqual(plan[0]["phase"], "prefix")
        self.assertEqual(plan[0]["prompt_tokens"], 8192)
        self.assertEqual(plan[0]["requested_diff"], 8192)
        self.assertEqual([p["requested_diff"] for p in plan[1:]], self.DIFFS)
        self.assertEqual([p["turn"] for p in plan], list(range(10)))
        self.assertEqual(plan[1]["prompt_tokens"], 8192)          # diff 0: the identical prompt again
        self.assertEqual(plan[2]["prompt_tokens"], 8193)          # diff 1
        self.assertEqual(plan[-1]["prompt_tokens"], 8192 + 4065)  # the sum of the diffs

    def test_the_plan_repeats_the_chain_per_round(self):
        plan = P.chain_plan(8192, [0, 1], rounds=2)
        self.assertEqual([p["round"] for p in plan], [0, 0, 0, 1, 1, 1])
        self.assertEqual([p["prompt_tokens"] for p in plan], [8192, 8192, 8193] * 2)

    def test_every_prompt_is_a_prefix_of_the_next_one(self):
        head, pool = [7, 8], list(range(100, 400))
        built = [P.build_chain_ids(head, pool, n) for n in (20, 20, 21, 30)]
        for short, long in zip(built, built[1:]):
            self.assertEqual(long[:len(short)], short)
        self.assertEqual(built[0], built[1])                      # the diff-0 turn sends the same ids

    def test_the_builders_are_deterministic_and_exact(self):
        head, pool = [1, 2, 3], list(range(50, 200))
        ids = P.build_chain_ids(head, pool, 64)
        self.assertEqual(len(ids), 64)
        self.assertEqual(ids[:3], head)
        self.assertEqual(ids, P.build_chain_ids(head, pool, 64))
        self.assertNotEqual(P.ids_sha256(ids), P.ids_sha256(P.build_chain_ids(head, pool, 65)))

    def test_a_prompt_too_small_for_the_tag_or_too_long_for_the_pool_is_refused(self):
        with self.assertRaises(SystemExit):
            P.build_chain_ids([1, 2, 3], list(range(10)), 3)
        with self.assertRaises(SystemExit):
            P.build_chain_ids([1], [1, 2], 10)

    def test_parse_diffs_keeps_the_order_and_rejects_bad_values(self):
        self.assertEqual(P.parse_diffs(" 0, 1 ,32"), [0, 1, 32])
        for bad in ("", " ", "0,x", "-1"):
            with self.assertRaises(SystemExit):
                P.parse_diffs(bad)


class PinTail(unittest.TestCase):
    """--chain-pin-tail: the request key pin=n-2, the largest value the engine honours."""

    def test_the_pin_is_the_largest_value_the_engine_honours(self):
        self.assertEqual(P.chain_pin(8192, True), 8190)
        self.assertLess(P.chain_pin(8192, True), 8192 - 1)     # pin >= n-1 is ignored by the engine
        self.assertIsNone(P.chain_pin(8192, False))
        self.assertIsNone(P.chain_pin(2, True))                # too short to pin


class CacheArgv(unittest.TestCase):
    """The chained arm's argv: caches on, every path private, the cold arm's argv unchanged."""

    BASE = {"args": ["--pack", "/m", "--kv", "int8", "--conversation-cache-disk", "/prod/conv",
                     "--conversation-cache-disk-gib", "25", "--conversation-cache-mib", "15360",
                     "--expert-profile-save", "/prod/profile.bin"],
            "gpu": [0, 1], "layer_split": 20}

    def chain(self, **kw):
        opts = {"prompt_cache": 6, "cache_every": 2048}
        opts.update(kw)
        return P.chain_engine_argv(self.BASE, "/tmp/engine", **opts)

    def test_the_chain_arm_turns_the_prompt_caches_on(self):
        argv, _ = self.chain()
        self.assertEqual(P.flag_value(argv, "--prompt-cache"), "6")
        self.assertEqual(P.flag_value(argv, "--prompt-cache-every"), "2048")
        self.assertEqual(P.flag_value(argv, "--prompt-cache-root"), "0")
        self.assertEqual(P.flag_count(argv, "--prompt-cache"), 1)

    def test_the_chain_arm_points_the_disk_store_at_a_private_directory(self):
        argv, isolated = self.chain(disk_dir="/out/conversations-arm")
        self.assertEqual(P.flag_value(argv, "--conversation-cache-disk"), "/out/conversations-arm")
        self.assertEqual(P.flag_value(argv, "--conversation-cache-disk-gib"), "25")
        self.assertEqual(isolated["disk_from"], "/prod/conv")
        self.assertEqual(isolated["disk_to"], "/out/conversations-arm")
        self.assertIsNone(isolated["disk_dropped"])
        self.assertNotIn("/prod/conv", argv)

    def test_the_chain_arm_mirrors_the_servers_own_disk_keys(self):
        # serve/server.py maps the two top-level keys to the flags; the chain arm keeps that shape
        cfg = {"args": ["--kv", "int8"], "gpu": [0, 1], "layer_split": 20,
               "conversation_cache_disk": "/prod/conv", "conversation_cache_disk_gib": 50,
               "conversation_cache_disk_slots": 8}
        argv, isolated = P.chain_engine_argv(cfg, "/tmp/engine", prompt_cache=6, cache_every=2048,
                                             disk_dir="/out/c")
        self.assertEqual(P.flag_value(argv, "--conversation-cache-disk"), "/out/c")
        self.assertEqual(P.flag_value(argv, "--conversation-cache-disk-gib"), "50")
        self.assertEqual(P.flag_value(argv, "--conversation-cache-disk-slots"), "8")
        self.assertEqual(isolated["disk_from"], "/prod/conv")
        self.assertNotIn("/prod/conv", argv)

    def test_a_disk_store_without_a_positive_budget_is_dropped(self):
        cfg = {"args": ["--conversation-cache-disk", "/prod/conv"], "gpu": [0, 1], "layer_split": 20}
        argv, isolated = P.chain_engine_argv(cfg, "/tmp/engine", prompt_cache=6, cache_every=2048,
                                             disk_dir="/out/c")
        self.assertNotIn("--conversation-cache-disk", argv)
        self.assertIsNone(isolated["disk_to"])
        self.assertTrue(isolated["disk_dropped"])

    def test_the_chain_arm_never_writes_a_profile(self):
        argv, isolated = self.chain()
        self.assertNotIn("--expert-profile-save", argv)
        self.assertEqual(isolated["dropped"], ["--expert-profile-save"])
        self.assertNotIn("/prod/profile.bin", argv)

    def test_the_chain_arm_can_drop_the_disk_tier(self):
        argv, isolated = self.chain(no_disk=True)
        self.assertNotIn("--conversation-cache-disk", argv)
        self.assertIsNone(isolated["disk_to"])
        self.assertEqual(isolated["disk_from"], "/prod/conv")
        self.assertTrue(isolated["disk_dropped"])

    def test_the_chain_arm_keeps_the_config_ram_budget_and_appends_the_knobs(self):
        argv, _ = self.chain(short_read=128, pipeline_windows=1, xargs=["--foo", "1"])
        self.assertEqual(P.flag_value(argv, "--conversation-cache-mib"), "15360")
        self.assertEqual(P.flag_value(argv, "--short-read"), "128")
        self.assertEqual(P.flag_value(argv, "--pipeline-windows"), "1")
        self.assertEqual(argv[-2:], ["--foo", "1"])

    def test_the_cold_arm_still_removes_the_caches_and_the_disk(self):
        argv = P.engine_argv(self.BASE, "/tmp/engine")
        self.assertEqual(P.flag_value(argv, "--prompt-cache"), "0")
        self.assertEqual(P.flag_value(argv, "--conversation-cache-mib"), "0")
        self.assertNotIn("--conversation-cache-disk", argv)

    def test_the_chain_arm_does_not_change_the_config(self):
        before = json.dumps(self.BASE, sort_keys=True)
        self.chain(disk_dir="/out/c")
        self.assertEqual(json.dumps(self.BASE, sort_keys=True), before)

    def test_xargs_splits_semicolons_and_spaces(self):
        self.assertEqual(P.parse_xargs(["--a 1;--b=2"]), ["--a", "1", "--b=2"])
        self.assertEqual(P.parse_xargs([]), [])

    def test_xargs_cannot_put_a_production_path_back_into_the_argv(self):
        # --xargs is appended last, so the private-path rule must be enforced on the FINAL argv
        argv, isolated = self.chain(disk_dir="/out/c",
                                    xargs=["--conversation-cache-disk", "/prod/conv2",
                                           "--expert-profile-save", "/prod/profile.bin"])
        self.assertEqual(P.flag_value(argv, "--conversation-cache-disk"), "/out/c")
        self.assertNotIn("/prod/conv2", argv)
        self.assertNotIn("--expert-profile-save", argv)
        self.assertNotIn("/prod/profile.bin", argv)
        self.assertEqual(len(isolated["xargs_sanitized"]), 2)

    def test_xargs_cannot_add_a_disk_store_to_an_arm_that_dropped_it(self):
        argv, isolated = self.chain(no_disk=True, xargs=["--conversation-cache-disk", "/prod/conv"])
        self.assertNotIn("--conversation-cache-disk", argv)
        self.assertIsNone(P.flag_value(argv, "--conversation-cache-disk-gib"))
        self.assertTrue(isolated["xargs_sanitized"])


class Settings(unittest.TestCase):
    """The settings check reads the value the engine keeps, and a divergence is reported beside it."""

    ENV = {"STRATA_EXPERT_PAIR": "1"}
    # The values REQUIRED_ARG_PAIRS demands, so this fixture can pass the whole check.
    ARGS = ["--kv", "int8", "--max-context", "524288", "--rope-scaling", "yarn", "--rope-scale", "2",
            "--spec", "8", "--vram-reserve-mib", "700", "--layer-split", "20", "--prefill", "auto",
            "--expert-cache", "auto", "--vision", "--mtp", "--expert-profile"]

    def argv(self, reserve: str = "700", extra: tuple = ()) -> list[str]:
        args = list(self.ARGS)
        args[args.index("--vram-reserve-mib") + 1] = reserve
        return P.engine_argv({"args": args + list(extra)}, "/tmp/e")

    def test_the_effective_value_is_the_last_occurrence(self):
        argv = ["--conversation-cache-mib", "15360", "--prompt-cache", "6",
                "--conversation-cache-mib", "0"]
        self.assertEqual(P.flag_value(argv, "--conversation-cache-mib"), "0")
        self.assertEqual(P.flag_value(["--prompt-cache=4"], "--prompt-cache"), "4")
        self.assertEqual(P.flag_count(argv, "--conversation-cache-mib"), 2)

    def test_the_cold_arm_reports_the_value_the_engine_keeps(self):
        # the config's 15360 then the cold arm's own 0: the engine keeps 0, so the check passes
        argv = self.argv(extra=("--conversation-cache-mib", "15360"))
        check = P.settings_check(argv, self.ENV)
        self.assertTrue(check["checks"]["conversation-cache-mib 0"])
        self.assertEqual(check["values"]["--conversation-cache-mib"], "0")

    def test_a_divergence_is_reported_next_to_its_check(self):
        argv = self.argv(reserve="750")        # the live config carries 750, the requirement is 700
        check = P.settings_check(argv, self.ENV)
        self.assertFalse(check["checks"]["--vram-reserve-mib 700"])
        self.assertEqual(check["values"]["--vram-reserve-mib"], "750")

    def test_the_chain_check_requires_the_caches_and_a_private_store(self):
        base = {"args": list(self.ARGS) + ["--conversation-cache-disk", "/prod/conv",
                                           "--conversation-cache-disk-gib", "25"], "gpu": [0, 1]}
        chain = {"prompt_cache": 6, "cache_every": 2048, "disk_dir": "/out/c",
                 "short_read": None, "pipeline_windows": None}
        argv, isolated = P.chain_engine_argv(base, "/tmp/e", prompt_cache=6, cache_every=2048,
                                             disk_dir="/out/c")
        self.assertEqual(isolated["disk_to"], "/out/c")
        self.assertTrue(P.chain_settings_check(argv, self.ENV, chain)["ok"])
        off = P.drop_flag(argv, "--prompt-cache") + ["--prompt-cache", "0"]
        self.assertFalse(P.chain_settings_check(off, self.ENV, chain)["checks"]["prompt cache on"])
        leaked = P.drop_flag(argv, "--conversation-cache-disk") + ["--conversation-cache-disk", "/prod/conv"]
        self.assertFalse(P.chain_settings_check(leaked, self.ENV, chain)["checks"]["no production disk store"])
        slow = P.drop_flag(argv, "--prompt-cache-every") + ["--prompt-cache-every", "8192"]
        self.assertFalse(P.chain_settings_check(slow, self.ENV, chain)["checks"]["prompt-cache-every 2048"])


class Accounting(unittest.TestCase):
    """The DONE fields and one chain turn's numbers: the engine's own counts, never the requested diff."""

    DEFAULTS = {"generated": 1, "prompt_tokens": 8225, "prompt_ms": 100.0, "decode_ms": 50.0,
                "finish": "length", "draft_accepted": 0, "draft_offered": 0, "reused": 8192,
                "hits": 1, "lookups": 2, "ram_blobs": 0, "file_blobs": 0, "file_mb": 0.0,
                "prompt_read": 33, "offloaded": 0}

    def fields(self, **kw):
        values = dict(self.DEFAULTS)
        values.update(kw)
        return [str(values[name]) for name in P.DONE_FIELDS if name in values]

    def row(self, turn=2, phase="chain", diff=32, prompt=8225, cache_every=2048, rnd=0, pin_tail=False,
            **kw):
        return P.chain_row_metrics(P.parse_done(self.fields(**kw)), round_index=rnd, turn=turn,
                                   phase=phase, requested_diff=diff, requested_prompt_tokens=prompt,
                                   cache_every=cache_every, pin_tail=pin_tail)

    def test_the_engine_counts_are_read_by_name(self):
        d = P.parse_done(self.fields())
        self.assertEqual(d["prompt_tokens"], 8225)
        self.assertEqual(d["reused"], 8192)
        self.assertEqual(d["prompt_read"], 33)
        self.assertEqual(d["finish"], "length")
        self.assertEqual(d["prompt_ms"], 100.0)

    def test_an_older_engine_that_stops_early_leaves_the_tail_absent(self):
        d = P.parse_done([str(v) for v in (1, 8192, 100.0, 50.0, "stop", 0, 0, 0)])
        self.assertEqual(d["reused"], 0)
        self.assertNotIn("prompt_read", d)

    def test_named_extensions_do_not_shift_prompt_accounting(self):
        fields = self.fields() + ["chunk=0", "experts_streamed=91"]
        done = P.parse_done(fields)
        row = P.chain_row_metrics(done, round_index=0, turn=1, phase="chain",
                                  requested_diff=32, requested_prompt_tokens=8225,
                                  cache_every=2048)
        self.assertEqual(row["prompt_read"], 33)
        self.assertEqual(row["reused"], 8192)
        self.assertEqual(row["prompt_tps"], 330.0)
        self.assertEqual(done["experts_streamed"], 91)

    def test_the_read_count_drives_the_rate_and_the_requested_diff_does_not(self):
        row = self.row()
        self.assertEqual(row["reused"], 8192)
        self.assertEqual(row["prompt_read"], 33)
        self.assertEqual(row["fresh_nominal"], 33)
        self.assertEqual(row["read_vs_requested"], 1)      # 33 read for a requested 32
        self.assertEqual(row["prompt_tps"], 330.0)         # 1000 * 33 / 100
        self.assertEqual(row["prompt_tps_nominal"], 330.0)
        self.assertEqual(row["decode_tps"], 20.0)          # 1000 * 1 / 50
        self.assertAlmostEqual(row["reuse_ratio"], 8192 / 8225)
        self.assertIsNone(row["error"])
        self.assertIsNone(row["note"])

    def test_the_diff_zero_turn_keeps_the_requested_zero_next_to_the_read(self):
        row = self.row(turn=1, diff=0, prompt=8192, prompt_tokens=8192, reused=6144, prompt_read=2048)
        self.assertEqual(row["requested_diff"], 0)
        self.assertEqual(row["prompt_read"], 2048)
        self.assertEqual(row["read_vs_requested"], 2048)
        self.assertTrue(row["note"])
        self.assertIsNone(row["error"])

    def test_a_turn_after_the_prefix_with_no_reuse_is_an_error(self):
        row = self.row(reused=0, prompt_read=8225)
        self.assertTrue(row["error"])
        self.assertEqual(row["reused"], 0)

    def test_the_first_request_must_be_cold(self):
        row = self.row(turn=0, phase="prefix", diff=8192, prompt=8192, prompt_tokens=8192,
                       reused=100, prompt_read=8092)
        self.assertTrue(row["error"])
        self.assertEqual(row["reused"], 100)

    def test_a_later_rounds_prefix_turn_is_a_warm_repeat(self):
        warm = self.row(turn=0, phase="prefix", diff=8192, prompt=8192, prompt_tokens=8192,
                        reused=6144, prompt_read=2048, rnd=1)
        self.assertIsNone(warm["error"])
        self.assertTrue(warm["note"])
        cold_again = self.row(turn=0, phase="prefix", diff=8192, prompt=8192, prompt_tokens=8192,
                              reused=0, prompt_read=8192, rnd=1)
        self.assertTrue(cold_again["error"])

    def test_a_prompt_the_engine_sized_differently_is_an_error(self):
        row = self.row(prompt_tokens=8224, reused=8192, prompt_read=32)
        self.assertTrue(row["error"])                 # the size mismatch is the primary error
        self.assertEqual(row["warnings"], [])

    def test_a_read_above_the_fresh_prompt_is_an_error(self):
        row = self.row(reused=8192, prompt_read=34)          # 33 fresh prompt tokens, 34 read
        self.assertTrue(row["error"])
        self.assertEqual(row["warnings"], [])

    def test_a_second_contradiction_becomes_a_warning(self):
        row = self.row(prompt_tokens=8200)                   # 8200 != 8225, and 33 read above 8 fresh
        self.assertTrue(row["error"])                 # the size error stays primary
        self.assertEqual(len(row["warnings"]), 1)     # the read contradiction is kept, as a warning

    def test_with_pin_tail_a_read_above_diff_plus_two_is_rejected(self):
        ok = self.row(pin_tail=True)                                  # diff 32, read 33 <= 34
        self.assertIsNone(ok["error"])
        late = self.row(pin_tail=True, prompt=8227, prompt_tokens=8227, reused=8192, prompt_read=35)
        self.assertTrue(late["error"])                                # 35 > 32 + 2: no pinned checkpoint

    def test_a_reuse_shortfall_beyond_one_interval_is_recorded_not_fatal(self):
        row = self.row(reused=4096, prompt_read=4129)
        self.assertTrue(row["warnings"])
        self.assertIsNone(row["error"])

    def test_the_summary_groups_the_turns_by_diff(self):
        rows = [self.row(turn=1, diff=0, prompt=8192, prompt_tokens=8192, reused=6144, prompt_read=2048,
                         prompt_ms=200.0),
                self.row(turn=2, diff=32, prompt=8224, prompt_tokens=8224, reused=8192, prompt_read=32,
                         prompt_ms=40.0),
                self.row(turn=3, diff=32, prompt=8225, prompt_tokens=8225, reused=8192, prompt_read=33,
                         prompt_ms=50.0)]
        for r in rows:
            r["output_ids"] = [7]
        summary = P.chain_summary(rows)
        self.assertEqual(summary["diff:32"]["turns"], 2)
        self.assertEqual(summary["diff:32"]["prompt_read_median"], 32.5)
        self.assertEqual(summary["diff:32"]["prompt_ms_median"], 45.0)
        self.assertEqual(summary["diff:0"]["turns"], 1)

    def test_a_warm_prefix_round_does_not_mix_with_the_cold_one(self):
        cold = self.row(turn=0, phase="prefix", diff=8192, prompt=8192, prompt_tokens=8192,
                        reused=0, prompt_read=8192, prompt_ms=5000.0)
        warm = self.row(turn=0, phase="prefix", diff=8192, prompt=8192, prompt_tokens=8192,
                        reused=8190, prompt_read=2, prompt_ms=2.0, rnd=1)
        for r in (cold, warm):
            r["output_ids"] = [7]
        summary = P.chain_summary([cold, warm])
        self.assertEqual(summary["prefix"]["turns"], 1)
        self.assertEqual(summary["prefix"]["prompt_read_median"], 8192)
        self.assertEqual(summary["prefix:r1"]["prompt_read_median"], 2)


class Compare(unittest.TestCase):
    """Two chained runs: the prompt hashes must match, and a different read count is a difference."""

    def result(self, label, **kw):
        row = {"round": 0, "turn": 2, "phase": "chain", "requested_diff": 32,
               "requested_prompt_tokens": 8224, "prompt_sha256": "h", "prompt_read": 32, "reused": 8192,
               "prompt_ms": 40.0, "prompt_tps": 800.0, "decode_tps": 20.0, "output_ids": [7]}
        row.update(kw)
        return {"mode": "chain", "arm_label": label, "build": {"engine_sha256": label}, "rows": [row]}

    def write(self, result, name):
        path = Path(self.home) / name
        path.write_text(json.dumps(result))
        return path

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_comparison_reports_the_engines_own_read_counts(self):
        out = P.compare_chain_results(self.result("a"), self.result("b"))
        self.assertTrue(out["read_parity"])
        self.assertEqual(out["turns"]["r0 t2 diff32"]["a_prompt_read"], 32)
        self.assertTrue(out["turns"]["r0 t2 diff32"]["output_parity"])

    def test_a_different_read_count_is_a_difference_not_parity(self):
        out = P.compare_chain_results(self.result("a"), self.result("b", prompt_read=33))
        self.assertFalse(out["read_parity"])
        self.assertEqual(out["turns"]["r0 t2 diff32"]["b_prompt_read"], 33)

    def test_a_prompt_hash_mismatch_is_collected(self):
        out = P.compare_chain_results(self.result("a"), self.result("b", prompt_sha256="x"))
        self.assertEqual(out["prompt_hash_mismatches"], ["r0 t2 diff32"])
        self.assertEqual(out["status"], "failed")
        self.assertIn("r0 t2 diff32", out["turns"])        # a mismatch is reported, not dropped

    def test_a_rejected_turn_is_reported_not_dropped(self):
        a = self.result("a", error="no prompt reuse after the prefix (reused=0)")
        out = P.compare_chain_results(a, self.result("b"))
        self.assertEqual(out["failed_turns"], ["r0 t2 diff32"])
        self.assertEqual(out["status"], "failed")
        self.assertIn("r0 t2 diff32", out["turns"])

    def test_different_plans_are_refused(self):
        with self.assertRaises(ValueError):
            P.compare_chain_results(self.result("a"), {"mode": "chain", "rows": []})

    def test_compare_results_dispatches_on_the_mode(self):
        a = self.write(self.result("a"), "a.json")
        b = self.write(self.result("b"), "b.json")
        self.assertEqual(P.compare_results(a, b)["mode"], "chain")

    def test_the_cold_comparison_is_unchanged(self):
        cold = {"arm_label": "a", "build": {"engine_sha256": "x"},
                "rows": [{"target": 2048, "repeat": 1, "prompt_sha256": "h", "prompt_ms": 10.0,
                          "prompt_tps": 100.0, "output_ids": [1]}]}
        other = json.loads(json.dumps(cold))
        other["arm_label"] = "b"
        out = P.compare_results(self.write(cold, "c.json"), self.write(other, "d.json"))
        self.assertEqual(sorted(out["targets"]), ["2048"])
        self.assertTrue(out["targets"]["2048"]["output_parity"])
        self.assertEqual(out["targets"]["2048"]["prompt_ms_change_pct"], 0.0)


class Sanitize(unittest.TestCase):
    def test_the_home_directory_is_scrubbed_from_the_result(self):
        home = str(Path.home())
        self.assertEqual(P.sanitize({"argv": [f"{home}/engine"], "n": 1}),
                         {"argv": ["$HOME/engine"], "n": 1})


if __name__ == "__main__":
    unittest.main()
