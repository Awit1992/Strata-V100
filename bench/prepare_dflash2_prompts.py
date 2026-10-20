#!/usr/bin/env python3
"""Prepare the DFlash2 acceptance cohort from the public benchmark datasets.

Source (pinned): https://github.com/z-lab/dflash/blob/265add3a0e4c62d02566dc44ad9a735f350084c2/dflash/benchmark.py
Revision: 265add3a0e4c62d02566dc44ad9a735f350084c2

SELECTION (the reference's _run_openai at concurrency 1)
  For every dataset, in the reference's order:
    order = list(range(num_rows_total)); random.Random(42).shuffle(order)
    selection[i] = order[i % num_rows_total]  for i in 0 .. num_prompts
  The first num_prompts entries are the measured cohort. The last entry is the single
  warmup request. The reference uses turns[0] only, even for MT-Bench. So a record's
  prompt is the first formatted turn. A short corpus cycles: MT-Bench has 80 rows, so
  the cohort wraps to the first rows of the shuffled order.

FORMAT (copied from the pinned reference's DATASETS table, applied to the row's fields)
  gsm8k      openai/gsm8k                     main     test   "{question}\\nPlease reason step by step, ... \\boxed{{}}."
  math500    HuggingFaceH4/MATH-500           default  test   "{problem}\\nPlease reason step by step, ... \\boxed{{}}."
  humaneval  openai/openai_humaneval          openai_humaneval test  "Write a solution to the following problem ... ```python\n{prompt}\n```"
  mbpp       google-research-datasets/mbpp    sanitized test  the row's "prompt"
  mt-bench   HuggingFaceH4/mt_bench_prompts   default  train  the row's "prompt" list, first turn

ROWS
  Rows come from the public datasets-server rows API (no dataset library, no login).
  Page 0 gives the authoritative num_rows_total. The program fetches only the 100-row
  pages that hold a selected index, and caches those pages in memory. It writes nothing
  else. There are no retries and no on-disk cache: a bad HTTP status fails the run.

OUTPUT
  A JSON array of records. Each record is:
    {"dataset": label, "row": selection ordinal, "source_row": corpus index,
     "turns": [first formatted prompt]}
  Each dataset's single warmup record comes first in that dataset's block:
    {"dataset": label, "row": num_prompts, "source_row": corpus index,
     "turns": [first formatted prompt], "warmup": true}
  The file holds public dataset prompts only. Do not commit it.

Usage:
  python bench/prepare_dflash2_prompts.py --out PROMPTS_JSON
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import requests

ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE = 100                     # the rows API page size this program requests and caches
NUM_PROMPTS = 128              # the model card's measured cohort at concurrency one
SHUFFLE_SEED = 42              # the reference's random.Random(42) selection seed

# The reference's DATASETS table. `turns` returns the reference's own turns list: the
# formatted string for a single-turn dataset, or the row's prompt list for MT-Bench.
# The API names HumanEval's implicit default builder "openai_humaneval".
DATASETS = {
    "gsm8k": {
        "dataset": "openai/gsm8k", "config": "main", "split": "test",
        "turns": lambda row: [
            "{question}\nPlease reason step by step, and put your final answer within "
            "\\boxed{{}}.".format(**row)],
    },
    "math500": {
        "dataset": "HuggingFaceH4/MATH-500", "config": "default", "split": "test",
        "turns": lambda row: [
            "{problem}\nPlease reason step by step, and put your final answer within "
            "\\boxed{{}}.".format(**row)],
    },
    "humaneval": {
        "dataset": "openai/openai_humaneval", "config": "openai_humaneval", "split": "test",
        "turns": lambda row: [
            "Write a solution to the following problem and make sure that it passes the "
            "tests:\n```python\n{prompt}\n```".format(**row)],
    },
    "mbpp": {
        "dataset": "google-research-datasets/mbpp", "config": "sanitized", "split": "test",
        "turns": lambda row: [row["prompt"]],
    },
    "mt-bench": {
        "dataset": "HuggingFaceH4/mt_bench_prompts", "config": "default", "split": "train",
        "turns": lambda row: row["prompt"],       # a list of turn strings
    },
}


class DatasetRows:
    """The rows API for one dataset: 100-row pages, fetched once each and held in memory."""

    def __init__(self, session: requests.Session, spec: dict):
        self.session = session
        self.spec = spec
        self.pages: dict[int, dict[int, dict]] = {}
        self.num_rows_total: int | None = None

    def page(self, offset: int) -> dict[int, dict]:
        if offset not in self.pages:
            r = self.session.get(ROWS_API, timeout=120, params={
                "dataset": self.spec["dataset"], "config": self.spec["config"],
                "split": self.spec["split"], "offset": offset, "length": PAGE})
            r.raise_for_status()
            body = r.json()
            if offset == 0:
                self.num_rows_total = body.get("num_rows_total")
            self.pages[offset] = {entry["row_idx"]: entry["row"] for entry in body.get("rows", [])}
        return self.pages[offset]

    def row(self, index: int) -> dict:
        rows = self.page((index // PAGE) * PAGE)
        if index not in rows:
            raise SystemExit(f"{self.spec['dataset']}: no row {index} on page {index // PAGE * PAGE}")
        return rows[index]


def record(dataset: str, spec: dict, rows: DatasetRows, ordinal: int, source: int,
           warmup: bool) -> dict:
    """One output record: the reference's turns[0] for the row at `source`."""
    turns = spec["turns"](rows.row(source))
    if not isinstance(turns, list) or not turns or not isinstance(turns[0], str):
        raise SystemExit(f"{dataset}: row {source} has no first turn")
    out = {"dataset": dataset, "row": ordinal, "source_row": source, "turns": [turns[0]]}
    if warmup:
        out["warmup"] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="output JSON file")
    ap.add_argument("--num-prompts", type=int, default=NUM_PROMPTS,
                    help=f"measured prompts per dataset (default: {NUM_PROMPTS})")
    ap.add_argument("--shuffle-seed", type=int, default=SHUFFLE_SEED,
                    help=f"selection shuffle seed (default: {SHUFFLE_SEED})")
    args = ap.parse_args()
    if args.num_prompts <= 0:
        raise SystemExit("--num-prompts must be positive")

    session = requests.Session()
    records = []
    for dataset, spec in DATASETS.items():
        rows = DatasetRows(session, spec)
        rows.page(0)
        total = rows.num_rows_total
        if not isinstance(total, int) or total <= 0:
            raise SystemExit(f"{dataset}: the rows API reported no num_rows_total")
        order = list(range(total))
        random.Random(args.shuffle_seed).shuffle(order)
        selection = [order[i % total] for i in range(args.num_prompts + 1)]
        # The dataset's warmup record comes first, then its measured cohort.
        records.append(record(dataset, spec, rows, args.num_prompts, selection[args.num_prompts], True))
        for ordinal in range(args.num_prompts):
            records.append(record(dataset, spec, rows, ordinal, selection[ordinal], False))
        print(f"{dataset}: {total} rows, {args.num_prompts} measured + 1 warmup", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} prompts to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
