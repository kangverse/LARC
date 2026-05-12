#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stand-alone scorer for LARC answer files.

Reports per-class accuracy, macro average, and flip statistics
(W -> R and R -> W) against the raw logit baseline written in the same
records. Accepts multiple shard files and de-duplicates them by
``question_id``.

Usage:
    python eval_larc.py answer/<model>/larc_emoset.jsonl
    python eval_larc.py answer/<model>/larc_emoset_shard*.jsonl
    python eval_larc.py --dataset emoset answer/<model>/larc_emoset.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import Dict, List

from larc.dataset_config import get_config, list_datasets


# Backward-compatible default for the 8-way EmoSet taxonomy.
_DEFAULT_CAT_MAP: Dict[str, str] = {
    "amusement": "A", "anger": "B", "awe": "C", "contentment": "D",
    "disgust": "E", "excitement": "F", "fear": "G", "sadness": "H",
}


def _load_records(paths: List[str]) -> List[Dict]:
    records: Dict[str, Dict] = {}
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    qid = rec.get("question_id")
                    if qid:
                        records[qid] = rec
                except json.JSONDecodeError:
                    pass
    return list(records.values())


def _evaluate(records: List[Dict], cat_map: Dict[str, str]):
    stats = defaultdict(lambda: {
        "total": 0, "baseline_correct": 0, "larc_correct": 0,
        "flip_ok": 0, "flip_bad": 0, "triggered": 0,
    })

    for rec in records:
        qid = rec["question_id"]
        cat = rec.get("category", "")
        if cat not in cat_map:
            cat = qid.split("/")[0]
        if cat not in cat_map:
            continue

        gt = cat_map[cat]
        larc_pred = rec.get("text", "?")
        logit_pred = rec.get("logit_prediction", larc_pred)
        triggered = rec.get("triggered", False)

        d = stats[cat]
        d["total"] += 1
        if logit_pred == gt:
            d["baseline_correct"] += 1
        if larc_pred == gt:
            d["larc_correct"] += 1
        if triggered:
            d["triggered"] += 1
        if logit_pred != gt and larc_pred == gt:
            d["flip_ok"] += 1
        elif logit_pred == gt and larc_pred != gt:
            d["flip_bad"] += 1
    return stats


def _print(stats):
    print(f"\n{'='*75}")
    print(f"{'Category':>12} | {'N':>5} | {'Baseline':>10} | {'LARC':>10} | "
          f"{'Δ':>5} | {'FlipOK':>6} | {'FlipBAD':>7} | {'Trig%':>6}")
    print("-" * 75)

    total = {"n": 0, "bl": 0, "larc": 0, "fok": 0, "fbad": 0, "trig": 0}
    for cat in sorted(stats.keys()):
        d = stats[cat]
        n, bl, lt = d["total"], d["baseline_correct"], d["larc_correct"]
        fok, fbad, trig = d["flip_ok"], d["flip_bad"], d["triggered"]
        delta = lt - bl
        total["n"] += n; total["bl"] += bl; total["larc"] += lt
        total["fok"] += fok; total["fbad"] += fbad; total["trig"] += trig

        bl_pct = bl / n * 100 if n else 0
        lt_pct = lt / n * 100 if n else 0
        trig_pct = trig / n * 100 if n else 0
        sign = "+" if delta > 0 else ""
        print(f"{cat:>12} | {n:>5} | {bl:>4} ({bl_pct:4.1f}%) | "
              f"{lt:>4} ({lt_pct:4.1f}%) | {sign}{delta:>4} | "
              f"{fok:>6} | {fbad:>7} | {trig_pct:5.1f}%")

    n = total["n"]
    bl = total["bl"]
    lt = total["larc"]
    delta = lt - bl
    bl_pct = bl / n * 100 if n else 0
    lt_pct = lt / n * 100 if n else 0
    trig_pct = total["trig"] / n * 100 if n else 0
    sign = "+" if delta > 0 else ""
    print("-" * 75)
    print(f"{'TOTAL':>12} | {n:>5} | {bl:>4} ({bl_pct:4.1f}%) | "
          f"{lt:>4} ({lt_pct:4.1f}%) | {sign}{delta:>4} | "
          f"{total['fok']:>6} | {total['fbad']:>7} | {trig_pct:5.1f}%")

    cats = list(stats.keys())
    if cats:
        macro_bl = sum(stats[c]["baseline_correct"] / max(stats[c]["total"], 1)
                       for c in cats) / len(cats) * 100
        macro_larc = sum(stats[c]["larc_correct"] / max(stats[c]["total"], 1)
                         for c in cats) / len(cats) * 100
        print(f"\n  Macro-Avg Baseline: {macro_bl:.2f}%")
        print(f"  Macro-Avg LARC:     {macro_larc:.2f}%")
        sign = "+" if macro_larc >= macro_bl else ""
        print(f"  Macro-Avg Δ:        {sign}{macro_larc - macro_bl:.2f}%")

    fok, fbad = total["fok"], total["fbad"]
    print(f"\n  Flip OK: {fok}, Flip BAD: {fbad}, Net: {fok - fbad}, "
          f"Ratio: {fok / max(fbad, 1):.1f}:1")
    print(f"  Total samples: {n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+",
                        help="Answer JSONL files (multiple shards are merged)")
    parser.add_argument("--dataset", type=str, default="",
                        choices=[""] + list_datasets(),
                        help="Use the category map of this dataset "
                             "(default: infer from the file or fall back to 8-way EmoSet)")
    args = parser.parse_args()

    cat_map = _DEFAULT_CAT_MAP
    if args.dataset:
        cat_map = get_config(args.dataset).emo2letter

    records = _load_records(args.files)
    print(f"Loaded {len(records)} records from {len(args.files)} file(s)")
    stats = _evaluate(records, cat_map)
    _print(stats)


if __name__ == "__main__":
    main()
