#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified LARC evaluation entry-point.

Run LARC on any supported emotion-recognition dataset. The script:
    1. Loads the question file, splits it into a small per-class calibration
       set and a (full) test set.
    2. Extracts the ASO directions on the calibration set, caching them to
       disk so re-runs are cheap.
    3. Runs the LARC inference pipeline on the test set with optional
       multi-GPU sharding and resume-from-checkpoint support.
    4. Prints a per-class accuracy table comparing the raw logit baseline
       against LARC.

Examples:
    # Single GPU, Emotion6
    CUDA_VISIBLE_DEVICES=0 python run_larc.py --dataset emotion6

    # 2-way shard on EmoSet
    CUDA_VISIBLE_DEVICES=0 python run_larc.py --dataset emoset --shard 0 --num_shards 2
    CUDA_VISIBLE_DEVICES=1 python run_larc.py --dataset emoset --shard 1 --num_shards 2

    # Resume an interrupted run
    CUDA_VISIBLE_DEVICES=0 python run_larc.py --dataset emoset --resume

Output layout:
    answer/{model_short}/larc_{dataset}.jsonl
    concept_cache/{model_short}/aso_{dataset}_n{n_calib}_l{layer_ratio}.json
    log/{dataset}/larc_{dataset}_{model_short}_{timestamp}.log
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(__file__))

from larc.vlm.builder import build_vlm, infer_model_name, get_model_short_name
from larc.aso import extract_aso, AffectiveSemanticOrientations
from larc.inference import larc_inference, NEUTRAL_PROBE_PROMPT  # noqa: F401
from larc.data import load_questions, load_image
from larc.dataset_config import get_config, list_datasets, DatasetConfig


# ============================================================
# Logging tee
# ============================================================

class _Tee:
    """Mirror stdout/stderr into a log file."""

    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._file = open(log_path, "a", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def _setup_logging(dataset: str, model_short: str,
                   shard: int, num_shards: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shard_suffix = f"_shard{shard}" if num_shards > 1 else ""
    log_name = f"larc_{dataset}_{model_short}_{ts}{shard_suffix}.log"
    log_path = os.path.join("log", dataset, log_name)
    tee = _Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee
    return log_path


# ============================================================
# Arg parsing
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="LARC unified evaluation")
    # Dataset
    p.add_argument("--dataset", type=str, required=True,
                   choices=list_datasets(),
                   help="Dataset name. Available: " + ", ".join(list_datasets()))
    # Model
    p.add_argument("--model_name", type=str, default="",
                   help="Model family (auto-inferred from --model_path if empty)")
    p.add_argument("--model_path", type=str,
                   default=os.environ.get(
                       "LARC_LLAVA_PATH",
                       "models/llava-v1.5-7b",
                   ))
    # IO overrides
    p.add_argument("--image_folder", type=str, default="",
                   help="Image root directory (defaults to dataset config)")
    p.add_argument("--question_file", type=str, default="",
                   help="Question JSONL path (defaults to dataset config)")
    p.add_argument("--answer_file", type=str, default="",
                   help="Output JSONL (defaults to answer/{model}/larc_{dataset}.jsonl)")
    # Calibration
    p.add_argument("--n_calib", type=int, default=20,
                   help="Number of calibration images per class")
    p.add_argument("--concept_cache", type=str, default="",
                   help="ASO cache path (auto-named if empty)")
    # Sharding
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    # Resume
    p.add_argument("--resume", action="store_true",
                   help="Skip already-processed question_ids")
    # LARC inference hyper-parameters
    p.add_argument("--temperature", type=float, default=0.05,
                   help="Latent softmax temperature tau")
    p.add_argument("--layer_ratio", type=float, default=0.65,
                   help="Target intermediate layer as a fraction of depth")
    p.add_argument("--margin_guard", type=float, default=0.15,
                   help="Protect logit top-1 if its top1-top2 margin exceeds this")
    p.add_argument("--geo_sigma", type=float, default=0.3,
                   help="RAER temperature (smaller -> stronger relation routing)")
    p.add_argument("--eta", type=float, default=0.0,
                   help="RAER residual smoothing (uniform floor)")
    p.add_argument("--fusion_mode", type=str, default="product",
                   choices=["product", "weighted"],
                   help="RGEM fusion form. 'product' matches the paper.")
    p.add_argument("--logit_exponent", type=float, default=1.0,
                   help="Exponent on P_logit in product fusion")
    p.add_argument("--topo_weight", type=float, default=0.3,
                   help="ASO mixing weight in weighted fusion")
    p.add_argument("--no_entropy_adaptive", action="store_true",
                   help="Disable entropy-adaptive logit exponent")
    p.add_argument("--no_auto_temperature", action="store_true",
                   help="Disable temperature scaling by class count")
    # Evaluation only
    p.add_argument("--eval_only", action="store_true",
                   help="Skip inference and only score an existing answer file")
    return p.parse_args()


# ============================================================
# Evaluation
# ============================================================

def evaluate_results(answer_file: str, config: DatasetConfig):
    """Compute per-class and macro accuracy from a JSONL answer file."""
    if not os.path.exists(answer_file):
        print(f"Answer file not found: {answer_file}")
        return

    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_correct_logit = defaultdict(int)
    triggered_count = 0
    total = 0

    emo2letter = config.emo2letter
    letter2emo = config.letter2emo

    with open(answer_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec["question_id"]

            cat = rec.get("category", "")
            if cat not in emo2letter:
                cat = qid.split("/")[0]
            gt_letter = emo2letter.get(cat)
            if gt_letter is None:
                continue

            pred = rec.get("text", "")
            logit_pred = rec.get("logit_prediction", "")

            class_total[gt_letter] += 1
            total += 1
            if pred == gt_letter:
                class_correct[gt_letter] += 1
            if logit_pred == gt_letter:
                class_correct_logit[gt_letter] += 1
            if rec.get("triggered", False):
                triggered_count += 1

    if total == 0:
        print("No valid records found!")
        return

    print(f"\n{'='*72}")
    print(f"  LARC Evaluation: {config.name} ({config.num_classes} classes)")
    print(f"  Total samples: {total}")
    print(f"  Triggered: {triggered_count} ({triggered_count/total*100:.1f}%)")
    print(f"{'='*72}")
    print(f"  {'Class':>15} | {'Letter':>6} | {'N':>6} | "
          f"{'Logit%':>7} | {'LARC%':>7} | {'Delta':>7}")
    print(f"  {'-'*15}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

    accs_logit, accs_larc = [], []
    for letter in config.labels:
        emo = letter2emo[letter]
        n = class_total.get(letter, 0)
        if n == 0:
            continue
        acc_logit = class_correct_logit.get(letter, 0) / n * 100
        acc_larc = class_correct.get(letter, 0) / n * 100
        delta = acc_larc - acc_logit
        accs_logit.append(acc_logit)
        accs_larc.append(acc_larc)
        sign = "+" if delta >= 0 else ""
        print(f"  {emo:>15} |    {letter:>1}   | {n:>5} | "
              f"{acc_logit:>6.1f} | {acc_larc:>6.1f} | {sign}{delta:>5.1f}")

    macro_logit = sum(accs_logit) / len(accs_logit) if accs_logit else 0
    macro_larc = sum(accs_larc) / len(accs_larc) if accs_larc else 0
    delta_macro = macro_larc - macro_logit
    sign = "+" if delta_macro >= 0 else ""

    print(f"  {'-'*15}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    print(f"  {'Macro Avg':>15} |        | {total:>5} | "
          f"{macro_logit:>6.1f} | {macro_larc:>6.1f} | {sign}{delta_macro:>5.1f}")
    print(f"{'='*72}\n")


# ============================================================
# Shard merging
# ============================================================

def _merge_shards(base_file: str, num_shards: int) -> str:
    """Concatenate shard files into ``base_file`` (deduped by question_id).
    Returns an empty string if any shard is still missing."""
    base, ext = os.path.splitext(base_file)
    shard_files = [f"{base}_shard{i}{ext}" for i in range(num_shards)]

    missing = [f for f in shard_files if not os.path.exists(f)]
    if missing:
        print(f"[Merge] Waiting for {len(missing)} shard(s): "
              + ", ".join(os.path.basename(f) for f in missing))
        return ""

    seen_ids = set()
    records = []
    for path in shard_files:
        n_shard = 0
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    qid = rec.get("question_id", "")
                    if qid not in seen_ids:
                        seen_ids.add(qid)
                        records.append(line)
                        n_shard += 1
                except json.JSONDecodeError:
                    pass
        print(f"  [Merge] {os.path.basename(path)}: {n_shard} records")

    os.makedirs(os.path.dirname(base_file) or ".", exist_ok=True)
    with open(base_file, "w") as f:
        for r in records:
            f.write(r + "\n")

    print(f"  [Merge] Total: {len(records)} records -> {base_file}")
    return base_file


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    model_name = args.model_name or infer_model_name(args.model_path)
    model_short = get_model_short_name(args.model_path)

    log_path = _setup_logging(args.dataset, model_short,
                              args.shard, args.num_shards)

    print(f"\n{'='*60}")
    print("  LARC Evaluation")
    print(f"  Model:   {model_short}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Log:     {log_path}")
    print(f"{'='*60}")

    config = get_config(args.dataset)
    print(f"\n[Model]   {model_short} (type={model_name})")
    print(f"[Dataset] {config.name}: {config.num_classes} classes")
    print(f"  Classes: {', '.join(config.classes)}")
    print(f"  Labels:  {config.labels}")

    image_folder = args.image_folder or config.image_folder
    question_file = args.question_file or config.question_file
    answer_file = args.answer_file or f"answer/{model_short}/larc_{config.name}.jsonl"

    if args.num_shards > 1:
        base, ext = os.path.splitext(answer_file)
        answer_file = f"{base}_shard{args.shard}{ext}"

    if args.eval_only:
        if args.num_shards > 1:
            base_file = args.answer_file or f"answer/{model_short}/larc_{config.name}.jsonl"
            merged = _merge_shards(base_file, args.num_shards)
            if merged:
                evaluate_results(merged, config)
        else:
            evaluate_results(answer_file, config)
        return

    os.makedirs(os.path.dirname(answer_file) or ".", exist_ok=True)

    # ---------- 1. Load questions ----------
    print(f"\nLoading questions from {question_file}")
    questions = load_questions(question_file)
    print(f"Total questions: {len(questions)}")

    emo2letter = config.emo2letter

    by_cat = defaultdict(list)
    for q in questions:
        cat = q.get("category", "")
        if cat not in emo2letter:
            cat = q["question_id"].split("/")[0]
        if cat in emo2letter:
            by_cat[cat].append(q)

    # The calibration set provides class-conditional statistics for ASO; it is
    # not removed from the test pool, matching the protocol used by other
    # training-free baselines for comparability.
    test_questions = []
    class_images_for_calib = {}

    for cat in sorted(emo2letter.keys()):
        letter = emo2letter[cat]
        qs = by_cat[cat]
        n_calib = min(args.n_calib, len(qs))
        calib_imgs = []
        for q in qs[:n_calib]:
            img = load_image(image_folder, q["image"])
            calib_imgs.append(img)
        class_images_for_calib[letter] = calib_imgs
        test_questions.extend(qs)

    print(f"Calibration: {sum(len(v) for v in class_images_for_calib.values())} "
          f"({args.n_calib}/class)")
    print(f"Test pool: {len(test_questions)} (full dataset)")

    test_questions.sort(key=lambda q: q["question_id"])
    shard_size = len(test_questions) // args.num_shards
    start = args.shard * shard_size
    end = (start + shard_size if args.shard < args.num_shards - 1
           else len(test_questions))
    shard_questions = test_questions[start:end]
    print(f"Shard {args.shard}/{args.num_shards}: "
          f"samples {start}-{end} ({len(shard_questions)})")

    processed_ids = set()
    if args.resume and os.path.exists(answer_file):
        with open(answer_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        processed_ids.add(rec["question_id"])
                    except json.JSONDecodeError:
                        pass
        print(f"Resume: {len(processed_ids)} already processed")

    remaining = [q for q in shard_questions if q["question_id"] not in processed_ids]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All samples processed.")
        evaluate_results(answer_file, config)
        return

    # ---------- 2. Load model ----------
    print(f"\nLoading model: {model_short} ...")
    vlm = build_vlm(model_name, args.model_path, "cuda")
    print(f"Model loaded. layers={vlm.get_num_layers()}, "
          f"hidden={vlm.get_hidden_size()}")

    # ---------- 3. ASO extraction (with disk cache) ----------
    concept_cache = (args.concept_cache or
                     f"concept_cache/{model_short}/"
                     f"aso_{config.name}_n{args.n_calib}_l{args.layer_ratio}.json")
    aso: Optional[AffectiveSemanticOrientations] = None

    if os.path.exists(concept_cache):
        print(f"\nLoading cached ASO from {concept_cache}")
        aso = AffectiveSemanticOrientations.load(concept_cache)
        expected_n = args.n_calib * config.num_classes
        if aso.num_calibration_images != expected_n:
            print(f"  Cache mismatch (cache={aso.num_calibration_images}, "
                  f"expected={expected_n}). Re-extracting.")
            aso = None

    if aso is None:
        print(f"\nExtracting ASO directions ({args.n_calib}/class)...")
        layer_idx = int(vlm.get_num_layers() * args.layer_ratio)
        aso = extract_aso(
            vlm, class_images_for_calib, layer_idx=layer_idx,
            letter2emo=config.letter2emo,
        )
        aso.save(concept_cache)

    print(f"ASO ready (layer={aso.layer_idx}, classes={len(aso.vectors)})")
    sim_matrix = aso.similarity_matrix()
    print("RAER relation matrix computed from ASO similarities.")

    # ---------- 4. Inference ----------
    print(f"\n{'='*60}")
    print(f"Starting LARC inference on {len(remaining)} samples")
    print(f"  model={model_short}, dataset={config.name}, "
          f"classes={config.num_classes}")
    print(f"  temperature={args.temperature}, margin_guard={args.margin_guard}, "
          f"geo_sigma={args.geo_sigma}, eta={args.eta}")
    print(f"  fusion_mode={args.fusion_mode}, "
          f"logit_exponent={args.logit_exponent}, topo_weight={args.topo_weight}")
    print(f"  entropy_adaptive={not args.no_entropy_adaptive}, "
          f"auto_temperature={not args.no_auto_temperature}")
    print(f"  answer_file={answer_file}")
    print(f"{'='*60}\n")

    t0 = time.time()
    n_done = 0
    n_error = 0
    n_triggered = 0

    with open(answer_file, "a") as fout:
        for q in remaining:
            qid = q["question_id"]
            cat = q.get("category", "")
            if cat not in emo2letter:
                cat = qid.split("/")[0]

            try:
                image = load_image(image_folder, q["image"])
                result = larc_inference(
                    vlm, aso, image,
                    config=config,
                    similarity_matrix=sim_matrix,
                    geo_sigma=args.geo_sigma,
                    eta=args.eta,
                    temperature=args.temperature,
                    auto_temperature=not args.no_auto_temperature,
                    fusion_mode=args.fusion_mode,
                    logit_exponent=args.logit_exponent,
                    topo_weight=args.topo_weight,
                    margin_guard=args.margin_guard,
                    entropy_adaptive=not args.no_entropy_adaptive,
                )

                gt_letter = emo2letter.get(cat, "?")
                record = {
                    "question_id": qid,
                    "dataset": config.name,
                    "category": cat,
                    "gt_letter": gt_letter,
                    "text": result["prediction"],
                    "logit_prediction": result["logit_prediction"],
                    "logit_confidence": round(result["logit_confidence"], 4),
                    "topo_prediction": result["topo_prediction"],
                    "topo_confidence": round(result["topo_confidence"], 4),
                    "triggered": result["triggered"],
                    "confidence": round(result["confidence"], 4),
                    "model_id": model_short,
                }
                if result["triggered"]:
                    n_triggered += 1

            except Exception as e:
                gt_letter = emo2letter.get(cat, "?")
                record = {
                    "question_id": qid,
                    "dataset": config.name,
                    "category": cat,
                    "gt_letter": gt_letter,
                    "text": "A",
                    "model_id": model_short,
                    "error": str(e),
                }
                n_error += 1
                print(f"[ERROR] {qid}: {e}")

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_done += 1

            if n_done % 50 == 0 or n_done == len(remaining):
                elapsed = time.time() - t0
                speed = n_done / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - n_done) / speed if speed > 0 else 0
                pred = record.get("text", "?")
                logit_p = record.get("logit_prediction", "?")
                topo_p = record.get("topo_prediction", "?")
                trig = "T" if record.get("triggered") else "."
                print(f"[{n_done:>5}/{len(remaining)}] {qid[:30]:>30} "
                      f"logit={logit_p} topo={topo_p} -> {pred} {trig} "
                      f"| {speed:.1f} samp/s, ETA {eta/60:.1f}min "
                      f"| triggered={n_triggered} err={n_error}")
                fout.flush()

            if n_done % 100 == 0:
                torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done! Shard {args.shard}/{args.num_shards}")
    print(f"  Processed: {n_done}")
    print(f"  Triggered: {n_triggered} ({n_triggered/max(n_done,1)*100:.1f}%)")
    print(f"  Errors: {n_error}")
    print(f"  Time: {elapsed/60:.1f} min ({n_done/elapsed:.1f} samp/s)")
    print(f"  Output: {answer_file}")

    if args.num_shards > 1:
        base_file = args.answer_file or f"answer/{model_short}/larc_{config.name}.jsonl"
        merged = _merge_shards(base_file, args.num_shards)
        if merged:
            print("\nAll shards complete. Evaluating merged results:")
            evaluate_results(merged, config)
        else:
            print("\nOther shards not yet complete. Evaluating current shard only:")
            evaluate_results(answer_file, config)
    else:
        evaluate_results(answer_file, config)


if __name__ == "__main__":
    main()
