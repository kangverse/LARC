#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate ``question/<dataset>_full.jsonl`` files for every supported dataset.

Each line of the output JSONL is ``{question_id, image, text, category}``,
where ``text`` is the multiple-choice prompt built from the dataset config.

Usage:
    python question/generate_all.py --dataset emotion6
    python question/generate_all.py --dataset webemo25
    python question/generate_all.py --all

Paths are resolved relative to the project root by default. Override the
dataset root via the ``LARC_DATA_ROOT`` environment variable.
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from larc.dataset_config import (  # noqa: E402
    get_config, WEBEMO_25_TO_7, DatasetConfig,
)


BASE_DATA = os.environ.get("LARC_DATA_ROOT", "data")


def generate_emotion6(config: DatasetConfig):
    """Emotion6: ``images/<class>/<id>.jpg``"""
    records = []
    img_dir = os.path.join(BASE_DATA, "Emotion6", "images")
    for cls in config.classes:
        cls_dir = os.path.join(img_dir, cls)
        images = sorted(glob.glob(os.path.join(cls_dir, "*.jpg")))
        print(f"  {cls}: {len(images)} images")
        for img_path in images:
            fname = os.path.basename(img_path)
            records.append({
                "question_id": f"{cls}/{fname}",
                "image": f"{cls}/{fname}",
                "text": config.get_classify_prompt(),
                "category": cls,
            })
    return records


def generate_abstract(config: DatasetConfig):
    """Abstract: flat folder + ``ABSTRACT_groundTruth.csv``."""
    records = []
    data_dir = os.path.join(BASE_DATA, "testImages_abstract")
    csv_path = os.path.join(data_dir, "ABSTRACT_groundTruth.csv")
    gt_classes = ["amusement", "anger", "awe", "contentment",
                  "disgust", "excitement", "fear", "sad"]

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].strip():
                continue
            fname = row[0].strip().strip("'")
            if not fname.endswith(".jpg"):
                continue
            votes = [int(v.strip()) for v in row[1:9]]
            cat = gt_classes[votes.index(max(votes))]
            if cat in config.emo2letter:
                records.append({
                    "question_id": f"{cat}/{fname}",
                    "image": fname,
                    "text": config.get_classify_prompt(),
                    "category": cat,
                })

    print(f"  Abstract: {len(records)} images")
    return records


def generate_artphoto(config: DatasetConfig):
    """ArtPhoto: ``<class>_<id>.jpg`` in a flat folder."""
    records = []
    data_dir = os.path.join(BASE_DATA, "testImages_artphoto")
    images = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    for img_path in images:
        fname = os.path.basename(img_path)
        cat = fname.split("_")[0]
        if cat not in config.emo2letter:
            continue
        records.append({
            "question_id": f"{cat}/{fname}",
            "image": fname,
            "text": config.get_classify_prompt(),
            "category": cat,
        })
    print(f"  ArtPhoto: {len(records)} images")
    return records


def generate_webemo25(config: DatasetConfig):
    """WEBEmo 25-way: ``images/test25/<0..24>/<file>``."""
    records = []
    img_dir = os.path.join(BASE_DATA, "WEBEmo", "images", "test25")
    idx2cls = {i: c for i, c in enumerate(config.classes)}

    for idx in range(25):
        cls = idx2cls[idx]
        cls_dir = os.path.join(img_dir, str(idx))
        if not os.path.isdir(cls_dir):
            print(f"  [WARN] {cls_dir} not found, skipping")
            continue
        images = sorted(glob.glob(os.path.join(cls_dir, "*.jpg")))
        print(f"  [{idx:>2}] {cls}: {len(images)} images")
        for img_path in images:
            fname = os.path.basename(img_path)
            records.append({
                "question_id": f"{cls}/{fname}",
                "image": f"{idx}/{fname}",
                "text": config.get_classify_prompt(),
                "category": cls,
            })
    return records


def generate_webemo7(config: DatasetConfig):
    """WEBEmo 7-way: roll 25 categories into 7 via ``WEBEMO_25_TO_7``."""
    records = []
    img_dir = os.path.join(BASE_DATA, "WEBEmo", "images", "test25")
    classes_25 = [
        "affection", "cheerfullness", "confusion", "contentment",
        "disappointment", "disgust", "enthrallment", "envy",
        "exasperation", "gratitude", "horror", "irritabilty",
        "lust", "neglect", "nervousness", "optimism",
        "pride", "rage", "relief", "sadness",
        "shame", "suffering", "surprise", "sympathy", "zest",
    ]

    for idx, cls25 in enumerate(classes_25):
        cls7 = WEBEMO_25_TO_7.get(cls25)
        if cls7 is None or cls7 not in config.emo2letter:
            continue
        cls_dir = os.path.join(img_dir, str(idx))
        if not os.path.isdir(cls_dir):
            continue
        for img_path in sorted(glob.glob(os.path.join(cls_dir, "*.jpg"))):
            fname = os.path.basename(img_path)
            records.append({
                "question_id": f"{cls7}/{idx}_{fname}",
                "image": f"{idx}/{fname}",
                "text": config.get_classify_prompt(),
                "category": cls7,
            })

    counts = Counter(r["category"] for r in records)
    for cls in config.classes:
        print(f"  {cls}: {counts.get(cls, 0)} images")
    return records


def generate_emoset(config: DatasetConfig):
    """EmoSet-118K: ``EmoSet-118K/image/<class>/<file>``.

    If a pre-built ``emoset_118k_full.jsonl`` already exists next to the
    question-file output path, reuse it instead of rescanning the dataset.
    """
    output_path = _resolve_question_path(config)
    existing = os.path.join(os.path.dirname(output_path),
                            "emoset_118k_full.jsonl")
    if os.path.exists(existing):
        print(f"  EmoSet question file already exists: {existing}")
        records = []
        with open(existing, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        print(f"  {len(records)} records")
        return records

    records = []
    img_dir = os.path.join(BASE_DATA, "EmoSet-118K", "image")
    for cls in config.classes:
        cls_dir = os.path.join(img_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [WARN] {cls_dir} not found, skipping")
            continue
        images = sorted(glob.glob(os.path.join(cls_dir, "*.jpg")))
        print(f"  {cls}: {len(images)} images")
        for img_path in images:
            fname = os.path.basename(img_path)
            records.append({
                "question_id": f"{cls}/{fname}",
                "image": f"{cls}/{fname}",
                "text": config.get_classify_prompt(),
                "category": cls,
            })
    return records


GENERATORS = {
    "emotion6": generate_emotion6,
    "abstract": generate_abstract,
    "artphoto": generate_artphoto,
    "webemo25": generate_webemo25,
    "webemo7":  generate_webemo7,
    "emoset":   generate_emoset,
}


def _resolve_question_path(config: DatasetConfig) -> str:
    output = config.question_file
    if not os.path.isabs(output):
        output = os.path.join(os.path.dirname(__file__),
                              os.path.basename(output))
    return output


def generate_dataset(name: str):
    config = get_config(name)
    output = _resolve_question_path(config)

    print(f"\n[{name}] Generating {config.num_classes}-class question file...")
    print(f"  Output: {output}")

    gen_fn = GENERATORS.get(name)
    if gen_fn is None:
        print(f"  [ERROR] No generator for {name}")
        return

    records = gen_fn(config)
    records.sort(key=lambda x: x["question_id"])

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Saved {len(records)} records to {output}")


def main():
    p = argparse.ArgumentParser(description="Generate question JSONL files.")
    p.add_argument("--dataset", type=str, default="",
                   help="Dataset name")
    p.add_argument("--all", action="store_true",
                   help="Generate question files for all supported datasets")
    args = p.parse_args()

    if args.all:
        for name in GENERATORS:
            generate_dataset(name)
    elif args.dataset:
        generate_dataset(args.dataset)
    else:
        print("Specify --dataset <name> or --all")
        print(f"Available: {', '.join(GENERATORS.keys())}")


if __name__ == "__main__":
    main()
