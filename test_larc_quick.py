# -*- coding: utf-8 -*-
"""
Quick LARC smoke test.

Loads LLaVA-v1.5, extracts ASO directions from a small number of EmoSet
calibration images, then runs LARC on a held-out subset to verify the
end-to-end pipeline. Prints the per-class accuracy and flip statistics.

Edit ``MODEL_PATH``, ``IMAGE_FOLDER`` and ``QUESTION_FILE`` to match your
local setup before running.
"""

import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(__file__))

from larc.vlm.builder import build_vlm
from larc.aso import extract_aso
from larc.inference import larc_inference
from larc.dataset_config import get_config
from larc.data import load_questions, load_image


# -------------------- user configuration --------------------
MODEL_PATH = os.environ.get("LARC_LLAVA_PATH", "models/llava-v1.5-7b")
DATASET = "emoset"  # any name registered in larc.dataset_config

N_CALIB_PER_CLASS = 20
N_TEST_PER_CLASS = 100

# ------------------------------------------------------------

print("Loading model...")
vlm = build_vlm("llava", MODEL_PATH, "cuda")
print(f"Model loaded. layers={vlm.get_num_layers()}, "
      f"hidden={vlm.get_hidden_size()}")

config = get_config(DATASET)
IMAGE_FOLDER = config.image_folder
QUESTION_FILE = config.question_file

questions = load_questions(QUESTION_FILE)

emo2letter = config.emo2letter
by_cat = defaultdict(list)
for q in questions:
    cat = q.get("category", q["question_id"].split("/")[0])
    if cat in emo2letter:
        by_cat[cat].append(q)

class_images = {}
test_set = []
for cat in sorted(emo2letter.keys()):
    letter = emo2letter[cat]
    qs = by_cat[cat]
    imgs = []
    for q in qs[:N_CALIB_PER_CLASS]:
        imgs.append(load_image(IMAGE_FOLDER, q["image"]))
    class_images[letter] = imgs
    for q in qs[N_CALIB_PER_CLASS:N_CALIB_PER_CLASS + N_TEST_PER_CLASS]:
        test_set.append((q, letter))

print(f"Calibration: {sum(len(v) for v in class_images.values())} "
      f"({N_CALIB_PER_CLASS}/class)")
print(f"Test: {len(test_set)} samples")

print("\n=== Extracting ASO directions ===")
aso = extract_aso(vlm, class_images, layer_idx=-1,
                  letter2emo=config.letter2emo)

print("\n=== ASO similarity matrix ===")
aso.print_similarity_matrix(letter2emo=config.letter2emo)
sim_matrix = aso.similarity_matrix()

print(f"\n=== Running LARC on {len(test_set)} samples ===")

stats = defaultdict(lambda: {"sys1": 0, "larc": 0, "total": 0})
flip_ok = flip_bad = 0

for i, (q, gt_letter) in enumerate(test_set):
    image = load_image(IMAGE_FOLDER, q["image"])
    cat = q.get("category", q["question_id"].split("/")[0])

    result = larc_inference(
        vlm, aso, image,
        config=config,
        similarity_matrix=sim_matrix,
    )

    logit_pred = result["logit_prediction"]
    larc_pred = result["prediction"]
    is_logit_ok = logit_pred == gt_letter
    is_larc_ok = larc_pred == gt_letter

    stats[cat]["total"] += 1
    if is_logit_ok:
        stats[cat]["sys1"] += 1
    if is_larc_ok:
        stats[cat]["larc"] += 1

    flip = ""
    if not is_logit_ok and is_larc_ok:
        flip_ok += 1
        flip = " +FLIP"
    elif is_logit_ok and not is_larc_ok:
        flip_bad += 1
        flip = " -FLIP"

    topo_p = result["topo_prediction"]
    topo_c = result["topo_confidence"]
    print(f"[{i+1:3d}] {cat[:5]:>5} logit={logit_pred}({result['logit_confidence']:.2f}) "
          f"topo={topo_p}({topo_c:.2f}) -> {larc_pred} GT={gt_letter}{flip}")

    torch.cuda.empty_cache()

print(f"\n{'='*45}")
print(f"{'Category':>12} | {'Baseline':>10} | {'LARC':>10} | {'Δ':>4}")
print("-" * 45)

total_s1 = total_larc = total_n = 0
for cat in sorted(stats.keys()):
    d = stats[cat]
    t, s1, lt = d["total"], d["sys1"], d["larc"]
    delta = lt - s1
    total_s1 += s1
    total_larc += lt
    total_n += t
    sign = "+" if delta > 0 else ""
    print(f"{cat:>12} | {s1:>2}/{t:>2} ({s1/t:.0%}) | "
          f"{lt:>2}/{t:>2} ({lt/t:.0%}) | {sign}{delta}")

print("-" * 45)
delta = total_larc - total_s1
print(f"{'TOTAL':>12} | {total_s1:>2}/{total_n} ({total_s1/total_n:.1%}) | "
      f"{total_larc:>2}/{total_n} ({total_larc/total_n:.1%}) | "
      f"{'+' if delta > 0 else ''}{delta}")
print(f"\nFlip OK: {flip_ok}, Flip BAD: {flip_bad}, Net: {flip_ok - flip_bad}")
print("Done.")
