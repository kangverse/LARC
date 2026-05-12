# LARC: Latent Affective Representation Calibration

**LARC** is a training-free inference framework that improves fine-grained visual emotion recognition with frozen multimodal large language models (MLLMs). Rather than only relying on the final logit head, LARC reads latent affective evidence from the intermediate layers of an MLLM, organises it into structured representations, and adaptively modulates its influence based on output reliability.

This repository contains the reference implementation used in our NeurIPS 2026 submission, focused on **LLaVA-v1.5-7B** as the backbone.

## Overview

LARC consists of three components that compose at inference time without any parameter updates:

- **ASO – Affective Semantic Orientation.** For each emotion category, we estimate a discriminative direction in the latent space using a small calibration set and a class-agnostic neutral probe.
- **RAER – Relation-Aware Affective Evidence Routing.** The pairwise cosine similarity between ASO directions defines a continuous affective relation graph; routing weights are derived from the row of the graph corresponding to the model's initial logit prediction.
- **RGEM – Reliability-Gated Evidence Modulation.** A Prediction Reliability Index (PRI), computed from the logit distribution's normalised entropy and Top-1 margin, gates how strongly the latent evidence influences the final answer.

Empirically, LARC improves average accuracy over strong training-free baselines (e.g. SEPM) on Emotion6, EmoSet, Abstract, WebEmo-7 and WebEmo-25, without modifying model parameters.

## Repository layout

```
LARC/
├── larc/                       # Python package
│   ├── __init__.py             # public re-exports
│   ├── aso.py                  # ASO extractors + relation matrix (RAER input)
│   ├── inference.py            # LARC pipeline: ASO + RAER + RGEM fusion
│   ├── dataset_config.py       # Dataset registry (EmoSet / Emotion6 / WebEmo / ...)
│   ├── data.py                 # JSONL + image loaders
│   └── vlm/                    # VLM wrappers (only LLaVA-v1.5 is shipped here)
│       ├── base.py
│       ├── builder.py
│       └── llava_wrapper.py
├── run_larc.py                 # Unified evaluation entry-point
├── eval_larc.py                # Stand-alone scorer for answer JSONLs
├── test_larc_quick.py          # End-to-end smoke test on a small subset
├── scripts/
│   ├── run_llava.sh            # Default LLaVA evaluation over all datasets
│   └── run_ablation_llava.sh   # Component / n_calib / layer-sweep ablations
├── question/
│   └── generate_all.py         # Build question JSONLs from image folders
├── requirements.txt
└── README.md
```

## Installation

LARC depends on the official **LLaVA** repository for model loading and prompt construction. It does not vendor LLaVA's source; you need to install it separately.

1. **Create a fresh environment** (Python 3.10+ recommended):

    ```bash
    conda create -n larc python=3.10 -y
    conda activate larc
    ```

2. **Install PyTorch** matching your CUDA version, e.g. for CUDA 12.1:

    ```bash
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
    ```

3. **Install the remaining Python dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Install LLaVA-v1.5** following https://github.com/haotian-liu/LLaVA. The simplest way is:

    ```bash
    git clone https://github.com/haotian-liu/LLaVA.git
    cd LLaVA
    pip install -e .
    ```

    Alternatively, point the wrapper at an existing checkout via the `LLAVA_HOME` environment variable:

    ```bash
    export LLAVA_HOME=/path/to/LLaVA
    ```

5. **Download the LLaVA-v1.5-7B checkpoint** (e.g. `liuhaotian/llava-v1.5-7b` on Hugging Face). By default the code looks in `./models/llava-v1.5-7b` (relative to the project root); either drop a symlink there or set `LARC_LLAVA_PATH`:

    ```bash
    # option 1: symlink
    mkdir -p models && ln -s /path/to/your/llava-v1.5-7b models/llava-v1.5-7b

    # option 2: env var
    export LARC_LLAVA_PATH=/path/to/your/llava-v1.5-7b
    ```

## Datasets

LARC supports six visual emotion-recognition datasets out of the box:

| name      | classes | source |
|-----------|---------|--------|
| `emoset`   | 8       | EmoSet-118K |
| `emotion6` | 6       | Emotion6 |
| `abstract` | 8       | IAPS-Abstract |
| `artphoto` | 8       | ArtPhoto |
| `webemo7`  | 7       | WebEmo (coarse) |
| `webemo25` | 25      | WebEmo (fine-grained) |

After downloading the raw images, place (or symlink) them under `./data/` so the layout matches `data/EmoSet-118K/image/...`, `data/Emotion6/images/...`, etc. The dataset root is configurable via `LARC_DATA_ROOT`:

```bash
# Option 1: drop datasets into ./data
# Option 2: point at an existing dataset directory
export LARC_DATA_ROOT=/path/to/your/visual_emotion_datasets

python question/generate_all.py --all
```

Each line of a question file is

```json
{"question_id": "anger/1.jpg", "image": "anger/1.jpg",
 "text": "Please focus on emotion ...", "category": "anger"}
```

You can override the default per-dataset paths from the CLI via `--image_folder` and `--question_file`.

## Quick start

After installation, the simplest end-to-end test is:

```bash
CUDA_VISIBLE_DEVICES=0 python test_larc_quick.py
```

This loads LLaVA-v1.5-7B, extracts ASO on 20 calibration images per class, prints the inter-class similarity matrix, and reports per-class accuracy on a small held-out subset.

## Reproducing the main results

The full evaluation over all five datasets is wrapped in one script:

```bash
# Single GPU, default n_calib=20, layer_ratio=0.65, fusion_mode=product
bash scripts/run_llava.sh

# Single dataset, single GPU
bash scripts/run_llava.sh emoset

# Single dataset, 2-way GPU shard
bash scripts/run_llava.sh emoset 2
```

Outputs:

* Per-sample predictions: `answer/<model_short>/larc_<dataset>.jsonl`
* ASO cache (re-used across runs): `concept_cache/<model_short>/aso_<dataset>_n20_l0.65.json`
* Logs: `log/<dataset>/larc_<dataset>_<model_short>_<timestamp>.log`

You can re-score an existing answer file at any time:

```bash
python eval_larc.py --dataset emoset answer/llava-v1.5-7b/larc_emoset.jsonl
```

The scorer prints a per-class accuracy table comparing the raw logit baseline against LARC, together with macro accuracy, the W -> R / R -> W flip statistics, and the LARC trigger rate.

## Reproducing the ablations

The component / n_calib / layer-sweep ablations from the paper are organised by `scripts/run_ablation_llava.sh`:

```bash
# All three ablation studies on Emotion6 (default)
bash scripts/run_ablation_llava.sh all

# Only component ablation
bash scripts/run_ablation_llava.sh component

# Run them on a different dataset
DATASET=emoset bash scripts/run_ablation_llava.sh component
```

Results are written to `answer/ablation/<tag>_<dataset>.jsonl`.

## Programmatic use

```python
from larc.vlm.builder import build_vlm
from larc.aso import extract_aso
from larc.inference import larc_inference
from larc.dataset_config import get_config
from larc.data import load_image

vlm = build_vlm("llava", "models/llava-v1.5-7b", "cuda")  # or absolute path
config = get_config("emotion6")

# class_images: {letter: [PIL.Image, ...]} -- e.g. 20 calibration images per class
aso = extract_aso(vlm, class_images, layer_idx=-1, letter2emo=config.letter2emo)
sim = aso.similarity_matrix()

image = load_image(config.image_folder, "joy/123.jpg")
result = larc_inference(vlm, aso, image, config=config, similarity_matrix=sim)
print(result["prediction"], result["confidence"])
```


## License

This repository is released under the MIT License. The LLaVA dependency retains its own license; see https://github.com/haotian-liu/LLaVA for details.
