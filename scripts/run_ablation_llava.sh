#!/bin/bash
# ==============================================================
# Ablation experiments on LLaVA-v1.5-7B.
# Mirrors the three ablation studies reported in the paper:
#   1. component ablation (ASO / RAER / RGEM)
#   2. n_calib sensitivity
#   3. intermediate-layer sweep
#
# Usage:
#   bash scripts/run_ablation_llava.sh component   # only the component ablation
#   bash scripts/run_ablation_llava.sh ncalib
#   bash scripts/run_ablation_llava.sh layer
#   bash scripts/run_ablation_llava.sh all         # all three
# ==============================================================
set -e

cd "$(dirname "$0")/.."

MODEL_PATH="${LARC_LLAVA_PATH:-models/llava-v1.5-7b}"
GPU="${GPU:-0}"
DATASET="${DATASET:-emotion6}"

run_exp() {
    local tag=$1; shift
    local ds=$1; shift
    echo ""
    echo "============================================="
    echo "  [LLaVA] $ds -- $tag"
    echo "============================================="
    CUDA_VISIBLE_DEVICES=$GPU python run_larc.py \
        --dataset "$ds" \
        --model_path "$MODEL_PATH" \
        --model_name llava \
        --answer_file "answer/ablation/${tag}_${ds}.jsonl" \
        "$@" \
        --resume
}

# ============================================================
# Component ablation: turn each LARC module off in turn.
# ============================================================
run_component_ablation() {
    echo "###  Component Ablation  ###"

    # w/o RGEM entropy-adaptive exponent
    run_exp "no_entropy" "$DATASET" \
        --n_calib 20 --layer_ratio 0.65 \
        --fusion_mode product --no_entropy_adaptive

    # w/o RAER relation routing (sigma -> infinity)
    run_exp "no_raer" "$DATASET" \
        --n_calib 20 --layer_ratio 0.65 \
        --fusion_mode product --geo_sigma 999.0

    # w/o RGEM margin guard
    run_exp "no_guard" "$DATASET" \
        --n_calib 20 --layer_ratio 0.65 \
        --fusion_mode product --margin_guard 999.0

    # ASO-only (latent path dominates, every other knob off)
    run_exp "aso_only" "$DATASET" \
        --n_calib 20 --layer_ratio 0.65 \
        --fusion_mode weighted --topo_weight 1.0 --margin_guard 999.0

    # Naive product fusion (none of the modulators enabled)
    run_exp "naive_product" "$DATASET" \
        --n_calib 20 --layer_ratio 0.65 \
        --fusion_mode product --no_entropy_adaptive \
        --geo_sigma 999.0 --margin_guard 999.0

    echo "=== Component ablation done ==="
}

# ============================================================
# n_calib sensitivity: how few labelled images can LARC tolerate?
# ============================================================
run_ncalib_sensitivity() {
    echo "###  n_calib sensitivity  ###"
    for n in 5 10 20 30 50; do
        run_exp "ncalib_${n}" "$DATASET" \
            --n_calib "$n" --layer_ratio 0.65 --fusion_mode product
    done
    echo "=== n_calib sensitivity done ==="
}

# ============================================================
# Layer sweep: where does affective evidence live?
# ============================================================
run_layer_sweep() {
    echo "###  Layer sweep  ###"
    for r in 0.30 0.40 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.90; do
        run_exp "layer_${r}" "$DATASET" \
            --n_calib 20 --layer_ratio "$r" --fusion_mode product
    done
    echo "=== Layer sweep done ==="
}

MODE="${1:-all}"
case "$MODE" in
    component) run_component_ablation ;;
    ncalib)    run_ncalib_sensitivity ;;
    layer)     run_layer_sweep ;;
    all)
        run_component_ablation
        run_ncalib_sensitivity
        run_layer_sweep
        ;;
    *) echo "Usage: $0 {component|ncalib|layer|all}"; exit 1 ;;
esac

echo ""
echo "========================================="
echo "  All requested ablations completed."
echo "========================================="
