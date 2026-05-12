#!/bin/bash
# ==============================================================
# LARC main evaluation on LLaVA-v1.5-7B.
#
# Usage:
#   bash scripts/run_llava.sh                    # run all five datasets
#   bash scripts/run_llava.sh emoset             # run a single dataset
#   bash scripts/run_llava.sh emoset 2           # 2-way GPU shard
#
# The script expects to be launched from the LARC project root or its
# scripts/ directory; it cd's to the parent of itself either way.
# ==============================================================
set -e

cd "$(dirname "$0")/.."

MODEL_PATH="${LARC_LLAVA_PATH:-models/llava-v1.5-7b}"
DATASETS="${1:-all}"
NUM_SHARDS="${2:-1}"

ALL_DATASETS="emotion6 emoset abstract artphoto webemo7 webemo25"

if [ "$DATASETS" = "all" ]; then
    DATASETS="$ALL_DATASETS"
fi

run_single() {
    local ds=$1
    echo ""
    echo "============================================="
    echo "  [LLaVA] LARC on $ds  (single GPU)"
    echo "============================================="
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python run_larc.py \
        --dataset "$ds" \
        --model_path "$MODEL_PATH" \
        --model_name llava \
        --n_calib 20 \
        --layer_ratio 0.65 \
        --fusion_mode product \
        --resume
}

run_sharded() {
    local ds=$1 n=$2
    echo ""
    echo "============================================="
    echo "  [LLaVA] LARC on $ds  ($n shards)"
    echo "============================================="
    pids=()
    for ((i=0; i<n; i++)); do
        CUDA_VISIBLE_DEVICES=$i python run_larc.py \
            --dataset "$ds" --model_path "$MODEL_PATH" --model_name llava \
            --n_calib 20 --layer_ratio 0.65 --fusion_mode product \
            --shard "$i" --num_shards "$n" --resume &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
}

for ds in $DATASETS; do
    if [ "$NUM_SHARDS" -gt 1 ]; then
        run_sharded "$ds" "$NUM_SHARDS"
    else
        run_single "$ds"
    fi
done

echo ""
echo "========================================="
echo "  All requested datasets completed."
echo "========================================="
