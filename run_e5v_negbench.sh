#!/bin/bash
set -e

# ==============================================================================
# Script: run_e5v_negbench.sh
# Purpose: Evaluate E5-V on NegBench MCQ and Retrieval tasks
# ==============================================================================

# ── Model settings ──
MODEL_NAME="royokong/e5-v"
DEVICE="cuda:0"         # Use cuda:0 or cuda:1
DTYPE="float16"         # float16 | bfloat16 | float32
QUANTIZE=""             # "" (no quantize, default) | "int4" | "int8"
LORA_PATH=""            # Optional: path to LoRA adapter

# ── Data settings ──
# Paths relative to the project root (negbench_for_colab/)
COCO_MCQ="COCO_val_mcq_llama3.1_rephrased.csv"
COCO_RETRIEVAL="COCO_val_retrieval.csv"
COCO_NEGATED_RETRIEVAL="COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"
DATA_ROOT="benchmarks"  # Root for resolving relative image paths in CSVs

# ── Batch sizes ──
TEXT_BATCH_SIZE=16
IMAGE_BATCH_SIZE=4

# ── Output ──
OUTPUT_DIR="logs/e5v/negbench"
NAME="e5v_eval"

echo "=========================================================="
echo "  E5-V NegBench Evaluation"
echo "  Model       : ${MODEL_NAME}"
echo "  Device      : ${DEVICE}"
echo "  Dtype       : ${DTYPE}"
echo "  Quantize    : ${QUANTIZE:-none}"
echo "  LoRA        : ${LORA_PATH:-none}"
echo "  COCO MCQ    : ${COCO_MCQ}"
echo "  COCO Ret    : ${COCO_RETRIEVAL}"
echo "  Output      : ${OUTPUT_DIR}/${NAME}"
echo "=========================================================="

# Build command
CMD="PYTHONPATH=benchmarks python -m src.e5v_analysis.eval_negbench_e5v \
    --model-name ${MODEL_NAME} \
    --device ${DEVICE} \
    --dtype ${DTYPE} \
    --data-root ${DATA_ROOT} \
    --text-batch-size ${TEXT_BATCH_SIZE} \
    --image-batch-size ${IMAGE_BATCH_SIZE} \
    --output-dir ${OUTPUT_DIR} \
    --name ${NAME}"

# Add optional flags
if [ -n "${QUANTIZE}" ]; then
    CMD="${CMD} --quantize ${QUANTIZE}"
fi

if [ -n "${LORA_PATH}" ]; then
    CMD="${CMD} --lora-path ${LORA_PATH}"
fi

# Add dataset flags
if [ -n "${COCO_MCQ}" ] && [ -f "${COCO_MCQ}" ]; then
    CMD="${CMD} --coco-mcq ${COCO_MCQ}"
fi

if [ -n "${COCO_RETRIEVAL}" ] && [ -f "${COCO_RETRIEVAL}" ]; then
    CMD="${CMD} --coco-retrieval ${COCO_RETRIEVAL}"
fi

if [ -n "${COCO_NEGATED_RETRIEVAL}" ] && [ -f "${COCO_NEGATED_RETRIEVAL}" ]; then
    CMD="${CMD} --coco-negated-retrieval ${COCO_NEGATED_RETRIEVAL}"
fi

eval ${CMD}

echo "=========================================================="
echo "  Evaluation complete!"
echo "  Results: ${OUTPUT_DIR}/${NAME}/results.json"
echo "  Predictions: ${OUTPUT_DIR}/${NAME}/predictions/"
echo "=========================================================="
