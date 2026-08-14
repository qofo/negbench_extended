#!/bin/bash
set -e

# ==============================================================================
# Script: run_beaf_multi_probing_sweep.sh
# Runs BEAF Flexible Probing across multiple linear classifier algorithms
# (Logistic Regression, Linear SVM, Ridge Classifier, SGD Log)
# ==============================================================================

CSV_PATH="benchmarks/data/images/beaf_counterfactual_6col.csv"
if [ ! -f "${CSV_PATH}" ]; then
    CSV_PATH="csvOLD/beaf_counterfactual_6col.csv"
fi

IMAGE_ROOT="."
BASE_OUT_DIR="logs/evaluation/beaf_multi_probe"
MODEL="ViT-B-32"
PRETRAINED="openai"

PROBES=("logistic" "svm_linear" "ridge" "sgd_log")
RUN_DIRS=()

for PROBE in "${PROBES[@]}"; do
    OUT_DIR="${BASE_OUT_DIR}_${PROBE}"
    RUN_DIRS+=("${OUT_DIR}")
    
    echo "=========================================================="
    echo "  Executing Flexible Probing with Classifier: [${PROBE^^}]"
    echo "  Output Directory: ${OUT_DIR}"
    echo "=========================================================="
    
    python -m benchmarks.src.analysis.run_beaf_flexible_probing \
        --csv_path "${CSV_PATH}" \
        --image_root "${IMAGE_ROOT}" \
        --output_dir "${OUT_DIR}" \
        --probe_type "${PROBE}" \
        --model "${MODEL}" \
        --pretrained "${PRETRAINED}"
done

echo "=========================================================="
echo "  Generating Multi-Classifier Probing Comparison Plot..."
echo "=========================================================="

python -m benchmarks.src.analysis.beaf.compare_beaf_probes \
    --input_dirs "${RUN_DIRS[@]}" \
    --output_dir "${BASE_OUT_DIR}_comparison"

echo "=========================================================="
echo "  Multi-Probing Sweep Complete! Summary plot saved to:"
echo "  ${BASE_OUT_DIR}_comparison/beaf_probing_classifier_comparison.png"
echo "=========================================================="
