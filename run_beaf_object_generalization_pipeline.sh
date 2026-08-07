#!/bin/bash
# ==============================================================================
# Pipeline: BEAF Single-Object & Multi-Object Generalization Evaluation
#
# Evaluates 1:1 balanced Present/Absent images across expanded 255 prompt templates
# (123 negative, 132 positive) per object category.
# ==============================================================================

set -e

BASE_DIR="."
BEAF_CSV="${BASE_DIR}/csvOLD/beaf_counterfactual_6col.csv"
if [ ! -f "${BEAF_CSV}" ]; then
    BEAF_CSV="${BASE_DIR}/benchmarks/data/images/beaf_counterfactual_6col.csv"
fi

TEMPLATE_JSON="${BASE_DIR}/benchmarks/data/beaf_expanded_templates.json"
OUT_DIR="${BASE_DIR}/logs/evaluation/beaf_object_generalization/openai_vit_b32"

MODEL="ViT-B-32"
PRETRAINED="openai"

echo "======================================================================"
echo "🚀 Running BEAF Single & Multi-Object Generalization Pipeline"
echo " Dataset CSV   : ${BEAF_CSV}"
echo " Template JSON : ${TEMPLATE_JSON}"
echo " Model         : ${MODEL} (${PRETRAINED})"
echo " Output Dir    : ${OUT_DIR}"
echo "======================================================================"

python -m benchmarks.src.analysis.run_beaf_object_generalization \
    --csv_path "${BEAF_CSV}" \
    --template_json "${TEMPLATE_JSON}" \
    --model "${MODEL}" \
    --pretrained "${PRETRAINED}" \
    --target_object "all" \
    --min_pairs 2 \
    --output_dir "${OUT_DIR}"

echo ""
echo "======================================================================"
echo "✅ Pipeline Completed Successfully!"
echo " Results saved to: ${OUT_DIR}"
echo "======================================================================"
