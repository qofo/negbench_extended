#!/bin/bash
# ==============================================================================
# Pipeline: BEAF Text Embedding Linear Probing & Unseen Object Generalization
# ==============================================================================

set -e

BASE_DIR="."
TEMPLATE_JSON="${BASE_DIR}/benchmarks/data/beaf_expanded_templates.json"
OUT_DIR="${BASE_DIR}/logs/evaluation/beaf_text_probe_generalization/openai_vit_b32"

MODEL="ViT-B-32"
PRETRAINED="openai"

echo "======================================================================"
echo "🚀 Running BEAF Text Linear Probing & Unseen Object Generalization"
echo " Template JSON : ${TEMPLATE_JSON}"
echo " Model         : ${MODEL} (${PRETRAINED})"
echo " Output Dir    : ${OUT_DIR}"
echo "======================================================================"

python -m benchmarks.src.analysis.run_beaf_text_probe_experiment \
    --template_json "${TEMPLATE_JSON}" \
    --model "${MODEL}" \
    --pretrained "${PRETRAINED}" \
    --source_object "person" \
    --C 1.0 \
    --output_dir "${OUT_DIR}"

echo ""
echo "======================================================================"
echo "✅ Text Probing Pipeline Completed Successfully!"
echo " Results saved to: ${OUT_DIR}"
echo "======================================================================"
