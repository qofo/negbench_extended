#!/bin/bash
# ==============================================================================
# Pipeline: BEAF Dual Classifier (+1/-1) Training & NegBench Evaluation
#
# Stage 1: Train f_V (Vision: Present=+1, Absent=-1) and f_T (Text: Positive=+1, Negative=-1)
#          on BEAF Counterfactual Dataset.
# Stage 2: Evaluate S(v, t) = f_V(v) * f_T(t) on NegBench benchmarks replacing standard
#          cosine similarity.
# ==============================================================================

set -e

BASE_DIR="."
DATA_DIR="${BASE_DIR}/benchmarks/data/images"

BEAF_CSV="${DATA_DIR}/beaf_counterfactual_6col.csv"
if [ ! -f "${BEAF_CSV}" ]; then
    BEAF_CSV="${BASE_DIR}/csvOLD/beaf_counterfactual_6col.csv"
fi

COCO_MCQ_CSV="${DATA_DIR}/COCO_val_mcq_llama3.1_rephrased.csv"
COCO_RETRIEVAL_CSV="${DATA_DIR}/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"
VOC_MCQ_CSV="${DATA_DIR}/VOC2007_mcq_llama3.1_rephrased.csv"
CHEXPERT_MCQ_CSV="${DATA_DIR}/chexpert_binary_mcq_control_valid_only.csv"

PROBE_OUT_DIR="${BASE_DIR}/logs/evaluation/beaf_dual_probe"
EVAL_OUT_DIR="${BASE_DIR}/logs/evaluation/beaf_dual_classifier_pipeline"
WEIGHTS_PATH="${PROBE_OUT_DIR}/beaf_dual_probe_weights.npz"

MODEL="ViT-B-32"
PRETRAINED="openai"

echo "======================================================================"
echo "🚀 STAGE 1: Training Dual Classifiers (+1/-1) on BEAF Data"
echo " Target BEAF CSV: ${BEAF_CSV}"
echo "======================================================================"

python -m benchmarks.src.analysis.train_beaf_dual_probes \
    --csv_path "${BEAF_CSV}" \
    --model_name "${MODEL}" \
    --pretrained "${PRETRAINED}" \
    --output_dir "${PROBE_OUT_DIR}" \
    --vision_type mlp \
    --vision_hidden_dim 64 \
    --C 1.0



if [ ! -f "${WEIGHTS_PATH}" ]; then
    echo "❌ Error: Dual classifier weights file not found at ${WEIGHTS_PATH}"
    exit 1
fi

echo ""
echo "======================================================================"
echo "🚀 STAGE 2: Evaluating Dual Classifier Product Scorer on NegBench"
echo " Saved Weights: ${WEIGHTS_PATH}"
echo " Output Directory: ${EVAL_OUT_DIR}"
echo "======================================================================"

BENCHMARKS=(
    "${BEAF_CSV}"
    "${COCO_MCQ_CSV}"
    "${COCO_RETRIEVAL_CSV}"
    "${VOC_MCQ_CSV}"
    "${CHEXPERT_MCQ_CSV}"
)


for csv in "${BENCHMARKS[@]}"; do
    if [ -f "$csv" ]; then
        echo ""
        echo "----------------------------------------------------------------------"
        echo "Evaluating: ${csv}"
        echo "----------------------------------------------------------------------"
        python -m benchmarks.src.evaluation.eval_zero_shot_transfer \
            --model "${MODEL}" \
            --pretrained "${PRETRAINED}" \
            --scorer-ckpt "${WEIGHTS_PATH}" \
            --target-mcq "${csv}" \
            --output-dir "${EVAL_OUT_DIR}/$(basename ${csv} .csv)"
    else
        echo "⚠️ Benchmark file not found, skipping: ${csv}"
    fi
done

echo ""
echo "======================================================================"
echo "✅ Pipeline Completed Successfully!"
echo " Results stored in: ${EVAL_OUT_DIR}"
echo "======================================================================"
