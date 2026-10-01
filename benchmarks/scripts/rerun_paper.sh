#!/usr/bin/env bash
# Re-run every experiment of the camera-ready paper from scratch.
#
# Fixed inputs: the OpenAI ViT-B/32 weights, the NegCLIP (CC12M NegFull) checkpoint and the data CSVs.
# Recomputed: every embedding, probe, rotation, trained W, benchmark score and figure.
#
# Old feature caches are never read: every cache-aware script gets --cache_dir $FC, a fresh directory
# under the run folder (train_and_save_narrow_rank32_w and score_delta_s_for_w force --use_cache on, so
# redirecting the cache is the only way to keep them off the old one). The first stage to encode a
# population writes it there and later stages reuse only those fresh arrays. The MCQ embeddings of
# Table 3 have their own cache keyed by CSV name only, redirected the same way to $RR/_mcq_cache.
#
# Usage (repo root), one stage or a chain of stages per call:
#     bash benchmarks/scripts/rerun_paper.sh f2_sweep ours
#     bash benchmarks/scripts/rerun_paper.sh t1_openai t1_random t1_negft labclip t2
#     bash benchmarks/scripts/rerun_paper.sh star t3 fig1 fig2 compare
#
# Stages and the paper item each one reproduces:
#     t1_abswap_openai t1_abswap_random t1_abswap_negft
#                                    Table 1 linear probing on the AB-swap population (image AB-swap 42 objects,
#                                    text diverse 58 objects; the 42 objects both share are what Table 1 reports)
#     t1_openai t1_random t1_negft   the same probes on the older 6col image set (sensitivity check only)
#     t2                             Table 2 probe-normal alignment and closed-form rotation (33 objects)
#     f2_sweep                       Sec 2.2 cosine 4.03% and the Figure 2 rank curve (AB-swap, 42 objects)
#     ours                           Table 3 Ours: rank-32 W, Delta hinge + warm start, all 42 objects
#     labclip                        LABCLIP recipe: identity-init full W, InfoNCE, all 5,542 AB-swap pairs
#     star                           Figure 2 LABCLIP star: 2x2 accuracy of that W on the 42 objects
#     t3                             Table 3: COCO / VOC2007 / CheXpert MCQ for cosine, Ours, LABCLIP
#     fig1 fig2                      Figure 1 scores + drawing, Figure 2 drawing
#     compare                        every paper number next to its re-run value
#
# Resource limits: GPU via CUDA_VISIBLE_DEVICES (default 1), CPU via nice 19 and 4 BLAS/torch threads.
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTHONPATH="$PWD:$PWD/benchmarks:$PWD/benchmarks/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export MPLBACKEND=Agg

PYTHON="${PYTHON:-python}"
RR="${RR:-logs/evaluation/01_paper/2026-09-19_rerun}"
FC="$RR/_feature_cache"
LOGS="$RR/_logs"
D=benchmarks/data/images
ABSWAP=$D/beaf_counterfactual_ab_swap_by_concept.csv
NEGFT=benchmarks/models/CLIP_CC12M_NegFull_ViT-B-32_lr1e-8_clw0.99_mlw0.01/checkpoint.pt
OURS=$RR/ours_rank32/delta_warmstart_rank32.pt
LABCLIP=$RR/labclip/labclip_official_recipe_bilinear.pt
mkdir -p "$FC" "$LOGS"

py() { nice -n 19 "$PYTHON" "$@"; }

run_stage() {
    case "$1" in
    t1_abswap_openai) py -m benchmarks.src.evaluation.eval_probe_failure_inspector \
                   --model ViT-B-32 --pretrained openai --seed 42 --vision_csv $ABSWAP \
                   --output_dir $RR/t1_abswap_openai ;;
    t1_abswap_random) py -m benchmarks.src.evaluation.eval_probe_failure_inspector \
                   --model ViT-B-32 --pretrained '' --seed 42 --vision_csv $ABSWAP \
                   --output_dir $RR/t1_abswap_random ;;
    t1_abswap_negft) py -m benchmarks.src.evaluation.eval_probe_failure_inspector \
                   --model ViT-B-32 --pretrained $NEGFT --seed 42 --vision_csv $ABSWAP \
                   --output_dir $RR/t1_abswap_negft ;;
    t1_openai) py -m benchmarks.src.evaluation.eval_probe_failure_inspector \
                   --model ViT-B-32 --pretrained openai --seed 42 --output_dir $RR/t1_probe_openai ;;
    t1_random) py -m benchmarks.src.evaluation.eval_probe_failure_inspector \
                   --model ViT-B-32 --pretrained '' --seed 42 --output_dir $RR/t1_probe_random ;;
    t1_negft)  py -m benchmarks.src.evaluation.eval_probe_failure_inspector \
                   --model ViT-B-32 --pretrained $NEGFT --seed 42 --output_dir $RR/t1_probe_negft ;;
    t2)        py -m benchmarks.src.evaluation.eval_per_object_alignment_intervention \
                   --model ViT-B-32 --pretrained openai --seed 42 --rank 32 --oof \
                   --restrict_objects logs/evaluation/00_concept_sets/paper33.txt \
                   --use_cache --cache_dir $FC --output_dir $RR/t2_rotation ;;
    # family order matches the accepted run (each fold reseeds with seed+fold, so order is cosmetic)
    f2_sweep)  py -m benchmarks.src.evaluation.eval_single_w_generalization \
                   --model ViT-B-32 --pretrained openai --seed 42 --csv_path $ABSWAP --loss delta --warmstart \
                   --families identity random diagonal lowrank_1 lowrank_2 lowrank_4 lowrank_8 lowrank_16 \
                              lowrank_32 full lowrank_64 lowrank_128 lowrank_256 lowrank_512 \
                   --use_cache --cache_dir $FC --output_dir $RR/f2_rank_sweep ;;
    # also trains the margin4 W, as the original run did; each W reseeds, so only the delta W is used
    ours)      py -m benchmarks.src.evaluation.train_and_save_narrow_rank32_w \
                   --csv_path $ABSWAP --seed 42 --cache_dir $FC --output_dir $RR/ours_rank32 ;;
    labclip)   py -m benchmarks.src.evaluation.train_labclip_official_recipe \
                   --csv_path $ABSWAP --seed 42 --output_dir $RR/labclip ;;
    star)      py -m benchmarks.src.evaluation.score_delta_s_for_w --ckpts $LABCLIP $OURS --names labclip ours \
                   --csv_path $ABSWAP --seed 42 --cache_dir $FC --output_dir $RR/f2_labclip_star ;;
    t3)        py -m benchmarks.src.evaluation.eval_w_ckpts_mcq --ckpts $OURS $LABCLIP \
                   --targets $D/COCO_val_mcq_llama3.1_rephrased.csv $D/VOC2007_mcq_llama3.1_rephrased.csv \
                             $D/chexpert_binary_mcq_control_valid_only.csv \
                   --seed 42 --cache_dir $RR/_mcq_cache --output $RR/t3_mcq/mcq_by_ckpt.json ;;
    fig1)      mkdir -p $RR/figures &&
               py paper_figures/compute_fig1_example.py --output $RR/figures/fig1_example_pair.json &&
               py paper_figures/mk_fig1_camera_ready.py $RR/figures/fig1_example_pair.json \
                   $RR/figures/fig1_camera_ready.png ;;
    fig2)      mkdir -p $RR/figures &&
               py paper_figures/mk_fig2_rank_heldout.py --run $RR/f2_rank_sweep \
                   --labclip $RR/f2_labclip_star/delta_s_report.json \
                   --out $RR/figures/fig2_rank_heldout.png --data_out $RR/figures/fig2_rank_heldout_data.csv ;;
    compare)   py -m benchmarks.src.evaluation.compare_paper_rerun --rerun $RR ;;
    *)         echo "unknown stage: $1" >&2; return 2 ;;
    esac
}

for stage in "$@"; do
    echo "[$(date '+%F %T')] start $stage"
    rm -f "$LOGS/$stage.done" "$LOGS/$stage.failed"
    if run_stage "$stage" > "$LOGS/$stage.log" 2>&1; then
        date '+%F %T' > "$LOGS/$stage.done"
        echo "[$(date '+%F %T')] done  $stage"
    else
        date '+%F %T' > "$LOGS/$stage.failed"
        echo "[$(date '+%F %T')] FAILED $stage (see $LOGS/$stage.log)" >&2
        exit 1
    fi
done
