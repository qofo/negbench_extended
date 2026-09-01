#!/bin/bash
# E2 Hadamard decomposition on the AB-swap captions for the nine models of PAPER.md section 4.3.
# Re-measures the coefficients on the text design section 2 actually describes; the earlier sweep
# used beaf_counterfactual_6col.csv (single-object captions).
# Usage:  bash benchmarks/scripts/run_negfam_e2_abswap.sh      (from the repo root)
cd /home/junseolee/negbench
PY=/home/junseolee/.conda/envs/clip_negation/bin/python
export PYTHONPATH="$PWD:$PWD/benchmarks:$PWD/benchmarks/src"
OUT=logs/evaluation/01_paper/2026-08-31_negfam_e2_abswap
M=benchmarks/models
CSV=benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv
mkdir -p $OUT

run () {  # $1=name  $2=model  $3=pretrained
  echo "=========== $1 ($2 / $3)"
  $PY -m benchmarks.src.evaluation.eval_e2_hadamard_decomposition \
      --model "$2" --pretrained "$3" \
      --csv_path "$CSV" --output_dir "$OUT/$1" \
      --use_cache --seed 42 --batch_size 128 \
      2>&1 | grep -E "Macro|Ratio|Concepts with|Median|Baseline 2x2|ablation|Identity|Verdict|Traceback|Error|out of memory" | cut -c1-160
}

run vitb32_openai      ViT-B-32          openai
run vitb16_openai      ViT-B-16          openai
run vitl14_openai      ViT-L-14          openai
run vitb32_laion2b     ViT-B-32          laion2b_s34b_b79k
run vitb16_siglip      ViT-B-16-SigLIP   webli
run conclip            ViT-B-32          $M/ConCLIP/conclip_b32_openclip_version.pt
run negclip            ViT-B-32          $M/NegCLIP/negclip.pth
run negclip_negfull    ViT-B-32          $M/NegCLIP_CC12M_NegFull_ViT-B-32_lr1e-8_clw0.99_mlw0.01/checkpoint.pt
run clip_negfull       ViT-B-32          $M/CLIP_CC12M_NegFull_ViT-B-32_lr1e-8_clw0.99_mlw0.01/checkpoint.pt
echo "=========== DONE"
