#!/bin/bash
cd /home/junseolee/negbench
PY=/home/junseolee/.conda/envs/clip_negation/bin/python
export PYTHONPATH="$PWD:$PWD/benchmarks:$PWD/benchmarks/src"
OUT=logs/evaluation/01_paper/2026-08-31_negfam_mcq
M=benchmarks/models
COCO=benchmarks/data/images/COCO_val_mcq_llama3.1_rephrased.csv
VOC=benchmarks/data/images/VOC2007_mcq_llama3.1_rephrased.csv
mkdir -p $OUT

run () {  # $1=name  $2=model  $3=pretrained
  echo "=========== $1 ($2 / $3)"
  $PY -m benchmarks.src.evaluation.eval_negation \
      --model "$2" --pretrained "$3" \
      --coco-mcq "$COCO" --voc2007-mcq "$VOC" \
      --name "$1" --logs "$OUT" --batch-size 64 --workers 4 --seed 42 \
      2>&1 | grep -E "Eval Epoch|Error|Traceback|error" | head -5
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
