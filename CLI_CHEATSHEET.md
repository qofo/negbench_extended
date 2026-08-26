# NegBench CLI Cheatsheet & Execution Guide

---
## 🇰🇷 한국어

### 1. 주요 직접 실행 Python 명령어 (Core Commands)

#### ① 텍스트 인코더 기하 구조 분석 (레이어 코사인 유사도 & 선형 프로브)
```bash
python -m benchmarks.src.analysis.run_analysis \
    --csv_path benchmarks/data/images/COCO_val_full_paired.csv \
    --model ViT-B-32 \
    --pretrained openai \
    --output_dir logs/analysis_modular/openai_vit_b32
```

#### ② BEAF 다중 프로빙 스윕 (Linear, MLP, Bilinear 전 레이어 평가)
```bash
bash run_beaf_comprehensive_probing_sweep.sh
```

#### ③ BEAF 2×2 Factorial ANOVA + Vision 분석
```bash
python -m benchmarks.src.analysis.run_beaf_analysis_v2 \
    --csv_path benchmarks/data/images/beaf_counterfactual_6col.csv \
    --output_dir logs/evaluation/beaf_counterfactual_v2/openai_vit_b32
```

#### ④ 객체 카테고리 기준 부정 부분공간 전이 평가
```bash
python -m benchmarks.src.analysis.subspace_analysis \
    --csv_path benchmarks/data/images/COCO_val_full_paired.csv \
    --split_by object_name \
    --output_dir logs/subspace_analysis
```

#### ⑤ 소수 차원 가설 (Sparse Dimensions) 검증
```bash
python -m benchmarks.src.analysis.eval_sparse_text_dimensions
```

#### ⑥ AB Swap 데이터셋 감사 (0순위) 및 인과성 평가 (1~4순위)
```bash
# 0순위: 데이터셋 감사
python benchmarks/src/analysis/beaf/audit_ab_swap_dataset.py

# 1~4순위: 2x2 Joint Consistency 및 진단 평가
python benchmarks/src/analysis/beaf/run_ab_swap_evaluation.py
```

#### ⑦ Bilinear W 행렬 가중치 및 Off-diagonal 에너지 분석
```bash
python -m benchmarks.src.analysis.analyze_internal_weights
```

#### ⑧ Negation Existence Probe (Exp A: Cosine Distance + Exp B: Polarity Probe)
```bash
python -m benchmarks.src.evaluation.eval_negation_existence_probe \
    --csv_path benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv \
    --output_dir logs/evaluation/negation_existence_probe \
    --model ViT-B-32 --pretrained openai
```

#### ⑨ Scoring Head 비교 평가 (6종, 5-Fold OOF)
```bash
python -m benchmarks.src.evaluation.eval_scoring_heads \
    --model ViT-B-32 --pretrained openai \
    --csv benchmarks/data/images/COCO_val_negation.csv \
    --output-dir logs/evaluation/scoring_head_experiments
```

#### ⑩ Word-Swap Counterfactual Probe (Token-Presence Bias 대응)
```bash
python -m benchmarks.src.evaluation.eval_word_swap_probe \
    --model ViT-B-32 --pretrained openai \
    --coco-mcq benchmarks/data/images/COCO_val_negation.csv \
    --output-dir logs/evaluation/top_priority_experiments
```

#### ⑪ Vision/Text Ablation 숏컷 진단
```bash
# 비전 임베딩 절제
python -m benchmarks.src.evaluation.eval_vision_ablation_shortcut \
    --model ViT-B-32 --pretrained openai \
    --csv benchmarks/data/images/COCO_val_negation.csv \
    --output-dir logs/evaluation/vision_ablation

# 텍스트 임베딩 절제
python -m benchmarks.src.evaluation.eval_text_ablation_shortcut \
    --model ViT-B-32 --pretrained openai \
    --csv benchmarks/data/images/COCO_val_negation.csv \
    --output-dir logs/evaluation/text_ablation
```

#### ⑫ Unary 4-Stage 메커니즘 분석 (E1~E4)
```bash
python -m benchmarks.src.evaluation.eval_unary_mechanistic_analysis \
    --model ViT-B-32 --pretrained openai \
    --beaf-csv benchmarks/data/images/beaf_counterfactual_6col.csv \
    --ab-swap-csv benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv \
    --output-dir logs/evaluation/unary_mechanistic
```

#### ⑬ Per-Object Alignment Causal Intervention (5종 조건)
```bash
python -m benchmarks.src.evaluation.eval_per_object_alignment_intervention \
    --model ViT-B-32 --pretrained openai \
    --beaf-csv benchmarks/data/images/beaf_counterfactual_6col.csv \
    --ab-swap-csv benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv \
    --output-dir logs/evaluation/per_object_intervention
```

---

### 2. 루트 셸 스크립트 모음 (`./*.sh`)

| 스크립트 파일 | 설명 및 주요 내용 |
|:---|:---|
| `run_hypothesis_experiments.sh` | 가설 검증 마스터 러너 (H1 Procrustes, H2 Hyperplane, H3 Scoring Head, H4 Subspace Bilinear 전체 실행) |
| `run_top_priority_experiments.sh` | 최우선 과제 실행 (Word-Swap Probe, Rank Sweep, Concept Vector Ablation) |
| `run_scoring_head_experiments.sh` | 6종 Scoring Head 학습 및 5-Fold OOF 교차 검증 자동화 |
| `run_all_benchmarks_zero_shot.sh` | 전체 벤치마크 (COCO, VOC, Synthetic, Video) Zero-Shot 평가 일괄 실행 |
| `run_analysis.sh` | 텍스트 인코더 16단계 기하 구조 분석 자동 실행 |
| `run_pipeline_breakdown.sh` | 파이프라인 단계별(Step 0~4) 분해 및 메트릭 산출 |
| `run_layerwise_linear_probe.sh` | 텍스트 인코더 전 레이어(Layer 0~12) Linear Probe 실행 |
| `run_sparse_dim_verification.sh` | 텍스트/비전 상위 k개 차원 절제 실험 (소수 차원 vs 분산 표현) |
| `run_bilinear_verification.sh` | Bilinear Scorer와 Low-Rank(k=512) 간의 수학적 동치성 검증 |
| `run_zero_shot_transfer.sh` | 학습된 Scorer의 외부 OOD 벤치마크 Zero-Shot 전이 성능 평가 |
| `run_ab_swap_diagnostic.sh` | AB Swap 합성 부정 진단 (Text Sanity Probe, Unary vs Compound, ΔS Margin) |
| `run_unary_mechanistic_pipeline.sh` | E1~E4 Unary 4단계 메커니즘 분석 파이프라인 |
| `run_per_object_alignment_intervention.sh` | 객체별 프로브 정렬 및 인과적 개입(5가지 조건) 실행 |
| `run_probe_failure_inspector.sh` | Vision/Text 프로브 OOF 실패 사례 및 에러 패턴 수집 |
| `run_e1_minimal_pair_auc.sh` | E1 Minimal Pair 원자적 개념 검출 AUC 측정 (Alshehri 0.88 가정 검증) |
| `run_vision_ablation_shortcut.sh` | 비전 입력 절제(Zero/Shuffle/Gaussian) 숏컷 진단 |
| `run_beaf_comprehensive_probing_sweep.sh` | BEAF 9종 프로빙 분류기 전 레이어 통합 스윕 |
| `run_beaf_multi_probing_sweep.sh` | BEAF 다양한 프로빙 모델 스윕 실행 |
| `run_beaf_text_probe_pipeline.sh` | BEAF 텍스트 프로빙 전용 파이프라인 |
| `run_beaf_dual_classifier_pipeline.sh` | Dual Classifier ($f_T, f_V$) 학습 및 결합 Scorer 평가 |
| `run_beaf_object_generalization_pipeline.sh` | BEAF 단일/다중 객체 및 255개 템플릿 전이 실험 |
| `run_cc12m_negfull_experiments.sh` | CC12M-NegFull 파인튜닝 모델 평가 및 벤치마크 스윕 |
| `run_e5v_negbench.sh` | E5-V 멀티모달 모델 대상 NegBench 평가 |
| `run_pca.sh`, `run_pca_full_v2.sh`, `run_pca_paired.sh`, `run_pca_v4.sh` | 텍스트 인코더 표현 PCA 및 부분공간 분석 스크립트 |

---

### 3. `benchmarks/scripts/` 평가 스크립트 모음

| 스크립트 파일 | 설명 및 용도 |
|:---|:---|
| `evaluate_images.sh` | 이미지 MCQ 및 Retrieval 표준 벤치마크 일괄 평가 |
| `evaluate_videos.sh` | MSR-VTT 비디오 MCQ 및 Retrieval 평가 |
| `run_openai_clip.sh` | 공식 OpenAI CLIP 모델 기본 평가 |
| `run_pretrained_model_evaluations.sh` | 사전학습 모델군 (OpenCLIP, ConCLIP, NegCLIP) 비교 평가 |
| `run_single_model_evaluations.sh` | 특정 모델 단일 체크포인트에 대한 전체 벤치마크 평가 |
| `run_single_model_evaluations_llava.sh` | LLaVA 멀티모달 모델 NegBench MCQ 평가 |
| `run_full_eval_with_scoring_head.sh` | 학습된 Scoring Head를 장착한 상태로 전체 벤치마크 평가 |
| `run_scoring_head_evaluations.sh` | Scoring Head 성능 평가 스크립트 |
| `run_category_generalization_eval.sh` | 카테고리 레벨 100% Unseen GroupKFold 평가 |
| `run_hypothesis_evaluations.sh` | 가설 검증용 모드별 평가 실행 |

---

---
## 🇺🇸 English

### 1. Core Python CLI Execution Commands

#### ① Text Encoder Geometric Analysis (Layer Cosine Similarity & Linear Probe)
```bash
python -m benchmarks.src.analysis.run_analysis \
    --csv_path benchmarks/data/images/COCO_val_full_paired.csv \
    --model ViT-B-32 \
    --pretrained openai \
    --output_dir logs/analysis_modular/openai_vit_b32
```

#### ② BEAF Comprehensive Probing Sweep (All-layer Linear, MLP, Bilinear)
```bash
bash run_beaf_comprehensive_probing_sweep.sh
```

#### ③ BEAF 2×2 Factorial ANOVA + Vision Analysis
```bash
python -m benchmarks.src.analysis.run_beaf_analysis_v2 \
    --csv_path benchmarks/data/images/beaf_counterfactual_6col.csv \
    --output_dir logs/evaluation/beaf_counterfactual_v2/openai_vit_b32
```

#### ④ Negation Subspace Cross-Category Transfer
```bash
python -m benchmarks.src.analysis.subspace_analysis \
    --csv_path benchmarks/data/images/COCO_val_full_paired.csv \
    --split_by object_name \
    --output_dir logs/subspace_analysis
```

#### ⑤ Sparse Dimensions Hypothesis Verification
```bash
python -m benchmarks.src.analysis.eval_sparse_text_dimensions
```

#### ⑥ AB Swap Dataset Audit (P0) & Causal Evaluation (P1-P4)
```bash
# Priority 0: Dataset Artifact Audit
python benchmarks/src/analysis/beaf/audit_ab_swap_dataset.py

# Priority 1-4: 2x2 Joint Consistency & Diagnostic Evaluation
python benchmarks/src/analysis/beaf/run_ab_swap_evaluation.py
```

#### ⑦ Bilinear W Matrix Energy & Weight Analysis
```bash
python -m benchmarks.src.analysis.analyze_internal_weights
```

#### ⑧ Negation Existence Probe (Exp A & Exp B)
```bash
python -m benchmarks.src.evaluation.eval_negation_existence_probe \
    --csv_path benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv \
    --output_dir logs/evaluation/negation_existence_probe \
    --model ViT-B-32 --pretrained openai
```

#### ⑨ Scoring Head Comparison (6 Heads, 5-Fold OOF)
```bash
python -m benchmarks.src.evaluation.eval_scoring_heads \
    --model ViT-B-32 --pretrained openai \
    --csv benchmarks/data/images/COCO_val_negation.csv \
    --output-dir logs/evaluation/scoring_head_experiments
```

#### ⑩ Word-Swap Counterfactual Probe
```bash
python -m benchmarks.src.evaluation.eval_word_swap_probe \
    --model ViT-B-32 --pretrained openai \
    --coco-mcq benchmarks/data/images/COCO_val_negation.csv \
    --output-dir logs/evaluation/top_priority_experiments
```

#### ⑪ Vision/Text Ablation Shortcut Diagnostics
```bash
# Vision ablation
python -m benchmarks.src.evaluation.eval_vision_ablation_shortcut \
    --model ViT-B-32 --pretrained openai \
    --csv benchmarks/data/images/COCO_val_negation.csv \
    --output-dir logs/evaluation/vision_ablation

# Text ablation
python -m benchmarks.src.evaluation.eval_text_ablation_shortcut \
    --model ViT-B-32 --pretrained openai \
    --csv benchmarks/data/images/COCO_val_negation.csv \
    --output-dir logs/evaluation/text_ablation
```

#### ⑫ Unary 4-Stage Mechanistic Analysis (E1~E4)
```bash
python -m benchmarks.src.evaluation.eval_unary_mechanistic_analysis \
    --model ViT-B-32 --pretrained openai \
    --beaf-csv benchmarks/data/images/beaf_counterfactual_6col.csv \
    --ab-swap-csv benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv \
    --output-dir logs/evaluation/unary_mechanistic
```

#### ⑬ Per-Object Alignment Causal Intervention (5 Conditions)
```bash
python -m benchmarks.src.evaluation.eval_per_object_alignment_intervention \
    --model ViT-B-32 --pretrained openai \
    --beaf-csv benchmarks/data/images/beaf_counterfactual_6col.csv \
    --ab-swap-csv benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv \
    --output-dir logs/evaluation/per_object_intervention
```

---

### 2. Root Shell Scripts (`./*.sh`)

| Script Name | Purpose & Description |
|:---|:---|
| `run_hypothesis_experiments.sh` | Master runner for 3-stage hypothesis verification pipeline (H1–H4) |
| `run_top_priority_experiments.sh` | Executes Word-Swap Probe, Rank Sweep, and Concept Ablation |
| `run_scoring_head_experiments.sh` | Trains 6 Scoring Heads with 5-Fold OOF evaluation |
| `run_all_benchmarks_zero_shot.sh` | Evaluates all benchmarks (COCO, VOC, Synthetic, Video) Zero-Shot |
| `run_analysis.sh` | Automated 16-step text encoder geometric analysis |
| `run_pipeline_breakdown.sh` | Decomposes pipeline steps (Step 0–4) and calculates shift metrics |
| `run_layerwise_linear_probe.sh` | Evaluates Linear Probe across all text layers (Layer 0–12) |
| `run_sparse_dim_verification.sh` | Top-k dimension ablation for text and vision probes |
| `run_bilinear_verification.sh` | Mathematical equivalence test between Bilinear and Low-Rank (k=512) |
| `run_zero_shot_transfer.sh` | Evaluates trained Scorer transfer on external OOD benchmarks |
| `run_ab_swap_diagnostic.sh` | AB Swap diagnostics (Text Sanity Probe, Unary vs Compound, ΔS Margin) |
| `run_unary_mechanistic_pipeline.sh` | E1–E4 4-stage unary mechanistic analysis pipeline |
| `run_per_object_alignment_intervention.sh` | Per-object probe alignment & causal intervention (5 conditions) |
| `run_probe_failure_inspector.sh` | Collects vision/text probe OOF failure cases and error patterns |
| `run_vision_ablation_shortcut.sh` | Vision input ablation (Zero/Shuffle/Gaussian) shortcut test |
| `run_beaf_comprehensive_probing_sweep.sh` | Full-layer sweep of 9 BEAF probing classifiers |
| `run_beaf_multi_probing_sweep.sh` | Multi-probing model evaluation on BEAF dataset |
| `run_beaf_text_probe_pipeline.sh` | BEAF text probing pipeline |
| `run_beaf_dual_classifier_pipeline.sh` | Trains dual classifiers ($f_T, f_V$) and evaluates product scorer |
| `run_beaf_object_generalization_pipeline.sh` | Single/multi-object and 255-template transfer experiment |
| `run_cc12m_negfull_experiments.sh` | Evaluations and benchmark sweeps for CC12M-NegFull models |
| `run_e5v_negbench.sh` | Evaluates E5-V multimodal model on NegBench |
| `run_pca.sh`, `run_pca_full_v2.sh`, `run_pca_paired.sh`, `run_pca_v4.sh` | PCA and subspace spectrum analysis scripts |

---

### 3. `benchmarks/scripts/` Evaluation Scripts

| Script Name | Purpose & Description |
|:---|:---|
| `evaluate_images.sh` | Standard benchmark evaluation for image MCQ and Retrieval |
| `evaluate_videos.sh` | MSR-VTT video MCQ and Retrieval evaluation |
| `run_openai_clip.sh` | Baseline evaluation for official OpenAI CLIP |
| `run_pretrained_model_evaluations.sh` | Comparative evaluation across pretrained models (OpenCLIP, ConCLIP, NegCLIP) |
| `run_single_model_evaluations.sh` | Complete benchmark suite evaluation for a single model checkpoint |
| `run_single_model_evaluations_llava.sh` | NegBench MCQ evaluation for LLaVA multimodal models |
| `run_full_eval_with_scoring_head.sh` | Full benchmark evaluation using a trained Scoring Head |
| `run_scoring_head_evaluations.sh` | Scoring head evaluation runner |
| `run_category_generalization_eval.sh` | 100% Unseen category split evaluation via GroupKFold |
| `run_hypothesis_evaluations.sh` | Mode-specific hypothesis evaluation runner |
