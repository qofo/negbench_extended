# Evaluation Module Technical Guide

---
## 🇰🇷 한국어

### 1. 모듈 개요

`benchmarks/src/evaluation/` 패키지는 NegBench의 **MCQ/Retrieval 평가 및 메커니즘 실험** 모듈입니다. 23개 파일이 5개 카테고리로 구분됩니다.

| 카테고리 | 파일 수 | 설명 |
|:---|:---:|:---|
| 기본 평가 인프라 | 6 | MCQ, Retrieval, 유틸리티, 모델 래퍼 |
| 핵심 실험 스크립트 | 7 | Scoring Head, Probe, 일반화, 검증 |
| 절제(Ablation) 및 진단 | 3 | 텍스트/비전 임베딩 절제, AB-Swap 진단 |
| 메커니즘 분석 | 4 | Unary 분석, 정렬 개입, 실패 검사 |
| 미구현 | 1 | `eval_per_object_polarity_probe.py` (빈 파일) |

---

### 2. 기본 평가 인프라 (6개 파일)

#### `eval_negation.py` (245줄)
- **역할**: NegBench 메인 MCQ/Retrieval 평가 엔트리포인트 (CLIP/NegCLIP)
- **의존성**: `training.data`, `training.params`, `evaluation.utils`
- **흐름**: CLI 파싱 (`training.params.parse_args()`) → 모델 로드 → `evaluate()` 호출

#### `eval_negation_llava.py` (287줄)
- **역할**: LLaVA 모델 MCQ 평가 엔트리포인트
- **특징**:
  - `LLaVAModularEvaluator` 사용 (HF processor 기반)
  - `--vision-encoder-path`로 파인튜닝된 CLIP 비전 인코더 교체 가능
  - `--shuffle-mcq-options` 위치 편향 제거 권장
- **의존성**: `llava/` 패키지

#### `mcq.py` (451줄)
- **역할**: MCQ 평가 로직 핵심
- **주요 함수**:
  - `evaluate_model(model, dataloader, args, ...)`: 모델에 대한 MCQ 정확도, 오답 유형별 통계, 질문 유형별 정확도 산출
  - `mcq_eval(model, data, epoch, args, ...)`: 여러 MCQ 데이터셋(COCO, VOC2007, Synthetic, CheXpert)에 대한 일괄 평가

#### `retrieval.py` (195줄)
- **역할**: Cross-Modal Image-Text Retrieval 평가 (Recall@k)
- **주요 함수**: `evaluate_model(model, dataloader, args, tokenizer, recall_k_list=[5])`
- **메트릭**: Image→Text Recall@k, Text→Image Recall@k, 전체 Recall 평균

#### `utils.py` (152줄)
- **역할**: 평가 유틸리티 래퍼
- **주요 함수**: `evaluate(model, data, epoch, args, ...)` — MCQ + Retrieval 통합 호출

#### `modified_clip.py` (276줄)
- **역할**: Negation-Aware OpenCLIP 래퍼
- **클래스**: `NegationAwareCLIPWrapper`
- **4가지 평가 모드**:

  | 모드 | 가설 | 설명 |
  |:---|:---|:---|
  | `baseline` | — | 표준 OpenCLIP forward pass |
  | `procrustes_orthogonal` | H1 | Orthogonal Procrustes 변환 $Q$ ($Q^TQ = I$) |
  | `hyperplane_projection` | H2 | 크로스모달 Hyperplane 투영 $t' = \text{L2Norm}(t + \lambda (t \cdot w) w)$ |
  | `subspace_bilinear` | H4 | Subspace-Constrained Bilinear Tensor $M = I + \alpha U_{\text{neg}}^T U_{\text{neg}}$ |

- **Scorer 통합**: `--scorer-checkpoint`로 학습된 Scoring Head 로드 가능
- **수식**: → [루트 GUIDE.md §5 참조](../../../GUIDE.md)

---

### 3. 핵심 실험 스크립트 (7개 파일)

#### `scoring_heads.py` (460줄)
- **역할**: 8종 Scoring Head 모듈 정의
- **클래스 목록**:

  | # | 클래스 | 표현력 | 캐싱 |
  |:---|:---|:---|:---|
  | 1 | `CosineScorer` | Very Low | O(1) |
  | 2 | `WeightedCosineScorer` | Low | O(1) |
  | 3 | `BilinearScorer` | High | ✕ |
  | 4 | `LogisticRegressionScorer` | Medium | ✕ |
  | 5 | `ShallowMLPScorer` | Medium-High | ✕ |
  | 6 | `DeepMLPScorer` | High | ✕ |
  | 7 | `LowRankBilinearScorer` | Medium | O(1) |
  | 8 | `NonLinearBiEncoderScorer` | Medium-High | O(1) |

- **수식**: → [루트 GUIDE.md §5 참조](../../../GUIDE.md)

#### `eval_scoring_heads.py` (504줄)
- **역할**: 6종 Scoring Head 5-Fold OOF 비교 평가
- **CV 전략**: `StratifiedKFold(n_splits=5)` Out-of-Fold
- **출력**: 모델별 정확도, 체크포인트 저장 (`checkpoints/`)

#### `eval_negation_existence_probe.py` (528줄)
- **역할**: 부정 존재 프로브 (2가지 실험)
- **Exp A**: Layer-wise Pairwise Cosine Distance — 레이어가 깊어질수록 유사도 하락 여부 확인
- **Exp B**: Per-Object Polarity Probe — 객체별 Affirmed vs Negated 극성 방향 존재 검증
  - CV: `StratifiedKFold(5)` 수동 Train/Val
- **수식**: → [루트 GUIDE.md §5 ⑧ 참조](../../../GUIDE.md)

#### `eval_word_swap_probe.py` (254줄)
- **역할**: Token-Presence Bias 대응 Word-Swap Counterfactual 검증
- **목적**: 'not' 토큰 존재 감지가 아닌 실제 의미적 부정 표현 학습 여부 확인

#### `eval_layerwise_linear_probe.py` (254줄)
- **역할**: 텍스트 인코더 전 레이어 Linear Probe
- **방법**: Layer 0~12 + Final Projected 각각에 대해 `StratifiedKFold(5)` LogReg
- **의존성**: `analysis.extractor.extract_all_features_unified` 재사용

#### `eval_bilinear_verification.py` (348줄)
- **역할**: Bilinear vs Low-Rank(k=512) 수학적 동치성 검증 (3종 실험)
- **실험**: Score 차이, 정확도 차이, 가중치 SVD 비교

#### `eval_category_generalization.py` (347줄)
- **역할**: 100% Unseen Category GroupKFold 교차 일반화 평가
- **목적**: Scorer가 범용 매칭 함수를 학습했는지 vs 객체별 패턴 암기인지 검증
- **CV 전략**: `GroupKFold` (그룹 키: `object_name`)

---

### 4. 절제(Ablation) 및 진단 (3개 파일)

#### `eval_concept_ablation.py` (514줄)
- **역할**: 부정 Hyperplane 방향 제거 후 Scorer 성능 측정 (RQ2)
- **방법**: $t_{\text{ablated}} = t - (t \cdot w_{\text{neg}}) w_{\text{neg}}$로 부정 방향 사영 제거
- **검증**: 절제 후 ≈ Cosine 기준선이면 $w_{\text{neg}}$가 필수 채널 (Proof of Necessity)

#### `eval_rank_sweep.py` (307줄)
- **역할**: Low-Rank Rank-k 스윕 포화점 탐색 (RQ3)
- **스윕 대상**: `LowRankBilinearScorer`, `NonLinearBiEncoderScorer`
- **Rank 범위**: $k = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512$

#### `eval_text_ablation_shortcut.py` (602줄)
- **역할**: 텍스트 임베딩 절제 (shuffle/zero) 숏컷 진단
- **절제 조건**: 텍스트를 Shuffle / Zero 처리 → Scorer 정확도 하락 정도로 텍스트 의존도 측정

#### `eval_vision_ablation_shortcut.py` (521줄)
- **역할**: 비전 임베딩 절제 (zero/shuffle/gaussian) 숏컷 진단
- **절제 조건**: 비전을 Zero / Shuffle / Gaussian 처리 → Scorer가 실제 비전 정보 사용 여부 진단

#### `eval_ab_swap_negation_diagnostic.py` (707줄)
- **역할**: AB-Swap 3종 실험
  1. Text Sanity Probe: T_XY vs T_YX 레이어별 프로브 (~50% 기대)
  2. Unary vs Compound: 원자/단항부정/복합부정 정확도 비교
  3. ΔS Margin: 점수 차이 히스토그램

---

### 5. 메커니즘 분석 (4개 파일)

#### `eval_unary_mechanistic_analysis.py` (700줄)
- **역할**: E1~E4 4단계 Unary 메커니즘 분석
  - E1: 이미지/텍스트 표현에 선형 신호(존재/극성) 포함 여부
  - E2: 시각적 부재와 텍스트 부정이 의미적으로 정렬되는가
  - E3: 프로브 정렬이 코사인 정확도 마진을 예측하는가
  - E4: 코사인이 왜 실패하고 Bilinear가 무엇을 복원하는가 (대각 vs 비대각 절제)
- **출력**: 4개 Figure PNG + CSV + JSON

#### `eval_per_object_alignment_intervention.py` (500줄)
- **역할**: Per-Object 프로브 정렬 인과 개입 (5종 조건 2×2 매칭)
- **5종 조건**: 원본, 부정 방향 제거, 랜덤 방향 제거, 비전 프로브 방향 정렬, 텍스트-비전 합동 정렬

#### `eval_probe_failure_inspector.py` (689줄)
- **역할**: Vision/Text 프로브 OOF 실패 사례 수집 및 패턴 분석
- **출력**:
  - `vision_probing_failures.csv`, `text_probing_failures.csv`
  - `top_failed_objects_breakdown.csv`
  - 객체별/부정 구문별 에러율 시각화 PNG

#### `eval_per_object_polarity_probe.py`
- **상태**: ⚠️ **빈 파일 (미구현)**

#### `eval_zero_shot_transfer.py` (425줄)
- **역할**: Pre-trained Scorer의 OOD 벤치마크 Zero-Shot 전이 평가
- **지원 벤치마크**: SugarCrepe, Winoground, BEAF Counterfactual, Medical/Video CSV
- **모드**: MCQ/Paired (Accuracy) + Retrieval (Recall@1, Recall@5)

---

### 6. Cross-Validation 전략 요약

| 실험 | CV 전략 | 그룹 키 |
|:---|:---|:---|
| `eval_scoring_heads.py` | `StratifiedKFold(5)` OOF | — |
| `eval_negation_existence_probe.py` Exp B | `StratifiedKFold(5)` 수동 | — |
| `eval_layerwise_linear_probe.py` | `StratifiedKFold(5)` | — |
| `eval_category_generalization.py` | `GroupKFold` | `object_name` |
| `eval_word_swap_probe.py` | `StratifiedKFold(5)` | — |
| `eval_bilinear_verification.py` | `StratifiedKFold(5)` OOF | — |
| `eval_rank_sweep.py` | `StratifiedKFold(5)` OOF | — |
| `eval_zero_shot_transfer.py` | Eval-Only (No Train) | — |

---

---
## 🇺🇸 English

### 1. Module Overview

The `benchmarks/src/evaluation/` package is NegBench's **MCQ/Retrieval evaluation and mechanism experimentation** module. Its 23 files are organized into 5 categories.

| Category | Files | Description |
|:---|:---:|:---|
| Core Evaluation Infra | 6 | MCQ, Retrieval, utilities, model wrappers |
| Key Experiment Scripts | 7 | Scoring Head, Probe, generalization, verification |
| Ablation & Diagnostics | 3 | Text/vision embedding ablation, AB-Swap diagnostics |
| Mechanism Analysis | 4 | Unary analysis, alignment intervention, failure inspection |
| Unimplemented | 1 | `eval_per_object_polarity_probe.py` (empty file) |

---

### 2. Core Evaluation Infrastructure (6 files)

#### `eval_negation.py` (245 lines)
- **Role**: NegBench main MCQ/Retrieval evaluation entrypoint (CLIP/NegCLIP)
- **Dependencies**: `training.data`, `training.params`, `evaluation.utils`

#### `eval_negation_llava.py` (287 lines)
- **Role**: LLaVA model MCQ evaluation entrypoint
- **Features**: HF processor-based, swappable vision encoder via `--vision-encoder-path`, `--shuffle-mcq-options` recommended

#### `mcq.py` (451 lines)
- **Role**: Core MCQ evaluation logic
- **Key Functions**: `evaluate_model()`, `mcq_eval()` — accuracy, wrong answer type stats, per-question-type accuracy

#### `retrieval.py` (195 lines)
- **Role**: Cross-modal Image-Text Retrieval evaluation (Recall@k)
- **Metrics**: Image→Text Recall@k, Text→Image Recall@k

#### `utils.py` (152 lines)
- **Role**: Evaluation utility wrapper — unified MCQ + Retrieval call

#### `modified_clip.py` (276 lines)
- **Role**: Negation-Aware OpenCLIP wrapper
- **4 Evaluation Modes**:

  | Mode | Hypothesis | Description |
  |:---|:---|:---|
  | `baseline` | — | Standard OpenCLIP forward pass |
  | `procrustes_orthogonal` | H1 | Orthogonal Procrustes transform $Q$ ($Q^TQ = I$) |
  | `hyperplane_projection` | H2 | Cross-modal hyperplane projection $t' = \text{L2Norm}(t + \lambda (t \cdot w) w)$ |
  | `subspace_bilinear` | H4 | Subspace-Constrained Bilinear Tensor $M = I + \alpha U_{\text{neg}}^T U_{\text{neg}}$ |

- **Formulas**: → [Root GUIDE.md §5 Reference](../../../GUIDE.md)

---

### 3. Key Experiment Scripts (7 files)

#### `scoring_heads.py` (460 lines)
- **Role**: 8 Scoring Head module definitions
- **Classes**:

  | # | Class | Expressiveness | Caching |
  |:---|:---|:---|:---|
  | 1 | `CosineScorer` | Very Low | O(1) |
  | 2 | `WeightedCosineScorer` | Low | O(1) |
  | 3 | `BilinearScorer` | High | ✕ |
  | 4 | `LogisticRegressionScorer` | Medium | ✕ |
  | 5 | `ShallowMLPScorer` | Medium-High | ✕ |
  | 6 | `DeepMLPScorer` | High | ✕ |
  | 7 | `LowRankBilinearScorer` | Medium | O(1) |
  | 8 | `NonLinearBiEncoderScorer` | Medium-High | O(1) |

- **Formulas**: → [Root GUIDE.md §5 Reference](../../../GUIDE.md)

#### `eval_scoring_heads.py` (504 lines)
- **Role**: 6-scorer 5-Fold OOF comparison evaluation
- **CV**: `StratifiedKFold(5)` Out-of-Fold

#### `eval_negation_existence_probe.py` (528 lines)
- **Role**: Negation existence probe (2 experiments)
- **Exp A**: Layer-wise pairwise cosine distance — similarity drop across layers
- **Exp B**: Per-Object Polarity Probe — polarity direction existence per object
- **Formulas**: → [Root GUIDE.md §5 ⑧ Reference](../../../GUIDE.md)

#### `eval_word_swap_probe.py` (254 lines)
- **Role**: Token-presence bias counterfactual — verifies genuine negation learning vs 'not' detection

#### `eval_layerwise_linear_probe.py` (254 lines)
- **Role**: Text encoder all-layer Linear Probe (Layer 0–12 + Final)
- **Reuses**: `analysis.extractor.extract_all_features_unified`

#### `eval_bilinear_verification.py` (348 lines)
- **Role**: Bilinear vs Low-Rank(k=512) mathematical equivalence verification (3 experiments)

#### `eval_category_generalization.py` (347 lines)
- **Role**: 100% unseen category GroupKFold cross-generalization evaluation
- **CV**: `GroupKFold` by `object_name`

---

### 4. Ablation & Diagnostics (5 files)

#### `eval_concept_ablation.py` (514 lines)
- **Role**: Negation hyperplane direction removal + scorer performance (RQ2)
- **Method**: $t_{\text{ablated}} = t - (t \cdot w_{\text{neg}}) w_{\text{neg}}$

#### `eval_rank_sweep.py` (307 lines)
- **Role**: Low-Rank rank-k sweep saturation search (RQ3)
- **Ranks**: $k = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512$

#### `eval_text_ablation_shortcut.py` (602 lines)
- **Role**: Text embedding ablation (shuffle/zero) shortcut diagnosis

#### `eval_vision_ablation_shortcut.py` (521 lines)
- **Role**: Vision embedding ablation (zero/shuffle/gaussian) shortcut diagnosis

#### `eval_ab_swap_negation_diagnostic.py` (707 lines)
- **Role**: AB-Swap 3 experiments — Text Sanity Probe, Unary vs Compound, ΔS Margin

---

### 5. Mechanism Analysis (4 files)

#### `eval_unary_mechanistic_analysis.py` (700 lines)
- **Role**: E1–E4 4-stage Unary mechanistic analysis
  - E1: Linear signal existence in image/text representations
  - E2: Visual absence ↔ textual negation semantic alignment
  - E3: Probe alignment predicts cosine margin
  - E4: Cosine failure diagnosis via W = D + O (Diagonal vs Off-Diagonal ablation)

#### `eval_per_object_alignment_intervention.py` (500 lines)
- **Role**: Per-object probe alignment causal intervention (5 conditions, 2×2 matching)

#### `eval_probe_failure_inspector.py` (689 lines)
- **Role**: Vision/Text probe OOF failure case collection and pattern analysis

#### `eval_per_object_polarity_probe.py`
- **Status**: ⚠️ **Empty file (unimplemented)**

#### `eval_zero_shot_transfer.py` (425 lines)
- **Role**: Pre-trained scorer OOD zero-shot transfer evaluation
- **Benchmarks**: SugarCrepe, Winoground, BEAF Counterfactual, Medical/Video CSVs
- **Modes**: MCQ/Paired (Accuracy) + Retrieval (Recall@1, Recall@5)

---

### 6. Cross-Validation Strategy Summary

| Experiment | CV Strategy | Group Key |
|:---|:---|:---|
| `eval_scoring_heads.py` | `StratifiedKFold(5)` OOF | — |
| `eval_negation_existence_probe.py` Exp B | `StratifiedKFold(5)` manual | — |
| `eval_layerwise_linear_probe.py` | `StratifiedKFold(5)` | — |
| `eval_category_generalization.py` | `GroupKFold` | `object_name` |
| `eval_word_swap_probe.py` | `StratifiedKFold(5)` | — |
| `eval_bilinear_verification.py` | `StratifiedKFold(5)` OOF | — |
| `eval_rank_sweep.py` | `StratifiedKFold(5)` OOF | — |
| `eval_zero_shot_transfer.py` | Eval-Only (No Train) | — |
