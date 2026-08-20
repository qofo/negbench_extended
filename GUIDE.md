# NegBench: Representation Analysis & BEAF Probing Subsystem Technical Guide (GUIDE.md)

> **목적**: 본 문서는 `NegBench` 프로젝트의 전체 아키텍처, 파일 목록, 데이터 흐름, 수학적 수식을 즉시 파악할 수 있도록 정리한 기술 가이드입니다.

---

## 1. 서브시스템 개요 및 핵심 연구 질문

* **연구 배경**: Vision-Language Model(CLIP, NegCLIP 등)은 부정어(Negation)가 포함된 텍스트와 이미지를 올바르게 매칭하지 못함 (CVPR 2025 NegBench).
* **핵심 질문 (Core Research Questions)**:
  1. **Representation Loss인가, Scoring Head(Cosine)의 한계인가?**
     * 텍스트 인코더와 비전 인코더의 잠재 공간에 부정 정보가 보존되어 있는가? (Linear Probing, SVD, Subspace Analysis)
     * 파라미터가 0개인 Cosine Similarity($\approx 1:1$ 대각 차원 매칭)가 병목인가, 아니면 차원 간 상호작용(Bilinear/MLP)이 필요한가?
  2. **소수 숏컷 차원인가, 분산 표현인가? (Sparse vs. Distributed)**
     * 프로브의 99.9% 정확도가 1~2개 소수 차원에 의존하는가, 512차원 전체에 고르게 분산되어 있는가?
  3. **BEAF (Benchmark Evaluation & Analysis Framework)**:
     * 원본 이미지($\text{Orig}$, 객체 존재)와 Inpainting 편집 이미지($\text{CF}$, 객체 부재)로 구성된 1:1 Counterfactual Pair를 사용하여 인코더 메커니즘을 인과적으로 정밀 검증.

---

## 2. 전체 파일 목록 및 모듈 맵

### 2.1 `benchmarks/src/analysis/` — 표현 분석 코어 (14개 파일)

```
benchmarks/src/analysis/
├── __init__.py                     # 분석 패키지 공개 API
│                                   #   Export: PipelineStep, MetadataKey, AnalysisConfig, RetrievalConfig,
│                                   #           to_bool, get_layer_features, l2_normalize,
│                                   #           batch_cosine_similarity, batch_dot_product, batch_l2_distance,
│                                   #           set_seed, DEFAULT_TUNING_GRIDS
├── config.py                       # PipelineStep, MetadataKey, AnalysisConfig, RetrievalConfig,
│                                   #   기하학 연산 (l2_normalize, batch_cosine_similarity, batch_dot_product, batch_l2_distance),
│                                   #   to_bool, get_layer_features, set_seed, DEFAULT_TUNING_GRIDS
├── extractor.py                    # 단일 패스(Single-pass) 전 레이어/단계별 특징 추출 엔진
├── metrics.py                      # 16단계 기하 메트릭, Welch's t-test, 선형 프로빙, PCA 유효 랭크
├── subspace_analysis.py            # 부정 차이 벡터(Δ) Global Subspace, SVD 스펙트럼, split_by 카테고리 전이
├── eval_sparse_text_dimensions.py  # 소수 차원(Sparse Dimensions) 가설 검증 (Top-k 절제 정확도)
├── analyze_internal_weights.py     # Linear Probe 가중치 및 Bilinear W 비대각 에너지(97.17%) 분석
├── reporter.py                     # PNG 시각화 렌더러 및 JSON/CSV 보고서 직렬화
├── run_analysis.py                 # 텍스트 인코더 기하 구조 분석 메인 오케스트레이터
├── run_beaf_analysis_v2.py         # BEAF 2x2 Factorial ANOVA + Vision 통합 실행 엔트리포인트
│                                   #   ⚠️ 현재 Step 6~10 (Direction Preservation, Image-Image Cosine,
│                                   #      4-Way Matrix, Pipeline Breakdown, Retrieval) 주석 처리 상태
├── train_beaf_dual_probes.py       # Dual Classifier (f_T, f_V) 학습 및 NPZ 가중치 저장
├── run_beaf_flexible_probing.py    # 9종 프로빙 분류기 전 레이어 스윕 (Nested GroupKFold)
├── run_beaf_train_val_per_object.py# 객체별 Train vs Val 일반화 갭(Gap) 측정
└── run_beaf_object_generalization.py# OOD 단일/다중 객체 및 255개 템플릿 전이 실험
```

### 2.2 `benchmarks/src/analysis/beaf/` — BEAF 프레임워크 (12개 파일)

```
benchmarks/src/analysis/beaf/
├── __init__.py                     # beaf 패키지 공개 API (프로브 클래스 4종, 팩토리, 시각화, 통계, 로더 export)
├── probe_factory.py                # [Single Source of Truth] PyTorch 프로브 4종 및 Sklearn 래퍼
│                                   #   Classes: LowRankBilinearPyTorch, FullBilinearPyTorch,
│                                   #            MLPVisionPyTorch, ElementWiseNonLinearPyTorch,
│                                   #            PyTorchProbeEstimator
│                                   #   SUPPORTED_PROBES: logistic, svm_linear, ridge, sgd_log, sgd_hinge,
│                                   #                     svm_rbf, mlp, bilinear_lowrank, bilinear_full
│                                   #   ⚠️ ElementWiseNonLinearPyTorch는 SUPPORTED_PROBES 미등록
├── beaf_loader.py                  # Counterfactual 6-column CSV 로더 및 엄격한 Pair 무결성 검증
├── vision_mechanisms.py            # 비전 트랜스포머 레이어 분해, SVD 스윕, 방향성 보존 분석
├── object_experiment.py            # 4-Way Cross Cosine, 문법 서식, 밸런스드 샘플링, 코사인 결합 Dual Probe 평가
├── beaf_stats.py                   # 2x2 Factorial ANOVA (Text/Visual/Interaction 효과) & Bootstrap 95% CI
├── audit_ab_swap_dataset.py        # AB-Swap 데이터셋 아티팩트/확장자 스큐/블러 감사 (0순위 Audit)
├── run_ab_swap_evaluation.py       # AB-Swap 1순위~4순위 인과성 평가 파이프라인 (2x2 Joint Consistency)
├── plot_beaf_correlations.py       # Image-fixed vs Text-fixed 상관관계 플롯
├── compare_beaf_probes.py          # 다중 프로빙 결과 JSON 비교 플롯 생성
├── generate_probing_comparison.py  # 프로빙 정확도 수평 막대그래프 렌더러
└── visualizer.py                   # BEAF 전용 플롯(히스토그램, 산점도, 2D 마진 공간, 4-Way 히트맵) 렌더러
```

### 2.3 `benchmarks/src/evaluation/` — MCQ 평가 및 메커니즘 실험 (23개 파일)

```
benchmarks/src/evaluation/
├── eval_negation.py                # NegBench 메인 MCQ/Retrieval 평가 엔트리포인트 (CLIP/NegCLIP)
├── eval_negation_llava.py          # LLaVA 모델 MCQ 평가 엔트리포인트
├── mcq.py                          # MCQ 평가 로직 (evaluate_model, mcq_eval)
├── retrieval.py                    # Cross-Modal Image-Text Retrieval 평가 (Recall@k)
├── utils.py                        # 평가 유틸리티 (evaluate 래퍼: MCQ + Retrieval 통합)
├── modified_clip.py                # Negation-Aware OpenCLIP 래퍼 (Procrustes, Hyperplane Projection, Subspace Bilinear)
├── scoring_heads.py                # 8종 Scoring Head 모듈 (Cosine, WCos, Bilinear, LR, ShallowMLP, DeepMLP,
│                                   #   LowRankBilinear, NonLinearBiEncoder)
│
│  ── 핵심 실험 스크립트 ──
├── eval_scoring_heads.py           # 6종 Scoring Head 5-Fold OOF 비교 평가
├── eval_negation_existence_probe.py# Exp A (Layer-wise Pairwise Cosine) + Exp B (Per-Object Polarity Probe)
├── eval_word_swap_probe.py         # Token-Presence Bias 대응 Word-Swap Counterfactual 검증
├── eval_layerwise_linear_probe.py  # 텍스트 인코더 전 레이어 Linear Probe (5-Fold Stratified CV)
├── eval_bilinear_verification.py   # Bilinear vs Low-Rank(k=512) 수학적 동치성 검증 (3종 실험)
├── eval_category_generalization.py # 100% Unseen Category GroupKFold 교차 일반화 평가
├── eval_concept_ablation.py        # 부정 Hyperplane 방향 제거 후 Scorer 성능 측정 (RQ2)
├── eval_rank_sweep.py              # Low-Rank Rank-k 스윕 포화점 탐색 (RQ3)
├── eval_zero_shot_transfer.py      # Pre-trained Scorer의 OOD 벤치마크 Zero-Shot 전이 평가
│
│  ── 절제(Ablation) 및 진단 ──
├── eval_text_ablation_shortcut.py  # 텍스트 임베딩 절제 (shuffle/zero) 숏컷 진단
├── eval_vision_ablation_shortcut.py# 비전 임베딩 절제 (zero/shuffle/gaussian) 숏컷 진단
├── eval_ab_swap_negation_diagnostic.py # AB-Swap 3종 실험 (Text Sanity Probe, Unary vs Compound, ΔS Margin)
│
│  ── 메커니즘 분석 ──
├── eval_unary_mechanistic_analysis.py  # E1~E4 4단계 Unary 메커니즘 분석 (Probe, Alignment, Margin, W Ablation)
├── eval_per_object_alignment_intervention.py # Per-Object 프로브 정렬 인과 개입 (5종 조건 2x2 매칭)
├── eval_probe_failure_inspector.py     # Vision/Text 프로브 OOF 실패 사례 수집 및 패턴 분석
└── eval_per_object_polarity_probe.py   # [빈 파일] 미구현
```

### 2.4 기타 디렉토리

```
benchmarks/src/data_generation/           # 데이터 생성 스크립트
├── create_full_coco_paired_v2.py         #   COCO 기반 Positive/Negative 페어 CSV 생성
├── generate_beaf_ab_swap_dataset.py      #   BEAF AB-Swap 기본 데이터셋 생성
└── generate_beaf_ab_swap_diverse_dataset.py # BEAF AB-Swap 다양화 데이터셋 생성

benchmarks/src/e5v_analysis/              # E5-V 모델 분석
├── __init__.py
├── e5v_wrapper.py                        #   E5-V 모델 래퍼
├── eval_negbench_e5v.py                  #   E5-V NegBench 평가
└── utils.py                              #   E5-V 유틸리티

benchmarks/src/training/                  # NegCLIP 파인튜닝 인프라
├── main.py                               #   학습 메인 엔트리포인트
├── train.py                              #   학습 루프
├── data.py                               #   CsvMCQDataset 및 데이터 로더
├── params.py                             #   하이퍼파라미터 설정
└── (scheduler, distributed, profiler, zero_shot 등)
```

---

## 3. 리팩토링 변경 이력

1. **PyTorch 프로빙 모델 단일화 (`probe_factory.py`)**:
   * `vision_mechanisms.py`, `object_experiment.py`, `train_beaf_dual_probes.py`에 난립해 있던 중복 클래스 정의를 `probe_factory.py` 하나로 일원화.
   * `LowRankBilinearPyTorch`, `FullBilinearPyTorch`, `MLPVisionPyTorch`, `ElementWiseNonLinearPyTorch` 4종이 표준 모델로 등록됨.
2. **공용 유틸리티 및 시드 일원화 (`config.py`)**:
   * `to_bool(v)`: 4개 파일에 중복 구현되어 있던 불리언 파서를 통합.
   * `get_layer_features(vis, key)`: 3개 파일에 중복 구현되어 있던 레이어 특징 추출 함수 `_get_feats`를 통합.
   * `set_seed(seed=42)`: Python/NumPy/PyTorch 일괄 시드 제어 함수 및 표준 튜닝 그리드(`DEFAULT_TUNING_GRIDS`) 등록.
   * ⚠️ **미완료**: `evaluation/` 디렉토리의 8개 파일에 `set_seed()`가 여전히 로컬 중복 정의됨.
3. **수학적 오류 및 실험 모호성 교정**:
   * `object_experiment.py`: `evaluate_dual_classifier_product_scorer`에서 텍스트 항($f_T$)이 단순 상수로 소거되던 버그를 수정하고, 코사인 유사도와 마진 곱 결합 $\text{Score} = \cos(v, t) \cdot (f_V(v) \cdot f_T(t))$ 적용 (시그모이드 제외하여 부호 일치성 보존).
   * `subspace_analysis.py`: `evaluate_cross_category_transfer`에 명시적 `split_by="object_name"` 인자를 추가하여 실제 80개 객체 카테고리 기준 분할 강제.
   * `run_beaf_analysis_v2.py`: `_classify_failure_mode`를 3-bit 진리표(8가지 경우) 완전 배타적 매핑으로 개편.
   * `run_ab_swap_evaluation.py`: 더미 이미지 벡터의 $T_{XY}$ 방향성 선호 메트릭 명칭을 `image_blind_xy_preference_pct`로 명확화.

---

## 4. 수학적 모델 및 알고리즘 데이터 흐름

### ① 텍스트 인코더 16단계 기하 구조 분석 (`run_analysis.py`)
1. **단일 패스 추출 (`extractor.py`)**:
   * `Step 0 (Embed)`: 토큰 + 위치 임베딩 ($h_0$)
   * `Layer 1 ~ 12`: 트랜스포머 잔차 블록 출력 중 EOT 토큰 풀링 ($h_1 \dots h_{12}$)
   * `Step 2 (LN)`: Layer 12 통과 후 LayerNorm ($z_{\text{LN}}$)
   * `Step 3 (Proj)`: 선형 투영 ($z_{\text{proj}} = z_{\text{LN}} W_{\text{proj}}$)
   * `Step 4 (L2Norm)`: 최종 정규화 ($z_{\text{final}} = z_{\text{proj}} / \|z_{\text{proj}}\|_2$)
2. **레이어별 코사인 유사도 계산 (`metrics.py`)**:
   $$\text{CosineSim}(h_{\text{pos}}^{(l)}, h_{\text{neg}}^{(l)}) = \frac{h_{\text{pos}}^{(l)} \cdot h_{\text{neg}}^{(l)}}{\|h_{\text{pos}}^{(l)}\|_2 \|h_{\text{neg}}^{(l)}\|_2}$$
   * 16개 단계별 평균 유사도 및 L2 거리를 Dual-Axis 꺾은선 그래프(`pipeline_step_lineplot.png`)로 렌더링.
3. **단계별 선형 프로빙 (`metrics.py`)**:
   * $X = [Z_{\text{pos}}; Z_{\text{neg}}]$, $y = [\mathbf{1}_N; \mathbf{0}_N]$ 구성 후 `StratifiedKFold(n_splits=5)`로 Logistic Regression 학습.
   * Step 0, Step 2, Step 4의 분류 정확도 막대그래프(`linear_probe_classification_acc.png`) 렌더링.

---

### ② BEAF 4-Way Cross Cosine & 2x2 Factorial ANOVA (`beaf_stats.py`)

4가지 상호 유사도 매트릭스:
* $A = \text{sim}(\text{Img}_{\text{orig}}, \text{Txt}_{\text{pos}})$ (원본 이미지 $\times$ 긍정 캡션)
* $B = \text{sim}(\text{Img}_{\text{orig}}, \text{Txt}_{\text{neg}})$ (원본 이미지 $\times$ 부정 캡션)
* $C = \text{sim}(\text{Img}_{\text{cf}}, \text{Txt}_{\text{pos}})$ (객체 부재 이미지 $\times$ 긍정 캡션)
* $D = \text{sim}(\text{Img}_{\text{cf}}, \text{Txt}_{\text{neg}})$ (객체 부재 이미지 $\times$ 부정 캡션)

3대 직교 주효과(Orthogonal Main Effects):
$$\text{Text Main Effect} = \frac{(A - B) + (C - D)}{2} \quad (\text{텍스트가 점수를 주도하는 편향})$$
$$\text{Visual Main Effect} = \frac{(A - C) + (B - D)}{2} \quad (\text{시각적 객체 유무가 점수를 주도하는 정도})$$
$$\text{Interaction Effect} = (A - B) - (C - D) \quad (\text{시각-텍스트가 결합하여 부정어를 올바르게 푸는 정도})$$

---

### ③ 프로브 모델 수식 정의 (`probe_factory.py`)

| 모델 클래스 | 수식 정의 | 특징 및 용도 |
|:---|:---|:---|
| `ElementWiseNonLinearPyTorch` | $f(x) = \sum_d w_d \cdot \text{GELU}(x_d) + b$ | 차원 간 혼합 0% (순수 비선형성 대조군) |
| `LowRankBilinearPyTorch` | $f(x) = \sum_{r=1}^R (x U_r)(x V_r) + x w_{\text{lin}} + b$ | Low-rank 교차 차원 상호작용 |
| `FullBilinearPyTorch` | $f(x) = x^T W x + x w_{\text{lin}} + b$ | Full $D \times D$ 2차 형식 상호작용 |
| `MLPVisionPyTorch` | $f(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$ | 2계층 비선형 MLP |

---

### ④ Bilinear $W$ 행렬 에너지 분해 (`analyze_internal_weights.py`)

* $\text{Total Energy} = \sum_{i, j} W_{i, j}^2$
* $\text{Diagonal Energy (Direct Matching)} = \sum_i W_{i, i}^2 \approx \mathbf{2.83\%}$
* $\text{Off-Diagonal Energy (Cross-Dimension Interaction)} = \text{Total} - \text{Diag} \approx \mathbf{97.17\%}$
* **결론**: 기존 코사인 유사도가 1:1 대각 차원만 매칭하는 것과 달리, 부정어 인식은 비대각 교차 차원 상호작용이 필수적임.

---

### ⑤ 부정 부분공간 및 유효 랭크 (`subspace_analysis.py`)

* 부정 차이 벡터 $D = X_{\text{pos}} - X_{\text{neg}}$의 공분산 고유값 $\lambda_i$로부터 산출:
  $$p_i = \frac{\lambda_i}{\sum \lambda_i}, \quad H = -\sum p_i \ln p_i, \quad r_{\text{eff}} = \exp(H)$$
* SVD를 통해 글로벌 부정 기저 $U_{\text{neg}}$를 추출하고, `--split_by object_name`을 통해 Unseen Category에 대한 Zero-shot 전이율 검증.

---

### ⑥ 소수 차원 가설 검증 (`eval_sparse_text_dimensions.py`)

* 가중치 벡터 $w$의 차원을 절대값 순으로 정렬 후, 상위 $k$개 차원만 남기고 나머지를 0으로 절제(Ablation):
  * $k=1, 2, 5, 10, 20, 50, 100, 512$에 대한 **Zero-shot Ablated Accuracy** 측정.
  * 소수 차원만으로는 성능이 급락함을 보여주어 **"부정 정보는 고차원에 고르게 분산(Distributed)되어 있다"**는 사실을 증명.

---

### ⑦ AB Compositional Swap 인과성 평가 (`run_ab_swap_evaluation.py`)

* **1순위 (2x2 Joint Consistency)**:
  $$\min(S(I_{XY}, T_{XY}), S(I_{YX}, T_{YX})) > \max(S(I_{XY}, T_{YX}), S(I_{YX}, T_{XY}))$$
* **2순위 (Text Separability)**: $T_{XY}$ vs $T_{YX}$의 코사인 유사도 및 프로빙 분리도.
* **3순위 (Vision Per-Pair Probing)**: $I_{XY}$ vs $I_{YX}$의 Base Scene GroupKFold 분리도.
* **4순위 (Image-Blind Forced-Choice)**: `image_blind_xy_preference_pct` (더미 이미지 벡터가 $T_{XY}$를 $T_{YX}$보다 선호하는 편향 비율 진단).

---

### ⑧ Negation Existence Probe (`eval_negation_existence_probe.py`)

* **Exp A (Layer-wise Pairwise Cosine Distance)**: Counterfactual Pair (T\_XY, T\_YX) 간 코사인 유사도를 전 레이어에 걸쳐 측정. 유사도가 레이어가 깊어질수록 하락하면 부정 바인딩이 Self-Attention에 의해 구축됨을 의미.
* **Exp B (Per-Object Polarity Probe)**: 각 객체 O에 대해 Affirmed vs Negated 문장을 수집 (양쪽 모두 단어 O 포함 → 어휘 숏컷 불가). Per-Object StratifiedKFold LogReg 프로브로 극성(Polarity) 방향 존재 여부 검증.
  * CV 전략: `StratifiedKFold(n_splits=5)` 수동 Train/Val (Train+Val 정확도 모두 보고)

---

### ⑨ Scoring Head 비교 및 숏컷 진단

* **`eval_scoring_heads.py`**: 6종 Scoring Head (Cosine, Weighted Cosine, Bilinear, LR, Shallow MLP, Deep MLP)를 `StratifiedKFold(n_splits=5)` Out-of-Fold 방식으로 비교.
* **`eval_vision_ablation_shortcut.py`**: 비전 임베딩을 Zero/Shuffle/Gaussian으로 절제하여 Scorer가 실제 비전 정보를 사용하는지 진단.
* **`eval_text_ablation_shortcut.py`**: 텍스트 임베딩을 Shuffle/Zero로 절제하여 Scorer의 텍스트 의존도 진단.
* **`eval_concept_ablation.py`**: 부정 Hyperplane 방향을 사영(Projection Out)하여 제거 후 모든 Scorer 성능 측정.

---

## 5. 실험별 Cross-Validation 전략 요약

| 실험 | CV 전략 | 그룹 키 |
|:---|:---|:---|
| `run_analysis.py` (텍스트 기하 프로브) | `StratifiedKFold(5)` | — |
| `run_beaf_flexible_probing.py` (BEAF 9종 프로빙) | `GroupKFold` (Nested inner CV) | `pair_id` |
| `eval_negation_existence_probe.py` Exp B | `StratifiedKFold(5)` 수동 | — |
| `eval_scoring_heads.py` | `StratifiedKFold(5)` OOF | — |
| `eval_category_generalization.py` | `GroupKFold` | `object_name` |
| `object_experiment.py` (single object) | `StratifiedKFold(5)` | — |
| `object_experiment.py` (train_val_experiment) | `train_test_split(70:30)` | — |
| `vision_mechanisms.py` (vision probe) | `StratifiedKFold(5)` 또는 `GroupKFold` | `pair_id` (옵션) |

---

## 6. ⚠️ 주의해야 할 실험 설계상 한계 및 리뷰어 대응 전략

1. **`DualClassifierProductScorer`의 무조건부(Unconditional) 모순**:
   * `train_beaf_dual_probes.py`에서 학습된 $f_V(v)$는 텍스트 입력이 없으므로 특정 이미지 $v$에 대해 단일 상수(예: $+1.5$)만 출력함.
   * MCQ 평가 시 $f_T(t)$가 긍정이면 $+1$, 부정이면 $-1$이 되어, **객체가 없는 이미지에서도 부정 선택지가 무조건 음수가 되어 오답(긍정 선택지)을 선택하는 이론적 모순**이 발생함.
   * $\rightarrow$ 해결책: $f(v, t)$는 무조건부 분리가 아닌 조건부 결합(Bilinear $v^T W t$ 등)이어야 함.
   * $\rightarrow$ 참고: `scoring_heads.py`에 8종 조건부 Scorer가 구현되어 있으나, BEAF 파이프라인에는 아직 미통합.
2. **Text Probe 99.9%의 Token-Presence 편향**:
   * Positive vs Negative 문장으로만 학습된 Linear Probe가 99.9%를 찍는 것은 'not' 토큰의 존재(Presence)만 감지한 것일 수 있음.
   * $\rightarrow$ 해결책: 'not' 토큰을 양쪽에 유지한 **Word-Swap Counterfactual Pair**(`eval_word_swap_probe.py`)로 평가해야 함.
3. **BEAF Inpainting Artifacts 간섭 가능성**:
   * $f_V(v)$가 객체의 의미적 부재가 아닌 Inpainting 생성 흔적(Blur/Brush)을 숏컷으로 학습했을 가능성 $\rightarrow$ `audit_ab_swap_dataset.py`로 아티팩트 감사 필수.
4. **`ElementWiseNonLinearPyTorch` 실험 통합 미완료**:
   * 클래스는 `probe_factory.py`에 정의되어 있으나, `SUPPORTED_PROBES` 리스트 및 `create_probe_classifier()` 분기에 미등록 상태.
   * 따라서 "차원 간 혼합 0% 대조군"이라는 실험적 주장에 대한 자동화된 정량적 비교가 불가능.
5. **`run_beaf_analysis_v2.py` 현재 실행 범위 제한**:
   * Step 6~10 (Vision Direction Preservation, Image-Image Cosine, 4-Way Matrix, Pipeline Breakdown, Retrieval)이 주석 처리되어 있으며, 현재 2x2 Factorial ANOVA + Per-Object Layerwise Stats만 실행됨.

---

## 7. 주요 실행 명령어 (CLI Cheatsheet)

```bash
# ──────────────────────────────────────────────────
# 1. 텍스트 인코더 기하 구조 분석 (레이어 코사인 유사도 & 선형 프로브)
# ──────────────────────────────────────────────────
python -m benchmarks.src.analysis.run_analysis \
    --csv_path benchmarks/data/images/COCO_val_full_paired.csv \
    --model ViT-B-32 \
    --pretrained openai \
    --output_dir logs/analysis_modular/openai_vit_b32

# ──────────────────────────────────────────────────
# 2. BEAF 다중 프로빙 스윕 (Linear, MLP, Bilinear 전 레이어 평가)
# ──────────────────────────────────────────────────
bash run_beaf_comprehensive_probing_sweep.sh

# ──────────────────────────────────────────────────
# 3. BEAF 2x2 Factorial ANOVA + Vision 분석
# ──────────────────────────────────────────────────
python -m benchmarks.src.analysis.run_beaf_analysis_v2 \
    --csv_path benchmarks/data/images/beaf_counterfactual_6col.csv \
    --output_dir logs/evaluation/beaf_counterfactual_v2/openai_vit_b32

# ──────────────────────────────────────────────────
# 4. 객체 카테고리 기준 부정 부분공간 전이 평가
# ──────────────────────────────────────────────────
python -m benchmarks.src.analysis.subspace_analysis \
    --csv_path benchmarks/data/images/COCO_val_full_paired.csv \
    --split_by object_name \
    --output_dir logs/subspace_analysis

# ──────────────────────────────────────────────────
# 5. 소수 차원 가설 (Sparse Dimensions) 검증
# ──────────────────────────────────────────────────
python -m benchmarks.src.analysis.eval_sparse_text_dimensions

# ──────────────────────────────────────────────────
# 6. AB Swap 데이터셋 감사 (0순위) 및 인과성 평가 (1~4순위)
# ──────────────────────────────────────────────────
python benchmarks/src/analysis/beaf/audit_ab_swap_dataset.py
python benchmarks/src/analysis/beaf/run_ab_swap_evaluation.py

# ──────────────────────────────────────────────────
# 7. Bilinear W 행렬 가중치 및 Off-diagonal 에너지 분석
# ──────────────────────────────────────────────────
python -m benchmarks.src.analysis.analyze_internal_weights

# ──────────────────────────────────────────────────
# 8. Negation Existence Probe (Exp A: Cosine Distance + Exp B: Polarity Probe)
# ──────────────────────────────────────────────────
python -m benchmarks.src.evaluation.eval_negation_existence_probe \
    --csv_path benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv \
    --output_dir logs/evaluation/negation_existence_probe \
    --model ViT-B-32 --pretrained openai

# ──────────────────────────────────────────────────
# 9. Scoring Head 비교 평가 (6종, 5-Fold OOF)
# ──────────────────────────────────────────────────
python -m benchmarks.src.evaluation.eval_scoring_heads \
    --model ViT-B-32 --pretrained openai \
    --csv benchmarks/data/images/COCO_val_negation.csv \
    --output-dir logs/evaluation/scoring_head_experiments

# ──────────────────────────────────────────────────
# 10. Word-Swap Counterfactual Probe (Token-Presence Bias 대응)
# ──────────────────────────────────────────────────
python -m benchmarks.src.evaluation.eval_word_swap_probe \
    --model ViT-B-32 --pretrained openai \
    --coco-mcq benchmarks/data/images/COCO_val_negation.csv \
    --output-dir logs/evaluation/top_priority_experiments

# ──────────────────────────────────────────────────
# 11. Vision/Text Ablation 숏컷 진단
# ──────────────────────────────────────────────────
python -m benchmarks.src.evaluation.eval_vision_ablation_shortcut \
    --model ViT-B-32 --pretrained openai \
    --csv benchmarks/data/images/COCO_val_negation.csv \
    --output-dir logs/evaluation/vision_ablation

python -m benchmarks.src.evaluation.eval_text_ablation_shortcut \
    --model ViT-B-32 --pretrained openai \
    --csv benchmarks/data/images/COCO_val_negation.csv \
    --output-dir logs/evaluation/text_ablation

# ──────────────────────────────────────────────────
# 12. Unary 4-Stage 메커니즘 분석 (E1~E4)
# ──────────────────────────────────────────────────
python -m benchmarks.src.evaluation.eval_unary_mechanistic_analysis \
    --model ViT-B-32 --pretrained openai \
    --beaf-csv benchmarks/data/images/beaf_counterfactual_6col.csv \
    --ab-swap-csv benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv \
    --output-dir logs/evaluation/unary_mechanistic

# ──────────────────────────────────────────────────
# 13. Per-Object Alignment Causal Intervention (5종 조건)
# ──────────────────────────────────────────────────
python -m benchmarks.src.evaluation.eval_per_object_alignment_intervention \
    --model ViT-B-32 --pretrained openai \
    --beaf-csv benchmarks/data/images/beaf_counterfactual_6col.csv \
    --ab-swap-csv benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv \
    --output-dir logs/evaluation/per_object_intervention
```
