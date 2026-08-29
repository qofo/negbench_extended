# BEAF Framework Technical Guide

---
## 🇰🇷 한국어

### 1. 프레임워크 개요

`benchmarks/src/analysis/beaf/` 패키지는 **BEAF (Benchmark Evaluation & Analysis Framework)**의 핵심 구현으로, Counterfactual Pair 기반 인과적 메커니즘 검증을 수행합니다.

**핵심 아이디어**: 원본 이미지($\text{Orig}$, 객체 존재)와 Inpainting 편집 이미지($\text{CF}$, 객체 부재)로 구성된 1:1 Counterfactual Pair를 사용하여, CLIP 인코더가 부정어를 처리하는 메커니즘을 인과적으로 정밀 검증합니다.

---

### 2. 파일별 상세 설명 (12개 파일)

#### `__init__.py` (112줄)
- **역할**: beaf 패키지 공개 API 정의
- **Export 카테고리**:
  - Visualizers (11개 함수): `render_image_image_histogram`, `render_4way_heatmap`, `render_text_vs_visual_scatter` 등
  - Vision Mechanisms (4개 함수): `extract_vision_features_unified`, `compute_vision_pipeline_breakdown` 등
  - Object Experiments (11개 함수/클래스): `run_single_object_analysis`, `VisionProbeWrapper` 등
  - Statistics (3개 함수): `compute_2x2_factorial_anova`, `compute_quadrant_bootstrap_ci`, `compute_per_object_layerwise_stats`
  - Data Loading (2개 함수): `load_beaf_csv`, `load_and_verify_counterfactual_pairs`
  - Probe Factory (8개): `SUPPORTED_PROBES`, 4개 PyTorch 모델 클래스, `PyTorchProbeEstimator`, `create_probe_classifier`, `get_param_candidates`, `format_params`

#### `probe_factory.py` (320줄) — **[Single Source of Truth]**
- **역할**: PyTorch 프로브 4종 및 Sklearn 래퍼의 통합 팩토리
- **PyTorch 프로브 모델**:

  | 클래스 | 용도 |
  |:---|:---|
  | `LowRankBilinearPyTorch` | Low-rank 교차 차원 상호작용 |
  | `FullBilinearPyTorch` | Full $D \times D$ 2차 형식 상호작용 |
  | `MLPVisionPyTorch` | 2계층 비선형 MLP |
  | `ElementWiseNonLinearPyTorch` | 차원 간 혼합 0% (순수 비선형성 대조군) |

- **SUPPORTED_PROBES**: `logistic`, `svm_linear`, `ridge`, `sgd_log`, `sgd_hinge`, `svm_rbf`, `mlp`, `bilinear_lowrank`, `bilinear_full`, `elementwise`
- **팩토리 함수**: `create_probe_classifier(probe_type, d_in, **params)` — Sklearn 또는 `PyTorchProbeEstimator` 반환
- **래퍼**: `PyTorchProbeEstimator` — PyTorch 모델을 Sklearn `BaseEstimator` API로 감싸서 `fit()`, `score()`, `predict()` 제공
- **수식**: → [루트 GUIDE.md §5 ③ 참조](../../../../GUIDE.md)

#### `beaf_loader.py` (136줄)
- **역할**: Counterfactual 6-column CSV 로더 및 엄격한 Pair 무결성 검증
- **주요 함수**:
  - `load_beaf_csv(csv_path, image_root)`: 기본 CSV 로드 + 절대 경로 해석
  - `load_and_verify_counterfactual_pairs(csv_path, image_root)`: 연속 행 쌍의 무결성 강제 검증
    - 검증 조건: 한 행은 `object_in_image=True` (orig), 다른 행은 `False` (cf)
    - 두 행은 동일한 `object_name`과 `source_template` 공유
- **CSV 스키마**: → [DATA_SCHEMA.md 참조](../../../../DATA_SCHEMA.md)
- **반환**: `(df_raw, df_pairs, pair_metadata)` — `df_pairs`에는 `pair_id`, `orig_path`, `cf_path`, `positive_caption`, `negative_caption` 포함

#### `beaf_stats.py` (266줄)
- **역할**: 2×2 Factorial ANOVA 및 Bootstrap 95% CI 통계 연산
- **주요 함수**:
  - `compute_2x2_factorial_anova(sim_A, sim_B, sim_C, sim_D, n_bootstraps=1000)`:
    - 입력: 4가지 유사도 벡터 (A=orig×pos, B=orig×neg, C=cf×pos, D=cf×neg)
    - 출력: Text Main Effect, Visual Main Effect, Interaction Effect + 95% Bootstrap CI
  - `compute_quadrant_bootstrap_ci(...)`: 사분면별 Bootstrap CI
  - `compute_per_object_layerwise_stats(vis_orig, vis_cf, df_pairs)`: 객체별 레이어별 코사인 유사도 & 선형 프로빙
- **수식**: → [루트 GUIDE.md §5 ② 참조](../../../../GUIDE.md)

#### `vision_mechanisms.py` (852줄) — 패키지 내 최대 파일
- **역할**: 비전 트랜스포머 레이어 분해, SVD 스윕, 방향성 보존 분석
- **주요 함수**:
  - `extract_vision_features_unified(model, preprocess, image_paths, ...)`: ViT 전 레이어 특징 추출 (layer0~12 + pre_proj + final_l2norm)
  - `compute_vision_pipeline_breakdown(vis_orig, vis_cf)`: 레이어별 코사인/L2 기하 메트릭
  - `compute_vision_svd_sweep(vis_orig, vis_cf, ...)`: SVD 투영 절단 스윕 (10%~90%)
  - `compute_vision_linear_probe(vis_orig, vis_cf, df_pairs, ...)`: 레이어별 선형/MLP/Bilinear 프로브 (StratifiedKFold 또는 GroupKFold)
  - `compute_vision_direction_preservation(vis_orig, vis_cf)`: 방향 보존율 & Welch's t-test
- **의존성**: `probe_factory.ElementWiseNonLinearPyTorch`, `LowRankBilinearPyTorch`

#### `object_experiment.py` (730줄)
- **역할**: 단일/다중 객체 일반화 실험 모듈
- **주요 기능**:
  - `format_object_name(name)`: 동적 문법 서식 (관사 a/an, 복수형 s/es/people)
  - `instantiate_templates(templates, object_name)`: 255개 확장 템플릿 인스턴스화
  - `get_balanced_beaf_object_df(df, object_name)`: 1:1 균형 Present/Absent 샘플링
  - `run_single_object_analysis(...)`: 단일 객체 4-Way Cross Cosine + 프로브
  - `run_leave_one_object_out_text_probe_experiment(...)`: Leave-One-Object-Out 텍스트 프로브
  - `evaluate_dual_classifier_product_scorer(...)`: 코사인 결합 Dual Probe 평가
  - `run_single_object_train_val_experiment(...)`: 70:30 Train/Val 분할 일반화
- **주요 클래스**: `VisionProbeWrapper` — Linear/Quadratic/MLP 프로브 래퍼 (decision_function + predict 통합)

#### `audit_ab_swap_dataset.py` (129줄)
- **역할**: AB-Swap 데이터셋 아티팩트 감사 (0순위 Audit)
- **감사 항목**:
  1. 저수준 이미지 아티팩트 분류 (파일 크기, 해상도, RGB 평균/표준편차, Laplacian 분산) — 5-Fold GroupKFold
  2. 파일 형식/확장자 분포 & Pair 스큐 분석
  3. 이미지 재사용 빈도 & 대칭 감사
  4. 텍스트 순서 & 단어 편향 점검
  5. Base Scene ID 유출 감사
- **목적**: Inpainting 생성 흔적(Blur/Brush)이 프로브의 숏컷이 되지 않는지 검증

#### `run_ab_swap_evaluation.py` (276줄)
- **역할**: AB-Swap 1순위~4순위 인과성 평가 파이프라인
- **실험 우선순위 체계**:
  - 1순위: 2×2 Joint Consistency (양방향 매칭)
  - 2순위: Text Separability (코사인 유사도 + 프로빙)
  - 3순위: Vision Per-Pair Probing (Base Scene GroupKFold)
  - 4순위: Image-Blind Forced-Choice (`image_blind_xy_preference_pct`)
- **수식**: → [루트 GUIDE.md §5 ⑦ 참조](../../../../GUIDE.md)

#### `visualizer.py` (513줄)
- **역할**: BEAF 전용 시각화 렌더러
- **렌더링 함수 (11개)**:
  - Part A (v1): `render_image_image_histogram`, `render_4way_heatmap`, `render_text_vs_visual_scatter`, `render_full_correct_by_object`
  - Part A (v2): `render_scatter_pos_vs_neg`, `render_scatter_delta_quadrant`, `render_scatter_img_orig_vs_img_cf`, `render_scatter_by_object_category`
  - 2×2 ANOVA: `render_2x2_factorial_anova_plots`
  - 마진 공간: `render_2d_margin_state_space`
  - 레이어별: `render_per_object_layerwise_plot`

#### `plot_beaf_correlations.py` (155줄)
- **역할**: Image-fixed vs Text-fixed 상관관계 산점도 생성
- **플롯 구성**:
  1. Image-Fixed: cos(image, pos_text) vs cos(image, neg_text) — Orig vs CF 그룹별
  2. Text-Fixed: cos(orig_img, text) vs cos(cf_img, text) — Positive vs Negative 그룹별

#### `compare_beaf_probes.py` (142줄)
- **역할**: 다중 프로빙 결과 JSON 비교 플롯 생성
- **입력**: 여러 프로브 출력 디렉토리의 `beaf_{probe_type}_layerwise.json`
- **출력**: Validation Accuracy & Generalization Gap 비교 꺾은선 그래프

#### `generate_probing_comparison.py` (113줄)
- **역할**: 프로빙 정확도 수평 막대그래프 렌더러
- **비교 대상**: Random Chance, Linear (Default & Tuned), MLP (Hidden=8, 32, 64), Low-Rank Bilinear (Rank=4, 16, 32)

---

### 3. 모듈 의존 관계

```
probe_factory.py ←── vision_mechanisms.py ←── run_beaf_analysis_v2.py (상위)
       ↑                     ↑
       │                     │
       └── object_experiment.py ←── run_beaf_object_generalization.py (상위)
       │
beaf_loader.py ←── run_beaf_flexible_probing.py (상위)
       ↑           run_beaf_train_val_per_object.py (상위)
       │
beaf_stats.py ←── run_beaf_analysis_v2.py (상위)

visualizer.py ←── run_beaf_analysis_v2.py (상위)

audit_ab_swap_dataset.py (독립)
run_ab_swap_evaluation.py (독립)
plot_beaf_correlations.py (독립)
compare_beaf_probes.py (독립)
generate_probing_comparison.py (독립)
```

---

### 4. AB-Swap 실험 실행 순서

```
0순위: audit_ab_swap_dataset.py  ─→  데이터 아티팩트 없음 확인
  ↓
1순위: run_ab_swap_evaluation.py ─→  2x2 Joint Consistency
  ↓
2순위: (동일 스크립트)            ─→  Text Separability
  ↓
3순위: (동일 스크립트)            ─→  Vision Per-Pair Probing
  ↓
4순위: (동일 스크립트)            ─→  Image-Blind Forced-Choice
```

---

---
## 🇺🇸 English

### 1. Framework Overview

The `benchmarks/src/analysis/beaf/` package is the core implementation of **BEAF (Benchmark Evaluation & Analysis Framework)**, performing causal mechanism verification using counterfactual pairs.

**Key Idea**: Using 1:1 Counterfactual Pairs composed of original images ($\text{Orig}$, object present) and inpainted images ($\text{CF}$, object absent), causally verify how CLIP encoders process negation.

---

### 2. File-by-File Details (12 files)

#### `__init__.py` (112 lines)
- **Role**: Public API for the beaf package
- **Export Categories**:
  - Visualizers (11 functions): `render_image_image_histogram`, `render_4way_heatmap`, etc.
  - Vision Mechanisms (4 functions): `extract_vision_features_unified`, etc.
  - Object Experiments (11 functions/classes): `run_single_object_analysis`, `VisionProbeWrapper`, etc.
  - Statistics (3 functions): `compute_2x2_factorial_anova`, `compute_quadrant_bootstrap_ci`, `compute_per_object_layerwise_stats`
  - Data Loading (2 functions): `load_beaf_csv`, `load_and_verify_counterfactual_pairs`
  - Probe Factory (8 items): `SUPPORTED_PROBES`, 4 PyTorch model classes, `PyTorchProbeEstimator`, `create_probe_classifier`, `get_param_candidates`, `format_params`

#### `probe_factory.py` (320 lines) — **[Single Source of Truth]**
- **Role**: Unified factory for 4 PyTorch probe models and Sklearn wrappers
- **PyTorch Probe Models**:

  | Class | Purpose |
  |:---|:---|
  | `LowRankBilinearPyTorch` | Low-rank cross-dimension interaction |
  | `FullBilinearPyTorch` | Full $D \times D$ quadratic form interaction |
  | `MLPVisionPyTorch` | 2-layer non-linear MLP |
  | `ElementWiseNonLinearPyTorch` | 0% cross-dimension mixing (pure non-linearity control) |

- **SUPPORTED_PROBES**: `logistic`, `svm_linear`, `ridge`, `sgd_log`, `sgd_hinge`, `svm_rbf`, `mlp`, `bilinear_lowrank`, `bilinear_full`, `elementwise`
- **Factory**: `create_probe_classifier(probe_type, d_in, **params)` — returns Sklearn or `PyTorchProbeEstimator`
- **Wrapper**: `PyTorchProbeEstimator` — wraps PyTorch models as Sklearn `BaseEstimator` API (`fit()`, `score()`, `predict()`)
- **Formulas**: → [Root GUIDE.md §5 ③ Reference](../../../../GUIDE.md)

#### `beaf_loader.py` (136 lines)
- **Role**: Counterfactual 6-column CSV loader with strict pair integrity verification
- **Key Functions**:
  - `load_beaf_csv(csv_path, image_root)`: Basic CSV load + absolute path resolution
  - `load_and_verify_counterfactual_pairs(csv_path, image_root)`: Strict consecutive row pair verification
    - Validation: one row `object_in_image=True` (orig), the other `False` (cf)
    - Both rows share the same `object_name` and `source_template`
- **CSV Schema**: → [DATA_SCHEMA.md Reference](../../../../DATA_SCHEMA.md)
- **Returns**: `(df_raw, df_pairs, pair_metadata)` — `df_pairs` includes `pair_id`, `orig_path`, `cf_path`, `positive_caption`, `negative_caption`

#### `beaf_stats.py` (266 lines)
- **Role**: 2×2 Factorial ANOVA and Bootstrap 95% CI computation
- **Key Functions**:
  - `compute_2x2_factorial_anova(sim_A, sim_B, sim_C, sim_D, n_bootstraps=1000)`:
    - Input: 4 similarity vectors (A=orig×pos, B=orig×neg, C=cf×pos, D=cf×neg)
    - Output: Text Main Effect, Visual Main Effect, Interaction Effect + 95% Bootstrap CI
  - `compute_quadrant_bootstrap_ci(...)`: Per-quadrant Bootstrap CI
  - `compute_per_object_layerwise_stats(vis_orig, vis_cf, df_pairs)`: Per-object layerwise cosine similarity & linear probing
- **Formulas**: → [Root GUIDE.md §5 ② Reference](../../../../GUIDE.md)

#### `vision_mechanisms.py` (852 lines) — Largest file in the package
- **Role**: Vision Transformer layer decomposition, SVD sweep, direction preservation analysis
- **Key Functions**:
  - `extract_vision_features_unified(model, preprocess, image_paths, ...)`: ViT full-layer feature extraction (layer0–12 + pre_proj + final_l2norm)
  - `compute_vision_pipeline_breakdown(vis_orig, vis_cf)`: Per-layer cosine/L2 geometric metrics
  - `compute_vision_svd_sweep(vis_orig, vis_cf, ...)`: SVD projection truncation sweep (10%–90%)
  - `compute_vision_linear_probe(vis_orig, vis_cf, df_pairs, ...)`: Per-layer linear/MLP/bilinear probes (StratifiedKFold or GroupKFold)
  - `compute_vision_direction_preservation(vis_orig, vis_cf)`: Direction preservation ratio & Welch's t-test
- **Dependencies**: `probe_factory.ElementWiseNonLinearPyTorch`, `LowRankBilinearPyTorch`

#### `object_experiment.py` (730 lines)
- **Role**: Single/multi-object generalization experiment module
- **Key Functions**:
  - `format_object_name(name)`: Dynamic grammar formatting (articles a/an, plurals s/es/people)
  - `instantiate_templates(templates, object_name)`: Instantiate 255 expanded templates
  - `get_balanced_beaf_object_df(df, object_name)`: 1:1 balanced Present/Absent sampling
  - `run_single_object_analysis(...)`: Single-object 4-Way Cross Cosine + probes
  - `run_leave_one_object_out_text_probe_experiment(...)`: Leave-One-Object-Out text probe
  - `evaluate_dual_classifier_product_scorer(...)`: Cosine-combined Dual Probe evaluation
  - `run_single_object_train_val_experiment(...)`: 70:30 Train/Val split generalization
- **Key Class**: `VisionProbeWrapper` — wraps Linear/Quadratic/MLP probes (unified decision_function + predict)

#### `audit_ab_swap_dataset.py` (129 lines)
- **Role**: AB-Swap dataset artifact audit (Priority 0 Audit)
- **Audit Items**:
  1. Low-level image artifact classification (file size, resolution, mean/std RGB, Laplacian variance) — 5-Fold GroupKFold
  2. File format/extension distribution & pair skew analysis
  3. Image reuse frequency & symmetry audit
  4. Text ordering & word bias check
  5. Base Scene ID leakage audit
- **Purpose**: Verify that inpainting artifacts (blur/brush strokes) don't become probe shortcuts

#### `run_ab_swap_evaluation.py` (276 lines)
- **Role**: AB-Swap Priority 1–4 causal evaluation pipeline
- **Priority System**:
  - P1: 2×2 Joint Consistency (bidirectional matching)
  - P2: Text Separability (cosine similarity + probing)
  - P3: Vision Per-Pair Probing (Base Scene GroupKFold)
  - P4: Image-Blind Forced-Choice (`image_blind_xy_preference_pct`)
- **Formulas**: → [Root GUIDE.md §5 ⑦ Reference](../../../../GUIDE.md)

#### `visualizer.py` (513 lines)
- **Role**: BEAF-specific visualization renderer
- **Rendering Functions (11)**:
  - Part A (v1): `render_image_image_histogram`, `render_4way_heatmap`, `render_text_vs_visual_scatter`, `render_full_correct_by_object`
  - Part A (v2): `render_scatter_pos_vs_neg`, `render_scatter_delta_quadrant`, `render_scatter_img_orig_vs_img_cf`, `render_scatter_by_object_category`
  - 2×2 ANOVA: `render_2x2_factorial_anova_plots`
  - Margin Space: `render_2d_margin_state_space`
  - Layerwise: `render_per_object_layerwise_plot`

#### `plot_beaf_correlations.py` (155 lines)
- **Role**: Image-fixed vs text-fixed correlation scatter plots
- **Plots**:
  1. Image-Fixed: cos(image, pos_text) vs cos(image, neg_text) — by Orig vs CF
  2. Text-Fixed: cos(orig_img, text) vs cos(cf_img, text) — by Positive vs Negative

#### `compare_beaf_probes.py` (142 lines)
- **Role**: Multi-probing result JSON comparison plot generation
- **Input**: `beaf_{probe_type}_layerwise.json` from multiple probe output directories
- **Output**: Validation Accuracy & Generalization Gap comparison line charts

#### `generate_probing_comparison.py` (113 lines)
- **Role**: Probing accuracy horizontal bar chart renderer
- **Comparison targets**: Random Chance, Linear (Default & Tuned), MLP (Hidden=8, 32, 64), Low-Rank Bilinear (Rank=4, 16, 32)

---

### 3. Module Dependency Graph

```
probe_factory.py ←── vision_mechanisms.py ←── run_beaf_analysis_v2.py (parent)
       ↑                     ↑
       │                     │
       └── object_experiment.py ←── run_beaf_object_generalization.py (parent)
       │
beaf_loader.py ←── run_beaf_flexible_probing.py (parent)
       ↑           run_beaf_train_val_per_object.py (parent)
       │
beaf_stats.py ←── run_beaf_analysis_v2.py (parent)

visualizer.py ←── run_beaf_analysis_v2.py (parent)

audit_ab_swap_dataset.py (standalone)
run_ab_swap_evaluation.py (standalone)
plot_beaf_correlations.py (standalone)
compare_beaf_probes.py (standalone)
generate_probing_comparison.py (standalone)
```

---

### 4. AB-Swap Experiment Execution Order

```
P0: audit_ab_swap_dataset.py  ─→  Verify no data artifacts
  ↓
P1: run_ab_swap_evaluation.py ─→  2x2 Joint Consistency
  ↓
P2: (same script)              ─→  Text Separability
  ↓
P3: (same script)              ─→  Vision Per-Pair Probing
  ↓
P4: (same script)              ─→  Image-Blind Forced-Choice
```
