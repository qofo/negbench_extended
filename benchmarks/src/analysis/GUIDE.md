# Analysis Module Technical Guide

---
## 🇰🇷 한국어

### 1. 모듈 개요

`benchmarks/src/analysis/` 패키지는 CLIP 텍스트 인코더의 부정어 처리 메커니즘을 정량적으로 분석하는 **표현 분석 코어**입니다.

**핵심 연구 질문과의 매핑:**
- **RQ1 (Representation vs Scoring Head)**: 텍스트 잠재 공간에 부정 정보가 보존되어 있는가? → `metrics.py` 선형 프로빙, `extractor.py` 레이어별 추출
- **RQ2 (Sparse vs Distributed)**: 소수 차원에 의존하는가? → `eval_sparse_text_dimensions.py`
- **RQ3 (Bilinear 필요성)**: 대각 매칭으로 충분한가? → `evaluation/analyze_internal_weights.py`
- **BEAF 통합**: Counterfactual Pair 기반 인과 검증 → `run_beaf_*.py` 시리즈

---

### 2. 파일별 상세 설명 (19개 파일)

#### `__init__.py`
- **역할**: 분석 패키지 공개 API 정의
- **Export**: `PipelineStep`, `MetadataKey`, `AnalysisConfig`, `RetrievalConfig`, `to_bool`, `layer_key`, `get_layer_features`, `l2_normalize`, `batch_cosine_similarity`, `batch_dot_product`, `batch_l2_distance`, `set_seed`, `DEFAULT_TUNING_GRIDS`, `filter_vision_dict`

#### `config.py`
- **역할**: 설정, 열거형, 기하학 연산 유틸리티의 단일 소스
- **주요 구성 요소**:
  - `PipelineStep(Enum)`: CLIP 텍스트 파이프라인 5단계 (`EMBEDDING`, `LAYER12_RAW`, `LAYER12_LN`, `PROJECTED_UNNORM`, `FINAL_L2NORM`)
  - `MetadataKey(Enum)`: 페어 캡션 CSV의 표준 메타데이터 키
  - `AnalysisConfig`: 분석 하이퍼파라미터 (`model_name`, `pretrained`, `target_token`, `max_samples`, `batch_size` 등)
  - `RetrievalConfig`: 크로스모달 검색 설정
  - 기하 연산: `l2_normalize()`, `batch_cosine_similarity()`, `batch_dot_product()`, `batch_l2_distance()`
  - `to_bool(v)`: 4개 파일에서 중복 구현되어 있던 불리언 파서 통합본
  - `layer_key(idx)`: Transformer 블록 이름을 만드는 유일한 함수 — 0은 `"Embedding"`, 나머지는 `"Layer <idx>"`.
    이전에는 세 곳이 각자 만들어 서로 어긋나 있었고(`layers`의 `"Layer 1"`, **같은 배열**을 담은 `pipeline`의 `"Layer1"`,
    그 주변 단계를 가리키는 `PipelineStep`), `get_layer_features`가 모르는 키에 최종 임베딩을 돌려주기 때문에
    불일치가 예외가 아니라 **평평한 "레이어별" 곡선**으로 나타났다.
  - `get_layer_features(vis, key)`: 3개 파일에서 중복 구현되어 있던 레이어 특징 추출 함수 통합본
  - `set_seed(seed=42)`: Python/NumPy/PyTorch 일괄 시드 제어
  - `DEFAULT_TUNING_GRIDS`: 프로브 하이퍼파라미터 튜닝 그리드 (logistic, ridge, svm_linear, svm_rbf, mlp, bilinear_lowrank)
  - `filter_vision_dict(vis, mask)`: 비전 특징 사전에 불리언 마스크 적용

#### `extractor.py`
- **역할**: 단일 패스(Single-Pass) 전 레이어/단계별 특징 추출 엔진
- **주요 함수**:
  - `extract_all_features_unified(model, tokenizer, texts, ...)`: 모든 Transformer 레이어와 파이프라인 단계를 단일 forward pass에서 추출
    - Step 0 (Embed): 토큰 + 위치 임베딩
    - Layer 1~12: Transformer 잔차 블록 출력 (EOT 토큰 풀링)
    - Step 2 (LN): LayerNorm 통과
    - Step 3 (Proj): 선형 투영
    - Step 4 (L2Norm): 최종 정규화
  - `assert_embedding_consistency(...)`: 수동 forward pass 출력과 `model.encode_text()` 결과의 동치성 검증
- **반환값**: `{"layers": {layer_name: ndarray}, "pipeline": {step_name: ndarray}, "final_l2norm": ndarray}`
  - `layers`의 키는 `layer_key(i)`(`i`는 `0..12`), `pipeline`은 `PipelineStep` 5개 값만 담는다. 이전에는
    `pipeline`이 블록 1–11을 `"Layer1"`이라는 두 번째 표기로 **한 벌 더** 들고 있었고, 그 배열은 `layers`와
    비트 단위로 동일했다. 블록은 `layers`에서 읽는다.

#### `paths.py`
- **역할**: CSV의 상대 이미지 경로를 실제 경로로 바꾸는 단 하나의 규칙
- **주요 함수**: `resolve_image_path()` — E1/E2 엔트리포인트 4개가 각자 복사본을 들고 있던 것을 통합

#### `feature_cache.py`
- **역할**: 디스크 특징 캐시와 실행 출처(provenance) 기록
- **주요 함수**:
  - `cached_encode(...)`: `(model, pretrained, kind, 항목 목록)` 해시로 인코더 호출을 메모이즈. 두 실험이
    "같은 임베딩"을 본다고 주장할 때 실제로 같음을 보장한다
  - `build_provenance(...)` / `inherit_upstream_provenance(...)`: 요약 JSON에 백본·시드·git 커밋·UTC 기록
  - `resolve_upstream_artifact(path, produced_by=...)`: 상류 산출물이 없을 때 **그것을 만드는 명령**을 이름으로 알려주는 에러
  - `load_object_restriction(...)`: `--restrict_objects` 개념 집합 로딩

#### `model_loader.py`
- **역할**: 평가용 CLIP 백본을 여는 단 한 곳
- **주요 함수**: `load_clip_for_eval(model_name, pretrained, device)` → `(model, preprocess_val, tokenizer)`.
  `create_model_and_transforms`는 `(model, preprocess_train, preprocess_val)`을 돌려주는데, 엔트리포인트 4개가
  두 번째(확률적 RandomResizedCrop)를 집어 임베딩이 매번 달라지고 있었다. 학습용 변환은 여기서 아예 도달 불가능하다

#### `cli.py`
- **역할**: 33개 `main()`이 각자 재발명하던 표준 플래그 선언
- **주요 함수**: `add_model_args`, `add_run_args`, `add_data_args`, `add_cache_args`,
  `add_restriction_args`, `add_concept_args(parser, min_pairs=DEFAULT_MIN_PAIRS)`, `add_bias_args`.
  `min_pairs`는 2026-08-30에 **`DEFAULT_MIN_PAIRS = 20`으로 통일**되었다. 이 임계값은 개념 모집단을
  조용히 결정하므로 스크립트마다 기본값이 다르면 같은 이름의 다른 실험이 된다 — 실제로
  `eval_unary_mechanistic_analysis`의 옛 기본값 6은 33개가 아니라 68개 개념을 평가하고 있었다.
  스윕을 위해 인자로 덮어쓸 수는 있고, `test_min_pairs_is_unified`가 엔트리포인트 9개를 검사한다

#### `plotting.py`
- **역할**: 헤드리스 백엔드 선택과, 두 개 이상의 실험이 그리는 그림
- **주요 구성 요소**:
  - `plt`: `matplotlib.use("Agg")`를 먼저 실행한 뒤 내보내는 pyplot. pyplot은 import 시점에 백엔드를 고르므로
    순서가 의미를 가진다
  - `render_top_objects_grid(raw_df, output_path, title, top_k)`: 세 모듈이 각자 정의하던(정규화 AST 95% 동일)
    객체별 레이어 Train/Val 정확도 그리드

#### `import_compat.py`
- **역할**: 이중 경로 import의 fallback 분기 보호
- **주요 함수**: `reraise_unless_standalone()` — `benchmarks` 패키지가 실제로 import 불가능할 때만 fallback으로
  넘어가고, 그 외의 진짜 ImportError(오타·누락 의존성·사라진 이름)는 다시 던진다

#### `metrics.py` (566줄)
- **역할**: 6가지 분석 차원에 걸친 기하 메트릭 계산 엔진
- **6대 분석 Stage**:
  1. **Stage 1-A**: 16단계 Multi-Metric Pipeline & Layer Breakdown (코사인, 내적, L2 거리)
  2. **Stage 1-B**: 방향 보존(Direction Preservation) & Welch's t-test (교란 대조군)
  3. **Stage 2**: Stratified 5-Fold 선형 프로빙 & 템플릿 숏컷 추정
  4. **Stage 3**: PCA 스펙트럼 유효 랭크 (Spectral Entropy, Participation Ratio)
  5. **Stage 4**: 투영 행렬 SVD 절단 스윕 (10%~90%)
  6. **Stage 5**: 마이크로배치 크로스모달 검색 & 대칭 객체 부재 평가
- **수식**: → [루트 GUIDE.md §5 ① 참조](../../../GUIDE.md)

#### `subspace_analysis.py`
- **역할**: 부정 차이 벡터(Δ) 글로벌 부분공간 분석 & 카테고리 전이 프로브
- **주요 함수**:
  - `compute_subspace_spectrum(diff_matrix)`: SVD 스펙트럼, 유효 랭크($r_{\text{eff}}$), Participation Ratio 산출
  - `evaluate_cross_category_transfer(...)`: `--split_by object_name`으로 80개 객체 카테고리 기준 Unseen 전이율 검증
- **출력**: 글로벌 부정 기저 $U_{\text{neg}}$ NPZ 파일, 선형 프로브 가중치 벡터
- **수식**: → [루트 GUIDE.md §5 ⑤ 참조](../../../GUIDE.md)

#### `eval_sparse_text_dimensions.py` (434줄)
- **역할**: 소수 차원(Sparse Dimensions) 가설 검증
- **방법론**: 가중치 벡터 $w$의 상위 $k$개 차원만 남기고 나머지를 0으로 절제
  - $k = 1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 512$
  - 텍스트 프로브 + 비전 프로브 모두 수행
- **출력**: `sparse_dim_analysis.png`, CSV, JSON
- **수식**: → [루트 GUIDE.md §5 ⑥ 참조](../../../GUIDE.md)

#### `reporter.py` (351줄)
- **역할**: PNG 시각화 렌더러 및 JSON/CSV 보고서 직렬화
- **주요 클래스**: `AnalysisReporter`
- **렌더링 메서드**:
  - `render_pipeline_breakdown()`: Dual-Axis 꺾은선 (`pipeline_step_lineplot.png`)
  - `render_direction_preservation()`: 방향 보존 막대그래프
  - `render_linear_probe_results()`: 선형 프로브 정확도 막대그래프 (`linear_probe_classification_acc.png`)
  - `render_pca_spectrum()`: PCA 스펙트럼 플롯
  - `render_projection_svd()`: SVD 절단 스윕 플롯
  - `render_retrieval_results()`: 검색 성능 테이블

#### `run_analysis.py` (148줄)
- **역할**: 텍스트 인코더 기하 구조 분석 **메인 오케스트레이터**
- **흐름**: CLI 파싱 → 모델 로드 → CSV 읽기 → `extract_all_features_unified` → `metrics.*` 호출 → `AnalysisReporter` 렌더링
- **CLI 인자**: `--model`, `--pretrained`, `--target_token`, `--csv_path`, `--output_dir`, `--max_samples`, `--image_root`, `--batch_size`, `--seed`
- **실행** (저장소 루트에서):
  ```bash
  python -m analysis.run_analysis --csv_path benchmarks/data/images/COCO_val_full_paired.csv \
      --output_dir logs/analysis_modular/openai_vit_b32
  ```

#### `run_beaf_analysis_v2.py` (441줄)
- **역할**: BEAF 2x2 Factorial ANOVA + Vision 통합 실행 엔트리포인트
- **파트 구성**:
  - Part A: 4-Axis BEAF 분석 (Text-Text, Image-Text, Image-Image, 4-Way Cross)
  - Part B: Vision 인코더 메커니즘 분석 (레이어 분해, SVD 스윕, 선형 프로브, 방향 보존)
- **⚠️ 현재 제한**: Step 6~10 주석 처리 상태
- **실행** (저장소 루트에서):
  ```bash
  python -m analysis.run_beaf_analysis_v2 --csv_path benchmarks/data/images/beaf_counterfactual_6col.csv \
      --image_root "" --output_dir logs/evaluation/beaf_counterfactual_v2/openai_vit_b32 \
      --model ViT-B-32 --pretrained openai
  ```

#### `train_beaf_dual_probes.py` (369줄)
- **역할**: Dual Classifier ($f_T$, $f_V$) 학습 및 NPZ 가중치 저장
- **출력**: LogisticRegression 가중치 ($w_v$, $b_v$, $w_t$, $b_t$)를 NPZ로 저장 → `DualClassifierProductScorer`에서 사용
- **⚠️ 설계 한계**: $f_V(v)$는 텍스트 무조건부 → [루트 GUIDE.md §7 경고 #1 참조](../../../GUIDE.md)

#### `run_beaf_flexible_probing.py` (316줄)
- **역할**: 9종 프로빙 분류기를 전 레이어에 걸쳐 스윕
- **CV 전략**: Nested `GroupKFold` (외부: `pair_id` 그룹, 내부: 하이퍼파라미터 튜닝)
- **프로브 종류**: `SUPPORTED_PROBES` (logistic, svm_linear, ridge, sgd_log, sgd_hinge, svm_rbf, mlp, bilinear_lowrank, bilinear_full)
- **의존성**: `beaf.probe_factory`, `beaf.beaf_loader`, `beaf.vision_mechanisms`

#### `run_beaf_train_val_per_object.py` (307줄)
- **역할**: 객체별 Train vs Val 일반화 갭(Gap) 측정
- **CV 전략**: 5-Fold `GroupKFold` (Linear Probe, Train Acc vs Val Acc 비교)
- **출력**: 객체별·레이어별 Train/Val 정확도 JSON, 히트맵 PNG

#### `run_beaf_object_generalization.py` (344줄)
- **역할**: OOD 단일/다중 객체 및 255개 템플릿 전이 실험
- **실험**: 1:1 균형 Present/Absent 이미지 쌍, 255개 확장 템플릿(123 부정, 132 긍정)
- **의존성**: `beaf.object_experiment`의 `run_single_object_analysis`, `run_leave_one_object_out_text_probe_experiment`

---

### 3. 모듈 의존 관계

```
config.py  ←─  extractor.py  ←─  metrics.py  ←─  run_analysis.py
     ↑              ↑                                    ↑
     │              │                                    │
     └── reporter.py ←───────────────────────────────────┘
     │
     ├── subspace_analysis.py (독립 엔트리포인트)
     ├── eval_sparse_text_dimensions.py (독립 엔트리포인트)
     │
     └── run_beaf_*.py  ──→  beaf/ 패키지 (→ beaf/GUIDE.md 참조)
```

---

---
## 🇺🇸 English

### 1. Module Overview

The `benchmarks/src/analysis/` package is the **representation analysis core** that quantitatively analyzes CLIP's text encoder negation processing mechanisms.

**Mapping to Core Research Questions:**
- **RQ1 (Representation vs Scoring Head)**: Is negation information preserved in text latent space? → `metrics.py` linear probing, `extractor.py` layer-wise extraction
- **RQ2 (Sparse vs Distributed)**: Does accuracy depend on a few dimensions? → `eval_sparse_text_dimensions.py`
- **RQ3 (Bilinear Necessity)**: Is diagonal matching sufficient? → `evaluation/analyze_internal_weights.py`
- **BEAF Integration**: Causal verification via counterfactual pairs → `run_beaf_*.py` series

---

### 2. File-by-File Details (19 files)

#### `__init__.py`
- **Role**: Public API for the analysis package
- **Exports**: `PipelineStep`, `MetadataKey`, `AnalysisConfig`, `RetrievalConfig`, `to_bool`, `layer_key`, `get_layer_features`, `l2_normalize`, `batch_cosine_similarity`, `batch_dot_product`, `batch_l2_distance`, `set_seed`, `DEFAULT_TUNING_GRIDS`, `filter_vision_dict`

#### `config.py`
- **Role**: Single source of truth for configuration, enumerations, and geometric operation utilities
- **Key Components**:
  - `PipelineStep(Enum)`: 5-stage CLIP text pipeline (`EMBEDDING`, `LAYER12_RAW`, `LAYER12_LN`, `PROJECTED_UNNORM`, `FINAL_L2NORM`)
  - `MetadataKey(Enum)`: Standardized metadata keys for paired caption CSVs
  - `AnalysisConfig`: Analysis hyperparameters (`model_name`, `pretrained`, `target_token`, `max_samples`, `batch_size`, etc.)
  - `RetrievalConfig`: Cross-modal retrieval settings
  - Geometric ops: `l2_normalize()`, `batch_cosine_similarity()`, `batch_dot_product()`, `batch_l2_distance()`
  - `to_bool(v)`: Unified boolean parser (previously duplicated across 4 files)
  - `layer_key(idx)`: The one producer of a transformer-block name — `"Embedding"` for 0, `"Layer <idx>"` otherwise.
    Three producers had drifted apart (`"Layer 1"` in `layers`, `"Layer1"` for the *same array* in `pipeline`,
    `PipelineStep` for the steps around them), and `get_layer_features` answers an unknown key with the final
    embedding, so the mismatch drew a flat "layerwise" curve instead of raising.
  - `get_layer_features(vis, key)`: Unified layer feature extraction (previously duplicated across 3 files)
  - `set_seed(seed=42)`: Global Python/NumPy/PyTorch seed control
  - `DEFAULT_TUNING_GRIDS`: Hyperparameter tuning grids for probes (logistic, ridge, svm_linear, svm_rbf, mlp, bilinear_lowrank)
  - `filter_vision_dict(vis, mask)`: Apply boolean mask to vision feature dictionaries

#### `extractor.py`
- **Role**: Single-pass unified feature extraction engine across all layers and pipeline steps
- **Key Functions**:
  - `extract_all_features_unified(model, tokenizer, texts, ...)`: Extracts all Transformer layer and pipeline step features in a single forward pass
    - Step 0 (Embed): Token + positional embedding
    - Layer 1–12: Transformer residual block outputs (EOT token pooling)
    - Step 2 (LN): LayerNorm
    - Step 3 (Proj): Linear projection
    - Step 4 (L2Norm): Final normalization
  - `assert_embedding_consistency(...)`: Validates equivalence between manual forward pass and `model.encode_text()`
- **Returns**: `{"layers": {layer_name: ndarray}, "pipeline": {step_name: ndarray}, "final_l2norm": ndarray}`
  - `layers` is keyed by `layer_key(i)` for `i` in `0..12`; `pipeline` holds exactly the five `PipelineStep`
    values. `pipeline` used to *also* carry blocks 1–11 under a second spelling (`"Layer1"`), holding arrays
    bit-identical to `layers`. Read a block from `layers`.

#### `paths.py`
- **Role**: The one rule for turning a CSV's relative image path into a real one
- **Key Function**: `resolve_image_path()` — the four E1/E2 entrypoints each carried a copy

#### `feature_cache.py`
- **Role**: On-disk feature cache and run provenance
- **Key Functions**:
  - `cached_encode(...)`: memoizes an encoder call keyed on `(model, pretrained, kind, exact item list)`,
    so two experiments claiming to share embeddings provably do
  - `build_provenance(...)` / `inherit_upstream_provenance(...)`: backbone, seed, git commit and UTC for summary JSONs
  - `resolve_upstream_artifact(path, produced_by=...)`: a missing upstream artifact raises an error naming
    **the command that produces it**
  - `load_object_restriction(...)`: loads the `--restrict_objects` concept set

#### `model_loader.py`
- **Role**: The single place an evaluation backbone is opened
- **Key Function**: `load_clip_for_eval(model_name, pretrained, device)` → `(model, preprocess_val, tokenizer)`.
  `create_model_and_transforms` returns `(model, preprocess_train, preprocess_val)` and four entrypoints took the
  second — a stochastic RandomResizedCrop, making every embedding a different sample of the same image. The train
  transform is not reachable from here.

#### `cli.py`
- **Role**: The standard flags 33 `main()` functions each reinvented
- **Key Functions**: `add_model_args`, `add_run_args`, `add_data_args`, `add_cache_args`,
  `add_restriction_args`, `add_concept_args(parser, min_pairs=DEFAULT_MIN_PAIRS)`, `add_bias_args`.
  `min_pairs` was unified on **`DEFAULT_MIN_PAIRS = 20`** on 2026-08-30. The threshold silently selects the
  concept population, so a per-script default is a per-script experiment — `eval_unary_mechanistic_analysis`
  had been evaluating 68 concepts at its default of 6, not the reported 33. It stays overridable for a sweep,
  and `test_min_pairs_is_unified` checks all nine entrypoints.

#### `plotting.py`
- **Role**: Headless backend selection, and the figures more than one experiment draws
- **Key Components**:
  - `plt`: pyplot re-exported after `matplotlib.use("Agg")`. pyplot picks its backend at import time, so the order matters
  - `render_top_objects_grid(raw_df, output_path, title, top_k)`: the per-object layerwise train/val accuracy grid
    that three modules each defined (95% identical by normalized AST)

#### `import_compat.py`
- **Role**: Guards the fallback branch of a dual-path import
- **Key Function**: `reraise_unless_standalone()` — falls through to the fallback only when the `benchmarks`
  package genuinely is not importable, and re-raises any other ImportError (a typo, a missing dependency, a
  name that no longer exists)

#### `metrics.py` (566 lines)
- **Role**: Geometric metric computation engine across 6 analytical dimensions
- **6 Analysis Stages**:
  1. **Stage 1-A**: 16-step multi-metric pipeline & layer breakdown (Cosine, Dot Product, L2 Distance)
  2. **Stage 1-B**: Direction preservation & Welch's t-test (deranged control pairs)
  3. **Stage 2**: Stratified 5-fold linear probing & template shortcut estimation
  4. **Stage 3**: PCA spectrum effective rank (Spectral Entropy, Participation Ratio)
  5. **Stage 4**: Projection matrix SVD truncation sweep (10%–90%)
  6. **Stage 5**: Micro-batched cross-modal retrieval & symmetric object absence evaluation
- **Formulas**: → [Root GUIDE.md §5 ① Reference](../../../GUIDE.md)

#### `subspace_analysis.py`
- **Role**: Global negation subspace analysis & cross-category transfer probe engine
- **Key Functions**:
  - `compute_subspace_spectrum(diff_matrix)`: SVD spectrum, effective rank ($r_{\text{eff}}$), Participation Ratio
  - `evaluate_cross_category_transfer(...)`: Zero-shot transfer verification across 80 object categories via `--split_by object_name`
- **Output**: Global negation basis $U_{\text{neg}}$ NPZ file, linear probe weight vector
- **Formulas**: → [Root GUIDE.md §5 ⑤ Reference](../../../GUIDE.md)

#### `eval_sparse_text_dimensions.py` (434 lines)
- **Role**: Sparse dimensions hypothesis verification
- **Methodology**: Retain only top-$k$ dimensions of weight vector $w$, zero out the rest
  - $k = 1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 512$
  - Both text and vision probes evaluated
- **Output**: `sparse_dim_analysis.png`, CSV, JSON
- **Formulas**: → [Root GUIDE.md §5 ⑥ Reference](../../../GUIDE.md)

#### `reporter.py` (351 lines)
- **Role**: PNG visualization renderer and JSON/CSV report serializer
- **Main Class**: `AnalysisReporter`
- **Rendering Methods**:
  - `render_pipeline_breakdown()`: Dual-axis line plot (`pipeline_step_lineplot.png`)
  - `render_direction_preservation()`: Direction preservation bar chart
  - `render_linear_probe_results()`: Linear probe accuracy bar chart (`linear_probe_classification_acc.png`)
  - `render_pca_spectrum()`: PCA spectrum plot
  - `render_projection_svd()`: SVD truncation sweep plot
  - `render_retrieval_results()`: Retrieval performance table

#### `run_analysis.py` (148 lines)
- **Role**: Main orchestrator for text encoder geometric structure analysis
- **Flow**: CLI parsing → model load → CSV read → `extract_all_features_unified` → `metrics.*` calls → `AnalysisReporter` rendering
- **CLI Args**: `--model`, `--pretrained`, `--target_token`, `--csv_path`, `--output_dir`, `--max_samples`, `--image_root`, `--batch_size`, `--seed`
- **Execution** (from repo root):
  ```bash
  python -m analysis.run_analysis --csv_path benchmarks/data/images/COCO_val_full_paired.csv \
      --output_dir logs/analysis_modular/openai_vit_b32
  ```

#### `run_beaf_analysis_v2.py` (441 lines)
- **Role**: BEAF 2×2 Factorial ANOVA + Vision unified execution entrypoint
- **Parts**:
  - Part A: 4-Axis BEAF Analysis (Text-Text, Image-Text, Image-Image, 4-Way Cross)
  - Part B: Vision encoder mechanism analysis (layer breakdown, SVD sweep, linear probe, direction preservation)
- **⚠️ Current Limitation**: Steps 6–10 commented out
- **Execution** (from repo root):
  ```bash
  python -m analysis.run_beaf_analysis_v2 --csv_path benchmarks/data/images/beaf_counterfactual_6col.csv \
      --image_root "" --output_dir logs/evaluation/beaf_counterfactual_v2/openai_vit_b32 \
      --model ViT-B-32 --pretrained openai
  ```

#### `train_beaf_dual_probes.py` (369 lines)
- **Role**: Train Dual Classifiers ($f_T$, $f_V$) and save NPZ weights
- **Output**: LogisticRegression weights ($w_v$, $b_v$, $w_t$, $b_t$) saved as NPZ → used by `DualClassifierProductScorer`
- **⚠️ Design Limitation**: $f_V(v)$ is text-unconditional → [Root GUIDE.md §7 Warning #1](../../../GUIDE.md)

#### `run_beaf_flexible_probing.py` (316 lines)
- **Role**: Sweep 9 probing classifiers across all layers
- **CV Strategy**: Nested `GroupKFold` (outer: `pair_id` groups, inner: hyperparameter tuning)
- **Probe Types**: `SUPPORTED_PROBES` (logistic, svm_linear, ridge, sgd_log, sgd_hinge, svm_rbf, mlp, bilinear_lowrank, bilinear_full)
- **Dependencies**: `beaf.probe_factory`, `beaf.beaf_loader`, `beaf.vision_mechanisms`

#### `run_beaf_train_val_per_object.py` (307 lines)
- **Role**: Per-object Train vs Val generalization gap measurement
- **CV Strategy**: 5-Fold `GroupKFold` (Linear Probe, Train Acc vs Val Acc comparison)
- **Output**: Per-object per-layer Train/Val accuracy JSON, heatmap PNG

#### `run_beaf_object_generalization.py` (344 lines)
- **Role**: OOD single/multi-object & 255-template transfer experiment
- **Experiment**: 1:1 balanced present/absent image pairs, 255 expanded templates (123 negative, 132 positive)
- **Dependencies**: `beaf.object_experiment`'s `run_single_object_analysis`, `run_leave_one_object_out_text_probe_experiment`

---

### 3. Module Dependency Graph

```
config.py  ←─  extractor.py  ←─  metrics.py  ←─  run_analysis.py
     ↑              ↑                                    ↑
     │              │                                    │
     └── reporter.py ←───────────────────────────────────┘
     │
     ├── subspace_analysis.py (standalone entrypoint)
     ├── eval_sparse_text_dimensions.py (standalone entrypoint)
     │
     └── run_beaf_*.py  ──→  beaf/ package (→ see beaf/GUIDE.md)
```
