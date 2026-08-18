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

## 2. 전체 파일 목록 및 모듈 맵 (총 27개 파일 전수 점검)

```
benchmarks/src/analysis/ (총 15개 파일)
├── __init__.py                     # 분석 패키지 공개 API (to_bool, get_layer_features, set_seed, config export)
├── config.py                       # PipelineStep, MetadataKey, 기하학 연산, to_bool, get_layer_features, set_seed, DEFAULT_TUNING_GRIDS
├── extractor.py                    # 단일 패스(Single-pass) 전 레이어/단계별 특징 추출 엔진
├── metrics.py                      # 16단계 기하 메트릭, Welch's t-test, 선형 프로빙, PCA 유효 랭크
├── subspace_analysis.py            # 부정 차이 벡터(Δ) Global Subspace, SVD 스펙트럼, split_by 카테고리 전이
├── eval_sparse_text_dimensions.py  # 소수 차원(Sparse Dimensions) 가설 검증 (Top-k 절제 정확도)
├── analyze_internal_weights.py     # Linear Probe 가중치 및 Bilinear W 비대각 에너지(97.17%) 분석
├── reporter.py                     # PNG 시각화 렌더러 및 JSON/CSV 보고서 직렬화
├── run_analysis.py                 # 텍스트 인코더 기하 구조 분석 메인 오케스트레이터
├── pca_text_encoder.py             # 모듈화 이전의 모놀리식 16-Step 원본 분석 스크립트 (독립 실행 가능)
├── run_beaf_analysis_v2.py         # BEAF Part A(4-Axis) + Part B(Vision) 통합 실행 엔트리포인트 (8-State 진리표)
├── train_beaf_dual_probes.py       # [이동됨] Dual Classifier (f_T, f_V) 학습 및 NPZ 가중치 저장
├── run_beaf_flexible_probing.py    # 9종 프로빙 분류기 전 레이어 스윕 (Nested GroupKFold)
├── run_beaf_train_val_per_object.py# 객체별 Train vs Val 일반화 갭(Gap) 측정
└── run_beaf_object_generalization.py# OOD 단일/다중 객체 및 255개 템플릿 전이 실험

benchmarks/src/analysis/beaf/ (총 12개 파일)
├── __init__.py                     # beaf 패키지 공개 API (프로브 클래스 4종 및 팩토리 export)
├── probe_factory.py                # [Single Source of Truth] PyTorch 프로브 4종 및 Sklearn 래퍼
├── beaf_loader.py                  # Counterfactual 6-column CSV 로더 및 엄격한 Pair 무결성 검증
├── vision_mechanisms.py            # 비전 트랜스포머 레이어 분해, SVD 스윕, 방향성 보존 분석
├── object_experiment.py            # 4-Way Cross Cosine 분석, 문법 서식, 밸런스드 샘플링, 코사인 결합 Dual Probe 평가
├── beaf_stats.py                   # 2x2 Factorial ANOVA (Text/Visual/Interaction 효과) & Bootstrap 95% CI
├── audit_ab_swap_dataset.py        # AB-Swap 데이터셋 아티팩트/확장자 스큐/블러 감사 (0순위 Audit)
├── run_ab_swap_evaluation.py       # AB-Swap 1순위~4순위 인과성 평가 파이프라인 (2x2 Joint Consistency)
├── plot_beaf_correlations.py       # Image-fixed vs Text-fixed 상관관계 플롯
├── compare_beaf_probes.py          # 다중 프로빙 결과 JSON 비교 플롯 생성
├── generate_probing_comparison.py  # 프로빙 정확도 수평 막대그래프 렌더러
└── visualizer.py                   # BEAF 전용 플롯(히스토그램, 산점도, 2D 마진 공간, 4-Way 히트맵) 렌더러
```

---

## 3. 리팩토링 변경 이력

1. **PyTorch 프로빙 모델 단일화 (`probe_factory.py`)**:
   * `vision_mechanisms.py`, `object_experiment.py`, `train_beaf_dual_probes.py`에 난립해 있던 중복 클래스 정의를 [`probe_factory.py`](file:///benchmarks/src/analysis/beaf/probe_factory.py) 하나로 일원화.
   * `LowRankBilinearPyTorch`, `FullBilinearPyTorch`, `MLPVisionPyTorch`, `ElementWiseNonLinearPyTorch` 4종이 표준 모델로 등록됨.
2. **공용 유틸리티 및 시드 일원화 (`config.py`)**:
   * `to_bool(v)`: 4개 파일에 중복 구현되어 있던 불리언 파서를 통합.
   * `get_layer_features(vis, key)`: 3개 파일에 중복 구현되어 있던 레이어 특징 추출 함수 `_get_feats`를 통합.
   * `set_seed(seed=42)`: Python/NumPy/PyTorch 일괄 시드 제어 함수 및 표준 튜닝 그리드(`DEFAULT_TUNING_GRIDS`) 등록.
3. **루트 스크립트 패키지화**:
   * 루트의 `train_beaf_dual_probes.py` $\rightarrow$ [`benchmarks/src/analysis/train_beaf_dual_probes.py`](file:///benchmarks/src/analysis/train_beaf_dual_probes.py)로 이동.
   * 루트 위치에는 하위 호환성을 위한 `DeprecationWarning` 스텁(Stub) 파일 유지.
4. **수학적 오류 및 실험 모호성 교정**:
   * `object_experiment.py`: `evaluate_dual_classifier_product_scorer`에서 텍스트 항($f_T$)이 단순 상수로 소거되던 버그를 수정하고, 코사인 유사도와 마진 곱 결합 $\text{Score} = \cos(v, t) \cdot (f_V(v) \cdot f_T(t))$ 적용 (시그모이드 제외하여 부호 일치성 보존).
   * `subspace_analysis.py`: `evaluate_cross_category_transfer`에 명시적 `split_by="object_name"` 인자를 추가하여 실제 80개 객체 카테고리 기준 분할 강제.
   * `run_beaf_analysis_v2.py`: `_classify_failure_mode`를 3-bit 진리표(8가지 경우) 완전 배타적 매핑으로 개편.
   * `run_ab_swap_evaluation.py`: 더미 이미지 벡터의 $T_{XY}$ 방향성 선호 메트릭 명칭을 `image_blind_xy_preference_pct`로 명확화.

---

## 4. 수학적 모델 및 알고리즘 데이터 흐름

### ① 텍스트 인코더 16단계 기하 구조 분석 (`run_analysis.py`)
1. **단일 패스 추출 ([`extractor.py`](file:///benchmarks/src/analysis/extractor.py))**:
   * `Step 0 (Embed)`: 토큰 + 위치 임베딩 ($h_0$)
   * `Layer 1 ~ 12`: 트랜스포머 잔차 블록 출력 중 EOT 토큰 풀링 ($h_1 \dots h_{12}$)
   * `Step 2 (LN)`: Layer 12 통과 후 LayerNorm ($z_{\text{LN}}$)
   * `Step 3 (Proj)`: 선형 투영 ($z_{\text{proj}} = z_{\text{LN}} W_{\text{proj}}$)
   * `Step 4 (L2Norm)`: 최종 정규화 ($z_{\text{final}} = z_{\text{proj}} / \|z_{\text{proj}}\|_2$)
2. **레이어별 코사인 유사도 계산 ([`metrics.py`](file:///benchmarks/src/analysis/metrics.py))**:
   $$\text{CosineSim}(h_{\text{pos}}^{(l)}, h_{\text{neg}}^{(l)}) = \frac{h_{\text{pos}}^{(l)} \cdot h_{\text{neg}}^{(l)}}{\|h_{\text{pos}}^{(l)}\|_2 \|h_{\text{neg}}^{(l)}\|_2}$$
   * 16개 단계별 평균 유사도 및 L2 거리를 Dual-Axis 꺾은선 그래프(`pipeline_step_lineplot.png`)로 렌더링.
3. **단계별 선형 프로빙 ([`metrics.py`](file:///benchmarks/src/analysis/metrics.py))**:
   * $X = [Z_{\text{pos}}; Z_{\text{neg}}]$, $y = [\mathbf{1}_N; \mathbf{0}_N]$ 구성 후 `StratifiedKFold(n_splits=5)`로 Logistic Regression 학습.
   * Step 0, Step 2, Step 4의 분류 정확도 막대그래프(`linear_probe_classification_acc.png`) 렌더링.

---

### ② BEAF 4-Way Cross Cosine & 2x2 Factorial ANOVA ([`beaf_stats.py`](file:///benchmarks/src/analysis/beaf/beaf_stats.py))

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

### ③ 프로브 모델 수식 정의 ([`probe_factory.py`](file:///benchmarks/src/analysis/beaf/probe_factory.py))

| 모델 클래스 | 수식 정의 | 특징 및 용도 |
|:---|:---|:---|
| `ElementWiseNonLinearPyTorch` | $f(x) = \sum_d w_d \cdot \text{GELU}(x_d) + b$ | 차원 간 혼합 0% (순수 비선형성 대조군) |
| `LowRankBilinearPyTorch` | $f(x) = \sum_{r=1}^R (x U_r)(x V_r) + x w_{\text{lin}} + b$ | Low-rank 교차 차원 상호작용 |
| `FullBilinearPyTorch` | $f(x) = x^T W x + x w_{\text{lin}} + b$ | Full $D \times D$ 2차 형식 상호작용 |
| `MLPVisionPyTorch` | $f(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$ | 2계층 비선형 MLP |

---

### ④ Bilinear $W$ 행렬 에너지 분해 ([`analyze_internal_weights.py`](file:///benchmarks/src/analysis/analyze_internal_weights.py))

* $\text{Total Energy} = \sum_{i, j} W_{i, j}^2$
* $\text{Diagonal Energy (Direct Matching)} = \sum_i W_{i, i}^2 \approx \mathbf{2.83\%}$
* $\text{Off-Diagonal Energy (Cross-Dimension Interaction)} = \text{Total} - \text{Diag} \approx \mathbf{97.17\%}$
* **결론**: 기존 코사인 유사도가 1:1 대각 차원만 매칭하는 것과 달리, 부정어 인식은 비대각 교차 차원 상호작용이 필수적임.

---

### ⑤ 부정 부분공간 및 유효 랭크 ([`subspace_analysis.py`](file:///benchmarks/src/analysis/subspace_analysis.py))

* 부정 차이 벡터 $D = X_{\text{pos}} - X_{\text{neg}}$의 공분산 고유값 $\lambda_i$로부터 산출:
  $$p_i = \frac{\lambda_i}{\sum \lambda_i}, \quad H = -\sum p_i \ln p_i, \quad r_{\text{eff}} = \exp(H)$$
* SVD를 통해 글로벌 부정 기저 $U_{\text{neg}}$를 추출하고, `--split_by object_name`을 통해 Unseen Category에 대한 Zero-shot 전이율 검증.

---

### ⑥ 소수 차원 가설 검증 ([`eval_sparse_text_dimensions.py`](file:///benchmarks/src/analysis/eval_sparse_text_dimensions.py))

* 가중치 벡터 $w$의 차원을 절대값 순으로 정렬 후, 상위 $k$개 차원만 남기고 나머지를 0으로 절제(Ablation):
  * $k=1, 2, 5, 10, 20, 50, 100, 512$에 대한 **Zero-shot Ablated Accuracy** 측정.
  * 소수 차원만으로는 성능이 급락함을 보여주어 **"부정 정보는 고차원에 고르게 분산(Distributed)되어 있다"**는 사실을 증명.

---

### ⑦ AB Compositional Swap 인과성 평가 ([`run_ab_swap_evaluation.py`](file:///benchmarks/src/analysis/beaf/run_ab_swap_evaluation.py))

* **1순위 (2x2 Joint Consistency)**:
  $$\min(S(I_{XY}, T_{XY}), S(I_{YX}, T_{YX})) > \max(S(I_{XY}, T_{YX}), S(I_{YX}, T_{XY}))$$
* **2순위 (Text Separability)**: $T_{XY}$ vs $T_{YX}$의 코사인 유사도 및 프로빙 분리도.
* **3순위 (Vision Per-Pair Probing)**: $I_{XY}$ vs $I_{YX}$의 Base Scene GroupKFold 분리도.
* **4순위 (Image-Blind Forced-Choice)**: `image_blind_xy_preference_pct` (더미 이미지 벡터가 $T_{XY}$를 $T_{YX}$보다 선호하는 편향 비율 진단).

---

## 5. ⚠️ 주의해야 할 실험 설계상 한계 및 리뷰어 대응 전략

1. **`DualClassifierProductScorer`의 무조건부(Unconditional) 모순**:
   * `train_beaf_dual_probes.py`에서 학습된 $f_V(v)$는 텍스트 입력이 없으므로 특정 이미지 $v$에 대해 단일 상수(예: $+1.5$)만 출력함.
   * MCQ 평가 시 $f_T(t)$가 긍정이면 $+1$, 부정이면 $-1$이 되어, **객체가 없는 이미지에서도 부정 선택지가 무조건 음수가 되어 오답(긍정 선택지)을 선택하는 이론적 모순**이 발생함.
   * $\rightarrow$ 해결책: $f(v, t)$는 무조건부 분리가 아닌 조건부 결합(Bilinear $v^T W t$ 등)이어야 함.
2. **Text Probe 99.9%의 Token-Presence 편향**:
   * Positive vs Negative 문장으로만 학습된 Linear Probe가 99.9%를 찍는 것은 'not' 토큰의 존재(Presence)만 감지한 것일 수 있음.
   * $\rightarrow$ 해결책: 'not' 토큰을 양쪽에 유지한 **Word-Swap Counterfactual Pair**([`eval_word_swap_probe.py`](file:///benchmarks/src/evaluation/eval_word_swap_probe.py))로 평가해야 함.
3. **BEAF Inpainting Artifacts 간섭 가능성**:
   * $f_V(v)$가 객체의 의미적 부재가 아닌 Inpainting 생성 흔적(Blur/Brush)을 숏컷으로 학습했을 가능성 $\rightarrow$ [`audit_ab_swap_dataset.py`](file:///benchmarks/src/analysis/beaf/audit_ab_swap_dataset.py)로 아티팩트 감사 필수.

---

## 6. 주요 실행 명령어 (CLI Cheatsheet)

```bash
# 1. 텍스트 인코더 기하 구조 분석 (레이어 코사인 유사도 & 선형 프로브)
python -m benchmarks.src.analysis.run_analysis \
    --csv_path benchmarks/data/images/COCO_val_full_paired.csv \
    --model ViT-B-32 \
    --pretrained openai \
    --output_dir logs/analysis_modular/openai_vit_b32

# 2. BEAF 다중 프로빙 스윕 (Linear, MLP, Bilinear 전 레이어 평가)
bash run_beaf_comprehensive_probing_sweep.sh

# 3. BEAF 통합 분석 실행기 (Part A 4-Axis + Part B Vision)
python -m benchmarks.src.analysis.run_beaf_analysis_v2 \
    --csv_path benchmarks/data/images/beaf_counterfactual_6col.csv \
    --output_dir logs/evaluation/beaf_counterfactual_v2/openai_vit_b32

# 4. 객체 카테고리 기준 부정 부분공간 전이 평가
python -m benchmarks.src.analysis.subspace_analysis \
    --csv_path benchmarks/data/images/COCO_val_full_paired.csv \
    --split_by object_name \
    --output_dir logs/subspace_analysis

# 5. 소수 차원 가설 (Sparse Dimensions) 검증 스크립트
python -m benchmarks.src.analysis.eval_sparse_text_dimensions

# 6. AB Swap 데이터셋 감사 (0순위) 및 인과성 평가 (1~4순위)
python benchmarks/src/analysis/beaf/audit_ab_swap_dataset.py
python benchmarks/src/analysis/beaf/run_ab_swap_evaluation.py

# 7. Bilinear W 행렬 가중치 및 Off-diagonal 에너지 분석
python -m benchmarks.src.analysis.analyze_internal_weights
```
