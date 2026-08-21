# Data Generation Module Technical Guide

---
## 🇰🇷 한국어

### 1. 모듈 개요

`benchmarks/src/data_generation/` 패키지는 NegBench 평가 및 분석에 사용되는 **Paired Caption CSV와 BEAF Counterfactual 데이터셋**을 생성합니다.

관련 모듈: 루트의 `synthetic_datasets/` 디렉토리는 대규모 파인튜닝 데이터셋(CC12M-NegCap, CC12M-NegMCQ)을 생성하며, 별도 `README.md`를 참조하세요.

---

### 2. 파일별 상세 설명 (3개 파일)

#### `create_full_coco_paired_v2.py` (306줄)
- **역할**: COCO 기반 Positive/Negative 페어 캡션 CSV 생성 (v2)
- **입력 소스**:
  1. `COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv` → 5,000 이미지의 positive/negative 객체 리스트
  2. `COCO_val_mcq_llama3.1_rephrased.csv` → LLaMA 생성 다양 pos/neg 캡션 쌍 (~1,869건)
  3. `COCO_val_retrieval.csv` (참조용)
- **전략**:
  - 기존 MCQ 부정 캡션 있으면 LLaMA 생성 캡션 직접 사용
  - 나머지는 10종 다양 pos/neg 템플릿 풀에서 라운드로빈 적용
  - COCO 80개 카테고리 문법 예외 처리 (skis, scissors, broccoli 등)
  - 동사 편향 방지: pos/neg 모두 동일 템플릿 풀
- **출력**: `COCO_val_full_paired_v2.csv`
  - 컬럼: `image_path`, `object_name`, `positive_caption`, `negative_caption`, `object_in_image`, `source_template`

#### `generate_beaf_ab_swap_dataset.py` (149줄)
- **역할**: BEAF Counterfactual A/B 객체 교환 데이터셋 기본 생성
- **개념**: 이미지 1에 객체 A 존재 + B 부재, 이미지 2에 객체 B 존재 + A 부재
- **출력**: `beaf_counterfactual_ab_swap.csv` (6-column 형식)

#### `generate_beaf_ab_swap_diverse_dataset.py` (421줄)
- **역할**: BEAF A/B Swap 다양화 데이터셋 생성
- **핵심 개선사항**:
  1. **위치 불변성**: 50:50 Pos-First / Neg-First 교환
  2. **어휘 다양성**: 4개 템플릿 패밀리 (Standard, Lacking, Absent, Free-of)
  3. **명시적 컬럼**: `object_a`, `object_b`, `pos_position`, `template_family` 등
- **템플릿 소스**: `benchmarks/data/beaf_expanded_templates.json`
- **출력**: `beaf_counterfactual_ab_swap_diverse.csv`

---

### 3. 생성 파이프라인 흐름

```
COCO annotations + LLaMA 캡션
    ↓
create_full_coco_paired_v2.py
    ↓
COCO_val_full_paired_v2.csv  →  analysis/run_analysis.py
                              →  analysis/subspace_analysis.py

beaf_counterfactual_6col.csv (수동 생성)
    ↓
generate_beaf_ab_swap_dataset.py
    ↓
beaf_counterfactual_ab_swap.csv  →  beaf/audit_ab_swap_dataset.py
                                 →  beaf/run_ab_swap_evaluation.py
    ↓
generate_beaf_ab_swap_diverse_dataset.py
    ↓
beaf_counterfactual_ab_swap_diverse.csv  →  eval_negation_existence_probe.py
                                         →  eval_ab_swap_negation_diagnostic.py
```

---

### 4. synthetic_datasets 상호 참조

루트의 `synthetic_datasets/` 디렉토리는 대규모 파인튜닝 데이터를 생성합니다:

| 하위 디렉토리 | 파일 수 | 역할 |
|:---|:---:|:---|
| `evaluation/` | 7 | NegBench 평가 데이터 생성 (MCQ, Retrieval, Video) |
| `finetuning/` | 8 | CC12M-NegCap / CC12M-NegMCQ 파인튜닝 데이터 생성 |

자세한 내용은 각 디렉토리의 `README.md`를 참조하세요:
- [synthetic_datasets/evaluation/README.md](../../../synthetic_datasets/evaluation/README.md)
- [synthetic_datasets/finetuning/README.md](../../../synthetic_datasets/finetuning/README.md)

---

---
## 🇺🇸 English

### 1. Module Overview

The `benchmarks/src/data_generation/` package generates **Paired Caption CSVs and BEAF Counterfactual datasets** used for NegBench evaluation and analysis.

Related: The root `synthetic_datasets/` directory generates large-scale fine-tuning datasets (CC12M-NegCap, CC12M-NegMCQ); see its separate `README.md`.

---

### 2. File-by-File Details (3 files)

#### `create_full_coco_paired_v2.py` (306 lines)
- **Role**: Generate COCO-based positive/negative paired caption CSV (v2)
- **Input Sources**:
  1. `COCO_val_negated_retrieval_...csv` → 5,000 images' pos/neg object lists
  2. `COCO_val_mcq_llama3.1_rephrased.csv` → LLaMA-generated diverse pos/neg caption pairs (~1,869)
  3. `COCO_val_retrieval.csv` (reference only)
- **Strategy**: Use LLaMA captions when available; otherwise round-robin 10 diverse template pools. Grammar exceptions for COCO 80 categories.
- **Output**: `COCO_val_full_paired_v2.csv` — columns: `image_path`, `object_name`, `positive_caption`, `negative_caption`, `object_in_image`, `source_template`

#### `generate_beaf_ab_swap_dataset.py` (149 lines)
- **Role**: Basic BEAF A/B object swap counterfactual dataset generation
- **Output**: `beaf_counterfactual_ab_swap.csv` (6-column format)

#### `generate_beaf_ab_swap_diverse_dataset.py` (421 lines)
- **Role**: Diverse BEAF A/B swap dataset generation
- **Enhancements**: 50:50 position swap, 4 template families, explicit metadata columns
- **Template Source**: `benchmarks/data/beaf_expanded_templates.json`
- **Output**: `beaf_counterfactual_ab_swap_diverse.csv`

---

### 3. Generation Pipeline Flow

```
COCO annotations + LLaMA captions
    ↓
create_full_coco_paired_v2.py
    ↓
COCO_val_full_paired_v2.csv  →  analysis/run_analysis.py
                              →  analysis/subspace_analysis.py

beaf_counterfactual_6col.csv (manually created)
    ↓
generate_beaf_ab_swap_dataset.py
    ↓
beaf_counterfactual_ab_swap.csv  →  beaf/audit_ab_swap_dataset.py
                                 →  beaf/run_ab_swap_evaluation.py
    ↓
generate_beaf_ab_swap_diverse_dataset.py
    ↓
beaf_counterfactual_ab_swap_diverse.csv  →  eval_negation_existence_probe.py
                                         →  eval_ab_swap_negation_diagnostic.py
```

---

### 4. synthetic_datasets Cross-Reference

The root `synthetic_datasets/` directory generates large-scale fine-tuning data:

| Subdirectory | Files | Purpose |
|:---|:---:|:---|
| `evaluation/` | 7 | NegBench evaluation data generation (MCQ, Retrieval, Video) |
| `finetuning/` | 8 | CC12M-NegCap / CC12M-NegMCQ fine-tuning data generation |

See each directory's `README.md` for details:
- [synthetic_datasets/evaluation/README.md](../../../synthetic_datasets/evaluation/README.md)
- [synthetic_datasets/finetuning/README.md](../../../synthetic_datasets/finetuning/README.md)
