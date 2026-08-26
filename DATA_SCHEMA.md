# Data Schema Reference

---
## 🇰🇷 한국어

### 1. 개요

이 문서는 `benchmarks/data/` 디렉토리에 있는 모든 데이터 파일의 스키마, 용도, 관련 실험을 정리합니다.

---

### 2. CSV 스키마 유형

데이터 파일은 **4가지 스키마 유형**으로 구분됩니다.

#### 스키마 A — Paired Caption (6-column)

| 컬럼 | 타입 | 설명 |
|:---|:---|:---|
| `image_path` | str | 이미지 파일 경로 |
| `object_name` | str | 대상 객체 이름 (COCO 80 카테고리) |
| `positive_caption` | str | 객체가 존재함을 서술하는 캡션 |
| `negative_caption` | str | 객체가 부재함을 서술하는 캡션 |
| `object_in_image` | bool | 이미지에 실제 객체 존재 여부 |
| `source_template` | str | 사용된 템플릿 식별자 |

**사용 파일**: `COCO_val_full_paired*.csv`, `beaf_counterfactual_6col.csv`, `beaf_counterfactual_ab_swap.csv`, `beaf_paired_*.csv`, `COCO_val_mcq_top100_paired.csv`

#### 스키마 B — MCQ (7-column)

| 컬럼 | 타입 | 설명 |
|:---|:---|:---|
| `image_path` | str | 이미지 파일 경로 |
| `correct_answer` | int | 정답 캡션 인덱스 (0~3) |
| `caption_0` ~ `caption_3` | str | 4개 선택지 캡션 |
| `correct_answer_template` | str | 정답 캡션 템플릿 유형 |

**사용 파일**: `COCO_val_mcq_llama3.1_rephrased.csv`, `VOC2007_mcq_llama3.1_rephrased.csv`, `synthetic_mcq_llama3.1_rephrased.csv`, `COCO_val_mcq_top100_uncovered.csv`

#### 스키마 C — Retrieval

| 컬럼 | 타입 | 설명 |
|:---|:---|:---|
| `filepath` | str | 이미지 파일 경로 |
| `captions` | str (list) | 연관 캡션 리스트 (Python literal) |
| `positive_objects` | str (list) | 이미지 내 존재 객체 (일부 파일) |
| `negative_objects` | str (list) | 이미지 내 부재 객체 (일부 파일) |
| `image_id` | int | COCO 이미지 ID (일부 파일) |

**사용 파일**: `COCO_val_retrieval.csv`, `COCO_val_negated_retrieval_*.csv`, `synthetic_retrieval_v*.csv`

#### 스키마 D — CheXpert Medical MCQ

| 컬럼 | 타입 | 설명 |
|:---|:---|:---|
| `correct_answer` | int | 정답 인덱스 (0~1) |
| `caption_0`, `caption_1` | str | 2개 선택지 (Binary MCQ) |
| `image_path` | str | 흉부 X-ray 경로 |

**사용 파일**: `chexpert_binary_mcq*.csv`

#### 스키마 E — AB Swap Diverse (12-column)

| 컬럼 | 타입 | 설명 |
|:---|:---|:---|
| `image_path` | str | 이미지 파일 경로 |
| `object_name` | str | 주 객체 이름 |
| `object_a`, `object_b` | str | A/B 교환 객체 |
| `object_a_present`, `object_b_present` | bool | 각 객체 존재 여부 |
| `positive_caption`, `negative_caption` | str | 긍정/부정 캡션 |
| `object_in_image` | bool | 이미지 내 주 객체 존재 |
| `pos_position` | str | 긍정 표현 위치 (first/second) |
| `template_family` | str | 템플릿 패밀리 (A~D) |
| `source_template` | str | 템플릿 식별자 |

**사용 파일**: `beaf_counterfactual_ab_swap_diverse.csv`, `beaf_clean_ab_swap_diverse.csv`

---

### 3. 파일별 용도 및 실험 매핑

| 파일 | 스키마 | 주요 사용처 |
|:---|:---:|:---|
| `COCO_val_full_paired.csv` | A | `run_analysis.py`, `subspace_analysis.py`, `eval_sparse_text_dimensions.py` |
| `COCO_val_full_paired_v2.csv` | A | `run_analysis.py` (v2 개선판) |
| `COCO_val_full_paired_3k.csv` | A | 빠른 테스트용 3,000건 서브셋 |
| `COCO_val_mcq_llama3.1_rephrased.csv` | B | `eval_negation.py`, MCQ 평가 |
| `VOC2007_mcq_llama3.1_rephrased.csv` | B | `eval_negation.py`, VOC MCQ |
| `synthetic_mcq_llama3.1_rephrased.csv` | B | `eval_negation.py`, Synthetic MCQ |
| `COCO_val_retrieval.csv` | C | `eval_negation.py`, Retrieval 평가 |
| `COCO_val_negated_retrieval_*.csv` | C | `eval_negation.py`, 부정 Retrieval |
| `synthetic_retrieval_v1.csv`, `v2.csv` | C | Synthetic Retrieval 평가 |
| `beaf_counterfactual_6col.csv` | A | `run_beaf_analysis_v2.py`, BEAF 핵심 데이터 |
| `beaf_counterfactual_ab_swap.csv` | A | `audit_ab_swap_dataset.py`, `run_ab_swap_evaluation.py` |
| `beaf_counterfactual_ab_swap_diverse.csv` | E | `eval_negation_existence_probe.py`, `eval_ab_swap_negation_diagnostic.py` (인페인팅 CF 기반) |
| `beaf_clean_ab_swap_diverse.csv` | E | `eval_negation_existence_probe.py`, `eval_ab_swap_negation_diagnostic.py` (순수 원본 이미지 기반 클린 벤치마크) |
| `beaf_paired_v2.csv` | A | BEAF Paired Caption |
| `beaf_paired_3k.csv` | A | BEAF 빠른 테스트용 3,000건 |
| `chexpert_binary_mcq.csv` | D | CheXpert Medical MCQ 평가 |
| `chexpert_binary_mcq_control.csv` | D | CheXpert 대조군 |
| `chexpert_binary_mcq_control_valid_only.csv` | D | CheXpert 대조군 (유효 샘플만) |
| `COCO_val_mcq_top100_paired.csv` | A | 상위 100 객체 Paired |
| `COCO_val_mcq_top100_uncovered.csv` | B | 상위 100 미포함 MCQ |

---

### 4. 기타 데이터 파일

#### `benchmarks/data/beaf_expanded_templates.json`
- **역할**: BEAF 255개 확장 템플릿 정의 (123 부정, 132 긍정)
- **구조**: JSON 배열, 각 항목은 `{"template": "...", "polarity": "positive"|"negative", "group": "A"|"B"|"C"|"D"}`
- **사용처**: `run_beaf_object_generalization.py`, `generate_beaf_ab_swap_diverse_dataset.py`

#### `benchmarks/data/videos/`
- MSR-VTT 비디오 평가 CSV 파일: `msr_vtt_retrieval.csv`, `msr_vtt_retrieval_rephrased_llama.csv`, `msr_vtt_mcq_rephrased_llama.csv`

---

### 5. 기존 문서 상호 참조

- [datasets.md](datasets.md) — 데이터셋 다운로드 가이드 (COCO, VOC2007, MSR-VTT, CheXpert)
- [models.md](models.md) — 모델 다운로드 가이드 (OpenCLIP, NegCLIP, ConCLIP)

---

---
## 🇺🇸 English

### 1. Overview

This document describes the schema, purpose, and experiment mapping of all data files in the `benchmarks/data/` directory.

---

### 2. CSV Schema Types

Data files fall into **4 schema types**.

#### Schema A — Paired Caption (6-column)

| Column | Type | Description |
|:---|:---|:---|
| `image_path` | str | Image file path |
| `object_name` | str | Target object name (COCO 80 categories) |
| `positive_caption` | str | Caption affirming object presence |
| `negative_caption` | str | Caption negating object presence |
| `object_in_image` | bool | Whether object is actually in the image |
| `source_template` | str | Template identifier used |

#### Schema B — MCQ (7-column)

| Column | Type | Description |
|:---|:---|:---|
| `image_path` | str | Image file path |
| `correct_answer` | int | Correct caption index (0–3) |
| `caption_0` – `caption_3` | str | 4 answer options |
| `correct_answer_template` | str | Correct answer template type |

#### Schema C — Retrieval

| Column | Type | Description |
|:---|:---|:---|
| `filepath` | str | Image file path |
| `captions` | str (list) | Associated caption list (Python literal) |
| `positive_objects` | str (list) | Objects present in image (some files) |
| `negative_objects` | str (list) | Objects absent from image (some files) |

#### Schema D — CheXpert Medical MCQ

| Column | Type | Description |
|:---|:---|:---|
| `correct_answer` | int | Correct index (0–1) |
| `caption_0`, `caption_1` | str | 2 options (Binary MCQ) |
| `image_path` | str | Chest X-ray path |

#### Schema E — AB Swap Diverse (12-column)

| Column | Type | Description |
|:---|:---|:---|
| `image_path` | str | Image file path |
| `object_name` | str | Primary object name |
| `object_a`, `object_b` | str | A/B swap objects |
| `object_a_present`, `object_b_present` | bool | Presence of each object |
| `positive_caption`, `negative_caption` | str | Affirming/negating captions |
| `object_in_image` | bool | Primary object presence |
| `pos_position` | str | Position of positive expression (first/second) |
| `template_family` | str | Template family (A–D) |
| `source_template` | str | Template identifier |

---

### 3. File-to-Experiment Mapping

| File | Schema | Primary Usage |
|:---|:---:|:---|
| `COCO_val_full_paired.csv` | A | `run_analysis.py`, `subspace_analysis.py`, `eval_sparse_text_dimensions.py` |
| `COCO_val_full_paired_v2.csv` | A | `run_analysis.py` (v2 improved) |
| `COCO_val_mcq_llama3.1_rephrased.csv` | B | `eval_negation.py`, MCQ evaluation |
| `VOC2007_mcq_llama3.1_rephrased.csv` | B | `eval_negation.py`, VOC MCQ |
| `synthetic_mcq_llama3.1_rephrased.csv` | B | `eval_negation.py`, Synthetic MCQ |
| `COCO_val_retrieval.csv` | C | `eval_negation.py`, Retrieval |
| `beaf_counterfactual_6col.csv` | A | `run_beaf_analysis_v2.py`, core BEAF data |
| `beaf_counterfactual_ab_swap.csv` | A | `audit_ab_swap_dataset.py`, `run_ab_swap_evaluation.py` |
| `beaf_counterfactual_ab_swap_diverse.csv` | E | `eval_negation_existence_probe.py`, `eval_ab_swap_negation_diagnostic.py` |
| `chexpert_binary_mcq*.csv` | D | CheXpert Medical MCQ evaluation |

---

### 4. Other Data Files

#### `benchmarks/data/beaf_expanded_templates.json`
- 255 expanded templates (123 negative, 132 positive)
- Used by: `run_beaf_object_generalization.py`, `generate_beaf_ab_swap_diverse_dataset.py`

#### `benchmarks/data/videos/`
- MSR-VTT evaluation CSVs

---

### 5. Cross-References

- [datasets.md](datasets.md) — Dataset download guide
- [models.md](models.md) — Model download guide
