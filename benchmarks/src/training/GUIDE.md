# Training Module Technical Guide

---
## 🇰🇷 한국어

### 1. 모듈 개요

`benchmarks/src/training/` 패키지는 NegCLIP 파인튜닝 인프라를 제공합니다. OpenCLIP 기반의 분산 학습, MCQ/Retrieval 평가, 비디오 처리를 지원합니다.

---

### 2. 파일별 상세 설명

#### 핵심 학습 파이프라인

| 파일 | 줄 수 | 역할 |
|:---|:---:|:---|
| `main.py` | 529 | 학습 메인 엔트리포인트. 모델 생성, 옵티마이저 설정, 체크포인트 로드/저장, 에포크 루프 관리 |
| `train.py` | 938 | 학습 루프 핵심. `train_one_epoch()`, `train_one_epoch_mixed()`, `evaluate()` 함수 제공. AMP, WandB, TensorBoard 로깅 포함 |
| `data.py` | 1130 | 데이터셋 & 데이터로더. `CsvDataset`, `CsvMCQDataset`, `CsvRetrievalDataset`, WebDataset 지원, 비디오 MCQ/Retrieval 데이터셋 |
| `params.py` | 529 | 하이퍼파라미터 CLI 파서. 모델, 학습률, 배치 크기, 정밀도, 분산 학습, 평가 관련 전체 인자 정의 |

#### 학습 인프라 유틸리티

| 파일 | 줄 수 | 역할 |
|:---|:---:|:---|
| `scheduler.py` | ~60 | 학습률 스케줄러: `cosine_lr()`, `const_lr()`, `const_lr_cooldown()` |
| `distributed.py` | ~130 | 분산 학습: `is_master()`, `init_distributed_device()`, `broadcast_object()` |
| `precision.py` | ~15 | AMP 정밀도: `get_autocast()` |
| `logger.py` | ~30 | 로깅 설정: `setup_logging()` |
| `file_utils.py` | ~80 | 파일 유틸: `pt_load()`, `check_exists()`, `start_sync_process()`, `remote_sync()` |
| `profiler.py` | ~120 | 학습 프로파일러 도구 |
| `zero_shot.py` | ~150 | Zero-shot 분류: ImageNet 등 다운스트림 평가 |
| `debug_data.py` | ~240 | 데이터 로더 디버깅 스크립트 |

#### 시각화

| 파일 | 줄 수 | 역할 |
|:---|:---:|:---|
| `visualize_image_dataset.py` | ~80 | 이미지 데이터셋 샘플 시각화 |
| `visualize_video_dataset.py` | ~170 | 비디오 데이터셋 프레임 시각화 |

#### `video_utils/` 서브패키지 (5개 파일)

| 파일 | 역할 |
|:---|:---|
| `frame_sampler.py` | 비디오 프레임 샘플링 전략 (Uniform, Random) |
| `model.py` | 비디오 인코더 래퍼 (프레임 평균 풀링) |
| `video_dataset.py` | `CsvVideoCaptionDataset`, `CsvVideoMCQDataset` |
| `video_reader.py` | 비디오 디코딩 (decord/PyAV 백엔드) |
| `utils.py` | 비디오 유틸리티 함수 |

---

### 3. 학습 파이프라인 데이터 흐름

```
params.py (CLI 파싱)
    ↓
main.py (엔트리포인트)
    ├── open_clip.create_model_and_transforms()  → 모델 생성
    ├── data.py → get_data()                     → DataLoader 생성
    │   ├── CsvMCQDataset (MCQ 학습)
    │   ├── CsvDataset (캡션 학습)
    │   └── WebDataset (대규모 학습)
    ├── scheduler.py → cosine_lr()               → LR 스케줄
    └── train.py
        ├── train_one_epoch()                    → Contrastive Loss
        ├── train_one_epoch_mixed()              → CL + MCQ Loss 혼합
        └── evaluate()                           → MCQ/Retrieval 평가
            └── evaluation.utils.evaluate()
```

---

---
## 🇺🇸 English

### 1. Module Overview

The `benchmarks/src/training/` package provides NegCLIP fine-tuning infrastructure. Supports OpenCLIP-based distributed training, MCQ/Retrieval evaluation, and video processing.

---

### 2. File-by-File Details

#### Core Training Pipeline

| File | Lines | Role |
|:---|:---:|:---|
| `main.py` | 529 | Main training entrypoint. Model creation, optimizer setup, checkpoint load/save, epoch loop |
| `train.py` | 938 | Core training loop. `train_one_epoch()`, `train_one_epoch_mixed()`, `evaluate()`. AMP, WandB, TensorBoard logging |
| `data.py` | 1130 | Datasets & DataLoaders. `CsvDataset`, `CsvMCQDataset`, `CsvRetrievalDataset`, WebDataset, Video MCQ/Retrieval |
| `params.py` | 529 | Hyperparameter CLI parser. Full argument definitions for model, learning rate, batch size, precision, distributed, evaluation |

#### Training Infrastructure Utilities

| File | Lines | Role |
|:---|:---:|:---|
| `scheduler.py` | ~60 | LR schedulers: `cosine_lr()`, `const_lr()`, `const_lr_cooldown()` |
| `distributed.py` | ~130 | Distributed training: `is_master()`, `init_distributed_device()`, `broadcast_object()` |
| `precision.py` | ~15 | AMP precision: `get_autocast()` |
| `logger.py` | ~30 | Logging setup: `setup_logging()` |
| `file_utils.py` | ~80 | File utilities: `pt_load()`, `check_exists()`, `start_sync_process()`, `remote_sync()` |
| `zero_shot.py` | ~150 | Zero-shot classification: ImageNet and downstream evaluation |
| `debug_data.py` | ~240 | Data loader debugging script |

#### Visualization

| File | Lines | Role |
|:---|:---:|:---|
| `visualize_image_dataset.py` | ~80 | Image dataset sample visualization |
| `visualize_video_dataset.py` | ~170 | Video dataset frame visualization |

#### `video_utils/` Subpackage (5 files)

| File | Role |
|:---|:---|
| `frame_sampler.py` | Video frame sampling strategies (Uniform, Random) |
| `model.py` | Video encoder wrapper (frame mean pooling) |
| `video_dataset.py` | `CsvVideoCaptionDataset`, `CsvVideoMCQDataset` |
| `video_reader.py` | Video decoding (decord/PyAV backends) |
| `utils.py` | Video utility functions |

---

### 3. Training Pipeline Data Flow

```
params.py (CLI parsing)
    ↓
main.py (entrypoint)
    ├── open_clip.create_model_and_transforms()  → Model creation
    ├── data.py → get_data()                     → DataLoader creation
    │   ├── CsvMCQDataset (MCQ training)
    │   ├── CsvDataset (caption training)
    │   └── WebDataset (large-scale training)
    ├── scheduler.py → cosine_lr()               → LR schedule
    └── train.py
        ├── train_one_epoch()                    → Contrastive Loss
        ├── train_one_epoch_mixed()              → CL + MCQ Loss mixed
        └── evaluate()                           → MCQ/Retrieval evaluation
            └── evaluation.utils.evaluate()
```
