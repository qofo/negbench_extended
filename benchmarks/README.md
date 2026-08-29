# NegBench Evaluation Benchmarks

This folder contains the code and scripts required to evaluate the negation understanding capabilities of CLIP models using the **NegBench** benchmark. The evaluations focus on two main tasks:
1. **Retrieval Tasks**: Evaluating the model's ability to retrieve images/videos given queries with negation.
2. **Multiple Choice Tasks (MCQ)**: Testing the model's understanding of negation through multiple-choice questions.

The benchmark code is based on **[OpenCLIP](https://github.com/mlfoundations/open_clip)** and extends it to include negation-focused evaluations.

---

## Installation

Follow the steps below to set up the environment and install the required dependencies.

### 1. Clone the Repository

First, clone the **NegBench** repository and navigate to the `benchmarks/` folder:
```bash
git clone https://github.com/m1k2zoo/negbench.git
cd NegBench/benchmarks/
```

### 2. Create and Activate a Conda Environment

Create a new Conda environment named `clip_negation`:
```bash
conda create -n clip_negation python=3.9 -y
conda activate clip_negation
```

### 3. Install the Requirements

Install the dependencies in development/editable mode:
```bash
pip install -e .
```

This command installs the `benchmarks` project and its dependencies in editable mode, allowing for modifications to the code.

---

## File Structure

The expected directory structure for running the benchmarks is as follows:

```
NegBench/
├── benchmarks/
│   ├── src/                    # Source code
│   │   ├── open_clip/          # OpenCLIP codebase
│   │   ├── training/           # Training-specific logic
│   │   └── evaluation/         # Evaluation-specific logic
│   ├── scripts/                # Bash scripts for running evaluations
│   ├── data/                   # Data for evaluation (user needs to download)
│   ├── logs/                   # Logs directory (generated during runs)
│   └── README.md
├── synthetic_datasets/         # Synthetic (evaluation and training) datasets construction
└── models/                     # Directory for models
├── datasets.md                 # Instructions for downloading and processing eval data
├── models.md                   # Instructions for downloading trained models
```

### Data Directory (`data/`)

The `data/` directory should include datasets for evaluation, organized as follows:
```
data/
├── images/
│   ├── COCO_val_mcq_llama3.1_rephrased.csv
│   ├── VOC2007_mcq_llama3.1_rephrased.csv
│   ├── synthetic_mcq_llama3.1_rephrased.csv
│   ├── COCO_val_retrieval.csv
│   ├── COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv
│   ├── synthetic_retrieval_v1.csv
│   └── synthetic_retrieval_v2.csv
├── videos/
│   ├── msr_vtt_retrieval.csv
│   ├── msr_vtt_retrieval_rephrased_llama.csv
│   └── msr_vtt_mcq_rephrased_llama.csv
```

The CSV files should include all the evaluation captions, but the images for each dataset need to be downloaded separately. The image paths may also need to be updated by the user according to their directory structure. The captions in these files can be used to reproduce the main evaluations in NegBench.

You can download these preprocessed CSV files **[here](https://drive.google.com/drive/folders/1kSEq0mkV1t1T8GuOAM65iz_iAA7e5gxB?usp=sharing)**

VOC2007 images are accessible through their **[official link](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)**

COCO 2017 Val Images can be found **[here](https://cocodataset.org/#download)**

The images corresponding to the stable diffusion generated dataset that was introduced in NegBench will be uploaded shortly. 

For mode details about preparing the datasets for evaluation, refer to [`dataset.md`](../datasets.md).


### Models Directory (`models/`)

The `models/` directory should follow this structure:
```
models/
├── ConCLIP/
│   ├── conclip_b32_openclip_version.pt
├── NegCLIP/
│   ├── negclip.pth
├── NegCLIP_CC12M_NegFull_ViT-B-32_lr1e-8_clw0.99_mlw0.01/
│   ├── finetuned_checkpoint.pt
```

Note that OpenCLIP models can be used directly without downloading.
For mode details about preparing the models for evaluation, refer to [`models.md`](../models.md).

---

## Running Evaluations

### 1. Evaluating a Single Model (Please start here!)

The script **`run_single_model_evaluations.sh`** facilitates quick testing of models, including pretrained models like CLIP OpenAI and fine-tuned models like those discussed in Section 5 of the paper. This script evaluates a single model for retrieval and MCQ tasks.

#### User-Defined Variables:
- **`BASE_DIR`**: Define the base directory in `run_single_model_evaluations.sh`:
  ```bash
  BASE_DIR="/path/to/your/research/project"
  ```
- **`MODEL`**: Specify the model architecture:
  ```bash
  MODEL="ViT-B-32"
  ```
- **`MODEL_NAME`**: Define the model name (used for logs and naming):
  ```bash
  MODEL_NAME="ViT_B_32_openai"
  ```
- **`PRETRAINED_MODEL`**: Specify the pretrained model.  
  - For OpenCLIP models, provide the **tag name** (e.g., `"openai"`):
    ```bash
    PRETRAINED_MODEL="openai"
    ```
  - For fine-tuned models, provide the **path to the checkpoint**. For example:
    ```bash
    PRETRAINED_MODEL="$MODELS_DIR/NegCLIP/negclip.pth"
    ```

#### Run Command:
```bash
bash scripts/run_single_model_evaluations.sh
```

For more details on OpenCLIP-supported models and pretrained tags, refer to the **[OpenCLIP documentation](https://github.com/mlfoundations/open_clip)**.

---

### 2. Evaluating Multiple Models (Optional: might need more advanced bash scripting experience)

The script **`run_pretrained_model_evaluations.sh`** is used to reproduce most of the results in Section 4 of the paper. It evaluates pretrained models (e.g., OpenAI's CLIP, LAION, or DataComp models) and supports both **main results** and **template results** (figures 3, 4, 5).

#### User-Defined Variables:
- **`BASE_DIR`**: Define the base directory in `run_pretrained_model_evaluations.sh`:
  ```bash
  BASE_DIR="/path/to/your/research/project"
  ```
- **`experiment_type`**: Specify the experiment type (`main` or `template`) in `run_pretrained_model_evaluations.sh`:
  ```bash
  experiment_type="main"  # Choose "main" or "template"
  ```
- **SLURM Options**: Adjust SLURM-specific options directly in the script:
  ```bash
  SBATCH_PARTITION="gpu_partition"
  SBATCH_GPUS=1
  SBATCH_CPUS=8
  SBATCH_MEM="20gb"
  SBATCH_TIME="2:00:00"
  ```

#### Run Command:
```bash
bash scripts/run_pretrained_model_evaluations.sh
```

---

### 2.1. Individual Image and Video Evaluations

**`evaluate_images.sh`** and **`evaluate_videos.sh`** are example SLURM scripts that perform the actual evaluation for image and video datasets, respectively. These scripts are called internally by the main script **`run_pretrained_model_evaluations.sh`** but can also be run directly.

#### User-Defined Variables:
- **SLURM Options**: Edit `SBATCH_PARTITION`, `SBATCH_CPUS`, etc., directly in the scripts:
  ```bash
  # Example in evaluate_images.sh
  SBATCH_PARTITION="gpu_partition"
  SBATCH_CPUS=8
  SBATCH_MEM="20gb"
  ```

#### Run Commands:
```bash
bash scripts/evaluate_images.sh
bash scripts/evaluate_videos.sh
```

---

## Logs and Results

Evaluation logs and results are saved in the **`logs/`** directory:
- **Image Evaluations**: `logs/eval_images/`
- **Video Evaluations**: `logs/eval_videos/`

---

For more detailed explanations, contact the developers.
