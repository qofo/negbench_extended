"""
E5-V NegBench Evaluation Entry Point
======================================
Evaluates the E5-V model on NegBench MCQ and Retrieval tasks.

This script is independent from the existing eval_negation.py and does NOT
modify any existing negbench code. It directly loads CSV datasets and uses
the E5VWrapper for embedding computation.

Example usage (MCQ only):
    python -m e5v_analysis.eval_negbench_e5v \
        --model-name royokong/e5-v \
        --coco-mcq COCO_val_mcq_llama3.1_rephrased.csv \
        --output-dir logs/e5v/negbench \
        --batch-size 4 \
        --device cuda

Example usage (MCQ + Retrieval):
    python -m e5v_analysis.eval_negbench_e5v \
        --model-name royokong/e5-v \
        --coco-mcq COCO_val_mcq_llama3.1_rephrased.csv \
        --coco-retrieval COCO_val_retrieval.csv \
        --coco-negated-retrieval COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv \
        --data-root benchmarks \
        --output-dir logs/e5v/negbench \
        --batch-size 4 \
        --device cuda
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from e5v_analysis.e5v_wrapper import E5VWrapper
from e5v_analysis.utils import resolve_image_path, setup_logging


# ---------------------------------------------------------------------------
# MCQ Evaluation
# ---------------------------------------------------------------------------

def evaluate_mcq(
    wrapper: E5VWrapper,
    csv_path: str,
    data_root: str = None,
    text_batch_size: int = 16,
    image_batch_size: int = 4,
    dataset_name: str = "coco-mcq",
) -> dict:
    """
    Evaluate E5-V on an MCQ task from a NegBench CSV.

    The CSV has columns: image_path, correct_answer, caption_0..caption_N,
    correct_answer_template.

    For each row:
    1. Encode the image with E5-V
    2. Encode all caption options with E5-V
    3. Select the caption with highest cosine similarity to the image
    4. Compare with ground truth

    Returns:
        dict with accuracy metrics and per-sample results.
    """
    df = pd.read_csv(csv_path)
    num_options = len([c for c in df.columns if c.startswith("caption_")])

    # Canonical caption type order (same as CsvMCQDataset.CAPTION_TYPES)
    caption_types_canonical = ['gt', 'hybrid', 'positive', 'negative']
    while len(caption_types_canonical) < num_options:
        caption_types_canonical.append(f"option_{len(caption_types_canonical)}")

    total = len(df)
    correct_total = 0
    correct_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
    total_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
    wrong_answer_counts = {'hybrid': 0, 'positive': 0, 'negative': 0}
    predictions_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
    wrong_by_qtype = {
        'positive': {'positive': 0, 'negative': 0, 'hybrid': 0},
        'negative': {'positive': 0, 'negative': 0, 'hybrid': 0},
        'hybrid': {'positive': 0, 'negative': 0, 'hybrid': 0},
    }
    sample_results = []

    print(f"Evaluating {dataset_name}: {total} samples, {num_options} options each")

    for idx in tqdm(range(total), desc=f"E5-V MCQ [{dataset_name}]"):
        row = df.iloc[idx]
        image_path = resolve_image_path(str(row['image_path']), data_root)
        captions = [str(row[f"caption_{i}"]) for i in range(num_options)]
        correct_answer = int(row['correct_answer']) if 'correct_answer' in row else 0
        question_type = str(row.get('correct_answer_template', 'mcq'))
        caption_types = list(caption_types_canonical[:num_options])

        # Load image
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Cannot open image {image_path}: {e}, skipping.")
            continue

        # Encode image (single)
        img_emb = wrapper.encode_image([pil_image], batch_size=1)  # (1, D)

        # Encode all caption options
        text_embs = wrapper.encode_text(captions, batch_size=text_batch_size)  # (N, D)

        # Compute cosine similarity and predict
        logits = (img_emb @ text_embs.t()).squeeze(0)  # (N,)
        predicted = logits.argmax().item()
        is_correct = (predicted == correct_answer)

        if is_correct:
            correct_total += 1

        # Build per-sample record
        sample = {
            "image_path": image_path,
            "question_type": question_type,
            "correct_answer": correct_answer,
            "predicted_answer": predicted,
            "is_correct": is_correct,
        }
        for j in range(num_options):
            sample[f"caption_{j}"] = captions[j]
            sample[f"logit_{j}"] = float(logits[j])

        sample_results.append(sample)

        # Aggregate metrics by question type
        if question_type in total_by_type:
            total_by_type[question_type] += 1
            predicted_type = caption_types[predicted]
            if is_correct:
                correct_by_type[question_type] += 1
                predictions_by_type[question_type] = predictions_by_type.get(question_type, 0) + 1
            else:
                wrong_type = predicted_type
                wrong_answer_counts[wrong_type] = wrong_answer_counts.get(wrong_type, 0) + 1
                predictions_by_type[wrong_type] = predictions_by_type.get(wrong_type, 0) + 1
                wrong_by_qtype[question_type][wrong_type] = \
                    wrong_by_qtype[question_type].get(wrong_type, 0) + 1

    # Compute accuracies
    total_accuracy = correct_total / total if total > 0 else 0.0
    positive_accuracy = (
        correct_by_type['positive'] / total_by_type['positive']
        if total_by_type['positive'] > 0 else float('nan')
    )
    negative_accuracy = (
        correct_by_type['negative'] / total_by_type['negative']
        if total_by_type['negative'] > 0 else float('nan')
    )
    hybrid_accuracy = (
        correct_by_type['hybrid'] / total_by_type['hybrid']
        if total_by_type['hybrid'] > 0 else float('nan')
    )

    total_wrong = sum(wrong_answer_counts.values())
    wrong_pcts = {
        k: (v / total_wrong * 100) if total_wrong > 0 else 0.0
        for k, v in wrong_answer_counts.items()
    }

    most_common_wrong = max(wrong_answer_counts, key=wrong_answer_counts.get) if total_wrong > 0 else "n/a"

    metrics = {
        f"{dataset_name}-total_accuracy": total_accuracy,
        f"{dataset_name}-positive_accuracy": positive_accuracy,
        f"{dataset_name}-negative_accuracy": negative_accuracy,
        f"{dataset_name}-hybrid_accuracy": hybrid_accuracy,
        f"{dataset_name}-most_common_wrong_answer_type": most_common_wrong,
        f"{dataset_name}-wrong_answer_percentages": list(wrong_pcts.items()),
        f"{dataset_name}-predictions_by_type": predictions_by_type,
        f"{dataset_name}-wrong_answers_by_question_type": wrong_by_qtype,
        f"{dataset_name}-sample_results": sample_results,
    }

    return metrics


# ---------------------------------------------------------------------------
# Retrieval Evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    wrapper: E5VWrapper,
    csv_path: str,
    data_root: str = None,
    text_batch_size: int = 16,
    image_batch_size: int = 4,
    recall_k_list: list = [1, 5],
    dataset_name: str = "coco-retrieval",
) -> dict:
    """
    Evaluate E5-V on a retrieval task from a NegBench CSV.

    The CSV has columns: filepath, captions (a Python list literal of strings).

    Returns:
        dict with recall@k metrics.
    """
    df = pd.read_csv(csv_path)

    # Detect column names
    img_col = "filepath" if "filepath" in df.columns else "image_path"
    caption_col = "captions" if "captions" in df.columns else "caption"

    # Collect all images and their captions
    image_paths = []
    all_captions = []
    texts_image_index = []  # maps each text to its image index

    for idx, row in df.iterrows():
        img_path = resolve_image_path(str(row[img_col]), data_root)
        image_paths.append(img_path)

        # captions column may be a Python list literal
        caps = row[caption_col]
        if isinstance(caps, str) and caps.startswith("["):
            caps = eval(caps)
        elif isinstance(caps, str):
            caps = [caps]

        for cap in caps:
            all_captions.append(cap)
            texts_image_index.append(idx)

    print(f"Retrieval [{dataset_name}]: {len(image_paths)} images, {len(all_captions)} captions")

    # Encode all images
    print("  Encoding images...")
    pil_images = []
    for p in tqdm(image_paths, desc="  Loading images"):
        pil_images.append(Image.open(p).convert("RGB"))

    img_embs = wrapper.encode_image(pil_images, batch_size=image_batch_size)

    # Encode all texts
    print("  Encoding texts...")
    text_embs = wrapper.encode_text(all_captions, batch_size=text_batch_size)

    # Compute similarity matrix: (num_texts, num_images)
    scores = text_embs @ img_embs.t()

    # Build positive pairs matrix
    positive_pairs = torch.zeros_like(scores, dtype=torch.bool)
    for text_idx, img_idx in enumerate(texts_image_index):
        positive_pairs[text_idx, img_idx] = True

    # Compute recall@k
    metrics = {}
    for k in recall_k_list:
        # Image retrieval: for each text, find matching image in top-k
        ir_recall = _recall_at_k(scores, positive_pairs, k)
        metrics[f"{dataset_name}-image_retrieval_recall@{k}"] = ir_recall

        # Text retrieval: for each image, find matching text in top-k
        tr_recall = _recall_at_k(scores.t(), positive_pairs.t(), k)
        metrics[f"{dataset_name}-text_retrieval_recall@{k}"] = tr_recall

    return metrics


def _recall_at_k(scores: torch.Tensor, positive_pairs: torch.Tensor, k: int) -> float:
    """Compute recall@k: fraction of queries with at least one hit in top-k."""
    nb_queries, nb_items = scores.shape
    topk_indices = torch.topk(scores, min(k, nb_items), dim=1)[1]
    topk_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_items)
    positive_reshaped = positive_pairs.view(nb_queries, 1, nb_items)
    hits = (topk_onehot * positive_reshaped).sum(dim=(1, 2))
    recall = (hits > 0).float().mean().item()
    return recall


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Evaluate E5-V on NegBench MCQ and Retrieval tasks."
    )

    # Model
    parser.add_argument("--model-name", type=str, default="royokong/e5-v",
                        help="HuggingFace model name or local path for E5-V.")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to a LoRA adapter to merge into the LLM.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="PyTorch device (e.g. cuda, cuda:0, cuda:1).")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model weight dtype.")
    parser.add_argument("--quantize", type=str, default=None,
                        choices=["int4", "int8"],
                        help="Enable bitsandbytes quantization (optional).")

    # Data
    parser.add_argument("--coco-mcq", type=str, default=None,
                        help="Path to COCO MCQ CSV.")
    parser.add_argument("--voc2007-mcq", type=str, default=None,
                        help="Path to VOC2007 MCQ CSV.")
    parser.add_argument("--synthetic-mcq", type=str, default=None,
                        help="Path to Synthetic MCQ CSV.")
    parser.add_argument("--coco-retrieval", type=str, default=None,
                        help="Path to COCO retrieval CSV.")
    parser.add_argument("--coco-negated-retrieval", type=str, default=None,
                        help="Path to COCO negated retrieval CSV.")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory for resolving relative image paths in CSVs.")

    # Batching
    parser.add_argument("--text-batch-size", type=int, default=16,
                        help="Batch size for text encoding.")
    parser.add_argument("--image-batch-size", type=int, default=4,
                        help="Batch size for image encoding.")

    # Output
    parser.add_argument("--output-dir", type=str, default="logs/e5v/negbench",
                        help="Directory to save results.")
    parser.add_argument("--name", type=str, default="e5v_eval",
                        help="Experiment name for log subdirectory.")

    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)

    # Setup output
    log_dir = os.path.join(args.output_dir, args.name)
    logger = setup_logging(log_dir)

    # Dtype mapping
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    # Load model
    logger.info(f"Loading E5-V model: {args.model_name}")
    logger.info(f"  Device: {args.device}, Dtype: {args.dtype}, Quantize: {args.quantize}")
    if args.lora_path:
        logger.info(f"  LoRA adapter: {args.lora_path}")

    wrapper = E5VWrapper(
        model_name=args.model_name,
        device=args.device,
        dtype=dtype_map[args.dtype],
        quantize=args.quantize,
        lora_path=args.lora_path,
    )

    all_metrics = {}

    # --- MCQ Evaluation ---
    mcq_configs = [
        (args.coco_mcq, "coco-mcq"),
        (args.voc2007_mcq, "voc2007-mcq"),
        (args.synthetic_mcq, "synthetic-mcq"),
    ]
    for csv_path, name in mcq_configs:
        if csv_path is None:
            continue
        logger.info(f"Evaluating MCQ: {name} from {csv_path}")
        metrics = evaluate_mcq(
            wrapper=wrapper,
            csv_path=csv_path,
            data_root=args.data_root,
            text_batch_size=args.text_batch_size,
            image_batch_size=args.image_batch_size,
            dataset_name=name,
        )

        # Save per-sample prediction CSV
        sample_key = f"{name}-sample_results"
        if sample_key in metrics:
            pred_dir = os.path.join(log_dir, "predictions")
            os.makedirs(pred_dir, exist_ok=True)
            df_pred = pd.DataFrame(metrics.pop(sample_key))
            csv_out = os.path.join(pred_dir, f"{name}_predictions.csv")
            df_pred.to_csv(csv_out, index=False)
            logger.info(f"  Saved predictions CSV: {csv_out}")

        all_metrics.update(metrics)

        # Print summary for this dataset
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")

    # --- Retrieval Evaluation ---
    retrieval_configs = [
        (args.coco_retrieval, "coco-retrieval"),
        (args.coco_negated_retrieval, "coco-negated-retrieval"),
    ]
    for csv_path, name in retrieval_configs:
        if csv_path is None:
            continue
        logger.info(f"Evaluating Retrieval: {name} from {csv_path}")
        metrics = evaluate_retrieval(
            wrapper=wrapper,
            csv_path=csv_path,
            data_root=args.data_root,
            text_batch_size=args.text_batch_size,
            image_batch_size=args.image_batch_size,
            dataset_name=name,
        )
        all_metrics.update(metrics)

        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")

    # --- Save overall results ---
    logger.info("=" * 60)
    logger.info("Evaluation complete. Summary:")
    for k, v in all_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        elif isinstance(v, str):
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    # Save JSON (filter out non-serializable items)
    serializable_metrics = {}
    for k, v in all_metrics.items():
        if isinstance(v, (int, float, str, list, dict, bool)):
            serializable_metrics[k] = v

    results_path = os.path.join(log_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(serializable_metrics, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
