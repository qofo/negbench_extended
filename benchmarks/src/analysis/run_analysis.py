"""
Main Command Line Orchestrator for Modular Representation Analysis.

This entrypoint script parses runtime arguments, initializes data loaders and pre-trained
multimodal models, executes feature extractions, invokes metric calculation engines,
and delegates reporting duties to the AnalysisReporter module.
"""

import os
import argparse
from typing import List
import pandas as pd
import torch
import open_clip

from .config import AnalysisConfig, MetadataKey, RetrievalConfig
from .extractor import extract_all_features_unified, assert_embedding_consistency
from .metrics import (
    compute_pipeline_and_layer_breakdown,
    compute_direction_preservation,
    compute_linear_probe_and_subsets,
    compute_pca_spectrum_compression,
    compute_projection_svd_ablation,
    compute_image_text_retrieval_metrics
)
from .reporter import AnalysisReporter


def main():
    """
    Main orchestration routine executing representation geometry analysis pipeline.
    """
    parser = argparse.ArgumentParser(description="Modular CLIP Negation Analysis Pipeline (Flat Architecture)")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="Pre-trained vision-language model architecture")
    parser.add_argument("--pretrained", type=str, default="openai", help="Pre-trained weights checkpoint tag")
    parser.add_argument("--target_token", type=str, default="eot", choices=["eot", "mean", "all"], help="Token pooling strategy")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to paired positive/negative caption CSV")
    parser.add_argument("--output_dir", type=str, default="logs/analysis_modular/openai_vit_b32", help="Output artifact directory")
    parser.add_argument("--max_samples", type=int, default=60000, help="Maximum number of paired samples to analyze")
    parser.add_argument("--image_root", type=str, default="", help="Root directory containing COCO image files")
    parser.add_argument("--batch_size", type=int, default=256, help="Text processing mini-batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for statistical reproducibility")
    parser.add_argument("--no_bias", "--no-bias", action="store_true", default=False,
                        help="Disable bias/intercept in linear probes (default: bias enabled)")
    args = parser.parse_args()

    # Construct configuration object
    cfg = AnalysisConfig(
        model_name=args.model,
        pretrained=args.pretrained,
        target_token=args.target_token,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        image_root=args.image_root,
        batch_size=args.batch_size,
        seed=args.seed
    )

    reporter = AnalysisReporter(cfg.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Use Bias: {not args.no_bias}")

    pos_texts = []
    neg_texts = []
    pair_metadata = []

    path_k = MetadataKey.IMAGE_PATH.value
    obj_name_k = MetadataKey.OBJECT_NAME.value
    obj_in_img_k = MetadataKey.OBJECT_IN_IMAGE.value
    tmpl_k = MetadataKey.SOURCE_TEMPLATE.value

    # Load paired caption dataset and metadata attributes
    if cfg.csv_path and os.path.exists(cfg.csv_path):
        df = pd.read_csv(cfg.csv_path).head(cfg.max_samples)
        pos_texts = df["positive_caption"].astype(str).tolist()
        neg_texts = df["negative_caption"].astype(str).tolist()
        for _, row in df.iterrows():
            meta = {
                path_k: str(row.get(path_k, "")),
                obj_name_k: str(row.get(obj_name_k, "")),
                obj_in_img_k: row.get(obj_in_img_k, None),
                tmpl_k: str(row.get(tmpl_k, ""))
            }
            if isinstance(meta[obj_in_img_k], str):
                meta[obj_in_img_k] = meta[obj_in_img_k].strip().lower() == "true"
            pair_metadata.append(meta)

    # Early validation assertion
    assert len(pos_texts) > 0, f"No valid caption pairs loaded! Check CSV path: {cfg.csv_path}"
    print(f"Loaded {len(pos_texts)} paired captions from: {cfg.csv_path}")

    # Initialize model weights and tokenizers
    print(f"Loading model {cfg.model_name} ({cfg.pretrained})...")
    model, preprocess, _ = open_clip.create_model_and_transforms(cfg.model_name, pretrained=cfg.pretrained)
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    model = model.to(device)

    # Stage 1: Single Unified Feature Extraction Pass
    print("\nExecuting Single-Pass Unified Feature Extraction...")
    pos_features = extract_all_features_unified(model, tokenizer, pos_texts, device, cfg.target_token, cfg.batch_size)
    neg_features = extract_all_features_unified(model, tokenizer, neg_texts, device, cfg.target_token, cfg.batch_size)

    # Assert embedding consistency against model.encode_text()
    assert_embedding_consistency(model, tokenizer, pos_texts, pos_features["final_l2norm"], device)

    # Stage 1-A: Multi-Metric Pipeline & Layer Breakdown
    pipeline_data = compute_pipeline_and_layer_breakdown(pos_features, neg_features)
    reporter.render_pipeline_breakdown(pipeline_data)

    # Stage 1-B: Direction Preservation Analysis
    dir_pres_report = compute_direction_preservation(pos_features, neg_features, seed=cfg.seed)
    reporter.render_direction_preservation(dir_pres_report)

    # Stage 1-C: Linear Probe & Sub-dataset Template Analysis
    probe_results = compute_linear_probe_and_subsets(pos_features, neg_features, pair_metadata, fit_intercept=not args.no_bias)
    reporter.render_linear_probe(probe_results)

    # Stage 1-D: Intrinsic Dimensionality & Negation Subspace Analysis
    pca_spec_report = compute_pca_spectrum_compression(pos_features, neg_features)
    reporter.render_pca_spectrum(pca_spec_report)

    # Stage 1-E: Layer-wise PCA Scatter Grid Rendering
    reporter.render_layerwise_pca_grid(pos_features, neg_features, cfg.target_token)

    # Stage 3: Projection SVD & 10%-90% Spectrum Truncation Sweep
    svd_report = compute_projection_svd_ablation(model, pos_features, neg_features, target_token=cfg.target_token)
    reporter.render_svd_ablation(svd_report)

    # Stage 2: Micro-Batched Retrieval Metrics & Symmetric Absence Evaluation
    if cfg.image_root:
        retrieval_cfg = RetrievalConfig(
            image_root=cfg.image_root,
            output_dir=cfg.output_dir,
            device=device,
            batch_size=cfg.batch_size,
            image_batch_size=cfg.image_batch_size
        )
        retrieval_data = compute_image_text_retrieval_metrics(
            model, tokenizer, preprocess, pair_metadata, pos_texts, neg_texts, retrieval_cfg
        )
        reporter.render_retrieval_metrics(retrieval_data)

    print(f"\n✅ Modular Analysis Complete! All output artifacts saved in: {cfg.output_dir}")


if __name__ == "__main__":
    main()
