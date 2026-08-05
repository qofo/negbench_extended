"""
CLIP Text Encoder Layer-wise Linear Probe Analysis Module.

Evaluates linear separability of positive vs negative caption representations
across ALL individual layers (Layer 0 to Layer 12) plus Final Projected Embedding
using LogisticRegression with 5-Fold Stratified Cross-Validation.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any

import open_clip


def extract_all_layers_features(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: str = "cpu",
    target_token: str = "eot",
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    """
    Extract hidden states for ALL layers (Layer 0 to 12) AND fine-grained post-Layer 12 steps:
      - Embedding (Layer 0)
      - Layer 1 .. Layer 12 (Raw Transformer block output)
      - Layer 12 + LN (ln_final applied)
      - Projected (Unnormalized, after text_projection matrix)
      - Final (L2 Normalized CLIP text embedding)

    Returns:
        feature_dict: Dict[step_name, np.ndarray of shape (N, D)]
    """
    model.eval()
    all_tokens = tokenizer(texts).to(device)

    text_tower = getattr(model, 'text', model)
    token_embedding = getattr(text_tower, 'token_embedding', None)
    positional_embedding = getattr(text_tower, 'positional_embedding', None)
    transformer = getattr(text_tower, 'transformer', None)
    ln_final = getattr(text_tower, 'ln_final', None)
    text_projection = getattr(text_tower, 'text_projection', None)
    attn_mask = getattr(text_tower, 'attn_mask', None)

    resblocks = transformer.resblocks if hasattr(transformer, 'resblocks') else []
    num_layers = 1 + len(resblocks)

    layer_lists = [[] for _ in range(num_layers)]
    ln_list = []
    proj_list = []
    final_list = []

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_tokens = all_tokens[start:end]

        with torch.no_grad():
            cast_dtype = transformer.get_cast_dtype() if hasattr(transformer, 'get_cast_dtype') else torch.float32
            eot_indices = batch_tokens.argmax(dim=-1)
            batch_idx = torch.arange(batch_tokens.shape[0], device=device)

            x = token_embedding(batch_tokens).to(cast_dtype)
            seq_len = batch_tokens.shape[1]
            x = x + positional_embedding[:seq_len].to(cast_dtype)

            hidden_states = [x]

            x_perm = x.permute(1, 0, 2)
            for block in resblocks:
                x_perm = block(x_perm, attn_mask=attn_mask)
                hidden_states.append(x_perm.permute(1, 0, 2))

            # Pool token representations per layer
            for l_idx, hs in enumerate(hidden_states):
                hs_cpu = hs.float().cpu()
                if target_token == "eot":
                    feat = hs_cpu[batch_idx.cpu(), eot_indices.cpu()]
                elif target_token == "mean":
                    feat = hs_cpu.mean(dim=1)
                else:
                    feat = hs_cpu[batch_idx.cpu(), eot_indices.cpu()]
                layer_lists[l_idx].append(feat)

            # Post Layer 12 steps:
            x_l12 = hidden_states[-1]
            x_ln = ln_final(x_l12)

            if target_token == "eot":
                eot_ln = x_ln[batch_idx, eot_indices]
            elif target_token == "mean":
                eot_ln = x_ln.mean(dim=1)
            else:
                eot_ln = x_ln[batch_idx, eot_indices]

            ln_list.append(eot_ln.float().cpu())

            # Text projection step
            if text_projection is not None:
                if isinstance(text_projection, nn.Linear):
                    eot_proj = text_projection(eot_ln.to(text_projection.weight.dtype))
                else:
                    eot_proj = eot_ln.to(text_projection.dtype) @ text_projection
            else:
                eot_proj = eot_ln.clone()

            proj_list.append(eot_proj.float().cpu())

            # Unit hyper-sphere normalization step
            eot_final = F.normalize(eot_proj.float(), dim=-1)
            final_list.append(eot_final.cpu())

    feature_dict = {}
    for l_idx, feats in enumerate(layer_lists):
        name = "Embedding" if l_idx == 0 else f"Layer {l_idx}"
        feature_dict[name] = torch.cat(feats, dim=0).numpy()

    feature_dict["Layer 12 + LN"] = torch.cat(ln_list, dim=0).numpy()
    feature_dict["Projected (Unnorm)"] = torch.cat(proj_list, dim=0).numpy()
    feature_dict["Final (L2 Normed)"] = torch.cat(final_list, dim=0).numpy()

    return feature_dict, feature_dict["Final (L2 Normed)"]


def run_layerwise_linear_probe(
    model: nn.Module,
    tokenizer: Any,
    pos_texts: List[str],
    neg_texts: List[str],
    output_dir: str,
    device: str = "cpu",
    target_token: str = "eot",
    batch_size: int = 256,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Run Stratified 5-Fold Cross-Validation LogisticRegression Linear Probe for every layer and pipeline step.
    """
    print("=" * 70)
    print(f"Executing Layer-wise & Pipeline-step Linear Probe Analysis ({n_splits}-Fold CV)")
    print("=" * 70)

    print("Extracting features for positive captions...")
    pos_layer_dict, _ = extract_all_layers_features(model, tokenizer, pos_texts, device, target_token, batch_size)
    
    print("Extracting features for negative captions...")
    neg_layer_dict, _ = extract_all_layers_features(model, tokenizer, neg_texts, device, target_token, batch_size)

    n_pos = len(pos_texts)
    n_neg = len(neg_texts)
    y = np.array([1] * n_pos + [0] * n_neg)

    results = []
    layer_names = list(pos_layer_dict.keys())

    for l_name in layer_names:
        X_pos = pos_layer_dict[l_name]
        X_neg = neg_layer_dict[l_name]
        X = np.vstack([X_pos, X_neg])

        # Standardize or L2 normalize features before linear probing for numerical stability
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_norm, y, cv=cv, scoring="accuracy")

        mean_acc = float(np.mean(scores)) * 100
        std_acc = float(np.std(scores)) * 100
        min_acc = float(np.min(scores)) * 100
        max_acc = float(np.max(scores)) * 100

        results.append({
            "layer": l_name,
            "mean_accuracy_pct": mean_acc,
            "std_accuracy_pct": std_acc,
            "min_accuracy_pct": min_acc,
            "max_accuracy_pct": max_acc,
            "feature_dim": X.shape[1],
        })

        print(f"  [{l_name:22s}] Acc: {mean_acc:6.2f}% (±{std_acc:4.2f}%) [Dim: {X.shape[1]}]")

    df_res = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "layerwise_linear_probe.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    json_path = os.path.join(output_dir, "layerwise_linear_probe.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Plot Layer-wise & Pipeline-step Linear Probe Accuracy Line Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = df_res["layer"].values
    accs = df_res["mean_accuracy_pct"].values
    stds = df_res["std_accuracy_pct"].values

    x_coords = list(range(len(layers)))
    ax.plot(x_coords, accs, "o-", color="#1f77b4", lw=2.5, ms=7, label="5-Fold CV Accuracy (%)")
    ax.fill_between(x_coords, accs - stds, accs + stds, color="#1f77b4", alpha=0.15)

    # Highlight Post-Layer 12 steps boundary
    if "Layer 12" in layers:
        l12_idx = list(layers).index("Layer 12")
        ax.axvline(x=l12_idx + 0.5, color="crimson", ls="--", alpha=0.7, label="Post-Layer 12 Transformations")

    ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=12)
    ax.set_xlabel("Transformer Layer / Pipeline Step", fontsize=12)
    ax.set_title("CLIP Text Encoder Layer-wise & Pipeline-step Linear Probe Accuracy", fontsize=13, fontweight="bold")
    ax.set_xticks(x_coords)
    ax.set_xticklabels(layers, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(min(accs) - 5, min(100, max(accs) + 5))
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "layerwise_linear_probe.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Plot: {plot_path}")

    return df_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Layer-wise Linear Probe Analysis")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--csv_path", type=str, default="COCO_val_full_paired.csv", help="Path to Paired CSV")
    parser.add_argument("--output_dir", type=str, default="logs/evaluation/linear_probe_layerwise")
    parser.add_argument("--target_token", type=str, default="eot", choices=["eot", "mean", "all"])
    parser.add_argument("--max_samples", type=int, default=60000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not os.path.exists(args.csv_path):
        if os.path.exists("COCO_val_mcq_top100_paired.csv"):
            args.csv_path = "COCO_val_mcq_top100_paired.csv"

    print(f"Loading paired dataset from: {args.csv_path}")
    df = pd.read_csv(args.csv_path).head(args.max_samples)
    pos_texts = df["positive_caption"].astype(str).tolist()
    neg_texts = df["negative_caption"].astype(str).tolist()
    print(f"Total positive/negative caption pairs: {len(pos_texts)}")

    print(f"Loading model {args.model} ({args.pretrained})...")
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    run_layerwise_linear_probe(
        model=model,
        tokenizer=tokenizer,
        pos_texts=pos_texts,
        neg_texts=neg_texts,
        output_dir=args.output_dir,
        device=device,
        target_token=args.target_token,
        batch_size=args.batch_size,
        n_splits=args.n_splits,
    )

