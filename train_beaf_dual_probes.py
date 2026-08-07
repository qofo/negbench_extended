"""
Train Dual Classifiers (+1 / -1) on BEAF Counterfactual Dataset.

Extracts CLIP image and text features from BEAF dataset, trains LogisticRegression classifiers:
- Vision Classifier f_V: object_in_image == True (+1) vs False (-1)
- Text Classifier f_T: positive_caption (+1) vs negative_caption (-1)

Saves weights (w_v, b_v, w_t, b_t) for DualClassifierProductScorer in NegBench evaluation.
"""

import os
import json
import argparse
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm


def load_beaf_data(csv_path: str, image_root: str = "") -> pd.DataFrame:
    """Load BEAF CSV and format object_in_image to boolean."""
    df = pd.read_csv(csv_path)

    def _to_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return bool(v)

    df["object_in_image"] = df["object_in_image"].apply(_to_bool)
    if image_root:
        df["abs_image_path"] = df["image_path"].apply(lambda p: os.path.join(image_root, p))
    else:
        df["abs_image_path"] = df["image_path"]
    return df


def extract_beaf_features(
    df: pd.DataFrame,
    model: torch.nn.Module,
    preprocess: callable,
    tokenizer: callable,
    device: str = "cuda",
    batch_size: int = 64,
    cache_path: str = "logs/evaluation/cached_embeddings/beaf_probe_features.npz"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract L2-normalized image and text embeddings from BEAF with disk caching."""
    if cache_path and os.path.exists(cache_path):
        print(f"\n⚡ Loading pre-cached BEAF features from disk: {cache_path}")
        try:
            data = np.load(cache_path)
            return data["X_text"], data["y_text"], data["X_vision"], data["y_vision"]
        except Exception as e:
            print(f"⚠️ Failed to load cache {cache_path}: {e}. Re-extracting...")

    model.eval()

    # 1. Text Features (Positive vs Negative)
    pos_captions = df["positive_caption"].tolist()
    neg_captions = df["negative_caption"].tolist()

    all_texts = pos_captions + neg_captions
    text_embeds_list = []

    with torch.no_grad():
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i : i + batch_size]
            tokens = tokenizer(batch_texts).to(device)
            t_emb = model.encode_text(tokens, normalize=True)
            text_embeds_list.append(t_emb.cpu().numpy())

    X_text = np.vstack(text_embeds_list)
    y_text = np.array([1] * len(pos_captions) + [-1] * len(neg_captions))

    # 2. Image Features (Original object_in_image=True vs Counterfactual object_in_image=False)
    img_paths = df["abs_image_path"].tolist()
    img_labels = np.array([1 if is_in else -1 for is_in in df["object_in_image"]])

    valid_img_embeds = []
    valid_img_labels = []

    print("Extracting BEAF image features...")
    with torch.no_grad():
        for i in tqdm(range(0, len(img_paths), batch_size)):
            batch_paths = img_paths[i : i + batch_size]
            batch_lbls = img_labels[i : i + batch_size]

            images_tensors = []
            lbls_batch = []
            for path, lbl in zip(batch_paths, batch_lbls):
                if os.path.exists(path):
                    try:
                        img = Image.open(path).convert("RGB")
                        images_tensors.append(preprocess(img))
                        lbls_batch.append(lbl)
                    except Exception as e:
                        pass

            if images_tensors:
                imgs_batch_t = torch.stack(images_tensors).to(device)
                v_emb = model.encode_image(imgs_batch_t, normalize=True)
                valid_img_embeds.append(v_emb.cpu().numpy())
                valid_img_labels.extend(lbls_batch)

    X_vision = np.vstack(valid_img_embeds)
    y_vision = np.array(valid_img_labels)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, X_text=X_text, y_text=y_text, X_vision=X_vision, y_vision=y_vision)
        print(f"✅ Saved BEAF features cache to: {cache_path}")

    return X_text, y_text, X_vision, y_vision


    return X_text, y_text, X_vision, y_vision


def train_dual_probes(
    X_text: np.ndarray,
    y_text: np.ndarray,
    X_vision: np.ndarray,
    y_vision: np.ndarray,
    C: float = 1.0,
    n_splits: int = 5
) -> dict:
    """Train LogisticRegression classifiers for text and vision."""
    print("\nExecuting 5-Fold CV for Text Classifier (f_T: Positive=+1, Negative=-1)...")
    clf_text = LogisticRegression(C=C, max_iter=1000, random_state=42)
    cv_text = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores_text = cross_val_score(clf_text, X_text, y_text, cv=cv_text, scoring="accuracy")
    mean_acc_text = float(np.mean(scores_text) * 100)
    std_acc_text = float(np.std(scores_text) * 100)
    print(f"✅ Text Classifier Accuracy: {mean_acc_text:.2f}% ± {std_acc_text:.2f}%")

    # Fit final text classifier on full dataset
    clf_text.fit(X_text, y_text)
    w_t = clf_text.coef_[0].astype(np.float32)
    b_t = float(clf_text.intercept_[0])

    print("\nExecuting 5-Fold CV for Vision Classifier (f_V: Present=+1, Absent=-1)...")
    clf_vision = LogisticRegression(C=C, max_iter=1000, random_state=42)
    cv_vision = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores_vision = cross_val_score(clf_vision, X_vision, y_vision, cv=cv_vision, scoring="accuracy")
    mean_acc_vision = float(np.mean(scores_vision) * 100)
    std_acc_vision = float(np.std(scores_vision) * 100)
    print(f"✅ Vision Classifier Accuracy: {mean_acc_vision:.2f}% ± {std_acc_vision:.2f}%")

    # Fit final vision classifier on full dataset
    clf_vision.fit(X_vision, y_vision)
    w_v = clf_vision.coef_[0].astype(np.float32)
    b_v = float(clf_vision.intercept_[0])

    return {
        "w_v": w_v,
        "b_v": b_v,
        "w_t": w_t,
        "b_t": b_t,
        "text_acc_mean": mean_acc_text,
        "text_acc_std": std_acc_text,
        "vision_acc_mean": mean_acc_vision,
        "vision_acc_std": std_acc_vision,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Dual Classifiers (+1/-1) on BEAF Data.")
    default_csv = "benchmarks/data/images/beaf_counterfactual_6col.csv"
    if not os.path.exists(default_csv) and os.path.exists("csvOLD/beaf_counterfactual_6col.csv"):
        default_csv = "csvOLD/beaf_counterfactual_6col.csv"

    parser.add_argument("--csv_path", type=str, default=default_csv, help="Path to BEAF CSV")

    parser.add_argument("--image_root", type=str, default="", help="Root directory for relative image paths")
    parser.add_argument("--model_name", type=str, default="ViT-B-32", help="OpenCLIP vision encoder architecture")
    parser.add_argument("--pretrained", type=str, default="openai", help="Pretrained weights")
    parser.add_argument("--output_dir", type=str, default="logs/evaluation/beaf_dual_probe", help="Output directory for weights")
    parser.add_argument("--C", type=float, default=1.0, help="Logistic Regression C parameter")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading OpenCLIP model ({args.model_name} / {args.pretrained}) on {args.device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained, device=args.device)
    tokenizer = open_clip.get_tokenizer(args.model_name)

    print(f"Loading BEAF CSV from {args.csv_path}...")
    df = load_beaf_data(args.csv_path, args.image_root)
    print(f"Total BEAF samples: {len(df)}")

    X_text, y_text, X_vision, y_vision = extract_beaf_features(
        df, model, preprocess, tokenizer, device=args.device, batch_size=args.batch_size
    )

    print(f"Features Extracted: Text {X_text.shape}, Vision {X_vision.shape}")

    results = train_dual_probes(X_text, y_text, X_vision, y_vision, C=args.C)

    out_weights_path = os.path.join(args.output_dir, "beaf_dual_probe_weights.npz")
    np.savez(
        out_weights_path,
        w_v=results["w_v"],
        b_v=np.array(results["b_v"], dtype=np.float32),
        w_t=results["w_t"],
        b_t=np.array(results["b_t"], dtype=np.float32),
    )
    print(f"\n🎉 Saved dual probe weights to: {out_weights_path}")

    info_path = os.path.join(args.output_dir, "beaf_dual_probe_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "pretrained": args.pretrained,
                "C": args.C,
                "text_cv_accuracy_pct": f"{results['text_acc_mean']:.2f} ± {results['text_acc_std']:.2f}%",
                "vision_cv_accuracy_pct": f"{results['vision_acc_mean']:.2f} ± {results['vision_acc_std']:.2f}%",
                "weights_path": out_weights_path,
            },
            f,
            indent=2,
        )
    print(f"Saved info JSON to: {info_path}")


if __name__ == "__main__":
    main()
