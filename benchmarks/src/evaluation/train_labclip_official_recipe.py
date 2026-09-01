"""
Train a LABCLIP-style text-side linear alignment (official recipe) on OUR BEAF
AB-swap dataset, then save it in the scoring_heads.py BilinearScorer checkpoint
format so eval_zero_shot_transfer.py can score it zero-shot on external NegBench
targets (COCO MCQ/Retrieval, VOC2007 MCQ, CheXpert MCQ).

Official LABCLIP recipe (from CLIP-not-BoW-unimodally/alignment/learning_alignment.py
and coco_alignment.py, read this session):
  - CLIPAlignment = a single full-rank nn.Linear(dim, dim, bias=False), identity-
    initialized, applied to the TEXT embedding only.
  - Trained with a symmetric InfoNCE / contrastive loss over image-caption pairs:
    cross-entropy over the whole batch treated as classes (image_embeddings @
    text_embeddings.T, both directions), i.e. every other item in the batch is an
    in-batch negative. Official default batch_size=1024 over ~118k COCO train images.

We cannot replicate batch_size=1024 faithfully (our AB-swap set has only ~1.7k
unique images), so we use the largest batch our data supports and are explicit
about that limitation in the printed report. Everything else (identity init,
architecture, loss shape) matches the official code.

The "true caption" for each row is whichever of positive_caption/negative_caption
actually matches object_in_image -- i.e. the caption that is TRUE of that image --
so this is a normal image/true-caption contrastive set, not our narrow 2x2 block
task.

Usage:
    python -m benchmarks.src.evaluation.train_labclip_official_recipe \
        --csv_path benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv \
        --output_dir logs/evaluation/01_paper/2026-09-01_labclip_official_recipe
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import open_clip

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from benchmarks.src.analysis.config import set_seed
from benchmarks.src.analysis.paths import resolve_image_path


def build_true_caption_pairs(csv_path: str) -> pd.DataFrame:
    from benchmarks.src.analysis.config import coerce_bool_column
    df = pd.read_csv(csv_path)
    df = coerce_bool_column(df, "object_in_image")
    df["true_caption"] = np.where(df["object_in_image"], df["positive_caption"], df["negative_caption"])
    pairs = df[["image_path", "true_caption"]].drop_duplicates().reset_index(drop=True)
    return pairs


@torch.no_grad()
def encode_all(pairs: pd.DataFrame, model, tokenizer, preprocess, device, image_root: str = ""):
    image_embs, text_embs = [], []
    unique_images = pairs["image_path"].unique().tolist()
    img_cache = {}
    for p in tqdm(unique_images, desc="encoding images"):
        real_path = resolve_image_path(p, image_root)
        img = preprocess(Image.open(real_path).convert("RGB")).unsqueeze(0).to(device)
        emb = model.encode_image(img)
        emb = F.normalize(emb, dim=-1)
        img_cache[p] = emb.squeeze(0).cpu()
    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="encoding text"):
        tok = tokenizer([row["true_caption"]]).to(device)
        emb = model.encode_text(tok)
        emb = F.normalize(emb, dim=-1)
        text_embs.append(emb.squeeze(0).cpu())
    image_embs = torch.stack([img_cache[p] for p in pairs["image_path"]])
    text_embs = torch.stack(text_embs)
    return image_embs, text_embs


def symmetric_infonce(img_emb: torch.Tensor, txt_emb: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Official-recipe loss: batch-internal cross entropy, both directions."""
    logits = logit_scale * img_emb @ txt_emb.t()
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-01_labclip_official_recipe")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch_size", type=int, default=256,
                     help="Official recipe uses 1024; capped down since our unique-image pool is ~1.7k.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = build_true_caption_pairs(args.csv_path)
    print(f"[data] {len(pairs)} unique (image, true_caption) pairs, "
          f"{pairs['image_path'].nunique()} unique images")

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    image_embs, text_embs = encode_all(pairs, model, tokenizer, preprocess, device, image_root="")
    feature_dim = image_embs.shape[1]
    image_embs = image_embs.to(device)
    text_embs = text_embs.to(device)
    print(f"[data] feature_dim={feature_dim}")

    # Official CLIPAlignment: identity-init full-rank Linear on TEXT side only.
    M = nn.Linear(feature_dim, feature_dim, bias=False).to(device)
    with torch.no_grad():
        M.weight.copy_(torch.eye(feature_dim))
    logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07)).float().to(device))
    opt = torch.optim.Adam(list(M.parameters()) + [logit_scale], lr=args.lr)

    n = image_embs.shape[0]
    bs = min(args.batch_size, n)
    print(f"[train] n={n} unique pairs, effective batch_size={bs} (official recipe: 1024)")

    history = []
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        epoch_losses = []
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            if idx.numel() < 2:
                continue
            v = image_embs[idx]
            t = F.normalize(M(text_embs[idx]), dim=-1)
            loss = symmetric_infonce(v, t, logit_scale.exp())
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        mean_loss = float(np.mean(epoch_losses))
        history.append(mean_loss)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  loss={mean_loss:.4f}")

    with torch.no_grad():
        W_final = M.weight.detach().cpu()
        drift = torch.norm(W_final - torch.eye(feature_dim)).item()
    print(f"[done] ||M - I||_F = {drift:.4f}")

    ckpt_path = os.path.join(args.output_dir, "labclip_official_recipe_bilinear.pt")
    torch.save({
        "model_name": "bilinear",
        "rank": None,
        "state_dict": {"W": W_final, "bias": torch.zeros(1)},
    }, ckpt_path)
    print(f"[saved] {ckpt_path}")

    with open(os.path.join(args.output_dir, "train_report.json"), "w") as f:
        json.dump({
            "csv_path": args.csv_path,
            "n_pairs": n,
            "n_unique_images": int(pairs["image_path"].nunique()),
            "batch_size_used": bs,
            "epochs": args.epochs,
            "final_loss": history[-1],
            "loss_history": history,
            "W_minus_I_frobenius": drift,
        }, f, indent=2)


if __name__ == "__main__":
    main()
