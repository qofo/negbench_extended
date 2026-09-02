"""
Compare our identity-distance regularization (train_residual_identity_infonce.py,
RESULTS.md 8-V.12) against BiCLIP's own regularization, using BiCLIP's OWN
official code (BilinearCLIP/models/bilinearclip.py, BilinearCLIP/losses.py) --
not a reimplementation of its recipe, the actual classes.

BiCLIP (arXiv 2603.08942, "BiCLIP: Domain Canonicalization via Structured
Geometric Transformation") trains the exact same bilinear form we use,
S(v,t) = v^T W t, full-rank, identity-initialized, via a symmetric batch
contrastive loss -- for few-shot IMAGE CLASSIFICATION (11 standard
benchmarks), never for retrieval or negation. Its only regularization is
standard AdamW weight_decay directly on W, which pulls W toward the ZERO
matrix, not toward the identity matrix the way our ||dW||_F^2 penalty on
W = I + dW does. This script asks: does BiCLIP's own regularization also
rescue gallery retrieval on our task, or is identity-proximity specifically
what matters (as RESULTS.md 8-V.12/13 argue)?

Fidelity notes:
  - Model/loss/optimizer/grad-clipping ARE BiCLIP's own code
    (models.bilinearclip.BilinearCLIP, losses.contrastive), imported
    directly from BilinearCLIP/, not copied.
  - initialization="identity", upper_triangle=False (full D x D, matching
    every other W in this project's comparison table -- BiCLIP's own
    configs default upper_triangle=True, which halves the free parameters;
    not used here so the only varying factor is the regularization).
  - Its forward() uses the frozen pretrained CLIP's OWN logit_scale
    (not a freshly learned one, unlike our train_residual_identity_infonce.py)
    -- kept as-is since that is literally what their code does.
  - Since freeze_clip=True, the encoder never receives gradients, so image/
    text embeddings are computed once with BiCLIP's own OpenAI-`clip`-package
    encoder and cached -- mathematically identical to running their per-step
    forward(), just faster.
  - Training budget (epochs, batch size, lr, data) is OURS (matches every
    other condition in RESULTS.md's comparison table) so that regularization
    mechanism is the only thing that differs across rows; BiCLIP's own
    configs use a different lr/epoch schedule tuned for few-shot classification.
  - Evaluation of the resulting W (score_delta_s_for_w.py, retrieval R@1/R@5)
    uses this project's standard open_clip-encoded embeddings, not BiCLIP's
    OpenAI-`clip`-package embeddings -- both load the same published
    ViT-B/32 weights, and this keeps the new row directly comparable to
    every other row already in the table (all scored on the same embeddings).

Usage:
    python -m benchmarks.src.evaluation.train_biclip_official_recipe \
        --weight_decays 0 0.001 0.01 0.05 0.1 \
        --output_dir logs/evaluation/01_paper/2026-09-02_biclip_official_recipe
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F

BILINEARCLIP_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "BilinearCLIP")
sys.path.insert(0, os.path.abspath(BILINEARCLIP_ROOT))
import clip  # noqa: E402  (OpenAI's official package, BiCLIP's own dependency)
from models.bilinearclip import BilinearCLIP  # noqa: E402
from losses import contrastive  # noqa: E402

from benchmarks.src.evaluation.train_labclip_official_recipe import build_true_caption_pairs
from benchmarks.src.analysis.paths import resolve_image_path
from benchmarks.src.analysis.config import set_seed
from PIL import Image
from tqdm import tqdm


@torch.no_grad()
def encode_with_biclip_model(pairs, biclip_model, device, image_root=""):
    """Encode (image, true_caption) pairs with BiCLIP's OWN frozen CLIP
    encoder + preprocess + tokenizer -- not open_clip."""
    clip_model = biclip_model.model
    preprocess = biclip_model.preprocess

    image_embs_cache = {}
    unique_images = pairs["image_path"].unique().tolist()
    for p in tqdm(unique_images, desc="[BiCLIP encoder] images"):
        real_path = resolve_image_path(p, image_root)
        img = preprocess(Image.open(real_path).convert("RGB")).unsqueeze(0).to(device)
        emb = clip_model.encode_image(img).float()
        emb = F.normalize(emb, dim=-1)
        image_embs_cache[p] = emb.squeeze(0).cpu()

    text_embs = []
    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="[BiCLIP encoder] text"):
        tok = clip.tokenize([row["true_caption"]], truncate=True).to(device)
        emb = clip_model.encode_text(tok).float()
        emb = F.normalize(emb, dim=-1)
        text_embs.append(emb.squeeze(0).cpu())

    image_embs = torch.stack([image_embs_cache[p] for p in pairs["image_path"]])
    text_embs = torch.stack(text_embs)
    return image_embs, text_embs


def train_one_weight_decay(wd: float, feature_dim: int, image_embs, text_embs,
                            logit_scale_fixed: float, device: str,
                            epochs: int, batch_size: int, lr: float, seed: int):
    set_seed(seed)
    W = torch.nn.Parameter(torch.eye(feature_dim, device=device))
    optimizer = torch.optim.AdamW([{"params": W, "lr": lr, "weight_decay": wd}])

    n = image_embs.shape[0]
    bs = min(batch_size, n)
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_losses = []
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            if idx.numel() < 2:
                continue
            I_f = image_embs[idx]
            T_f = text_embs[idx]
            logits_per_image = (I_f @ W @ T_f.t()) * logit_scale_fixed
            logits_per_text = logits_per_image.t()
            ground_truth = torch.arange(len(idx), device=device)
            # losses.contrastive -- BiCLIP's own loss function, unmodified
            loss = contrastive(logits_per_image, logits_per_text, ground_truth)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W], max_norm=1.0)  # matches their train.py
            optimizer.step()
            epoch_losses.append(loss.item())
        history.append(float(np.mean(epoch_losses)))
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"    [wd={wd}] epoch {epoch:3d}  loss={history[-1]:.4f}  ||W||_F={W.norm().item():.2f}")

    with torch.no_grad():
        W_final = W.detach().cpu()
        drift = torch.norm(W.detach() - torch.eye(feature_dim, device=device)).item()
    print(f"  [wd={wd}] done, ||W-I||_F = {drift:.4f}")
    return W_final, drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-02_biclip_official_recipe")
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decays", type=float, nargs="+", default=[0, 0.001, 0.01, 0.05, 0.1],
                     help="BiCLIP's own regularization knob (AdamW weight_decay on W, pulls toward 0)")
    ap.add_argument("--upper_triangle", action="store_true", default=False,
                     help="BiCLIP's own configs default this True (halves free params); off here for a full-rank apples-to-apples comparison")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = build_true_caption_pairs(args.csv_path)
    print(f"[data] {len(pairs)} unique (image, true_caption) pairs, "
          f"{pairs['image_path'].nunique()} unique images")

    print(f"[model] instantiating BiCLIP's own BilinearCLIP({args.backbone}, "
          f"initialization='identity', upper_triangle={args.upper_triangle})")
    biclip_model = BilinearCLIP(args.backbone, device=device, freeze_clip=True,
                                 upper_triangle=args.upper_triangle, initialization="identity").to(device)
    biclip_model.float()
    feature_dim = biclip_model.model.visual.output_dim
    logit_scale_fixed = biclip_model.model.logit_scale.exp().detach().item()
    print(f"[model] feature_dim={feature_dim}, frozen pretrained logit_scale={logit_scale_fixed:.2f} "
          f"(BiCLIP's forward() uses this fixed value, not a freshly trained one)")

    image_embs, text_embs = encode_with_biclip_model(pairs, biclip_model, device)
    image_embs = image_embs.to(device)
    text_embs = text_embs.to(device)

    for wd in args.weight_decays:
        print(f"\n=== weight_decay={wd} ===")
        W_final, drift = train_one_weight_decay(
            wd, feature_dim, image_embs, text_embs, logit_scale_fixed, device,
            args.epochs, args.batch_size, args.lr, args.seed,
        )
        ckpt_path = os.path.join(args.output_dir, f"biclip_wd{wd}.pt")
        torch.save({
            "model_name": "bilinear",
            "rank": None,
            "weight_decay": wd,
            "dW_norm": drift,
            "state_dict": {"W": W_final, "bias": torch.zeros(1)},
        }, ckpt_path)
        print(f"  [saved] {ckpt_path}")


if __name__ == "__main__":
    main()
