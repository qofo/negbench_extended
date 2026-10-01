"""
Reproduce LABCLIP-neg -- the in-distribution NegBench setting the authors of [4]
reported in their ICLR 2026 rebuttal -- and then score that same W on the two
things the rebuttal never measured: open-gallery retrieval and our controlled 2x2
success condition.

Why this experiment exists
--------------------------
In the public OpenReview discussion for "CLIP Behaves like a Bag-of-Words Model
Cross-modally but not Uni-modally" (ICLR 2026, submission 4454), reviewers dUnx and
kJCN both asked whether LABCLIP generalises past attribute-object binding. The
authors answered with a negation table:

    Model         COCO MCQ   VOC MCQ
    CLIP              39.3      38.7
    LABCLIP           34.5      38.7
    LABCLIP-neg       70.1      81.4

and described LABCLIP-neg as "we use half of the NegBench subsets for training and
the remaining half for testing. With this explicit negation supervision, accuracy
increases to 70.1 on COCO MCQ and 81.4 on VOC MCQ."

70.1 is far above anything in our own table, so it has to be addressed rather than
ignored. But it is measured on one axis only. Our claim is not that no W can score
well on NegBench MCQ -- it is that the score function cannot carry polarity-object
binding, so gains bought with in-distribution MCQ supervision should not survive
either (a) an open gallery, where the same W has to rank 5,000 images, or (b) the
controlled 2x2 minimal-pair condition Delta(S) > 0.

`train_labclip_official_recipe.py` already reproduces the official recipe on OUR
BEAF AB-swap data (a transfer setting: train on BEAF, score NegBench zero-shot).
This script keeps that architecture and loss byte-for-byte and changes only the
training population, so the difference between the two runs isolates exactly one
variable: whether the supervision comes from the evaluation distribution.

Protocol
--------
- Split the NegBench MCQ csv by ``image_path`` (not by row): the csv has 5,914 rows
  over 3,926 unique COCO images, so a row-level split would put the same image in
  both halves. The held-out half is written out as its own csv so
  ``eval_zero_shot_transfer.py`` can score it with the shared tie-aware MCQ harness.
- The positive for each row is ``caption_{correct_answer}``. Two losses are offered:
  ``infonce`` is the literal official recipe (in-batch negatives only), and
  ``infonce_hn`` additionally puts each row's three MCQ distractors into the
  image->text denominator. The official LABCLIP COCO recipe trains with NegCLIP-style
  hard negatives, and for NegBench the distractors are the corresponding hard
  negatives, so ``infonce_hn`` is the closer analogue of "explicit negation
  supervision" and is the default.
- Everything else matches the official recipe: a single full-rank
  ``nn.Linear(dim, dim, bias=False)``, identity-initialised, applied to the TEXT
  embedding only, symmetric contrastive loss, learned logit scale.
- The argument only works if LABCLIP-neg is given its best shot, so the learning
  rate and the stopping epoch are chosen on a further 10% of images held out of the
  *training* half -- never on the evaluation half. Our training half is 2,934 rows
  against the official recipe's 118k COCO images, so a fixed 60-epoch budget would
  be 180 optimiser steps and would undertrain the very baseline this experiment
  exists to give a fair hearing.

The checkpoint is saved in the ``scoring_heads.BilinearScorer`` format, i.e.
S(v, t) = v^T W t, so it drops straight into ``eval_zero_shot_transfer.py`` and
``score_delta_s_for_w.py`` without conversion.

Usage (from the repo root):
    python -m benchmarks.src.evaluation.train_labclip_neg_indist \
        --csv_path benchmarks/data/images/COCO_val_mcq_llama3.1_rephrased.csv \
        --output_dir logs/evaluation/01_paper/2026-09-08_labclip_neg_indist_coco
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
from benchmarks.src.evaluation.train_labclip_official_recipe import symmetric_infonce

CAPTION_COLS = ["caption_0", "caption_1", "caption_2", "caption_3"]


def split_by_image(df: pd.DataFrame, seed: int, train_frac: float = 0.5):
    """Half/half split on unique images.

    1,374 of the 3,926 COCO images carry more than one question, so a row-level
    split puts the same scene on both sides. ``--split_by row`` is offered because
    the rebuttal does not say which was used, and the gap between the two is itself
    the measurement: it separates "the transform learned negation" from "the
    transform learned these scenes".
    """
    images = np.sort(df["image_path"].unique())
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(images))
    n_train = int(round(len(images) * train_frac))
    train_images = set(images[perm[:n_train]])
    is_train = df["image_path"].isin(train_images)
    return df[is_train].reset_index(drop=True), df[~is_train].reset_index(drop=True)


def split_by_row(df: pd.DataFrame, seed: int, train_frac: float = 0.5):
    """Half/half split on rows, letting a scene appear on both sides."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(df))
    n_train = int(round(len(df) * train_frac))
    train_rows = np.zeros(len(df), dtype=bool)
    train_rows[perm[:n_train]] = True
    return df[train_rows].reset_index(drop=True), df[~train_rows].reset_index(drop=True)


def make_split(df, seed, train_frac, split_by):
    return (split_by_image if split_by == "image" else split_by_row)(df, seed, train_frac)


@torch.no_grad()
def encode_split(df: pd.DataFrame, model, tokenizer, preprocess, device, image_root: str):
    """Encode every image once and all four captions of every row."""
    unique_images = df["image_path"].unique().tolist()
    img_cache = {}
    for p in tqdm(unique_images, desc="encoding images"):
        real_path = resolve_image_path(p, image_root)
        img = preprocess(Image.open(real_path).convert("RGB")).unsqueeze(0).to(device)
        emb = F.normalize(model.encode_image(img), dim=-1)
        img_cache[p] = emb.squeeze(0).cpu()

    caps = []
    for col in CAPTION_COLS:
        caps.append(df[col].astype(str).tolist())
    flat = [c for col_caps in caps for c in col_caps]
    text_embs = []
    bs = 256
    for start in tqdm(range(0, len(flat), bs), desc="encoding text"):
        tok = tokenizer(flat[start:start + bs]).to(device)
        emb = F.normalize(model.encode_text(tok), dim=-1)
        text_embs.append(emb.cpu())
    text_embs = torch.cat(text_embs, dim=0).view(len(CAPTION_COLS), len(df), -1).permute(1, 0, 2)

    image_embs = torch.stack([img_cache[p] for p in df["image_path"]])
    return image_embs, text_embs


def infonce_with_hard_negatives(v, t_pos, t_neg, logit_scale):
    """Symmetric contrastive loss with each row's own distractors in the i2t denominator.

    This is the NegCLIP-shaped objective the official LABCLIP COCO recipe uses; the
    t2i direction is left untouched because the hard negatives are text-side only
    and have no image of their own to be retrieved by.
    """
    logits_i2t = logit_scale * v @ t_pos.t()                      # (B, B)
    logits_hn = logit_scale * torch.einsum("bd,bkd->bk", v, t_neg)  # (B, K)
    labels = torch.arange(v.shape[0], device=v.device)
    loss_i2t = F.cross_entropy(torch.cat([logits_i2t, logits_hn], dim=1), labels)
    loss_t2i = F.cross_entropy(logits_i2t.t(), labels)
    return (loss_i2t + loss_t2i) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", nargs="+",
                    default=["benchmarks/data/images/COCO_val_mcq_llama3.1_rephrased.csv"],
                    help="One or more NegBench MCQ csvs. The rebuttal says 'half of the NegBench "
                         "subsets' (plural), so passing COCO and VOC together is the literal reading; "
                         "each csv is split independently and the held-out halves are written separately.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image_root", default="")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--loss", choices=["infonce", "infonce_hn"], default="infonce_hn")
    ap.add_argument("--batch_size", type=int, default=1024,
                    help="Official recipe uses 1024 and the training half supports it.")
    ap.add_argument("--epochs", type=int, default=3000,
                    help="Epoch budget per lr. Features are precomputed, so an epoch is a few matmuls.")
    ap.add_argument("--lr_grid", type=float, nargs="+", default=[1e-4, 5e-4, 1e-3, 5e-3])
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--split_by", choices=["image", "row"], default="image",
                    help="'image' keeps a scene on one side only; 'row' lets it appear on both.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_parts, held_out_csvs = [], []
    for csv_path in args.csv_path:
        sub = pd.read_csv(csv_path)
        sub_train, sub_test = make_split(sub, args.seed, args.train_frac, args.split_by)
        print(f"[data] {os.path.basename(csv_path)}: {len(sub)} rows / {sub['image_path'].nunique()} images -> "
              f"train {len(sub_train)} / test {len(sub_test)}")
        base = os.path.splitext(os.path.basename(csv_path))[0]
        test_csv = os.path.join(args.output_dir, f"{base}_heldout_half_{args.split_by}.csv")
        sub_test.to_csv(test_csv, index=False)
        sub_train.to_csv(os.path.join(args.output_dir, f"{base}_train_half_{args.split_by}.csv"), index=False)
        held_out_csvs.append(test_csv)
        train_parts.append(sub_train)
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat([pd.read_csv(c) for c in held_out_csvs], ignore_index=True)
    print(f"[data] pooled training half: {len(train_df)} rows / {train_df['image_path'].nunique()} images")
    print(f"[data] held-out halves written to {held_out_csvs}")

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    image_embs, all_text = encode_split(train_df, model, tokenizer, preprocess, device, args.image_root)
    feature_dim = image_embs.shape[1]
    correct = torch.as_tensor(train_df["correct_answer"].to_numpy(), dtype=torch.long)
    pos_text = all_text[torch.arange(len(train_df)), correct]
    neg_mask = torch.ones(len(train_df), len(CAPTION_COLS), dtype=torch.bool)
    neg_mask[torch.arange(len(train_df)), correct] = False
    neg_text = all_text[neg_mask].view(len(train_df), len(CAPTION_COLS) - 1, feature_dim)

    image_embs, pos_text, neg_text = image_embs.to(device), pos_text.to(device), neg_text.to(device)
    print(f"[data] feature_dim={feature_dim}, {neg_text.shape[1]} hard negatives per row, loss={args.loss}")

    # Inner split: pick lr and stopping epoch on images held out of the training half.
    inner_train_df, val_df = make_split(train_df, args.seed + 1, 1.0 - args.val_frac, args.split_by)
    if args.split_by == "image":
        is_inner = train_df["image_path"].isin(set(inner_train_df["image_path"])).to_numpy()
    else:
        key = train_df["image_path"] + "\u0000" + train_df["caption_0"].astype(str)
        inner_key = set(inner_train_df["image_path"] + "\u0000" + inner_train_df["caption_0"].astype(str))
        is_inner = key.isin(inner_key).to_numpy()
    inner_idx = torch.as_tensor(np.where(is_inner)[0])
    val_idx = torch.as_tensor(np.where(~is_inner)[0])
    print(f"[tune] inner train {len(inner_idx)} rows / val {len(val_idx)} rows "
          f"({val_df['image_path'].nunique()} val images)")

    all_text_dev = all_text.to(device)

    def mcq_acc(W: torch.Tensor, idx: torch.Tensor) -> float:
        with torch.no_grad():
            t = F.normalize(all_text_dev[idx].reshape(-1, feature_dim) @ W.t(), dim=-1)
            t = t.reshape(idx.numel(), len(CAPTION_COLS), feature_dim)
            scores = torch.einsum("bd,bkd->bk", image_embs[idx], t)
            return (scores.argmax(dim=1).cpu() == correct[idx]).float().mean().item() * 100

    best = {"val_acc": -1.0, "lr": None, "epoch": None, "W": None, "logit_scale": None}
    tuning_log = []
    n_inner = inner_idx.numel()
    bs = min(args.batch_size, n_inner)
    print(f"[train] inner n={n_inner} rows, effective batch_size={bs}, "
          f"lr grid={args.lr_grid}, budget={args.epochs} epochs")

    for lr in args.lr_grid:
        set_seed(args.seed)
        M = nn.Linear(feature_dim, feature_dim, bias=False).to(device)
        with torch.no_grad():
            M.weight.copy_(torch.eye(feature_dim))
        logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07)).float().to(device))
        opt = torch.optim.Adam(list(M.parameters()) + [logit_scale], lr=lr)

        history = []
        for epoch in range(args.epochs):
            perm = inner_idx[torch.randperm(n_inner)]
            epoch_losses = []
            for start in range(0, n_inner, bs):
                idx = perm[start:start + bs].to(device)
                if idx.numel() < 2:
                    continue
                v = image_embs[idx]
                t = F.normalize(M(pos_text[idx]), dim=-1)
                if args.loss == "infonce_hn":
                    t_hn = F.normalize(M(neg_text[idx]), dim=-1)
                    loss = infonce_with_hard_negatives(v, t, t_hn, logit_scale.exp())
                else:
                    loss = symmetric_infonce(v, t, logit_scale.exp())
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_losses.append(loss.item())
            history.append(float(np.mean(epoch_losses)))

            if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
                W_now = M.weight.detach()
                va = mcq_acc(W_now, val_idx)
                tuning_log.append({"lr": lr, "epoch": epoch + 1, "loss": history[-1], "val_mcq_acc": va})
                if va > best["val_acc"]:
                    best = {"val_acc": va, "lr": lr, "epoch": epoch + 1,
                            "W": W_now.clone().cpu(),
                            "logit_scale": float(logit_scale.detach().cpu())}
                if (epoch + 1) % (args.eval_every * 10) == 0 or epoch == args.epochs - 1:
                    print(f"  lr={lr:<8g} epoch {epoch+1:5d}  loss={history[-1]:.4f}  val_mcq={va:.2f}%")

    W_final = best["W"]
    drift = torch.norm(W_final - torch.eye(feature_dim)).item()
    train_acc = mcq_acc(W_final.to(device), inner_idx)
    print(f"[select] best lr={best['lr']} epoch={best['epoch']} val_mcq={best['val_acc']:.2f}%")
    print(f"[done] ||M - I||_F = {drift:.4f}   in-sample MCQ acc on inner train = {train_acc:.2f}%")

    cosine_val = mcq_acc(torch.eye(feature_dim, device=device), val_idx)
    print(f"[ref] cosine (W=I) MCQ on the same val rows = {cosine_val:.2f}%")

    tag = f"{args.loss}_{args.split_by}" + ("_pooled" if len(args.csv_path) > 1 else "")
    ckpt_path = os.path.join(args.output_dir, f"labclip_neg_indist_{tag}.pt")
    torch.save({
        "model_name": "bilinear",
        "rank": None,
        "state_dict": {"W": W_final, "bias": torch.zeros(1)},
    }, ckpt_path)
    print(f"[saved] {ckpt_path}")

    with open(os.path.join(args.output_dir, f"train_report_{tag}.json"), "w") as f:
        json.dump({
            "csv_path": args.csv_path,
            "held_out_csvs": held_out_csvs,
            "loss": args.loss,
            "split_by": args.split_by,
            "seed": args.seed,
            "train_frac": args.train_frac,
            "n_train_rows": int(len(train_df)),
            "n_train_images": int(train_df["image_path"].nunique()),
            "n_test_rows": int(len(test_df)),
            "n_test_images": int(test_df["image_path"].nunique()),
            "batch_size_used": bs,
            "epoch_budget": args.epochs,
            "lr_grid": args.lr_grid,
            "selected_lr": best["lr"],
            "selected_epoch": best["epoch"],
            "val_mcq_acc_pct": best["val_acc"],
            "cosine_val_mcq_acc_pct": cosine_val,
            "W_minus_I_frobenius": drift,
            "in_sample_mcq_acc_inner_train_pct": train_acc,
            "tuning_log": tuning_log,
        }, f, indent=2)


if __name__ == "__main__":
    main()
