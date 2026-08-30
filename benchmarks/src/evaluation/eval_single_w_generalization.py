"""
Does a *single* scoring matrix W generalise to concepts it never saw?

Why this experiment exists
--------------------------
The main-effect ablation reaches 67.43% on the 2x2 task by projecting alpha and
beta out of each pair. That projection is different for every pair, i.e. it is a
different W per pair, while a deployed retriever uses one W for everything. So
67.43% is the reachable point of removing main effects near W = I, not something
any retriever can implement -- as `eval_e2_hadamard_decomposition` already notes
in its own summary.

"Does one W reproduce 67.43%?" is the wrong question: it cannot, by construction.
The right one is **how far a single shared W actually gets on concepts held out of
its training**, and that is what this measures. The answer is not assumed to be
encouraging. Fitting a 512x512 W *per concept* scored 88.23% in-sample and 8.69%
out-of-fold (median 26 pairs per concept), and Full Bilinear fell below chance on
28 of 33 concepts. A single W sees ~52x more pairs per parameter, but 1,357 pairs
against 262,144 parameters is still lopsided.

So the experiment is a ladder of degrees of freedom rather than one model:
identity (0) -> diagonal (D) -> low rank r (2rD) -> full (D^2). The result is not
"the full matrix failed" but **which rung, if any, survives on unseen concepts**.

Protocol
--------
- Quads are built exactly as `eval_e2_hadamard_decomposition` builds them -- same
  rows, same positional pairing, same cache keys -- so this shares its embeddings
  and its 0.88% / 67.43% reference points refer to the same population.
- ``GroupKFold`` on ``object_name``: a held-out fold's concepts appear nowhere in
  training. Splitting *within* a concept would answer a different question, since
  each concept's pairs share captions and scenes.
- Training loss is the four-term margin ranking loss the per-concept matchers
  already use, unchanged, so the only difference from that experiment is which
  pairs the fit sees.
- Success on a quad is ``min(S++, S--) > max(S+-, S-+)``; chance is 100/6 = 16.67%.

Note on bias: a scalar bias shared by all four scores cancels in
``min(...) - max(...)``, so it cannot change 2x2 accuracy. The matchers here carry
none, and ``--no_bias`` would be a no-op; ``test_shared_bias_cancels_in_2x2``
locks that reasoning in rather than leaving it as a claim in a comment.

Usage:
    python -m benchmarks.src.evaluation.eval_single_w_generalization \
        --model ViT-B-32 --pretrained openai --seed 42 --use_cache \
        --restrict_objects logs/evaluation/00_concept_sets/paper33.txt \
        --output_dir logs/evaluation/01_paper/2026-08-30_i2_single_w

Outputs:
    - single_w_per_family.csv       one row per (family, fold)
    - single_w_per_concept.csv      out-of-fold accuracy per (family, concept)
    - single_w_summary.json         headline numbers + provenance
    - fig_single_w_ladder.png       accuracy vs degrees of freedom
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

try:
    from benchmarks.src.analysis.cli import (
        add_model_args, add_run_args, add_data_args, add_cache_args,
        add_restriction_args, add_concept_args,
    )
    from benchmarks.src.analysis.config import set_seed, coerce_bool_column
    from benchmarks.src.analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction, DEFAULT_CACHE_DIR,
    )
    from benchmarks.src.analysis.model_loader import load_clip_for_eval, get_embed_dim
    from benchmarks.src.analysis.paths import resolve_image_path as resolve_path
    from benchmarks.src.analysis.plotting import plt
    from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified,
    )
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.cli import (
        add_model_args, add_run_args, add_data_args, add_cache_args,
        add_restriction_args, add_concept_args,
    )
    from analysis.config import set_seed, coerce_bool_column
    from analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction, DEFAULT_CACHE_DIR,
    )
    from analysis.model_loader import load_clip_for_eval, get_embed_dim
    from analysis.paths import resolve_image_path as resolve_path
    from analysis.plotting import plt
    from evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified,
    )

CHANCE_PCT = 100.0 / 6.0


# ============================================================
# 1. The ladder of W families
# ============================================================
class IdentityMatcher(nn.Module):
    """W = I, i.e. plain cosine. Zero parameters; the baseline this must beat."""

    trainable = False

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sum(v * t, dim=-1)


class DiagonalMatcher(nn.Module):
    """W = diag(w): per-dimension reweighting, no cross-dimensional terms."""

    trainable = True

    def __init__(self, embed_dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(embed_dim))

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sum(v * t * self.w, dim=-1)


class LowRankMatcher(nn.Module):
    """W = U V^T with U, V in R^(D x r). Initialised near identity via a scaled draw."""

    trainable = True

    def __init__(self, embed_dim: int, rank: int):
        super().__init__()
        self.rank = rank
        self.proj_v = nn.Linear(embed_dim, rank, bias=False)
        self.proj_t = nn.Linear(embed_dim, rank, bias=False)
        nn.init.normal_(self.proj_v.weight, std=0.02)
        nn.init.normal_(self.proj_t.weight, std=0.02)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.proj_v(v) * self.proj_t(t), dim=-1)


class FullMatcher(nn.Module):
    """Unconstrained W, initialised at the identity so training starts from cosine."""

    trainable = True

    def __init__(self, embed_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.eye(embed_dim))

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.matmul(v, self.W) * t, dim=-1)


class RandomMatcher(nn.Module):
    """A fixed random W, never trained: the control for "any matrix would do"."""

    trainable = False

    def __init__(self, embed_dim: int, seed: int = 42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(embed_dim, embed_dim, generator=g) / np.sqrt(embed_dim)
        self.register_buffer("W", W)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.matmul(v, self.W) * t, dim=-1)


def build_family(name: str, embed_dim: int, seed: int) -> Tuple[nn.Module, int]:
    """Return (module, trainable parameter count) for one rung of the ladder."""
    if name == "identity":
        return IdentityMatcher(), 0
    if name == "random":
        return RandomMatcher(embed_dim, seed=seed), 0
    if name == "diagonal":
        return DiagonalMatcher(embed_dim), embed_dim
    if name.startswith("lowrank_"):
        r = int(name.split("_")[1])
        return LowRankMatcher(embed_dim, r), 2 * embed_dim * r
    if name == "full":
        return FullMatcher(embed_dim), embed_dim * embed_dim
    raise ValueError(f"unknown W family: {name}")


DEFAULT_FAMILIES = [
    "identity", "random", "diagonal",
    "lowrank_1", "lowrank_2", "lowrank_4", "lowrank_8", "lowrank_16", "lowrank_32",
    "full",
]


# ============================================================
# 2. Training and scoring
# ============================================================
def train_matcher(
    model: nn.Module,
    quads: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    epochs: int = 150,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    margin: float = 0.1,
) -> nn.Module:
    """
    Fit W with the four-term margin ranking loss the per-concept matchers use.

    Keeping the objective, optimiser and schedule identical to
    ``eval_per_object_alignment_intervention.train_bilinear_matcher`` is the point:
    the only thing that differs between that experiment and this one is which pairs
    the fit sees, so any gap is attributable to the fitting population.
    """
    if not getattr(model, "trainable", True):
        return model

    v_pos, v_neg, t_pos, t_neg = quads
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MarginRankingLoss(margin=margin)
    target = torch.ones(v_pos.shape[0], device=v_pos.device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        s_pp = model(v_pos, t_pos)
        s_mm = model(v_neg, t_neg)
        s_pm = model(v_pos, t_neg)
        s_mp = model(v_neg, t_pos)
        loss = (criterion(s_pp, s_pm, target) + criterion(s_mm, s_mp, target)
                + criterion(s_pp, s_mp, target) + criterion(s_mm, s_pm, target)) / 4.0
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def joint_correct(
    model: nn.Module,
    quads: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> np.ndarray:
    """Boolean per quad: min(S++, S--) > max(S+-, S-+)."""
    v_pos, v_neg, t_pos, t_neg = quads
    with torch.no_grad():
        s_pp = model(v_pos, t_pos)
        s_mm = model(v_neg, t_neg)
        s_pm = model(v_pos, t_neg)
        s_mp = model(v_neg, t_pos)
        ok = torch.minimum(s_pp, s_mm) > torch.maximum(s_pm, s_mp)
    return ok.cpu().numpy()


def concept_bootstrap_ci(
    per_concept_acc: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Percentile CI resampling *concepts*, which is the unit of generalisation here.

    Resampling pairs would treat two quads of the same concept as independent
    evidence; they share a scene and a caption template, so they are not.
    """
    if len(per_concept_acc) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_concept_acc), size=(n_boot, len(per_concept_acc)))
    means = per_concept_acc[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ============================================================
# 3. Data assembly -- identical to the E2 quad construction
# ============================================================
def load_quads(args, model, preprocess, tokenizer, device, cache_kw):
    """
    Build the same (v+, v-, t+, t-) quads `eval_e2_hadamard_decomposition` builds.

    Same rows, same positional pairing within a concept, same cache ``kind`` strings
    -- so the embeddings are literally the arrays that experiment used, and the
    0.88% baseline and 67.43% ceiling refer to this exact population.
    """
    df = pd.read_csv(args.csv_path)
    coerce_bool_column(df, "object_in_image")

    restrict = load_object_restriction(args.restrict_objects)
    target_objects = sorted(df["object_name"].unique().tolist())
    target_objects = [o for o in target_objects if "," not in str(o)]
    if restrict is not None:
        keep = set(restrict)
        target_objects = [o for o in target_objects if o in keep]
        print(f"  Restricted to {len(target_objects)} of {len(keep)} requested concepts")

    V_POS, V_NEG, T_POS, T_NEG, GROUPS = [], [], [], [], []
    for obj in target_objects:
        df_obj = df[df["object_name"] == obj].reset_index(drop=True)
        df_true = df_obj[df_obj["object_in_image"] == True].reset_index(drop=True)
        df_false = df_obj[df_obj["object_in_image"] == False].reset_index(drop=True)

        n_pairs = min(len(df_true), len(df_false))
        if n_pairs < args.min_pairs:
            continue

        img_pres = [resolve_path(p, args.image_root) for p in df_true["image_path"].tolist()[:n_pairs]]
        img_abs = [resolve_path(p, args.image_root) for p in df_false["image_path"].tolist()[:n_pairs]]
        t_pos_texts = df_true["positive_caption"].tolist()[:n_pairs]
        t_neg_texts = df_true["negative_caption"].tolist()[:n_pairs]

        v_pres, _, mask_vp = cached_encode(
            lambda: encode_images_unified(model, preprocess, img_pres, device, args.batch_size),
            kind="image_pres@norm+raw+flags", items=img_pres, **cache_kw)
        v_abs, _, mask_va = cached_encode(
            lambda: encode_images_unified(model, preprocess, img_abs, device, args.batch_size),
            kind="image_abs@norm+raw+flags", items=img_abs, **cache_kw)

        valid = np.where(mask_vp & mask_va)[0]
        if len(valid) < args.min_pairs:
            continue

        t_pos_texts = [t_pos_texts[i] for i in valid]
        t_neg_texts = [t_neg_texts[i] for i in valid]
        t_pos, _ = cached_encode(
            lambda: encode_texts_unified(model, tokenizer, t_pos_texts, device, args.batch_size),
            kind="text_pos@norm+raw", items=t_pos_texts, **cache_kw)
        t_neg, _ = cached_encode(
            lambda: encode_texts_unified(model, tokenizer, t_neg_texts, device, args.batch_size),
            kind="text_neg@norm+raw", items=t_neg_texts, **cache_kw)

        V_POS.append(v_pres[valid])
        V_NEG.append(v_abs[valid])
        T_POS.append(t_pos)
        T_NEG.append(t_neg)
        GROUPS.extend([obj] * len(valid))

    if not V_POS:
        raise RuntimeError(
            "no concept met --min_pairs with both images loading; nothing to fit")

    return (np.concatenate(V_POS), np.concatenate(V_NEG),
            np.concatenate(T_POS), np.concatenate(T_NEG), np.array(GROUPS))


# ============================================================
# 4. Experiment
# ============================================================
def run(args) -> Dict[str, Any]:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  I-2: does ONE shared W generalise to unseen concepts?")
    print("=" * 70)
    print(f"  Model     : {args.model} ({args.pretrained}) on {device}")
    print(f"  Seed      : {args.seed} | Min pairs: {args.min_pairs} | Folds: {args.n_splits}")
    print(f"  Chance    : {CHANCE_PCT:.2f}%\n")

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    embed_dim = get_embed_dim(model)
    cache_kw = dict(model=args.model, pretrained=str(args.pretrained),
                    cache_dir=args.cache_dir, enabled=args.use_cache)

    print("  Loading quads (same construction as E2)...")
    v_pos, v_neg, t_pos, t_neg, groups = load_quads(
        args, model, preprocess, tokenizer, device, cache_kw)
    concepts = sorted(set(groups))
    print(f"  -> {len(concepts)} concepts, {len(groups)} pairs, embed_dim={embed_dim}\n")

    T = lambda a: torch.from_numpy(a).float().to(device)
    quads_all = (T(v_pos), T(v_neg), T(t_pos), T(t_neg))

    n_splits = min(args.n_splits, len(concepts))
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(np.zeros(len(groups)), groups=groups))

    fold_rows, concept_rows = [], []
    for family in args.families:
        _, n_params = build_family(family, embed_dim, args.seed)
        oof = np.zeros(len(groups), dtype=bool)
        insample_accs = []

        for fold, (tr, te) in enumerate(splits):
            set_seed(args.seed + fold)
            net, _ = build_family(family, embed_dim, args.seed)
            net = net.to(device)
            q_tr = tuple(x[tr] for x in quads_all)
            q_te = tuple(x[te] for x in quads_all)
            net = train_matcher(net, q_tr, epochs=args.epochs, lr=args.lr,
                                weight_decay=args.weight_decay, margin=args.margin)

            ok_te = joint_correct(net, q_te)
            ok_tr = joint_correct(net, q_tr)
            oof[te] = ok_te
            insample_accs.append(100.0 * ok_tr.mean())
            fold_rows.append(dict(
                family=family, n_trainable_params=n_params, fold=fold,
                n_train_pairs=len(tr), n_test_pairs=len(te),
                n_train_concepts=len(set(groups[tr])), n_test_concepts=len(set(groups[te])),
                in_sample_acc_pct=100.0 * ok_tr.mean(),
                oof_acc_pct=100.0 * ok_te.mean(),
            ))

        per_concept = np.array([100.0 * oof[groups == c].mean() for c in concepts])
        for c, acc in zip(concepts, per_concept):
            concept_rows.append(dict(family=family, object_name=c,
                                     n_pairs=int((groups == c).sum()), oof_acc_pct=acc))

        lo, hi = concept_bootstrap_ci(per_concept, seed=args.seed)
        pooled = 100.0 * oof.mean()
        insample = float(np.mean(insample_accs))
        verdict = "ABOVE CHANCE" if lo > CHANCE_PCT else "not above chance"
        print(f"  {family:12s} params={n_params:>7,}  in-sample {insample:6.2f}%  "
              f"OOF {pooled:6.2f}%  macro {per_concept.mean():6.2f}% "
              f"CI[{lo:5.2f}, {hi:5.2f}]  {verdict}")

    df_folds = pd.DataFrame(fold_rows)
    df_concepts = pd.DataFrame(concept_rows)
    df_folds.to_csv(os.path.join(args.output_dir, "single_w_per_family.csv"), index=False)
    df_concepts.to_csv(os.path.join(args.output_dir, "single_w_per_concept.csv"), index=False)

    families = []
    for family in args.families:
        sub_f = df_folds[df_folds["family"] == family]
        per_concept = df_concepts[df_concepts["family"] == family]["oof_acc_pct"].values
        lo, hi = concept_bootstrap_ci(per_concept, seed=args.seed)
        n_pairs_total = int(sub_f["n_test_pairs"].sum())
        pooled = float((sub_f["oof_acc_pct"] * sub_f["n_test_pairs"]).sum() / n_pairs_total)
        families.append(dict(
            family=family,
            n_trainable_params=int(sub_f["n_trainable_params"].iloc[0]),
            in_sample_acc_pct=float(sub_f["in_sample_acc_pct"].mean()),
            oof_pooled_acc_pct=pooled,
            oof_macro_acc_pct=float(per_concept.mean()),
            oof_concept_ci_lower=lo,
            oof_concept_ci_upper=hi,
            beats_chance=bool(lo > CHANCE_PCT),
            generalisation_gap_pp=float(sub_f["in_sample_acc_pct"].mean() - pooled),
        ))

    winners = [f for f in families if f["beats_chance"] and f["family"] not in ("identity", "random")]
    summary = {
        "n_concepts": len(concepts),
        "n_pairs": int(len(groups)),
        "n_splits": n_splits,
        "chance_pct": CHANCE_PCT,
        "cv_strategy": "GroupKFold(object_name) -- held-out concepts appear nowhere in training",
        "families": families,
        "verdict": (
            "(a) at least one W family generalises above chance to unseen concepts"
            if winners else
            "(b) no W family beats chance out-of-fold; the per-pair 67.43% is an upper-bound "
            "demonstration, not an implementable path"
        ),
        "smallest_family_above_chance": winners[0]["family"] if winners else None,
        "reference_points": {
            "cosine_baseline_2x2_pct": 0.88,
            "per_pair_projection_ceiling_2x2_pct": 67.43,
            "per_concept_full_bilinear_in_sample_pct": 88.23,
            "per_concept_full_bilinear_oof_pct": 8.69,
            "note": (
                "The ceiling is a per-pair projection, i.e. a different W per pair; it is "
                "not reachable by any single W and is listed as the upper reference only."
            ),
        },
        "provenance": build_provenance(args, extra={"families": args.families,
                                                    "epochs": args.epochs}),
    }
    with open(os.path.join(args.output_dir, "single_w_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _render_ladder(families, args.output_dir, args.model)

    print("\n" + "=" * 70)
    print(f"  VERDICT: {summary['verdict']}")
    if winners:
        print(f"  Smallest family above chance: {winners[0]['family']} "
              f"({winners[0]['n_trainable_params']:,} params, "
              f"OOF {winners[0]['oof_pooled_acc_pct']:.2f}%)")
    print(f"  Results: {args.output_dir}")
    print("=" * 70)
    return summary


def _render_ladder(families: List[Dict[str, Any]], output_dir: str, model_name: str) -> None:
    """Accuracy against degrees of freedom, with chance and the two reference points."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [f["family"] for f in families]
    x = np.arange(len(labels))
    ax.plot(x, [f["in_sample_acc_pct"] for f in families], "o-", color="#1f77b4",
            lw=2, ms=6, label="In-sample (fitted concepts)")
    ax.plot(x, [f["oof_pooled_acc_pct"] for f in families], "s--", color="#d62728",
            lw=2, ms=6, label="Out-of-fold (unseen concepts)")
    ax.fill_between(x, [f["oof_concept_ci_lower"] for f in families],
                    [f["oof_concept_ci_upper"] for f in families],
                    color="#d62728", alpha=0.15, label="OOF concept-level 95% CI")
    ax.axhline(CHANCE_PCT, color="black", ls=":", lw=2, label=f"Chance ({CHANCE_PCT:.2f}%)")
    ax.axhline(67.43, color="#2ca02c", ls="-.", lw=1.5,
               label="Per-pair projection ceiling (different W per pair)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n({f['n_trainable_params']:,}p)"
                        for l, f in zip(labels, families)], fontsize=8)
    ax.set_ylabel("2x2 joint accuracy (%)", fontsize=11)
    ax.set_title(f"A single shared W across concepts -- {model_name}", fontsize=13, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    out = os.path.join(output_dir, "fig_single_w_ladder.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Does one shared W generalise to concepts it never saw?")
    add_model_args(parser, "ViT-B-32", "openai")
    add_run_args(parser, "logs/evaluation/single_w_generalization", batch_size=128)
    add_data_args(parser, csv_path="benchmarks/data/images/beaf_counterfactual_6col.csv")
    add_cache_args(parser)
    add_restriction_args(parser)
    add_concept_args(parser)
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES,
                        help="W families to run, in ladder order")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="GroupKFold splits over object_name (default: 5)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.1)
    run(parser.parse_args())
