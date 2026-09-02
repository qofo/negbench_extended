"""
Audit PeakPatch on the controlled 2x2, in the factor coordinates.

PeakPatch (Lu et al., ECCV 2026, arXiv 2607.23271) reaches 74.3% on NegBench COCO
MCQ against CLIP's 39.2% while keeping the backbone frozen and preserving standard
retrieval. NegBench MCQ fixes the image and picks a caption, which in the factor
coordinates S_ab = C + a*beta + b*alpha + ab*gamma is the single condition
gamma > |alpha|. This script scores the same modules on counterfactual image pairs,
where the other condition gamma > |beta| and their conjunction (the Winoground group
score) are also defined, and reports what moved.

Three conditions, all on the same pairs and the same frozen ViT-B/32:

  cosine    the untouched backbone -- reproduces the e2 decomposition
  ecn       the ECN's corrected text embedding, scored with plain cosine. A drop-in
            replacement for the text embedding, so the score stays an inner product
            and the decomposition is unchanged in kind
  ecn+scn   the full system: the ECN's embedding fed to the SCN as its final-layer
            slot, scored with ScoreCorrector.score_mcq over the block's two captions

The SCN's score is not an inner product of two vectors. That does not matter here:
the decomposition needs only the four numbers S_ab, so it applies to any scorer.
score_mcq is also pool-independent -- it calls forward() per option and the
confidence gate only engages when baseline_sim has a candidate axis (score_pairwise
and score_contrastive), so a K=2 block and a K=4 MCQ give the same per-pair score.

Usage:
    python -m benchmarks.src.evaluation.eval_peakpatch_2x2_audit \\
        --peakpatch_root PeakPatch \\
        --restrict_objects logs/evaluation/01_paper/2026-08-28_e2_hadamard_decomposition/e2_per_concept_decomposition.csv \\
        --use_cache --output_dir logs/evaluation/01_paper/2026-09-02_peakpatch_2x2_audit
"""
import os
import sys
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.cli import (
        add_model_args, add_run_args, add_data_args, add_cache_args,
        add_restriction_args, add_concept_args,
    )
    from benchmarks.src.analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction, DEFAULT_CACHE_DIR,
    )
    from benchmarks.src.analysis.config import set_seed, coerce_bool_column
    from benchmarks.src.analysis.paths import resolve_image_path as resolve_path
    from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, compute_hadamard_coordinates,
    )
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.model_loader import load_clip_for_eval
    from analysis.cli import (
        add_model_args, add_run_args, add_data_args, add_cache_args,
        add_restriction_args, add_concept_args,
    )
    from analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction, DEFAULT_CACHE_DIR,
    )
    from analysis.config import set_seed, coerce_bool_column
    from analysis.paths import resolve_image_path as resolve_path
    from evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, compute_hadamard_coordinates,
    )

CONDITIONS = ("cosine", "ecn", "ecn+scn")
CHANCE_PAIR = 25.0
CHANCE_GROUP = 100 / 6


# ============================================================
# PeakPatch loading -- their modules, imported, not reimplemented
# ============================================================
def load_peakpatch(root: str, device: str):
    """
    Load the released ECN and SCN checkpoints and the backbone they were built on.

    Mirrors scripts/eval_mcq.py:build_peakpatch_system, but returns the backbone the
    caller already holds rather than creating a second copy -- the checkpoints are
    ViT-B/32 (openai), which is the model this repo's e2 runs also use, so the two
    decompositions are measured on identical image embeddings.
    """
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from peakpatch.model import EmbeddingCorrector, ScoreCorrector

    ec_path = os.path.join(root, "checkpoints", "peakpatch_ecn.pt")
    sc_path = os.path.join(root, "checkpoints", "peakpatch_scn.pt")
    ckpt = torch.load(ec_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    arch = cfg.get("clip_model", "ViT-B-32")
    pretrained = cfg.get("clip_pretrained", "openai")

    ec = EmbeddingCorrector.from_checkpoint(ec_path, device)
    sc = ScoreCorrector.from_checkpoint(sc_path, device)
    ec_alpha = torch.exp(ec.log_alpha).item()

    print(f"  PeakPatch backbone declared by checkpoint: {arch} ({pretrained})")
    print(f"  ECN: {ec.count_parameters():,} params, peak layer "
          f"{cfg.get('target_layer', 12)}, anchor layer {cfg.get('anchor_layer', 10)}, "
          f"alpha {ec_alpha:.4f}")
    print(f"  SCN: {sum(p.numel() for p in sc.parameters()):,} params, layers "
          f"{sc.selected_layers}, max_correction {sc.max_correction}")

    return dict(ec=ec, sc=sc, cfg=cfg, ec_alpha=ec_alpha, arch=arch,
                pretrained=pretrained, sc_layers=sc.selected_layers)


def walk_text_tower(tokens, clip_model, layers) -> Tuple[Dict[int, torch.Tensor],
                                                         Dict[int, torch.Tensor]]:
    """
    One traversal of the frozen text tower returning, for each requested layer, both
    the full token sequence (B, N, D) and the projected, normalized [EOS] embedding.

    This is PeakPatch's extract_token_sequences and extract_sc_features fused into a
    single pass, with one correction: **this repository's vendored open_clip runs its
    text transformer in LND**, permuting NLD -> LND before the residual blocks and
    back after (`transformer.py`, the text tower). PeakPatch's extractors were written
    against an upstream open_clip that keeps NLD throughout, so calling them here feeds
    a (B, 77, D) tensor to blocks that read dim 0 as the sequence -- which fails
    outright whenever the batch size differs from 77, and would silently transpose the
    computation if it ever matched. Their repository is left untouched; the layout fix
    lives here, and assert_encode_text_consistency() below verifies the result against
    the model's own forward.
    """
    from peakpatch.clip_utils import _get_text_encoder, _apply_text_projection, get_eos_positions

    text_enc = _get_text_encoder(clip_model)
    cast_dtype = text_enc.transformer.get_cast_dtype()
    eos_pos = get_eos_positions(tokens, clip_model)
    batch_idx = torch.arange(tokens.shape[0], device=tokens.device)
    attn_mask = getattr(text_enc, "attn_mask", None)

    x = text_enc.token_embedding(tokens).to(cast_dtype)
    x = x + text_enc.positional_embedding[:x.size(1)].to(cast_dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND

    seqs, eos_feats = {}, {}
    for layer_idx, block in enumerate(text_enc.transformer.resblocks):
        x = block(x, attn_mask=attn_mask)
        layer = layer_idx + 1
        if layer in layers:
            nld = x.permute(1, 0, 2)  # LND -> NLD, the layout both modules expect
            seqs[layer] = text_enc.ln_final(nld).float()
            eos = text_enc.ln_final(nld[batch_idx, eos_pos])
            eos_feats[layer] = F.normalize(
                _apply_text_projection(text_enc, eos).float(), dim=-1)
    return seqs, eos_feats


def assert_encode_text_consistency(clip_model, tokenizer, device, tol=1e-4):
    """
    The manual traversal must reproduce the model's own encode_text at the final layer.

    Without this the LND/NLD fix above is unverified, and a wrong layout does not
    raise -- it returns plausible embeddings computed over the wrong axis.
    """
    n_layers = len(_text_encoder(clip_model).transformer.resblocks)
    probe = ["There is a train in this image.",
             "There is no truck in this image.",
             "A photo of a dog and a cat."]
    tokens = tokenizer(probe).to(device)
    with torch.no_grad():
        _, eos = walk_text_tower(tokens, clip_model, [n_layers])
        reference = F.normalize(clip_model.encode_text(tokens).float(), dim=-1)
    err = (eos[n_layers] - reference).abs().max().item()
    if err > tol:
        raise SystemExit(
            f"manual text-tower traversal disagrees with encode_text by {err:.3e} "
            f"(tolerance {tol:.0e}). The layer walk is not reproducing the encoder, so "
            "every corrected embedding downstream would be wrong.")
    print(f"  [check] manual traversal == encode_text at layer {n_layers} "
          f"(max abs err {err:.2e})")


def _text_encoder(clip_model):
    from peakpatch.clip_utils import _get_text_encoder
    return _get_text_encoder(clip_model)


def encode_texts_peakpatch(clip_model, tokenizer, texts: List[str], pp, device: str,
                           batch_size: int = 128):
    """
    Returns (plain, ecn_corrected, sc_layer_features) for a list of captions.

    The ECN call is the one in scripts/eval_mcq.py: hidden states at the anchor and
    peak layers, the frozen final embedding, and the padding mask.
    """
    from peakpatch.clip_utils import compute_padding_mask

    ec, cfg = pp["ec"], pp["cfg"]
    target_layer = cfg.get("target_layer", 12)
    anchor_layer = cfg.get("anchor_layer", 10)
    ec_layers = [target_layer] if ec.no_anchor else [anchor_layer, target_layer]
    wanted = sorted(set(ec_layers) | set(pp["sc_layers"]))

    plain, corrected = [], []
    sc_feats: Dict[int, List[torch.Tensor]] = {l: [] for l in pp["sc_layers"]}

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[i:i + batch_size]).to(device)

            seqs, eos = walk_text_tower(tokens, clip_model, wanted)
            text_cls = F.normalize(clip_model.encode_text(tokens).float(), dim=-1)
            H_anchor = seqs[target_layer] if ec.no_anchor else seqs[anchor_layer]
            ec_out, _ = ec.correct_embedding(
                text_cls, H_anchor, seqs[target_layer],
                compute_padding_mask(tokens, clip_model), alpha=pp["ec_alpha"])

            for l in pp["sc_layers"]:
                sc_feats[l].append(eos[l].cpu())
            plain.append(text_cls.cpu())
            corrected.append(ec_out.cpu())

    return (torch.cat(plain).numpy(),
            torch.cat(corrected).numpy(),
            {l: torch.cat(v) for l, v in sc_feats.items()})


def score_block_with_scn(pp, v_img: np.ndarray, sc_layers_pos, sc_layers_neg,
                         ecn_pos: np.ndarray, ecn_neg: np.ndarray, device: str):
    """
    SCN scores for one image state against the block's two captions.

    Builds the (B, K=2, D) option tensor score_mcq expects and overwrites its
    final-layer slot with the ECN output, exactly as scripts/eval_mcq.py does.
    """
    sc = pp["sc"]
    img = torch.from_numpy(v_img).float().to(device)
    mcq = {l: torch.stack([sc_layers_pos[l], sc_layers_neg[l]], dim=1).to(device)
           for l in pp["sc_layers"]}
    mcq[max(pp["sc_layers"])] = torch.stack(
        [torch.from_numpy(ecn_pos).float(), torch.from_numpy(ecn_neg).float()],
        dim=1).to(device)
    with torch.no_grad():
        scores, _ = sc.score_mcq(mcq, img)
    return scores[:, 0].cpu().numpy(), scores[:, 1].cpu().numpy()


# ============================================================
# Metrics
# ============================================================
def three_conditions(h: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The Winoground triple, each a direct inequality on the coefficients."""
    return dict(
        caption=h["gamma"] > h["abs_alpha"],
        image=h["gamma"] > h["abs_beta"],
        group=h["gamma"] > np.maximum(h["abs_alpha"], h["abs_beta"]),
    )


def validate_against_negbench_mcq(args, model, preprocess, tokenizer, pp, device,
                                  cache_kw) -> Dict[str, Dict[str, float]]:
    """
    Reproduce PeakPatch's published NegBench MCQ number through this adaptation.

    The traversal check proves the layer-12 [EOS] matches encode_text, but it says
    nothing about the layer-6 and layer-8 *sequences* the ECN actually reads -- a
    wrong tensor there still yields a plausible-looking correction. Scoring the
    benchmark PeakPatch reports on is the check that closes that gap: their paper
    and README give 39.3% for frozen CLIP and 74.3% for PeakPatch on COCO MCQ, so
    the adaptation is only trustworthy if those two numbers come back.
    """
    csv_path = os.path.join(args.negbench_csv_dir, "COCO_val_mcq_llama3.1_rephrased.csv")
    df = pd.read_csv(csv_path)
    print(f"\n  [validation] NegBench COCO MCQ: {len(df)} questions from {csv_path}")

    paths = [resolve_path(p, args.image_root) for p in df["image_path"].tolist()]
    img, _, ok = cached_encode(
        lambda: encode_images_unified(model, preprocess, paths, device, args.batch_size),
        kind="image_mcq@norm+raw+flags", items=paths, **cache_kw)
    if not ok.all():
        print(f"    {(~ok).sum()} images failed to load; they are dropped")
    df, img = df[ok].reset_index(drop=True), img[ok]

    correct = {c: np.zeros(len(df), dtype=bool) for c in CONDITIONS}
    by_type: Dict[str, Dict[str, List[bool]]] = {c: {} for c in CONDITIONS}

    for start in range(0, len(df), args.batch_size):
        block = df.iloc[start:start + args.batch_size]
        b = len(block)
        flat = [block[f"caption_{i}"].tolist() for i in range(4)]
        flat = [c for opt in flat for c in opt]  # option-major, as eval_mcq.py builds it

        plain, ecn, sc = encode_texts_peakpatch(
            model, tokenizer, flat, pp, device, args.batch_size * 4)
        v = torch.from_numpy(img[start:start + b]).float().to(device)

        opts = {"cosine": torch.from_numpy(plain).float().view(4, b, -1).permute(1, 0, 2),
                "ecn": torch.from_numpy(ecn).float().view(4, b, -1).permute(1, 0, 2)}
        logits = {c: torch.einsum("bd,bkd->bk", v, t.to(device)) for c, t in opts.items()}

        mcq = {l: sc[l].view(4, b, -1).permute(1, 0, 2).to(device) for l in pp["sc_layers"]}
        mcq[max(pp["sc_layers"])] = opts["ecn"].to(device)
        with torch.no_grad():
            logits["ecn+scn"], _ = pp["sc"].score_mcq(mcq, v)

        gold = block["correct_answer"].to_numpy()
        for c in CONDITIONS:
            hit = (logits[c].argmax(dim=1).cpu().numpy() == gold)
            correct[c][start:start + b] = hit
            for t, h in zip(block["correct_answer_template"], hit):
                by_type[c].setdefault(t, []).append(bool(h))

    print(f"    {'condition':<10}{'total':>8}" +
          "".join(f"{t:>12}" for t in sorted(by_type['cosine'])))
    out = {}
    for c in CONDITIONS:
        row = {"total": float(correct[c].mean() * 100)}
        row.update({t: float(np.mean(v) * 100) for t, v in sorted(by_type[c].items())})
        out[c] = row
        print(f"    {c:<10}{row['total']:8.2f}" +
              "".join(f"{row[t]:12.2f}" for t in sorted(by_type[c])))
    print("    published (Lu et al., Table 1): CLIP 39.2 total / 6.6 negation, "
          "PeakPatch 74.3 total / 63.2 negation")
    return out


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 66)
    print("  PeakPatch on the controlled 2x2 -- factor-coordinate audit")
    print("=" * 66)

    df = pd.read_csv(args.csv_path)
    df = coerce_bool_column(df, "object_in_image")
    restriction = load_object_restriction(args.restrict_objects)
    objects = sorted(df["object_name"].unique().tolist())
    if restriction:
        objects = [o for o in objects if o in set(restriction)]
    print(f"\n  Concepts after restriction: {len(objects)}")

    print("\n  Loading PeakPatch...")
    pp = load_peakpatch(args.peakpatch_root, device)
    if (args.model, args.pretrained) != (pp["arch"], pp["pretrained"]):
        raise SystemExit(
            f"backbone mismatch: --model {args.model}/{args.pretrained} but the ECN "
            f"checkpoint was built on {pp['arch']}/{pp['pretrained']}. The corrections "
            "are tied to that encoder's layer geometry; scoring them on another "
            "backbone would not be PeakPatch.")

    print(f"\n  Loading CLIP '{args.model}' ({args.pretrained})...")
    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    assert_encode_text_consistency(model, tokenizer, device)
    cache_kw = dict(model=args.model, pretrained=args.pretrained,
                    enabled=args.use_cache, cache_dir=args.cache_dir)

    per_pair, per_concept = [], []
    for obj in objects:
        d = df[df["object_name"] == obj].reset_index(drop=True)
        d_t = d[d["object_in_image"] == True].reset_index(drop=True)
        d_f = d[d["object_in_image"] == False].reset_index(drop=True)
        n = min(len(d_t), len(d_f))
        if n < args.min_pairs:
            continue

        p_pres = [resolve_path(p, args.image_root) for p in d_t["image_path"].tolist()[:n]]
        p_abs = [resolve_path(p, args.image_root) for p in d_f["image_path"].tolist()[:n]]
        t_pos = d_t["positive_caption"].tolist()[:n]
        t_neg = d_t["negative_caption"].tolist()[:n]

        v_pres, _, m_p = cached_encode(
            lambda: encode_images_unified(model, preprocess, p_pres, device, args.batch_size),
            kind="image_pres@norm+raw+flags", items=p_pres, **cache_kw)
        v_abs, _, m_a = cached_encode(
            lambda: encode_images_unified(model, preprocess, p_abs, device, args.batch_size),
            kind="image_abs@norm+raw+flags", items=p_abs, **cache_kw)

        keep = np.where(m_p & m_a)[0]
        if len(keep) < args.min_pairs:
            continue
        v_pres, v_abs = v_pres[keep], v_abs[keep]
        t_pos = [t_pos[i] for i in keep]
        t_neg = [t_neg[i] for i in keep]

        # Text is encoded three ways in one pass per polarity.
        pos_plain, pos_ecn, pos_sc = encode_texts_peakpatch(
            model, tokenizer, t_pos, pp, device, args.batch_size)
        neg_plain, neg_ecn, neg_sc = encode_texts_peakpatch(
            model, tokenizer, t_neg, pp, device, args.batch_size)

        quads = {}
        for cond, tp, tn in (("cosine", pos_plain, neg_plain), ("ecn", pos_ecn, neg_ecn)):
            quads[cond] = (np.sum(v_pres * tp, -1), np.sum(v_abs * tp, -1),
                           np.sum(v_pres * tn, -1), np.sum(v_abs * tn, -1))

        s11, s21 = score_block_with_scn(pp, v_pres, pos_sc, neg_sc, pos_ecn, neg_ecn, device)
        s12, s22 = score_block_with_scn(pp, v_abs, pos_sc, neg_sc, pos_ecn, neg_ecn, device)
        quads["ecn+scn"] = (s11, s12, s21, s22)

        rec = {"object_name": obj, "n_pairs": len(keep)}
        for cond in CONDITIONS:
            h = compute_hadamard_coordinates(*quads[cond])
            flags = three_conditions(h)
            for i in range(len(keep)):
                per_pair.append(dict(
                    object_name=obj, pair_idx=i, condition=cond,
                    alpha=h["alpha"][i], abs_alpha=h["abs_alpha"][i],
                    beta=h["beta"][i], abs_beta=h["abs_beta"][i], gamma=h["gamma"][i],
                    **{k: bool(v[i]) for k, v in flags.items()}))
            rec.update({
                f"{cond}_abs_alpha": float(h["abs_alpha"].mean()),
                f"{cond}_abs_beta": float(h["abs_beta"].mean()),
                f"{cond}_alpha_signed": float(h["alpha"].mean()),
                f"{cond}_gamma": float(h["gamma"].mean()),
                **{f"{cond}_{k}": float(v.mean() * 100) for k, v in flags.items()},
            })
        per_concept.append(rec)
        print(f"    {obj:<16} n={len(keep):<4} "
              + "  ".join(f"{c}: grp {rec[c + '_group']:5.2f}%" for c in CONDITIONS))

    dp = pd.DataFrame(per_pair)
    dc = pd.DataFrame(per_concept)
    dp.to_csv(os.path.join(args.output_dir, "peakpatch_per_pair.csv"), index=False)
    dc.to_csv(os.path.join(args.output_dir, "peakpatch_per_concept.csv"), index=False)

    print("\n" + "=" * 66)
    print(f"  {len(dc)} concepts, {dp.pair_idx.nunique() and len(dp) // len(CONDITIONS)} pairs")
    print(f"  {'condition':<10}{'|alpha|':>11}{'|beta|':>11}{'gamma':>11}"
          f"{'caption':>10}{'image':>9}{'group':>8}")
    summary = {}
    for cond in CONDITIONS:
        s = dp[dp.condition == cond]
        row = dict(
            abs_alpha=float(s.abs_alpha.mean()), abs_beta=float(s.abs_beta.mean()),
            gamma=float(s.gamma.mean()),
            caption=float(s.caption.mean() * 100), image=float(s.image.mean() * 100),
            group=float(s.group.mean() * 100),
            concepts_meeting_success=int((dc[f"{cond}_group"] > 0).sum()),
        )
        summary[cond] = row
        print(f"  {cond:<10}{row['abs_alpha']:11.3e}{row['abs_beta']:11.3e}"
              f"{row['gamma']:11.3e}{row['caption']:10.2f}{row['image']:9.2f}"
              f"{row['group']:8.2f}")
    print(f"  {'chance':<10}{'':>11}{'':>11}{'':>11}{CHANCE_PAIR:10.2f}"
          f"{CHANCE_PAIR:9.2f}{CHANCE_GROUP:8.2f}")

    mcq = None
    if args.validate_mcq:
        mcq = validate_against_negbench_mcq(
            args, model, preprocess, tokenizer, pp, device, cache_kw)

    out = dict(summary=summary, negbench_mcq=mcq, n_concepts=len(dc),
               provenance=build_provenance(args, extra=dict(
                   peakpatch_root=os.path.abspath(args.peakpatch_root),
                   ecn_params=pp["ec"].count_parameters(),
                   scn_layers=list(pp["sc_layers"]),
                   ec_alpha=pp["ec_alpha"],
                   conditions=list(CONDITIONS))))
    with open(os.path.join(args.output_dir, "peakpatch_2x2_summary.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser, "ViT-B-32", "openai")
    add_run_args(parser, "logs/evaluation/peakpatch_2x2_audit", seed=42, batch_size=128)
    add_data_args(parser, csv_path="benchmarks/data/images/beaf_counterfactual_6col.csv",
                  image_root="benchmarks/data/images")
    add_cache_args(parser)
    add_restriction_args(parser, "Concept set to share with the e2 runs")
    add_concept_args(parser, help_text="Minimum counterfactual pairs per object")
    parser.add_argument("--peakpatch_root", type=str, default="PeakPatch",
                        help="Clone of the official PeakPatch repository")
    parser.add_argument("--negbench_csv_dir", type=str, default="benchmarks/data/images",
                        help="Directory holding COCO_val_mcq_llama3.1_rephrased.csv")
    parser.add_argument("--validate_mcq", action="store_true",
                        help="Also score NegBench COCO MCQ, to check this adaptation "
                             "against the numbers Lu et al. publish for it")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
