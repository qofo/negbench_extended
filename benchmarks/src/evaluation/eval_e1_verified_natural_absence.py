"""E1 extension: does the image-side AUC survive when "absent" means a genuinely
unedited photo, not an inpainted one?

review4.md (six independent reviews, all six raised this) argues the E1 macro AUC
(0.759, `eval_e1_minimal_pair_auc.py`) may reflect an inpainting-editing signature
rather than object-presence semantics, since the "absent" image in every
counterfactual pair is BEAF's inpainted edit of the "present" image. The existing
placebo test (`eval_e1_placebo_test.py`) answers a *different* question -- whether
an unrelated query picks up the same edit inside the SAME pair -- and already
passes for 28 of 32 placebo-eligible concepts. `eval_e1_minimal_pair_auc.py`
separately reports a "natural between-image" AUC (0.783), but its negative class
is *unverified*: it is just another concept's present-image, assumed (not checked)
to lack the target concept.

This script closes that gap with a *verified* natural-absence negative class:
COCO val2014 images that were never edited at all (no BEAF suffix -- the raw
`COCO_val2014_<id>.jpg` files already on disk) and that COCO's own instance
annotations confirm do not contain the target category. The same "present" images
and the same concept text prompt as E1 are reused; only the negative class changes,
from "inpainted-absent" to "confirmed-never-present, unedited".

Category labels are not shipped with this repo and were not locally available
(no `pycocotools`, no `instances_val2014.json`, and the official cocodataset.org
annotation mirror was unreachable from this environment). They were fetched via
the Hugging Face-hosted `detection-datasets/coco` mirror instead -- column-pruned
so only `image_id` + `objects` (not the embedded image bytes) were downloaded --
and cached locally as two parquet files. Provenance: HF dataset
`detection-datasets/coco`, train+val splits, ~122k rows, fetched 2026-09-01.

Usage:
    python -m benchmarks.src.evaluation.eval_e1_verified_natural_absence \\
        --restrict_objects logs/evaluation/00_concept_sets/paper33.txt \\
        --output_dir logs/evaluation/01_paper/2026-09-01_e1_verified_natural_absence \\
        --use_cache
"""

import os
import re
import glob
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.cli import (
        add_model_args, add_run_args, add_data_args, add_cache_args, add_restriction_args,
    )
    from benchmarks.src.analysis.beaf.beaf_loader import load_and_verify_counterfactual_pairs
    from benchmarks.src.analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction,
    )
    from benchmarks.src.analysis.config import set_seed, COCO80
    from benchmarks.src.evaluation.eval_e1_minimal_pair_auc import (
        extract_normalized_image_features, extract_normalized_text_features,
    )
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.model_loader import load_clip_for_eval
    from analysis.cli import (
        add_model_args, add_run_args, add_data_args, add_cache_args, add_restriction_args,
    )
    from analysis.beaf.beaf_loader import load_and_verify_counterfactual_pairs
    from analysis.feature_cache import cached_encode, build_provenance, load_object_restriction
    from analysis.config import set_seed, COCO80
    from evaluation.eval_e1_minimal_pair_auc import (
        extract_normalized_image_features, extract_normalized_text_features,
    )


_ID_RE = re.compile(r"COCO_val2014_0*(\d+)(?:_\d+)?\.\w+$")


def image_id_of(path: str) -> int:
    m = _ID_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"cannot parse a COCO val2014 image id from: {path}")
    return int(m.group(1))


def fetch_coco_category_labels(cache_dir: str) -> Dict[int, set]:
    """image_id -> set(category names present), from the cached HF-mirror parquet files.

    Re-fetches from `detection-datasets/coco` (train+val, image_id + objects columns
    only) if the cache files are missing. Network access to huggingface.co is required
    only on a cache miss.
    """
    os.makedirs(cache_dir, exist_ok=True)
    val_path = os.path.join(cache_dir, "coco_val_labels.parquet")
    train_path = os.path.join(cache_dir, "coco_train_labels.parquet")
    if not (os.path.exists(val_path) and os.path.exists(train_path)):
        import pyarrow.parquet as pq
        import pyarrow as pa
        from huggingface_hub import HfApi, HfFileSystem
        print("  Fetching COCO category labels from detection-datasets/coco (HF mirror)...")
        api = HfApi()
        info = api.dataset_info("detection-datasets/coco")
        fs = HfFileSystem()
        for split, out_path in (("val", val_path), ("train", train_path)):
            paths = sorted(f"datasets/detection-datasets/coco/{s.rfilename}"
                           for s in info.siblings if s.rfilename.startswith(f"data/{split}"))
            tables = [pq.read_table(p, columns=["image_id", "objects"], filesystem=fs) for p in paths]
            pa.concat_tables(tables).to_pandas().to_parquet(out_path)
            print(f"    {split}: {len(paths)} shard(s) -> {out_path}")

    lbl = pd.concat([pd.read_parquet(val_path), pd.read_parquet(train_path)], ignore_index=True)
    lbl = lbl.drop_duplicates(subset="image_id")
    id2cats = {int(row.image_id): {COCO80[c] for c in row.objects["category"]}
               for row in lbl.itertuples()}
    return id2cats


def find_pristine_images(image_root: str) -> List[str]:
    """Unedited COCO val2014 files: no BEAF `_NN` suffix on the filename.

    These live at ``benchmarks/data/coco/images/val2014/`` regardless of
    ``--image_root`` (which points at the BEAF-specific ``data/images`` tree the
    counterfactual pairs resolve against) -- so this walks up from image_root to
    the repo's ``data`` root rather than nesting under it.
    """
    data_root = os.path.dirname(image_root.rstrip("/")) if image_root.rstrip("/").endswith("images") \
        else image_root
    pattern = os.path.join(data_root, "coco", "images", "val2014", "COCO_val2014_*.jpg")
    files = glob.glob(pattern)
    if not files:
        # Fall back to the repo-root-relative path used throughout the codebase.
        pattern = "benchmarks/data/coco/images/val2014/COCO_val2014_*.jpg"
        files = glob.glob(pattern)
    return [f for f in files if re.match(r".*COCO_val2014_\d+\.jpg$", f)]


def main():
    parser = argparse.ArgumentParser(
        description="E1 with a verified natural-absence negative class (review4 E2)")
    add_model_args(parser, "ViT-B-32", "openai")
    add_run_args(parser, "logs/evaluation/e1_verified_natural_absence", seed=42, batch_size=128)
    add_data_args(parser, csv_path="benchmarks/data/images/beaf_counterfactual_6col.csv",
                  image_root="benchmarks/data/images")
    add_cache_args(parser)
    add_restriction_args(parser, "Concept set (paper33.txt shares E1's own population)")
    parser.add_argument("--prompt_template", type=str, default="a photo of a {}")
    parser.add_argument("--min_pairs", type=int, default=20)
    parser.add_argument("--min_natural_pairs", type=int, default=15,
                        help="Skip a concept if fewer verified-absent pristine images survive")
    parser.add_argument("--label_cache_dir", type=str,
                        default="logs/evaluation/00_concept_sets/coco_category_labels")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_kw = dict(model=args.model, pretrained=args.pretrained,
                    cache_dir=args.cache_dir, enabled=args.use_cache)
    rng = np.random.default_rng(args.seed)

    print("=" * 74)
    print("  E1 with a verified (COCO-annotation-checked) natural-absence negative class")
    print("=" * 74)

    df_raw, df_pairs, _ = load_and_verify_counterfactual_pairs(args.csv_path, args.image_root)
    restrict = load_object_restriction(args.restrict_objects)
    if restrict is not None:
        df_pairs = df_pairs[df_pairs["object_name"].isin(set(restrict))].reset_index(drop=True)
    unique_concepts = sorted(df_pairs["object_name"].unique().tolist())
    print(f"  {len(df_pairs)} counterfactual pairs across {len(unique_concepts)} concepts")

    id2cats = fetch_coco_category_labels(args.label_cache_dir)
    pristine_files = find_pristine_images(args.image_root)
    pristine = {image_id_of(f): f for f in pristine_files if image_id_of(f) in id2cats}
    print(f"  {len(pristine)} unedited local images with verified COCO category labels")

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)

    paths_pres = df_pairs["orig_path"].tolist()
    feats_pres, flags_pres = cached_encode(
        lambda: extract_normalized_image_features(model, preprocess, paths_pres, device, args.batch_size),
        kind="image_pres@l2norm+flags", items=paths_pres, **cache_kw)

    def _encode_concepts():
        d = extract_normalized_text_features(model, tokenizer, unique_concepts, device,
                                             prompt_template=args.prompt_template)
        return (np.stack([d[c] for c in unique_concepts]),)
    (concept_matrix,) = cached_encode(
        _encode_concepts, kind=f"text_concept@l2norm|{args.prompt_template}|ens=0",
        items=unique_concepts, **cache_kw)
    concept_text_feats = {c: concept_matrix[i] for i, c in enumerate(unique_concepts)}

    rows = []
    all_y_true, all_y_score = [], []
    for c in unique_concepts:
        mask = (df_pairs["object_name"] == c).values & flags_pres
        n_c = int(mask.sum())
        if n_c < args.min_pairs:
            continue
        idx = np.where(mask)[0]
        own_ids = {image_id_of(p) for p in df_pairs["orig_path"].iloc[idx]} | \
                  {image_id_of(p) for p in df_pairs["cf_path"].iloc[idx]}

        candidates = [iid for iid in pristine if c not in id2cats[iid] and iid not in own_ids]
        if len(candidates) < args.min_natural_pairs:
            print(f"  [{c:16s}] skipped: only {len(candidates)} verified-absent pristine images")
            continue
        n_neg = min(n_c, len(candidates))
        chosen = rng.choice(candidates, size=n_neg, replace=False)
        neg_paths = [pristine[iid] for iid in chosen]

        feats_neg, flags_neg = cached_encode(
            lambda: extract_normalized_image_features(model, preprocess, neg_paths, device, args.batch_size),
            kind="image_natural_absent@l2norm+flags", items=neg_paths, **cache_kw)
        feats_neg = feats_neg[flags_neg]
        if len(feats_neg) < args.min_natural_pairs:
            print(f"  [{c:16s}] skipped: only {len(feats_neg)} natural images loaded")
            continue

        t_vec = concept_text_feats[c]
        n_use = min(n_c, len(feats_neg))
        s_pres = np.dot(feats_pres[idx][:n_use], t_vec)
        s_neg = np.dot(feats_neg[:n_use], t_vec)

        y_true = np.concatenate([np.ones(n_use), np.zeros(n_use)])
        y_score = np.concatenate([s_pres, s_neg])
        auc = float(roc_auc_score(y_true, y_score))
        all_y_true.extend(y_true.tolist())
        all_y_score.extend(y_score.tolist())
        rows.append(dict(object_name=c, n_pairs=n_use, n_candidates_available=len(candidates),
                         natural_absence_auc=auc))
        print(f"  [{c:16s}] n={n_use:3d}  verified-natural-absence AUC = {auc:.3f}  "
              f"(candidates available: {len(candidates)})")

    if not rows:
        raise SystemExit("No concept had enough verified-absent pristine images; nothing to report.")

    df_out = pd.DataFrame(rows)
    macro_auc = float(df_out["natural_absence_auc"].mean())
    pooled_auc = float(roc_auc_score(all_y_true, all_y_score))

    summary = {
        "n_concepts": int(len(df_out)),
        "macro_natural_absence_auc": macro_auc,
        "pooled_natural_absence_auc": pooled_auc,
        "reference_counterfactual_macro_auc_33concepts": 0.759,
        "reference_counterfactual_pooled_auc_33concepts": 0.569,
        "reference_unverified_natural_between_image_macro_auc_33concepts": 0.783,
        "label_source": "detection-datasets/coco (HF mirror), train+val, fetched 2026-09-01",
        "provenance": build_provenance(args, n_concepts=int(len(df_out)),
                                       n_pairs=int(df_out["n_pairs"].sum())),
    }
    df_out.to_csv(os.path.join(args.output_dir, "per_concept_natural_absence_auc.csv"), index=False)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    vals = [0.759, 0.783, macro_auc]
    labels = ["CF (inpainted)\nmacro 0.759", "Natural\n(unverified)\nmacro 0.783",
              "Natural\n(verified absent)\nmacro"]
    bars = ax.bar(range(3), vals, color=["#e9c46a", "#8d99ae", "#2a9d8f"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="crimson", ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("macro AUC")
    ax.set_title("E1 image-side AUC: three negative classes")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "fig_natural_absence_comparison.png"), dpi=150)

    print("=" * 74)
    print(f"  concepts evaluated       : {summary['n_concepts']}")
    print(f"  macro AUC (verified)     : {macro_auc:.3f}")
    print(f"  pooled AUC (verified)    : {pooled_auc:.3f}")
    print(f"  reference: CF (inpainted) macro 0.759, unverified-natural macro 0.783")
    print("=" * 74)


if __name__ == "__main__":
    main()
