"""
Create improved full COCO paired caption dataset (v2).

Sources:
  1. COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv
     → 5,000 images: positive_objects, negative_objects lists
  2. COCO_val_mcq_llama3.1_rephrased.csv
     → LLaMA-generated diverse pos/neg caption pairs for ~1,869 (image, absent_obj) combos
  3. COCO_val_retrieval.csv (reference only)

Strategy:
  - For absent objects with existing MCQ 'negative' row: use LLaMA-generated (caption_2, caption_0) directly
  - For all other (image, object) pairs: apply 10 diverse pos/neg template pools via round-robin
  - Grammar exceptions for plural/uncountable nouns (skis, scissors, broccoli, etc.)
  - No verb bias: same template pool for both present and absent objects

Output: COCO_val_full_paired_v2.csv
  Columns: image_path, object_name, positive_caption, negative_caption, object_in_image, source_template
"""

import pandas as pd
import ast
import os
import re
import argparse
from collections import defaultdict


# ============================================================
# Grammar: article & verb exceptions for COCO 80 categories
# ============================================================
# Nouns that should not take "a/an" (plural or uncountable)
NO_ARTICLE_NOUNS = {"skis", "scissors", "broccoli"}


def get_article_obj(obj: str) -> str:
    """Return 'a/an obj' or just 'obj' for no-article nouns."""
    if obj.lower() in NO_ARTICLE_NOUNS:
        return obj
    article = "an" if obj[0].lower() in "aeiou" else "a"
    return f"{article} {obj}"


def get_be_verb(obj: str) -> str:
    """Return 'are' for plural nouns, 'is' for singular."""
    if obj.lower() in {"skis", "scissors"}:
        return "are"
    return "is"


def capitalize_first(s: str) -> str:
    """Capitalize only the first character."""
    if not s:
        return s
    return s[0].upper() + s[1:]


# ============================================================
# 10 Negation Templates (from MCQ LLaMA distribution analysis)
# ============================================================
NEG_TEMPLATES = [
    lambda obj: f"There {get_be_verb(obj)} no {obj} in this image." if obj.lower() not in NO_ARTICLE_NOUNS else f"There are no {obj} in this image.",
    lambda obj: f"{capitalize_first(get_article_obj(obj))} {get_be_verb(obj)} not present in this image.",
    lambda obj: f"No {obj} {get_be_verb(obj)} present in this image.",
    lambda obj: f"{capitalize_first(get_article_obj(obj))} {get_be_verb(obj)} not included in this image.",
    lambda obj: f"No {obj} {get_be_verb(obj)} visible in this image.",
    lambda obj: f"This image does not feature {get_article_obj(obj)}.",
    lambda obj: f"{capitalize_first(get_article_obj(obj))} {get_be_verb(obj)} absent from this image.",
    lambda obj: f"No {obj} {get_be_verb(obj)} included in this image.",
    lambda obj: f"{capitalize_first(get_article_obj(obj))} {get_be_verb(obj)} not in this image.",
    lambda obj: f"This image does not have {get_article_obj(obj)}.",
]

# ============================================================
# 10 Positive Templates (from MCQ LLaMA distribution analysis)
# ============================================================
POS_TEMPLATES = [
    lambda obj: f"This image shows {get_article_obj(obj)}.",
    lambda obj: f"{capitalize_first(get_article_obj(obj))} {get_be_verb(obj)} present in this image.",
    lambda obj: f"This image features {get_article_obj(obj)}.",
    lambda obj: f"{capitalize_first(get_article_obj(obj))} {get_be_verb(obj)} included in this image.",
    lambda obj: f"This image contains {get_article_obj(obj)}.",
    lambda obj: f"This image depicts {get_article_obj(obj)}.",
    lambda obj: f"{capitalize_first(get_article_obj(obj))} {get_be_verb(obj)} visible in this image.",
    lambda obj: f"This image includes {get_article_obj(obj)}.",
    lambda obj: f"{capitalize_first(get_article_obj(obj))} {get_be_verb(obj)} shown in this image.",
    lambda obj: f"{capitalize_first(get_article_obj(obj))} {get_be_verb(obj)} featured in this image.",
]


# ============================================================
# MCQ caption pair extraction with object validation
# ============================================================
def extract_object_from_positive(caption: str) -> str:
    """Extract object name from positive caption patterns."""
    cap = str(caption).strip().rstrip(".")
    # Pattern: "This image shows/features/includes/depicts/contains a/an X"
    m = re.match(
        r"This image (?:shows|features|includes|depicts|contains) (?:a |an )?(.+)",
        cap, re.I
    )
    if m:
        return m.group(1).strip()
    # Pattern: "A/An X is present/included/shown/visible/featured/depicted in this image"
    m = re.match(
        r"(?:A |An )?(.+?) is (?:present|included|shown|visible|featured|depicted|contained) in this image",
        cap, re.I
    )
    if m:
        return m.group(1).strip()
    return None


def validate_object_in_negation(neg_caption: str, obj: str) -> bool:
    """Check if the negation caption actually refers to the same object."""
    return obj.lower() in str(neg_caption).lower()


def build_mcq_lookup(mcq_csv_path: str) -> dict:
    """
    Build lookup: (image_path, object_name_lower) -> (pos_caption, neg_caption)
    Only from 'negative' template rows where object validation passes.
    """
    try:
        df = pd.read_csv(mcq_csv_path, encoding="latin-1")
    except Exception:
        df = pd.read_csv(mcq_csv_path)

    neg_rows = df[df["correct_answer_template"] == "negative"]

    lookup = {}
    skipped = 0

    for _, row in neg_rows.iterrows():
        pos_cap = str(row["caption_2"]).strip()
        neg_cap = str(row["caption_0"]).strip()
        image_path = row["image_path"]

        # Extract object from positive caption
        obj = extract_object_from_positive(pos_cap)
        if not obj:
            skipped += 1
            continue

        # Validate: same object must appear in negation caption
        if not validate_object_in_negation(neg_cap, obj):
            skipped += 1
            continue

        key = (image_path, obj.lower())
        # Keep first occurrence (don't overwrite)
        if key not in lookup:
            lookup[key] = (pos_cap, neg_cap, obj)

    print(f"  MCQ lookup built: {len(lookup)} validated pairs ({skipped} skipped)")
    return lookup


# ============================================================
# Main dataset creation
# ============================================================
def create_full_paired_v2(
    neg_csv_path: str,
    mcq_csv_path: str,
    output_csv_path: str,
):
    print(f"Loading negated retrieval dataset: {neg_csv_path}")
    df_neg = pd.read_csv(neg_csv_path)

    print(f"Building MCQ lookup from: {mcq_csv_path}")
    mcq_lookup = build_mcq_lookup(mcq_csv_path)

    all_pairs = []
    template_idx = 0  # Global round-robin counter
    mcq_used = 0

    for i in range(len(df_neg)):
        row = df_neg.iloc[i]
        image_path = row["filepath"]
        pos_objs = ast.literal_eval(row["positive_objects"])
        neg_objs = ast.literal_eval(row["negative_objects"])

        # ── 1. Present Objects (object_in_image = True) ──
        for obj in pos_objs:
            if not obj or not isinstance(obj, str):
                continue
            obj = obj.strip()

            pos_cap = POS_TEMPLATES[template_idx % len(POS_TEMPLATES)](obj)
            neg_cap = NEG_TEMPLATES[template_idx % len(NEG_TEMPLATES)](obj)
            template_idx += 1

            all_pairs.append({
                "image_path": image_path,
                "object_name": obj,
                "positive_caption": pos_cap,
                "negative_caption": neg_cap,
                "object_in_image": True,
                "source_template": "generated_template",
            })

        # ── 2. Absent Objects (object_in_image = False) ──
        for obj in neg_objs:
            if not obj or not isinstance(obj, str):
                continue
            obj = obj.strip()

            # Check if MCQ has a LLaMA-generated pair for this (image, obj)
            key = (image_path, obj.lower())
            if key in mcq_lookup:
                pos_cap, neg_cap, _ = mcq_lookup[key]
                source = "mcq_llama_original"
                mcq_used += 1
            else:
                pos_cap = POS_TEMPLATES[template_idx % len(POS_TEMPLATES)](obj)
                neg_cap = NEG_TEMPLATES[template_idx % len(NEG_TEMPLATES)](obj)
                source = "generated_template"
                template_idx += 1

            all_pairs.append({
                "image_path": image_path,
                "object_name": obj,
                "positive_caption": pos_cap,
                "negative_caption": neg_cap,
                "object_in_image": False,
                "source_template": source,
            })

    pairs_df = pd.DataFrame(all_pairs)

    # ── Summary ──
    n_present = len(pairs_df[pairs_df["object_in_image"] == True])
    n_absent = len(pairs_df[pairs_df["object_in_image"] == False])
    n_mcq = len(pairs_df[pairs_df["source_template"] == "mcq_llama_original"])
    n_gen = len(pairs_df[pairs_df["source_template"] == "generated_template"])

    print("\n" + "=" * 60)
    print("  COCO_val_full_paired_v2 Dataset Summary")
    print("=" * 60)
    print(f"  Total images          : {pairs_df['image_path'].nunique()}")
    print(f"  Total caption pairs   : {len(pairs_df)}")
    print(f"  Present objects (True) : {n_present}")
    print(f"  Absent objects (False) : {n_absent}")
    print(f"  ─────────────────────────────────────")
    print(f"  MCQ LLaMA originals   : {n_mcq}")
    print(f"  Generated templates   : {n_gen}")
    print(f"  Unique objects        : {pairs_df['object_name'].nunique()}")

    # ── Template distribution check ──
    gen_df = pairs_df[pairs_df["source_template"] == "generated_template"]
    if len(gen_df) > 0:
        print("\n  --- Negation Template Distribution (generated) ---")
        neg_starts = gen_df["negative_caption"].apply(lambda c: c.split(" in ")[0][:40] if " in " in c else c[:40])
        for pat, cnt in neg_starts.value_counts().head(10).items():
            pct = cnt / len(gen_df) * 100
            print(f"    {pat:<40}: {cnt:>5} ({pct:>5.1f}%)")

    # ── Grammar check for problematic nouns ──
    print("\n  --- Grammar Spot Check ---")
    for obj_check in ["skis", "scissors", "broccoli", "oven", "umbrella"]:
        rows = pairs_df[pairs_df["object_name"] == obj_check]
        if len(rows) > 0:
            r = rows.iloc[0]
            print(f"    {obj_check}: POS='{r['positive_caption']}' | NEG='{r['negative_caption']}'")

    # ── Sample pairs ──
    print("\n  --- Sample MCQ LLaMA Pairs ---")
    mcq_samples = pairs_df[pairs_df["source_template"] == "mcq_llama_original"].head(3)
    for _, r in mcq_samples.iterrows():
        print(f"    [{r['object_name']}] (+) {r['positive_caption']}")
        print(f"    [{r['object_name']}] (-) {r['negative_caption']}")

    print("\n  --- Sample Generated Template Pairs ---")
    gen_samples = pairs_df[pairs_df["source_template"] == "generated_template"].head(5)
    for _, r in gen_samples.iterrows():
        oi = "IN" if r["object_in_image"] else "NOT"
        print(f"    [{r['object_name']} ({oi})] (+) {r['positive_caption']}")
        print(f"    [{r['object_name']} ({oi})] (-) {r['negative_caption']}")

    # ── Save ──
    os.makedirs(os.path.dirname(output_csv_path) if os.path.dirname(output_csv_path) else ".", exist_ok=True)
    pairs_df.to_csv(output_csv_path, index=False)
    print(f"\n  Saved to: {output_csv_path}")
    print("=" * 60)

    return pairs_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create improved full COCO paired dataset (v2)")
    parser.add_argument(
        "--neg_csv", type=str,
        default="COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv",
    )
    parser.add_argument(
        "--mcq_csv", type=str,
        default="COCO_val_mcq_llama3.1_rephrased.csv",
    )
    parser.add_argument(
        "--output_csv", type=str,
        default="COCO_val_full_paired_v2.csv",
    )
    args = parser.parse_args()

    create_full_paired_v2(args.neg_csv, args.mcq_csv, args.output_csv)
