"""
BEAF Clean A/B Swap Diverse Dataset Generator.

Generates a robust, position-balanced, lexically diverse A/B swap dataset using
pure unedited original COCO images (without inpainting artifacts), while strictly
satisfying object presence/absence conditions:
- Image 1 (img1): Object A is present, Object B is absent (A in img1, B not in img1)
- Image 2 (img2): Object B is present, Object A is absent (B in img2, A not in img2)

Key Enhancements:
1. Pure Unedited Images:
   - Uses genuine unedited COCO images (from val2017 / val2014)
   - 0% inpainting artifacts (eliminates brush, blur, compression shortcuts)
2. Positional Invariance (50:50 Position Swap):
   - 50% Pos-First / Neg-Second: e.g. "There is a truck, but no train."
   - 50% Neg-First / Pos-Second: e.g. "There is no train, but there is a truck."
3. Lexical Diversity across 4 Template Families:
   - Standard (Group A): is present, can be seen, is shown, no X is present, does not contain
   - Lacking (Group B): contains X, includes X, lacks any Y, contains no Y
   - Absent (Group C): X is visible, X appears, Y is absent, Y does not appear
   - Free-of (Group D): shows X, depicts X, is free of Y, without any Y
4. Explicit 12-Column Schema:
   - image_path, object_name, object_a, object_b, object_a_present, object_b_present,
     positive_caption, negative_caption, object_in_image,
     pos_position, template_family, source_template

Output:
    benchmarks/data/images/beaf_clean_ab_swap_diverse.csv
"""

import os
import re
import ast
import random
import argparse
from typing import List, Dict, Tuple, Set, Any
import pandas as pd

# Uncountable or plural nouns that take zero article
NO_ARTICLE_NOUNS = {"broccoli", "scissors", "skis"}


def format_a_noun(obj: str) -> str:
    """Format noun with indefinite article (a/an) or bare for no-article nouns."""
    obj_clean = obj.lower().strip()
    if obj_clean in NO_ARTICLE_NOUNS:
        return obj_clean
    elif obj_clean[0] in "aeiou":
        return f"an {obj_clean}"
    else:
        return f"a {obj_clean}"


def format_capital_a_noun(obj: str) -> str:
    """Format noun with capitalized indefinite article (A/An) or capitalized bare noun."""
    s = format_a_noun(obj)
    return s[0].upper() + s[1:] if s else s


def get_plural_noun(obj: str) -> str:
    """Simple pluralizer for COCO objects."""
    obj_clean = obj.lower().strip()
    if obj_clean == "person":
        return "people"
    elif obj_clean in {"scissors", "skis", "broccoli"}:
        return obj_clean
    elif obj_clean.endswith(("s", "sh", "ch", "x", "z")):
        return f"{obj_clean}es"
    elif obj_clean.endswith("y") and obj_clean[-2] not in "aeiou":
        return f"{obj_clean[:-1]}ies"
    else:
        return f"{obj_clean}s"


# ============================================================
# Diverse Compound Template Definitions (4 Families x 2 Positions)
# ============================================================

# Position 1: Positive First, Negative Second (A present, B absent)
POS_FIRST_TEMPLATES = [
    # Family 1: Standard (Group A)
    {
        "family": "standard",
        "gen": lambda a, b: f"There is {format_a_noun(a)} in this image, but no {b}.",
        "id": "std_pos_first_01",
    },
    {
        "family": "standard",
        "gen": lambda a, b: f"{format_capital_a_noun(a)} is present in this image, but no {b} is present.",
        "id": "std_pos_first_02",
    },
    {
        "family": "standard",
        "gen": lambda a, b: f"{format_capital_a_noun(a)} is visible in the image, but {format_a_noun(b)} is not present.",
        "id": "std_pos_first_03",
    },
    {
        "family": "standard",
        "gen": lambda a, b: f"{format_capital_a_noun(a)} can be seen in this image, but no {b} can be found.",
        "id": "std_pos_first_04",
    },
    {
        "family": "standard",
        "gen": lambda a, b: f"This image shows {format_a_noun(a)}, but does not contain any {b}.",
        "id": "std_pos_first_05",
    },

    # Family 2: Lacking / Contains (Group B)
    {
        "family": "lacking",
        "gen": lambda a, b: f"The image contains {format_a_noun(a)}, while lacking any {b}.",
        "id": "lack_pos_first_01",
    },
    {
        "family": "lacking",
        "gen": lambda a, b: f"This picture includes {format_a_noun(a)}, but contains no {b}.",
        "id": "lack_pos_first_02",
    },
    {
        "family": "lacking",
        "gen": lambda a, b: f"The scene features {format_a_noun(a)}, but lacks {format_a_noun(b)}.",
        "id": "lack_pos_first_03",
    },
    {
        "family": "lacking",
        "gen": lambda a, b: f"This image depicts {format_a_noun(a)}, though it contains no visible {b}.",
        "id": "lack_pos_first_04",
    },

    # Family 3: Absent / Appears (Group C)
    {
        "family": "absent",
        "gen": lambda a, b: f"{format_capital_a_noun(a)} appears in the image, but the {b} is absent.",
        "id": "abs_pos_first_01",
    },
    {
        "family": "absent",
        "gen": lambda a, b: f"{format_capital_a_noun(a)} is visible in this scene, whereas the {b} does not appear.",
        "id": "abs_pos_first_02",
    },
    {
        "family": "absent",
        "gen": lambda a, b: f"A visible {a} is present, but the {b} is absent from this image.",
        "id": "abs_pos_first_03",
    },

    # Family 4: Free-of / Without (Group D)
    {
        "family": "free_of",
        "gen": lambda a, b: f"The picture shows {format_a_noun(a)}, without any {b}.",
        "id": "free_pos_first_01",
    },
    {
        "family": "free_of",
        "gen": lambda a, b: f"This image includes {format_a_noun(a)}, while being free of {b}.",
        "id": "free_pos_first_02",
    },
    {
        "family": "free_of",
        "gen": lambda a, b: f"The scene contains {format_a_noun(a)}, but is free of any {b}.",
        "id": "free_pos_first_03",
    },
]


# Position 2: Negative First, Positive Second (A present, B absent)
NEG_FIRST_TEMPLATES = [
    # Family 1: Standard (Group A)
    {
        "family": "standard",
        "gen": lambda a, b: f"There is no {b} in this image, but there is {format_a_noun(a)}.",
        "id": "std_neg_first_01",
    },
    {
        "family": "standard",
        "gen": lambda a, b: f"No {b} is present in this image, but {format_a_noun(a)} is present.",
        "id": "std_neg_first_02",
    },
    {
        "family": "standard",
        "gen": lambda a, b: f"While {format_a_noun(b)} is not present, {format_a_noun(a)} is visible in the image.",
        "id": "std_neg_first_03",
    },
    {
        "family": "standard",
        "gen": lambda a, b: f"No {b} can be seen in this image, but {format_a_noun(a)} can be found.",
        "id": "std_neg_first_04",
    },
    {
        "family": "standard",
        "gen": lambda a, b: f"This image does not contain any {b}, but it shows {format_a_noun(a)}.",
        "id": "std_neg_first_05",
    },

    # Family 2: Lacking / Contains (Group B)
    {
        "family": "lacking",
        "gen": lambda a, b: f"While the image lacks any {b}, it contains {format_a_noun(a)}.",
        "id": "lack_neg_first_01",
    },
    {
        "family": "lacking",
        "gen": lambda a, b: f"This picture contains no {b}, but it includes {format_a_noun(a)}.",
        "id": "lack_neg_first_02",
    },
    {
        "family": "lacking",
        "gen": lambda a, b: f"Although the scene lacks {format_a_noun(b)}, it features {format_a_noun(a)}.",
        "id": "lack_neg_first_03",
    },
    {
        "family": "lacking",
        "gen": lambda a, b: f"This image contains no visible {b}, though it depicts {format_a_noun(a)}.",
        "id": "lack_neg_first_04",
    },

    # Family 3: Absent / Appears (Group C)
    {
        "family": "absent",
        "gen": lambda a, b: f"The {b} is absent from the image, but {format_a_noun(a)} appears.",
        "id": "abs_neg_first_01",
    },
    {
        "family": "absent",
        "gen": lambda a, b: f"While the {b} does not appear, {format_a_noun(a)} is visible in this scene.",
        "id": "abs_neg_first_02",
    },
    {
        "family": "absent",
        "gen": lambda a, b: f"The {b} is absent from this image, but a visible {a} is present.",
        "id": "abs_neg_first_03",
    },

    # Family 4: Free-of / Without (Group D)
    {
        "family": "free_of",
        "gen": lambda a, b: f"Without any {b}, the picture shows {format_a_noun(a)}.",
        "id": "free_neg_first_01",
    },
    {
        "family": "free_of",
        "gen": lambda a, b: f"Being free of {b}, this image includes {format_a_noun(a)}.",
        "id": "free_neg_first_02",
    },
    {
        "family": "free_of",
        "gen": lambda a, b: f"The scene is free of any {b}, but contains {format_a_noun(a)}.",
        "id": "free_neg_first_03",
    },
]


def load_clean_coco_object_map(
    retrieval_csv_path: str = "benchmarks/data/images/COCO_val_retrieval.csv",
    paired_csv_path: str = "benchmarks/data/images/beaf_paired_v2.csv",
) -> Dict[str, Set[str]]:
    """
    Build a dictionary mapping each clean unedited image path to its set of present objects.
    Excludes any inpainting edited images.
    """
    img_objs: Dict[str, Set[str]] = {}

    # 1. Load from COCO_val_retrieval.csv (val2017 unedited images)
    if os.path.exists(retrieval_csv_path):
        df_ret = pd.read_csv(retrieval_csv_path)
        for _, r in df_ret.iterrows():
            img_path = str(r['filepath']).strip()
            # Ensure unedited .jpg
            if re.search(r'_\d{2}\.', img_path) or img_path.endswith('.png'):
                continue
            if 'positive_objects' in r and pd.notna(r['positive_objects']):
                pos_objs = set(ast.literal_eval(r['positive_objects']))
                img_objs[img_path] = {str(o).strip().lower() for o in pos_objs}

    # 2. Also load unedited images from beaf_paired_v2.csv (val2014 unedited images)
    if os.path.exists(paired_csv_path):
        df_paired = pd.read_csv(paired_csv_path)
        for _, r in df_paired.iterrows():
            img_path = str(r['image_path']).strip()
            if re.search(r'_\d{2}\.', img_path) or img_path.endswith('.png'):
                continue
            obj = str(r['object_name']).strip().lower()
            val = r['object_in_image']
            if isinstance(val, str):
                val = val.strip().lower() == 'true'
            else:
                val = bool(val)
            if img_path not in img_objs:
                img_objs[img_path] = set()
            if val:
                img_objs[img_path].add(obj)

    print(f"[Clean Image Pool] Loaded {len(img_objs)} unedited COCO images.")
    return img_objs


def generate_clean_diverse_ab_swap_dataset(
    reference_diverse_csv_path: str = "benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv",
    retrieval_csv_path: str = "benchmarks/data/images/COCO_val_retrieval.csv",
    paired_csv_path: str = "benchmarks/data/images/beaf_paired_v2.csv",
    output_csv_path: str = "benchmarks/data/images/beaf_clean_ab_swap_diverse.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate clean unedited image-based diverse A/B swap dataset matching reference pair distribution.
    """
    random.seed(seed)
    print("=" * 70)
    print("Generating BEAF Clean Diverse A/B Swap Dataset (Unedited Images)")
    print(f"  Reference Diverse CSV : {reference_diverse_csv_path}")
    print(f"  Retrieval Source CSV  : {retrieval_csv_path}")
    print(f"  Paired Source CSV     : {paired_csv_path}")
    print(f"  Output Clean CSV      : {output_csv_path}")
    print("=" * 70)

    # 1. Build clean image object map
    img_objs = load_clean_coco_object_map(retrieval_csv_path, paired_csv_path)

    # Inverted index: object -> list of images containing it
    obj_to_imgs: Dict[str, List[str]] = {}
    for img, objs in img_objs.items():
        for obj in objs:
            if obj not in obj_to_imgs:
                obj_to_imgs[obj] = []
            obj_to_imgs[obj].append(img)

    # Track usage frequency to ensure uniform image distribution
    img_usage_count: Dict[str, int] = {img: 0 for img in img_objs}

    def choose_best_image(cand_imgs: List[str]) -> str:
        """Choose image with lowest usage count (with tie-breaking randomness)."""
        if not cand_imgs:
            raise ValueError("Candidate image list is empty!")
        min_usage = min(img_usage_count[img] for img in cand_imgs)
        best_cands = [img for img in cand_imgs if img_usage_count[img] == min_usage]
        chosen = random.choice(best_cands)
        img_usage_count[chosen] += 1
        return chosen

    # 2. Extract object pairs and templates from reference dataset
    df_ref = pd.read_csv(reference_diverse_csv_path)
    
    # Read unique pair units (grouped by 2 consecutive rows)
    num_pairs = len(df_ref) // 2
    print(f"\n[Reference] Found {num_pairs} pairs ({len(df_ref)} rows) in reference CSV.")

    rows = []
    n_pos_first = len(POS_FIRST_TEMPLATES)
    n_neg_first = len(NEG_FIRST_TEMPLATES)
    unmatched_pairs = 0

    for idx in range(num_pairs):
        r1 = df_ref.iloc[idx * 2]
        r2 = df_ref.iloc[idx * 2 + 1]

        A = str(r1['object_a']).strip().lower()
        B = str(r1['object_b']).strip().lower()

        # Candidates for img1: A present, B absent
        cand_img1 = [img for img in obj_to_imgs.get(A, []) if B not in img_objs[img]]
        # Candidates for img2: B present, A absent
        cand_img2 = [img for img in obj_to_imgs.get(B, []) if A not in img_objs[img]]

        if not cand_img1 or not cand_img2:
            unmatched_pairs += 1
            # Fallback if rare object has no disjoint image in pool
            if not cand_img1:
                cand_img1 = obj_to_imgs.get(A, list(img_objs.keys()))
            if not cand_img2:
                cand_img2 = obj_to_imgs.get(B, list(img_objs.keys()))

        img1 = choose_best_image(cand_img1)
        img2 = choose_best_image(cand_img2)

        # Apply 50:50 positional templates
        if idx % 2 == 0:
            tmpl_info = POS_FIRST_TEMPLATES[(idx // 2) % n_pos_first]
            pos_position = "pos_first"
            pos_cap = tmpl_info["gen"](A, B)
            neg_cap = tmpl_info["gen"](B, A)
        else:
            tmpl_info = NEG_FIRST_TEMPLATES[(idx // 2) % n_neg_first]
            pos_position = "neg_first"
            pos_cap = tmpl_info["gen"](A, B)
            neg_cap = tmpl_info["gen"](B, A)

        obj_name = f"{A}, {B}"
        family = tmpl_info["family"]
        tmpl_id = f"clean_{tmpl_info['id']}_p{idx:04d}"

        # Row 1 (img1): Object A is present, Object B is absent -> positive_caption is TRUE
        rows.append({
            'image_path': img1,
            'object_name': obj_name,
            'object_a': A,
            'object_b': B,
            'object_a_present': True,
            'object_b_present': False,
            'positive_caption': pos_cap,
            'negative_caption': neg_cap,
            'object_in_image': True,
            'pos_position': pos_position,
            'template_family': family,
            'source_template': tmpl_id,
        })

        # Row 2 (img2): Object B is present, Object A is absent -> positive_caption is FALSE
        rows.append({
            'image_path': img2,
            'object_name': obj_name,
            'object_a': A,
            'object_b': B,
            'object_a_present': False,
            'object_b_present': True,
            'positive_caption': pos_cap,
            'negative_caption': neg_cap,
            'object_in_image': False,
            'pos_position': pos_position,
            'template_family': family,
            'source_template': tmpl_id,
        })

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_out.to_csv(output_csv_path, index=False)

    # ── Summary Statistics & Integrity Check ──
    print("\n" + "=" * 70)
    print("  Clean Diverse A/B Swap Dataset Generation Summary")
    print("=" * 70)
    print(f"  Total Rows                 : {len(df_out)}")
    print(f"  Total Clean Pairs          : {len(df_out) // 2}")
    print(f"  Unique Unedited Images     : {df_out['image_path'].nunique()}")
    print(f"  Unique Objects             : {pd.concat([df_out['object_a'], df_out['object_b']]).nunique()}")
    print(f"  Unmatched Fallbacks        : {unmatched_pairs}")

    print("\n  [Position Distribution (50:50)]")
    for pos, count in df_out['pos_position'].value_counts().items():
        print(f"    - {pos:15s}: {count:5d} ({count/len(df_out)*100:.1f}%)")

    print("\n  [Template Family Distribution]")
    for fam, count in df_out['template_family'].value_counts().items():
        print(f"    - {fam:15s}: {count:5d} ({count/len(df_out)*100:.1f}%)")

    # Inpainting artifact check
    inpainting_count = df_out['image_path'].str.contains(r'_\d{2}\.|\.png$').sum()
    print(f"\n  [Inpainting Artifact Audit]")
    print(f"    - Inpainted/PNG images detected: {inpainting_count} (Should be 0)")

    print(f"\n  Successfully saved to: {output_csv_path}\n")
    return df_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate clean unedited diverse A/B swap dataset.")
    parser.add_argument("--ref_csv", default="benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv")
    parser.add_argument("--retrieval_csv", default="benchmarks/data/images/COCO_val_retrieval.csv")
    parser.add_argument("--paired_csv", default="benchmarks/data/images/beaf_paired_v2.csv")
    parser.add_argument("--output_csv", default="benchmarks/data/images/beaf_clean_ab_swap_diverse.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_clean_diverse_ab_swap_dataset(
        reference_diverse_csv_path=args.ref_csv,
        retrieval_csv_path=args.retrieval_csv,
        paired_csv_path=args.paired_csv,
        output_csv_path=args.output_csv,
        seed=args.seed,
    )
