"""
BEAF Counterfactual A/B Swap Diverse Dataset Generator.

Generates a robust, position-balanced, lexically diverse A/B swap dataset using
templates from `beaf_expanded_templates.json`.

Key Enhancements:
1. Positional Invariance (50:50 Position Swap):
   - 50% Pos-First / Neg-Second: e.g. "There is a truck, but no train."
   - 50% Neg-First / Pos-Second: e.g. "There is no train, but there is a truck."
2. Lexical Diversity across 4 Template Families:
   - Standard (Group A): is present, can be seen, is shown, no X is present, does not contain
   - Lacking (Group B): contains X, includes X, lacks any Y, contains no Y
   - Absent (Group C): X is visible, X appears, Y is absent, Y does not appear
   - Free-of (Group D): shows X, depicts X, is free of Y, without any Y
3. Explicit Columns:
   - image_path, object_a, object_b, object_a_present, object_b_present,
     positive_caption, negative_caption, object_in_image,
     pos_position, template_family, source_template

Output:
    benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv
"""

import os
import re
import json
import pandas as pd
from typing import List, Dict, Tuple, Any

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
# Pattern: [Pos clause for A] + [Connective] + [Neg clause for B]
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
# Pattern: [Neg clause for B] + [Connective] + [Pos clause for A]
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


def get_base_id(path: str) -> str:
    """Extract COCO base image identifier from path."""
    m = re.search(r'(COCO_val2014_\d{12})', str(path))
    if m:
        return m.group(1)
    return str(path)


def generate_diverse_ab_swap_dataset(
    paired_csv_path: str = "benchmarks/data/images/beaf_paired_v2.csv",
    cf_csv_path: str = "benchmarks/data/images/beaf_counterfactual_6col.csv",
    output_csv_path: str = "benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv",
) -> pd.DataFrame:
    """
    Build position-balanced and lexically diverse counterfactual A/B swap dataset.
    """
    print("=" * 65)
    print("Generating BEAF Diverse A/B Swap Dataset")
    print(f"  Input Paired CSV : {paired_csv_path}")
    print(f"  Input CF CSV     : {cf_csv_path}")
    print(f"  Output CSV       : {output_csv_path}")
    print("=" * 65)

    df_paired = pd.read_csv(paired_csv_path)
    df_cf = pd.read_csv(cf_csv_path)

    # Build comprehensive object map per image
    img_objs = {}

    def _update_map(df):
        for _, r in df.iterrows():
            img = r['image_path']
            obj = str(r['object_name']).strip()
            val = r['object_in_image']
            if isinstance(val, str):
                val = val.strip().lower() == 'true'
            else:
                val = bool(val)
            if img not in img_objs:
                img_objs[img] = {}
            img_objs[img][obj] = val

    _update_map(df_paired)
    _update_map(df_cf)

    base_groups = {}
    for img in img_objs.keys():
        bid = get_base_id(img)
        if bid not in base_groups:
            base_groups[bid] = []
        base_groups[bid].append(img)

    all_swaps = []
    for bid, imgs in base_groups.items():
        for img1 in imgs:
            for img2 in imgs:
                if img1 == img2:
                    continue
                objs1 = img_objs[img1]
                objs2 = img_objs[img2]

                cand_A = [obj for obj, is_in in objs1.items() if is_in is True and objs2.get(obj) is False]
                cand_B = [obj for obj, is_in in objs1.items() if is_in is False and objs2.get(obj) is True]

                for A in cand_A:
                    for B in cand_B:
                        if A != B:
                            all_swaps.append({
                                'base_id': bid,
                                'img1': img1,
                                'img2': img2,
                                'A': A,
                                'B': B
                            })

    # Deduplicate symmetric pairs into canonical counterfactual pairs
    unique_pairs = []
    seen = set()

    for item in all_swaps:
        img1, img2, A, B = item['img1'], item['img2'], item['A'], item['B']
        key = tuple(sorted([(img1, A), (img2, B)]))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(item)

    print(f"\nExtracted {len(unique_pairs)} canonical counterfactual pairs across {len(base_groups)} scenes.")

    rows = []
    n_pos_first = len(POS_FIRST_TEMPLATES)
    n_neg_first = len(NEG_FIRST_TEMPLATES)

    for idx, p in enumerate(unique_pairs):
        img1 = p['img1']
        img2 = p['img2']
        A = p['A']
        B = p['B']

        # Alternate between Pos-First (50%) and Neg-First (50%)
        if idx % 2 == 0:
            tmpl_info = POS_FIRST_TEMPLATES[(idx // 2) % n_pos_first]
            pos_position = "pos_first"
            # pos_cap: A present, B absent (matches img1)
            pos_cap = tmpl_info["gen"](A, B)
            # neg_cap: B present, A absent (matches img2)
            neg_cap = tmpl_info["gen"](B, A)
        else:
            tmpl_info = NEG_FIRST_TEMPLATES[(idx // 2) % n_neg_first]
            pos_position = "neg_first"
            # pos_cap: B absent, A present (matches img1)
            pos_cap = tmpl_info["gen"](A, B)
            # neg_cap: A absent, B present (matches img2)
            neg_cap = tmpl_info["gen"](B, A)

        obj_name = f"{A}, {B}"
        family = tmpl_info["family"]
        tmpl_id = f"diverse_{tmpl_info['id']}_p{idx:04d}"

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

    # ── Summary Statistics ──
    print("\n" + "=" * 65)
    print("  Diverse A/B Swap Dataset Generation Summary")
    print("=" * 65)
    print(f"  Total Rows                 : {len(df_out)}")
    print(f"  Total Counterfactual Pairs : {len(df_out) // 2}")
    print(f"  Unique Objects             : {pd.concat([df_out['object_a'], df_out['object_b']]).nunique()}")
    print("\n  [Position Distribution (50:50)]")
    for pos, count in df_out['pos_position'].value_counts().items():
        print(f"    - {pos:15s}: {count:5d} ({count/len(df_out)*100:.1f}%)")
    print("\n  [Template Family Distribution]")
    for fam, count in df_out['template_family'].value_counts().items():
        print(f"    - {fam:15s}: {count:5d} ({count/len(df_out)*100:.1f}%)")

    print(f"\n  Successfully saved to: {output_csv_path}\n")
    return df_out


if __name__ == '__main__':
    generate_diverse_ab_swap_dataset()
