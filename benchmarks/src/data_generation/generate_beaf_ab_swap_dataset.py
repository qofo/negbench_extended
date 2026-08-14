"""
BEAF Counterfactual A/B Object Swap Dataset Generator.

Generates a 6-column CSV matching beaf_counterfactual_6col.csv format for cases where:
- Image 1 (img1): Object A is present, Object B is absent ("There is A, but no B").
- Image 2 (img2): Object B is present, Object A is absent ("There is B, but no A").
"""

import os
import re
import pandas as pd

def format_a_noun(obj: str) -> str:
    """Format noun with proper English article or zero article for uncountables/plurals."""
    obj_clean = obj.lower().strip()
    if obj_clean in ['broccoli', 'scissors', 'skis']:
        return obj_clean
    elif obj_clean[0] in 'aeiou':
        return f"an {obj_clean}"
    else:
        return f"a {obj_clean}"

def get_base_id(path: str) -> str:
    """Extract COCO base image identifier from path."""
    m = re.search(r'(COCO_val2014_\d{12})', path)
    if m:
        return m.group(1)
    return path

def generate_ab_swap_dataset(
    paired_csv_path: str,
    cf_csv_path: str,
    output_csv_path: str,
):
    """Generate the counterfactual A/B object swap dataset."""
    df_paired = pd.read_csv(paired_csv_path)
    df_cf = pd.read_csv(cf_csv_path)

    # Build comprehensive object map per image
    img_objs = {}

    for _, r in df_paired.iterrows():
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

    for _, r in df_cf.iterrows():
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

    rows = []
    for idx, p in enumerate(unique_pairs):
        img1 = p['img1']
        img2 = p['img2']
        A = p['A']
        B = p['B']

        pos_cap = f"There is {format_a_noun(A)} in this image, but no {B}."
        neg_cap = f"There is {format_a_noun(B)} in this image, but no {A}."
        obj_name = f"{A}, {B}"
        tmpl = f"ab_swap_t{idx:04d}"

        # Row 1 (img1): object_in_image = True (for pos_cap)
        rows.append({
            'image_path': img1,
            'object_name': obj_name,
            'positive_caption': pos_cap,
            'negative_caption': neg_cap,
            'object_in_image': True,
            'source_template': tmpl
        })
        # Row 2 (img2): object_in_image = False (for pos_cap)
        rows.append({
            'image_path': img2,
            'object_name': obj_name,
            'positive_caption': pos_cap,
            'negative_caption': neg_cap,
            'object_in_image': False,
            'source_template': tmpl
        })

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_out.to_csv(output_csv_path, index=False)
    print(f"Successfully generated {len(df_out)} rows ({len(unique_pairs)} pairs) at: {output_csv_path}")
    return df_out

if __name__ == '__main__':
    paired_csv = r"benchmarks/data/images/beaf_paired_v2.csv"
    cf_csv = r"benchmarks/data/images/beaf_counterfactual_6col.csv"
    out_csv = r"benchmarks/data/images/beaf_counterfactual_ab_swap.csv"
    generate_ab_swap_dataset(paired_csv, cf_csv, out_csv)
