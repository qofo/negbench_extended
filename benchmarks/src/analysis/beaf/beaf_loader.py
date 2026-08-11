"""
BEAF Counterfactual Dataset Loader Module.

Provides data loading, path resolution, and strict pair integrity
verification for beaf_counterfactual_6col.csv datasets.
"""

import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from analysis.config import MetadataKey


def load_beaf_csv(csv_path: str, image_root: str) -> Tuple[pd.DataFrame, List[dict]]:
    """Load beaf_counterfactual_6col.csv and resolve absolute image paths for Axis 1-4."""
    df = pd.read_csv(csv_path)
    if image_root:
        df["abs_image_path"] = df["image_path"].apply(lambda p: os.path.join(image_root, p))
    else:
        df["abs_image_path"] = df["image_path"]

    def _to_bool(v) -> Optional[bool]:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return None

    df["object_in_image"] = df["object_in_image"].apply(_to_bool)

    pair_metadata = []
    for _, row in df.iterrows():
        pair_metadata.append({
            MetadataKey.IMAGE_PATH.value:      row["image_path"],
            MetadataKey.OBJECT_NAME.value:     str(row.get("object_name", "")),
            MetadataKey.OBJECT_IN_IMAGE.value: row["object_in_image"],
            MetadataKey.SOURCE_TEMPLATE.value: str(row.get("source_template", "")),
        })

    return df, pair_metadata


def load_and_verify_counterfactual_pairs(
    csv_path: str,
    image_root: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    """Load beaf_counterfactual_6col.csv, group by source_template, and enforce strict pairing integrity.

    Each consecutive row pair must satisfy:
      - Exactly one row has object_in_image=True (orig), the other False (cf).
      - Both rows share the same object_name and source_template.

    Returns:
        (df_raw, df_pairs, pair_metadata)
          df_raw       : full original DataFrame
          df_pairs     : paired DataFrame with columns:
                         pair_id, source_template, object_name,
                         orig_path, cf_path, positive_caption, negative_caption
          pair_metadata: list of per-row metadata dicts (MetadataKey)
    """
    df = pd.read_csv(csv_path)

    def _resolve_path(p: str, root: str) -> str:
        """Resolve a relative image path against image_root with multiple fallback strategies."""
        p_str = str(p).strip()
        if os.path.exists(p_str):
            return p_str
        if root:
            candidate1 = os.path.join(root, p_str)
            if os.path.exists(candidate1):
                return candidate1
            if p_str.startswith("data/images/"):
                candidate2 = os.path.join(root, p_str[len("data/images/"):])
                if os.path.exists(candidate2):
                    return candidate2
            if p_str.startswith("data/"):
                candidate3 = os.path.join(root, p_str[len("data/"):])
                if os.path.exists(candidate3):
                    return candidate3
        return os.path.join(root, p_str) if root else p_str

    def _to_bool(v) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return False

    df["abs_image_path"] = df["image_path"].apply(lambda p: _resolve_path(p, image_root))
    df["object_in_image"] = df["object_in_image"].apply(_to_bool)

    pair_metadata = []
    for _, row in df.iterrows():
        pair_metadata.append({
            MetadataKey.IMAGE_PATH.value:      row["image_path"],
            MetadataKey.OBJECT_NAME.value:     str(row.get("object_name", "")),
            MetadataKey.OBJECT_IN_IMAGE.value: row["object_in_image"],
            MetadataKey.SOURCE_TEMPLATE.value: str(row.get("source_template", "")),
        })

    pairs = []
    num_pairs = len(df) // 2
    for i in range(num_pairs):
        row1 = df.iloc[2 * i]
        row2 = df.iloc[2 * i + 1]

        b1 = row1["object_in_image"]
        b2 = row2["object_in_image"]

        assert (b1 and not b2) or (not b1 and b2), (
            f"Row pair {i} object_in_image mismatch: {b1}, {b2}"
        )

        orig_row = row1 if b1 else row2
        cf_row   = row2 if b1 else row1

        # Strict Assertion Checks
        assert orig_row["object_in_image"] == True, (
            f"Orig row for pair {i} must have object_in_image == True"
        )
        assert cf_row["object_in_image"] == False, (
            f"CF row for pair {i} must have object_in_image == False"
        )
        assert str(orig_row.get("object_name")) == str(cf_row.get("object_name")), (
            f"Object name mismatch in pair {i}"
        )
        assert str(orig_row.get("source_template")) == str(cf_row.get("source_template")), (
            f"Source template mismatch in pair {i}"
        )

        pairs.append({
            "pair_id":          i,
            "source_template":  str(orig_row.get("source_template", "")),
            "object_name":      str(orig_row.get("object_name", "")),
            "orig_path":        orig_row["abs_image_path"],
            "cf_path":          cf_row["abs_image_path"],
            "positive_caption": str(orig_row["positive_caption"]),
            "negative_caption": str(orig_row["negative_caption"]),
        })

    df_pairs = pd.DataFrame(pairs)
    print(
        f"  ✅ [Unified Pairing Verified] Extracted all {len(df_pairs)} exact counterfactual pairs "
        f"with 100% strict assertion checks."
    )
    return df, df_pairs, pair_metadata
