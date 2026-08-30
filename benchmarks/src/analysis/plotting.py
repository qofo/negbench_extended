"""
Headless plotting backend and the figures more than one experiment draws.

Two things lived in copies:

1. **The backend.** Every plotting module repeated
   ``import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt``
   -- and four of them repeated only the last line. Those four worked by luck:
   pyplot picks a backend at import time and falls back to Agg when ``DISPLAY``
   is unset, so on a GPU node with X11 forwarding they would try to open a window
   instead. Importing ``plt`` from here selects Agg first, once.

2. **The top-objects grid.** Three modules each defined
   ``render_top_objects_grid`` with the same body -- 95% identical by normalized
   AST -- differing only in title, output filename, and, in one copy, where the
   legend sits. That last one is drift, not intent: the copies are one figure.
"""

import math
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

__all__ = ["plt", "render_top_objects_grid"]


def render_top_objects_grid(
    raw_df: pd.DataFrame,
    output_path: str,
    title: str,
    top_k: int = 16,
    cols: int = 4,
) -> Optional[str]:
    """
    Draw per-layer train/val accuracy for the ``top_k`` best-sampled objects.

    Expects the long-format per-object frame every layerwise probe emits: one row
    per (``object_name``, ``layer_name``) with ``n_pairs``, ``train_acc_pct`` and
    ``val_acc_pct``. Layers are drawn in first-appearance order, so the caller
    controls the x-axis by the order it wrote its rows.

    Args:
        raw_df: Per-object, per-layer accuracy frame.
        output_path: Full path of the PNG to write.
        title: Figure suptitle.
        top_k: How many objects to draw, ranked by ``n_pairs``.
        cols: Subplots per row.

    Returns:
        str: ``output_path``, or None when the frame has no objects to draw.
    """
    obj_counts = raw_df.groupby("object_name")["n_pairs"].first().sort_values(ascending=False)
    top_objects = obj_counts.head(top_k).index.tolist()
    if not top_objects:
        print("  [skipped] top-objects grid: no objects in the frame")
        return None

    rows = math.ceil(len(top_objects) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.5 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    layer_names = raw_df["layer_name"].unique().tolist()
    x = np.arange(len(layer_names))

    for i, obj in enumerate(top_objects):
        ax = axes[i]
        sub = raw_df[raw_df["object_name"] == obj].set_index("layer_name").reindex(layer_names)
        n_pairs = int(sub["n_pairs"].iloc[0])

        ax.plot(x, sub["train_acc_pct"], "o-", color="#1f77b4", lw=2, ms=5, label="Train Acc")
        ax.plot(x, sub["val_acc_pct"], "s--", color="#d62728", lw=2, ms=5, label="Val Acc")

        ax.set_title(f"{obj} (N={n_pairs} pairs)", fontsize=11, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.3)
        ax.set_ylim(35, 105)

        if i % cols == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=10)
        if i >= (rows - 1) * cols:
            ax.set_xticks(x)
            ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=8)

    for j in range(len(top_objects), len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=12)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path
