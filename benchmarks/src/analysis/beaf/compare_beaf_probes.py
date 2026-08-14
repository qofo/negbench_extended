"""
Compare Multiple Probing Classifiers Script for BEAF.

Scans specified directories for beaf_{probe_type}_layerwise.json files
and generates comparative line plots (Validation Accuracy & Generalization Gap per Probe Classifier).
"""

import os
import glob
import json
import argparse
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compare_probe_results(input_dirs: List[str], output_dir: str) -> None:
    """Load JSON reports from multiple probe output directories and plot comparison."""
    os.makedirs(output_dir, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}

    for d in input_dirs:
        # Search for any beaf_*_layerwise.json or beaf_per_object_train_val_layerwise.json
        json_files = glob.glob(os.path.join(d, "beaf_*_layerwise.json"))
        for jf in json_files:
            bname = os.path.basename(jf)
            probe_name = bname.replace("beaf_", "").replace("_layerwise.json", "")
            if probe_name == "per_object_train_val":
                probe_name = "logistic_baseline"
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results[probe_name] = data
                print(f"  Loaded probe result: '{probe_name}' from {jf}")
            except Exception as ex:
                print(f"  [Warning] Failed to load {jf}: {ex}")

    if not results:
        print("  [Error] No valid probe JSON reports found!")
        return

    # Render Comparative Plots
    sample_key = next(iter(results))
    layer_names = list(results[sample_key].keys())
    x = np.arange(len(layer_names))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "p", "h"]

    # 1. Standalone Validation Accuracy Comparison Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    for idx, (probe_name, layer_data) in enumerate(results.items()):
        val_accs = [layer_data[lk]["val_acc_mean"] for lk in layer_names]
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(
            x,
            val_accs,
            marker=marker,
            color=color,
            lw=2.5,
            ms=7,
            label=f"{probe_name.upper()} (Val Acc %)"
        )

    if "Pre-Projection" in layer_names:
        pre_idx = layer_names.index("Pre-Projection")
        ax.axvline(x=pre_idx - 0.5, color="gray", ls=":", lw=1.5, alpha=0.7, label="Post-Transformer Transformations")

    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Vision Transformer Layer / Pipeline Step", fontsize=12, fontweight="bold")
    ax.set_title("Layerwise Probing Classifier Comparison (Validation Accuracy across Objects)", fontsize=13, fontweight="bold")
    ax.set_ylim(45, 95)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    out_val_png = os.path.join(output_dir, "beaf_probing_classifier_val_acc_comparison.png")
    plt.savefig(out_val_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  ✅ Val Acc Comparison Plot saved: {out_val_png}")

    # 2. Dual-Panel Plot: (a) Validation Accuracy, (b) Generalization Gap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 7))

    for idx, (probe_name, layer_data) in enumerate(results.items()):
        val_accs = [layer_data[lk]["val_acc_mean"] for lk in layer_names]
        gap_accs = [layer_data[lk].get("gap_mean", layer_data[lk].get("train_acc_mean", 0) - layer_data[lk].get("val_acc_mean", 0)) for lk in layer_names]
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        ax1.plot(x, val_accs, marker=marker, color=color, lw=2.2, ms=6, label=probe_name.upper())
        ax2.plot(x, gap_accs, marker=marker, color=color, lw=2.2, ms=6, label=probe_name.upper())

    for ax_curr, title_text, y_label, y_lim in [
        (ax1, "(a) Validation Accuracy (%) [Higher is Better]", "Validation Accuracy (%)", (45, 95)),
        (ax2, "(b) Generalization Gap (Train - Val Acc %) [Lower is Better]", "Gap (%)", (-5, 35)),
    ]:
        if "Pre-Projection" in layer_names:
            pre_idx = layer_names.index("Pre-Projection")
            ax_curr.axvline(x=pre_idx - 0.5, color="gray", ls=":", lw=1.5, alpha=0.7)

        ax_curr.set_xticks(x)
        ax_curr.set_xticklabels(layer_names, rotation=35, ha="right", fontsize=9)
        ax_curr.set_ylabel(y_label, fontsize=11, fontweight="bold")
        ax_curr.set_xlabel("Vision Transformer Layer / Pipeline Step", fontsize=11, fontweight="bold")
        ax_curr.set_title(title_text, fontsize=12, fontweight="bold")
        ax_curr.set_ylim(y_lim)
        ax_curr.grid(True, ls="--", alpha=0.4)
        ax_curr.legend(fontsize=9, loc="best")

    fig.suptitle("BEAF Multi-Classifier Probing Comprehensive Benchmark (Linear, Bilinear, Non-Linear)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_comp_png = os.path.join(output_dir, "beaf_probing_classifier_comparison.png")
    plt.savefig(out_comp_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Comprehensive 2-Panel Comparison Plot saved: {out_comp_png}")


def main():
    parser = argparse.ArgumentParser(description="Compare Multiple BEAF Probing Classifiers")
    parser.add_argument("--input_dirs", type=str, nargs="+", required=True, help="List of probe output directories")
    parser.add_argument("--output_dir", type=str, default="logs/evaluation/beaf_probe_comparison")
    args = parser.parse_args()

    print("=" * 60)
    print("  BEAF Multi-Classifier Probing Comparison")
    print("=" * 60)
    compare_probe_results(args.input_dirs, args.output_dir)


if __name__ == "__main__":
    main()

