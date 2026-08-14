"""
Compare Multiple Probing Classifiers Script for BEAF.

Scans specified directories for beaf_{probe_type}_layerwise.json files
and generates comparative line plots (Validation Accuracy per Probe Classifier).
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
        # Search for any beaf_*_layerwise.json
        json_files = glob.glob(os.path.join(d, "beaf_*_layerwise.json"))
        for jf in json_files:
            bname = os.path.basename(jf)
            probe_name = bname.replace("beaf_", "").replace("_layerwise.json", "")
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

    # Render Comparative Plot
    sample_key = next(iter(results))
    layer_names = list(results[sample_key].keys())
    x = np.arange(len(layer_names))

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "P"]

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

    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Vision Transformer Layer / Pipeline Step", fontsize=12, fontweight="bold")
    ax.set_title("Layerwise Linear Probe Classifier Algorithm Comparison (Validation Accuracy)", fontsize=13, fontweight="bold")
    ax.set_ylim(45, 80)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=11, loc="lower right")

    plt.tight_layout()
    out_png = os.path.join(output_dir, "beaf_probing_classifier_comparison.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n  ✅ Classifier Comparison Plot saved: {out_png}")


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
