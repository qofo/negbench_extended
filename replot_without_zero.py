import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_vision_ablation_results_no_zero(results_path, output_dir):
    """
    Reads the existing vision_ablation_results.json and replots the graphs
    EXCLUDING the 'zero' condition to focus on Shuffle and Gaussian.
    """
    if not os.path.exists(results_path):
        print(f"Error: Could not find {results_path}")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    models = list(results.keys())
    
    # Exclude 'zero'
    conditions = ["original", "shuffle", "gaussian"]
    condition_labels = ["Original", "Shuffle Vision", "Gaussian Vision"]
    
    # Use consistent colors/hatches (removed red/zero)
    colors = ["#1f77b4", "#ff7f0e", "#9467bd"] 
    hatches = ["", "\\\\", "xx"]

    os.makedirs(output_dir, exist_ok=True)

    # ── Plot 1: Total Accuracy ──
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(models))
    width = 0.25

    for i, (cond, label, color, hatch) in enumerate(zip(conditions, condition_labels, colors, hatches)):
        accs = [results[m][cond]["total_accuracy"] for m in models]
        bars = ax.bar(
            x + (i - 1) * width, accs, width,
            label=label, color=color, alpha=0.85, hatch=hatch, edgecolor="white"
        )
        for bar, acc in zip(bars, accs):
            h = bar.get_height()
            ax.annotate(
                f"{acc:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=11)
    ax.set_ylabel("Out-of-Fold Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Vision Ablation Shortcut Diagnostic (Excluding 'Zero' Artifact)\n"
        "If ablated approx original -> scorer uses text-only shortcut",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    out_path1 = os.path.join(output_dir, "vision_ablation_total_accuracy_no_zero.png")
    plt.savefig(out_path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path1}")

    # ── Plot 2: Delta (Accuracy Drop) Heatmap ──
    delta_data = []
    for m in models:
        row = []
        for cond in ["shuffle", "gaussian"]:
            delta = results[m][cond]["total_accuracy"] - results[m]["original"]["total_accuracy"]
            row.append(delta)
        delta_data.append(row)

    delta_arr = np.array(delta_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Adjust vmin/vmax for better contrast without zero
    im = ax.imshow(delta_arr, cmap="RdYlGn", aspect="auto", vmin=-45, vmax=5)

    ax.set_xticks(range(2))
    ax.set_xticklabels(["Shuffle Vision", "Gaussian Vision"], fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    for i in range(len(models)):
        for j in range(2):
            val = delta_arr[i, j]
            color = "white" if abs(val) > 20 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    ax.set_title(
        "Accuracy Drop from Vision Ablation (Delta %)\n"
        "Green approx 0 -> TEXT SHORTCUT  |  Red < 0 -> Vision is used",
        fontsize=12, fontweight="bold"
    )
    plt.colorbar(im, ax=ax, label="Delta Accuracy (%)")
    plt.tight_layout()
    out_path2 = os.path.join(output_dir, "vision_ablation_delta_heatmap_no_zero.png")
    plt.savefig(out_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path2}")


if __name__ == "__main__":
    RESULTS_JSON = "logs/evaluation/vision_ablation_shortcut/vision_ablation_results.json"
    OUTPUT_DIR = "logs/evaluation/vision_ablation_shortcut"
    
    print("Generating updated plots (excluding 'zero' condition)...")
    plot_vision_ablation_results_no_zero(RESULTS_JSON, OUTPUT_DIR)
    print("Done!")
