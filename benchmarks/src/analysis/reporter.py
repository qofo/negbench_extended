"""
Visualization and Experimental Artifact Serialization Engine.

This module provides the AnalysisReporter class responsible for rendering
publication-grade figures (PNGs) and serializing structured JSON/CSV experimental logs.
"""

import os
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class AnalysisReporter:
    """
    Renders empirical visualization charts and serializes experimental logs.

    Attributes:
        output_dir (str): Target directory path for storing generated artifacts.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the reporter and guarantee output directory existence.

        Args:
            output_dir (str): Destination directory path.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def render_pipeline_breakdown(self, data: Dict[str, Any]):
        """
        Render dual-axis line plot demonstrating geometric shift across 16 pipeline steps.

        Args:
            data (Dict[str, Any]): Step-wise pipeline and layer breakdown statistics.
        """
        print("\n" + "="*60)
        print("Stage 1-A: Full 16-Step Multi-Metric Pipeline & Layer Breakdown Analysis")
        print("="*60)

        for row in data["pipeline"]:
            print(f"  [{row['step_name']:20s}] Cosine Sim: {row['mean_cosine_sim']:.4f} | Dot Prod: {row['mean_dot_product']:.2f} | L2 Dist: {row['mean_l2_distance']:.4f}")

        df_pipeline = pd.DataFrame(data["pipeline"])
        df_pipeline.to_csv(os.path.join(self.output_dir, "full_pipeline_step_breakdown.csv"), index=False)

        df_layer = pd.DataFrame(data["layers"])
        df_layer.to_csv(os.path.join(self.output_dir, "layerwise_cosine_breakdown.csv"), index=False)

        fig, ax1 = plt.subplots(figsize=(12, 6))
        x_labels = df_pipeline["step_name"].values
        means_cos = df_pipeline["mean_cosine_sim"].values

        ax1.plot(x_labels, means_cos, "o-", color="crimson", lw=2.5, ms=7, label="Mean Cosine Sim")
        ax1.set_ylabel("Cosine Similarity", color="crimson", fontsize=11)
        ax1.tick_params(axis="y", labelcolor="crimson")
        ax1.set_title("Full Sequence Pipeline Breakdown: Representation Geometry Shift Across All Layers & Projection", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        ax1.grid(True, ls="--", alpha=0.5)

        ax2 = ax1.twinx()
        means_l2 = df_pipeline["mean_l2_distance"].values
        ax2.plot(x_labels, means_l2, "s--", color="dodgerblue", lw=2, ms=6, label="Mean L2 Distance")
        ax2.set_ylabel("L2 Distance", color="dodgerblue", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="dodgerblue")

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "pipeline_step_lineplot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    def render_direction_preservation(self, report: Dict[str, Any]):
        """
        Render comparative histogram of distance ratios for negation vs control random pairs.

        Args:
            report (Dict[str, Any]): Distance ratio distributions and Welch's t-test metrics.
        """
        print("\n" + "="*60)
        print("Stage 1-B: Direction Preservation Analysis (Negation vs Random Control)")
        print("="*60)

        print(f"Negation Pairs  : Pre Dist={report['negation_mean_dist_pre']:.4f} -> Post={report['negation_mean_dist_post']:.4f} (Ratio={report['negation_mean_ratio']:.4f})")
        print(f"Control Pairs   : Pre Dist={report['control_mean_dist_pre']:.4f} -> Post={report['control_mean_dist_post']:.4f} (Ratio={report['control_mean_ratio']:.4f})")
        print(f"Welch's T-test  : t={report['ttest_t_stat']:.4f}, p-value={report['ttest_p_value']:.2e}")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(report["ratio_neg"], bins=35, alpha=0.6, color="crimson", edgecolor="black", label=f"Negation Pairs (Mean: {report['negation_mean_ratio']:.4f})")
        ax.hist(report["ratio_ctrl"], bins=35, alpha=0.6, color="gray", edgecolor="black", label=f"Control Random Pairs (Mean: {report['control_mean_ratio']:.4f})")
        ax.set_title(f"Direction Preservation: Negation vs Control Pairs (p={report['ttest_p_value']:.1e})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Distance Ratio (Post-Proj Dist / Pre-Proj Dist)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, ls="--", alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, "direction_preservation_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        serializable_report = {k: v for k, v in report.items() if k not in ["ratio_neg", "ratio_ctrl"]}
        rpt_path = os.path.join(self.output_dir, "direction_preservation_report.json")
        with open(rpt_path, "w", encoding="utf-8") as f:
            json.dump(serializable_report, f, indent=2)

    def render_linear_probe(self, results: Dict[str, Any]):
        """
        Render bar chart displaying cross-validated linear probing accuracies pre vs post projection.

        Args:
            results (Dict[str, Any]): Overall probing accuracies and sub-dataset template breakdowns.
        """
        print("\n" + "="*60)
        print("Stage 1-C: Linear Probe & Sub-Dataset Template Shortcut Analysis")
        print("="*60)

        for slabel, info in results["overall_probe"].items():
            print(f"  [{slabel:22s}] Linear Probe Accuracy: {info['mean_accuracy']:.2f}% (±{info['std_accuracy']:.2f}%)")

        if results["template_breakdown"]:
            print("\n  --- Sub-dataset Breakdown by source_template ---")
            for tmpl, info in results["template_breakdown"].items():
                print(f"    [{tmpl:25s}] (N={info['sample_count']:5d}) Probe Acc: {info['linear_probe_accuracy_pct']:6.2f}% (±{info['linear_probe_accuracy_std_pct']:.1f}%) | Cosine Sim: {info['mean_cosine_sim']:.4f}")

        report_path = os.path.join(self.output_dir, "linear_probe_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(results["overall_probe"].keys(), [v["mean_accuracy"] for v in results["overall_probe"].values()],
                      color=["gray", "seagreen", "crimson"], alpha=0.85, edgecolor="black")
        ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=11)
        ax.set_title("Linear Probe: Separability Pre vs Post Projection", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", ls="--", alpha=0.3)

        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "linear_probe_accuracy.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    def render_pca_spectrum(self, report: Dict[str, Any]):
        """
        Render PCA variance spectrum and serialize intrinsic dimensionality metrics.

        Args:
            report (Dict[str, Any]): Effective rank and PCA variance ratio metrics.
        """
        print("\n" + "="*60)
        print("Stage 1-D: Intrinsic Dimensionality & Negation Subspace Geometry")
        print("="*60)

        print(f"Full Space Pre-Proj (Layer12+LN) : Eff Rank={report['pre_effective_rank']:.2f}, PR={report['pre_participation_ratio']:.2f}, PC1={report['var_pre'][0]*100:.2f}%")
        print(f"Full Space Post-Proj (Final L2)  : Eff Rank={report['post_effective_rank']:.2f}, PR={report['post_participation_ratio']:.2f}, PC1={report['var_post'][0]*100:.2f}%")
        print(f"Negation Diff Subspace Pre-Proj  : Eff Rank={report['diff_subspace_pre_effective_rank']:.2f}")
        print(f"Negation Diff Subspace Post-Proj : Eff Rank={report['diff_subspace_post_effective_rank']:.2f}")

        fig, ax = plt.subplots(figsize=(8, 5))
        n_comp = len(report["var_pre"])
        indices = np.arange(1, n_comp + 1)
        ax.plot(indices, np.array(report["var_pre"]) * 100, "o-", color="seagreen", lw=2, label=f"Pre-Projection (r_eff={report['pre_effective_rank']:.1f}, PR={report['pre_participation_ratio']:.1f})")
        ax.plot(indices, np.array(report["var_post"]) * 100, "s-", color="crimson", lw=2, label=f"Post-Projection (r_eff={report['post_effective_rank']:.1f}, PR={report['post_participation_ratio']:.1f})")
        ax.set_xlabel("Principal Component Index", fontsize=11)
        ax.set_ylabel("Explained Variance Ratio (%)", fontsize=11)
        ax.set_title("PCA Variance Spectrum & Intrinsic Dimensionality", fontsize=11, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=10)
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, "pca_spectrum_compression.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        serializable = {k: v for k, v in report.items() if k not in ["var_pre", "var_post"]}
        rpt_path = os.path.join(self.output_dir, "pca_spectrum_report.json")
        with open(rpt_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def render_svd_ablation(self, svd_report: Dict[str, Any]):
        """
        Render SVD 10%-90% singular value truncation sweep curve.

        Args:
            svd_report (Dict[str, Any]): Singular vectors, alignment scores, and sweep curves.
        """
        if not svd_report:
            print("Model does not have a text_projection matrix. Skipping SVD Ablation.")
            return

        print("\n" + "="*60)
        print(f"Stage 3: Projection SVD & Singular Value Spectrum Sweep (Target Token: '{svd_report['target_token']}')")
        print("="*60)

        print(f"  SVD Singular Values S (Top 5): {np.array(svd_report['singular_values_top10'][:5]).round(3)}")
        print(f"  Negation Direction Alignment with Top Singular Vector V1: {svd_report['top1_alignment']:.4f}")
        print(f"  Max Alignment: {svd_report['max_alignment']:.4f} (with Singular Vector #{svd_report['max_alignment_singular_vector_idx']+1})")

        print("\n  --- SVD Spectrum Sweep Results ---")
        for sr in svd_report["spectrum_sweep"]:
            print(f"    Keep Ratio {sr['keep_ratio']*100:2.0f}% ({sr['k_singular_values']:3d} SVs) | Top-k Sim: {sr['cosine_sim_top_k']:.4f} | Bottom-k Sim: {sr['cosine_sim_bottom_k']:.4f}")

        fig, ax = plt.subplots(figsize=(8, 5))
        pcts = [sr["keep_ratio"] * 100 for sr in svd_report["spectrum_sweep"]]
        sim_tops = [sr["cosine_sim_top_k"] for sr in svd_report["spectrum_sweep"]]
        sim_bots = [sr["cosine_sim_bottom_k"] for sr in svd_report["spectrum_sweep"]]

        ax.plot(pcts, sim_tops, "o-", color="crimson", lw=2, label="Keep Top-k Singular Values")
        ax.plot(pcts, sim_bots, "s--", color="dodgerblue", lw=2, label="Keep Bottom-k Singular Values")
        ax.axhline(svd_report["cosine_sim_original"], color="black", ls=":", label=f"Original W_proj Sim ({svd_report['cosine_sim_original']:.4f})")
        ax.set_xlabel("Singular Values Retained (%)", fontsize=11)
        ax.set_ylabel("Final Cosine Similarity", fontsize=11)
        ax.set_title("SVD Spectrum Sweep: Top-k vs Bottom-k Singular Value Truncation", fontsize=12, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=10)
        plt.tight_layout()

        sweep_plot_path = os.path.join(self.output_dir, "projection_svd_spectrum_sweep.png")
        plt.savefig(sweep_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        svd_path = os.path.join(self.output_dir, "projection_svd_report.json")
        with open(svd_path, "w", encoding="utf-8") as f:
            json.dump(svd_report, f, indent=2)

    def render_retrieval_metrics(self, data: Dict[str, Any]):
        """
        Serialize cross-modal retrieval summary JSON and detailed pair similarities CSV.

        Args:
            data (Dict[str, Any]): Retrieval summary dictionary and pair pandas DataFrame.
        """
        if not data or data.get("results_df") is None:
            print("No valid images processed for Retrieval Metrics.")
            return

        res_df = data["results_df"]
        summary = data["summary"]
        skipped_count = data["skipped_count"]

        res_df.to_csv(os.path.join(self.output_dir, "image_text_similarity.csv"), index=False)

        sum_path = os.path.join(self.output_dir, "retrieval_metrics_summary.json")
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n=== Retrieval Metrics Summary ===")
        print(f"  Evaluated Pairs                     : {summary['total_pairs_evaluated']} (Skipped: {skipped_count})")
        print(f"  Pearson r                           : {summary['pearson_r']:.4f}")
        print(f"  [Object Present] Pos Caption Acc    : {summary['object_present_subgroup']['positive_caption_accuracy_pct']:.1f}% (Flip: {summary['object_present_subgroup']['ranking_flip_rate_pct']:.1f}%, Tie: {summary['object_present_subgroup']['tie_rate_pct']:.1f}%)")
        print(f"  [Object Absent ] Neg Caption Acc    : {summary['object_absent_subgroup']['negative_caption_accuracy_pct']:.1f}% (Pos Preference: {summary['object_absent_subgroup']['positive_caption_flip_rate_pct']:.1f}%, Tie: {summary['object_absent_subgroup']['tie_rate_pct']:.1f}%)")

    def render_layerwise_pca_grid(
        self,
        pos_features: Dict[str, Any],
        neg_features: Dict[str, Any],
        target_token: str = "eot"
    ):
        """
        Render 2D Principal Component Analysis (PCA) scatter plots across all 13 layer representations.

        Args:
            pos_features (Dict[str, Any]): Positive caption features.
            neg_features (Dict[str, Any]): Negative caption features.
            target_token (str): Pooling strategy label.
        """
        print("\n" + "="*60)
        print(f"Layer-wise PCA Grid Visualization (Target Token: '{target_token}')")
        print("="*60)

        layer_names = list(pos_features["layers"].keys())
        num_layers = len(layer_names)
        n_pos = len(pos_features["layers"][layer_names[0]])

        cols = min(4, num_layers)
        rows = (num_layers + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
        if num_layers == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        analysis_report = [
            "=== CLIP Text Encoder Layer-wise PCA Analysis Report ===",
            f"Target Token Strategy: {target_token}",
            f"Total Layers analyzed: {num_layers}\n"
        ]

        for l_idx, l_name in enumerate(layer_names):
            pos_f = pos_features["layers"][l_name]
            neg_f = neg_features["layers"][l_name]

            combined = np.vstack([pos_f, neg_f])
            pca = PCA(n_components=min(combined.shape[0], combined.shape[1], 2))
            combined_pca = pca.fit_transform(combined)

            pos_pca = combined_pca[:n_pos]
            neg_pca = combined_pca[n_pos:]

            var_ratio = pca.explained_variance_ratio_
            total_var_2d = float(np.sum(var_ratio[:2])) if len(var_ratio) >= 2 else float(np.sum(var_ratio))

            pos_mean_orig = pos_f.mean(axis=0)
            neg_mean_orig = neg_f.mean(axis=0)
            centroid_dist_orig = float(np.linalg.norm(pos_mean_orig - neg_mean_orig))

            pos_mean_pca = pos_pca.mean(axis=0)
            neg_mean_pca = neg_pca.mean(axis=0)

            report_str = (f"[{l_name}] 2D Explained Variance: {total_var_2d*100:.2f}% "
                          f"(PC1: {var_ratio[0]*100:.1f}%, PC2: {var_ratio[1]*100:.1f}%) | "
                          f"Group Centroid Dist (Orig Dim): {centroid_dist_orig:.4f}")
            analysis_report.append(report_str)

            ax = axes[l_idx]
            ax.scatter(pos_pca[:, 0], pos_pca[:, 1], c="dodgerblue", label="Positive", alpha=0.75, edgecolors="k", linewidth=0.5, s=40)
            ax.scatter(neg_pca[:, 0], neg_pca[:, 1], c="crimson", label="Negative", alpha=0.75, edgecolors="k", linewidth=0.5, marker="^", s=40)
            ax.scatter(pos_mean_pca[0], pos_mean_pca[1], c="navy", s=120, marker="X", label="Pos Centroid", edgecolors="w")
            ax.scatter(neg_mean_pca[0], neg_mean_pca[1], c="darkred", s=120, marker="X", label="Neg Centroid", edgecolors="w")

            ax.set_title(f"{l_name}\n(Var: {total_var_2d*100:.1f}%)", fontsize=11, fontweight="bold")
            ax.set_xlabel("PC 1", fontsize=9)
            ax.set_ylabel("PC 2", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)

            if l_idx == 0:
                ax.legend(fontsize=8, loc="best")

        for l_idx in range(num_layers, len(axes)):
            fig.delaxes(axes[l_idx])

        plt.tight_layout()
        plot_filename = os.path.join(self.output_dir, f"pca_grid_{target_token}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close()

        report_filename = os.path.join(self.output_dir, f"pca_report_{target_token}.txt")
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(analysis_report))
