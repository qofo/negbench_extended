"""
BEAF (Benchmark Evaluation & Analysis Framework) Subpackage.
"""

from analysis.beaf.visualizer import (
    render_image_image_histogram,
    render_4way_heatmap,
    render_text_vs_visual_scatter,
    render_full_correct_by_object,
    render_scatter_pos_vs_neg,
    render_scatter_delta_quadrant,
    render_scatter_img_orig_vs_img_cf,
    render_scatter_by_object_category,
    render_2x2_factorial_anova_plots,
    render_2d_margin_state_space,
)
from analysis.beaf.vision_mechanisms import (
    extract_vision_features_unified,
    compute_vision_pipeline_breakdown,
    compute_vision_svd_sweep,
    compute_vision_linear_probe,
    compute_vision_direction_preservation,
)
from analysis.beaf.object_experiment import (
    format_object_name,
    instantiate_templates,
    get_balanced_beaf_object_df,
    run_single_object_analysis,
    evaluate_text_linear_probe_single_object,
    evaluate_text_linear_probe_cross_object,
    run_leave_one_object_out_text_probe_experiment,
    evaluate_unseen_template_group_text_probe,
    train_eval_vision_linear_probe,
    train_eval_vision_high_order_probe,
    VisionProbeWrapper,
    evaluate_dual_classifier_product_scorer,
    run_single_object_train_val_experiment,
)
from analysis.beaf.beaf_stats import (
    compute_2x2_factorial_anova,
    compute_quadrant_bootstrap_ci,
)
from analysis.beaf.beaf_loader import (
    load_beaf_csv,
    load_and_verify_counterfactual_pairs,
)

__all__ = [
    # Visualizers
    "render_image_image_histogram",
    "render_4way_heatmap",
    "render_text_vs_visual_scatter",
    "render_full_correct_by_object",
    "render_scatter_pos_vs_neg",
    "render_scatter_delta_quadrant",
    "render_scatter_img_orig_vs_img_cf",
    "render_scatter_by_object_category",
    "render_2x2_factorial_anova_plots",
    "render_2d_margin_state_space",
    # Vision Mechanisms
    "extract_vision_features_unified",
    "compute_vision_pipeline_breakdown",
    "compute_vision_svd_sweep",
    "compute_vision_linear_probe",
    "compute_vision_direction_preservation",
    # Object Experiments
    "format_object_name",
    "instantiate_templates",
    "get_balanced_beaf_object_df",
    "run_single_object_analysis",
    "evaluate_text_linear_probe_single_object",
    "evaluate_text_linear_probe_cross_object",
    "run_leave_one_object_out_text_probe_experiment",
    "evaluate_unseen_template_group_text_probe",
    "train_eval_vision_linear_probe",
    "train_eval_vision_high_order_probe",
    "VisionProbeWrapper",
    "evaluate_dual_classifier_product_scorer",
    "run_single_object_train_val_experiment",
    # Statistics
    "compute_2x2_factorial_anova",
    "compute_quadrant_bootstrap_ci",
    # Data Loading
    "load_beaf_csv",
    "load_and_verify_counterfactual_pairs",
]
