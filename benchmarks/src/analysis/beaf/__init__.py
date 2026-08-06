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
    compute_vision_non_linear_probe,
    compute_vision_direction_preservation,
)

__all__ = [
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
    "extract_vision_features_unified",
    "compute_vision_pipeline_breakdown",
    "compute_vision_svd_sweep",
    "compute_vision_linear_probe",
    "compute_vision_non_linear_probe",
    "compute_vision_direction_preservation",
]
