"""
CLIP Negation Representation & Mechanism Analysis Package (Modular Flat Architecture).
"""

from .config import (
    PipelineStep,
    MetadataKey,
    AnalysisConfig,
    RetrievalConfig,
    PRE_PROJECTION_KEY,
    FINAL_L2NORM_KEY,
    EMBEDDING_KEY,
    layer_key,
    to_bool,
    coerce_bool_column,
    get_layer_features,
    l2_normalize,
    batch_cosine_similarity,
    batch_dot_product,
    batch_l2_distance,
    set_seed,
    DEFAULT_TUNING_GRIDS,
    filter_vision_dict,
)
from .paths import resolve_image_path
from .feature_cache import (
    cached_encode,
    make_cache_key,
    build_provenance,
    inherit_upstream_provenance,
    get_git_commit,
    load_object_restriction,
    DEFAULT_CACHE_DIR,
)

__all__ = [
    "PipelineStep",
    "MetadataKey",
    "AnalysisConfig",
    "RetrievalConfig",
    "PRE_PROJECTION_KEY",
    "FINAL_L2NORM_KEY",
    "EMBEDDING_KEY",
    "layer_key",
    "to_bool",
    "coerce_bool_column",
    "get_layer_features",
    "resolve_image_path",
    "l2_normalize",
    "batch_cosine_similarity",
    "batch_dot_product",
    "batch_l2_distance",
    "set_seed",
    "DEFAULT_TUNING_GRIDS",
    "filter_vision_dict",
    "cached_encode",
    "make_cache_key",
    "build_provenance",
    "inherit_upstream_provenance",
    "get_git_commit",
    "load_object_restriction",
    "DEFAULT_CACHE_DIR",
]
