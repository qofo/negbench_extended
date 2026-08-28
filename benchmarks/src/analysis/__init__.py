"""
CLIP Negation Representation & Mechanism Analysis Package (Modular Flat Architecture).
"""

from analysis.config import (
    PipelineStep,
    MetadataKey,
    AnalysisConfig,
    RetrievalConfig,
    to_bool,
    get_layer_features,
    l2_normalize,
    batch_cosine_similarity,
    batch_dot_product,
    batch_l2_distance,
    set_seed,
    DEFAULT_TUNING_GRIDS,
    filter_vision_dict,
)
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
    "to_bool",
    "get_layer_features",
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
