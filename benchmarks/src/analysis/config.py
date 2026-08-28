"""
CLIP Negation Representation & Mechanism Analysis Configuration.

This module provides data structures, domain enumerations, and fundamental
algebraic utility functions for analyzing multi-modal vision-language representation
geometry and negation processing mechanisms.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np


class PipelineStep(str, Enum):
    """Enumeration of key sequential transformation steps across the CLIP text pipeline."""
    EMBEDDING = "Step0_Embedding"
    LAYER12_RAW = "Step1_Layer12_Raw"
    LAYER12_LN = "Step2_Layer12_LN"
    PROJECTED_UNNORM = "Step3_Projected_Unnorm"
    FINAL_L2NORM = "Step4_Final_L2Norm"


class MetadataKey(str, Enum):
    """Enumeration of standardized metadata attributes in paired caption datasets."""
    IMAGE_PATH = "image_path"
    OBJECT_NAME = "object_name"
    OBJECT_IN_IMAGE = "object_in_image"
    SOURCE_TEMPLATE = "source_template"


@dataclass
class AnalysisConfig:
    """Hyperparameters and runtime parameters for representation analysis."""
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    target_token: str = "eot"
    csv_path: Optional[str] = None
    output_dir: str = "logs/analysis_modular/openai_vit_b32"
    max_samples: int = 60000
    image_root: str = ""
    batch_size: int = 256
    image_batch_size: int = 64
    seed: int = 42


@dataclass
class RetrievalConfig:
    """Hyperparameters for cross-modal image-text retrieval evaluation."""
    image_root: str
    output_dir: str
    device: str = "cpu"
    batch_size: int = 256
    image_batch_size: int = 64


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Project feature vectors onto the unit hyper-sphere via L2 normalization.

    Args:
        x (np.ndarray): Tensor of shape (..., D).
        eps (float): Numerical stability hyperparameter to prevent division by zero.

    Returns:
        np.ndarray: L2-normalized feature matrix of shape (..., D).
    """
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def batch_cosine_similarity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute element-wise cosine similarity between aligned feature vectors.

    Args:
        u (np.ndarray): Matrix of shape (N, D).
        v (np.ndarray): Matrix of shape (N, D).

    Returns:
        np.ndarray: Pairwise cosine similarity vector of shape (N,).
    """
    return np.sum(l2_normalize(u) * l2_normalize(v), axis=-1)


def batch_dot_product(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute unnormalized inner product between aligned feature vectors.

    Args:
        u (np.ndarray): Matrix of shape (N, D).
        v (np.ndarray): Matrix of shape (N, D).

    Returns:
        np.ndarray: Inner product vector of shape (N,).
    """
    return np.sum(u * v, axis=-1)


def batch_l2_distance(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean (L2) distance between aligned feature vectors.

    Args:
        u (np.ndarray): Matrix of shape (N, D).
        v (np.ndarray): Matrix of shape (N, D).

    Returns:
        np.ndarray: Euclidean distance vector of shape (N,).
    """
    return np.linalg.norm(u - v, axis=-1)


def to_bool(v: Optional[object], default: bool = False) -> bool:
    """
    Parse a boolean or string representation into a clean boolean value.

    Args:
        v: Input value (bool, str, int, None, etc.).
        default (bool): Fallback value if v is None or unrecognized.

    Returns:
        bool: Normalized boolean value.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "t", "yes", "y"):
            return True
        if s in ("false", "0", "f", "no", "n"):
            return False
    if v is None:
        return default
    return bool(v)


def coerce_bool_column(df, column: str = "object_in_image", default: bool = False):
    """
    Normalize a truthy CSV column to real booleans, in place.

    Thirteen entrypoints each inlined their own version of this, and they did not
    agree: most accepted only the literal ``"true"``, one also accepted ``"1"``,
    ``"t"`` and ``"yes"``. Today's CSVs are written as ``True``/``False`` so pandas
    types the column as bool and every variant happens to agree, but a CSV written
    as ``1``/``0`` would split the experiments into two incompatible readings of the
    same file. Routing all of them through :func:`to_bool` removes that fork.

    Args:
        df: DataFrame to modify in place.
        column: Column name; a no-op when the column is absent.
        default: Value for entries :func:`to_bool` cannot classify.

    Returns:
        The same DataFrame, for chaining.
    """
    if column in df.columns:
        df[column] = df[column].apply(lambda v: to_bool(v, default=default))
    return df


PRE_PROJECTION_KEY = "Pre-Projection"
FINAL_L2NORM_KEY = "+Final L2Norm"

_warned_layer_keys = set()


def get_layer_features(vis: dict, key: str) -> np.ndarray:
    """
    Extract intermediate representations by layer/step key from unified feature dictionaries.

    ``FINAL_L2NORM_KEY`` is the intended sentinel for the final embedding, and any
    other unrecognized key falls back to it too. That fallback used to be silent,
    which turned a typo'd or renamed layer into a "layerwise" curve that was really
    the same final layer repeated. It is kept so callers do not break, but it now
    warns once per unknown key so the substitution shows up in the run log.

    Args:
        vis (dict): Unified feature dictionary returned by vision/text extractor.
        key (str): Layer key (e.g. 'Layer 1', 'Pre-Projection', '+Final L2Norm').

    Returns:
        np.ndarray: Feature tensor corresponding to the specified layer/transformation step.
    """
    if "layers" in vis and key in vis["layers"]:
        return vis["layers"][key]
    if key == PRE_PROJECTION_KEY and "pre_proj" in vis:
        return vis["pre_proj"]
    if key != FINAL_L2NORM_KEY and key not in _warned_layer_keys:
        _warned_layer_keys.add(key)
        available = sorted(vis.get("layers", {}).keys())
        print(f"[WARNING] get_layer_features: unknown layer key {key!r}; returning 'final_l2norm' "
              f"instead. Available layer keys: {available}")
    return vis["final_l2norm"]


def set_seed(seed: int = 42) -> None:
    """
    Globally set random seeds across Python random, NumPy, and PyTorch (CPU & CUDA) for strict reproducibility.

    Args:
        seed (int): Random seed integer.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


DEFAULT_TUNING_GRIDS = {
    "logistic": {"C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    "ridge": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    "svm_linear": {"C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
    "svm_rbf": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto", 0.01, 0.1]},
    "mlp": {"hidden_dim": [32, 64, 128], "lr": [1e-2, 1e-3], "weight_decay": [1e-4, 1e-3]},
    "bilinear_lowrank": {"rank": [2, 4, 8, 16], "lr": [1e-2, 1e-3], "weight_decay": [1e-4, 1e-3]},
}


def filter_vision_dict(vis: dict, mask: np.ndarray) -> dict:
    """
    Apply a boolean mask to all arrays in a unified vision feature dictionary.

    Filters all ndarray entries in 'layers', 'pre_proj', 'final_l2norm', and any
    other ndarray values at the top level, so that new feature keys are never silently dropped.

    Args:
        vis (dict): Unified vision feature dictionary from extract_vision_features_unified.
        mask (np.ndarray): Boolean mask of shape (N,).

    Returns:
        dict: Filtered copy of the vision feature dictionary.
    """
    result = {}
    for key, val in vis.items():
        if key == "layers" and isinstance(val, dict):
            result["layers"] = {k: v[mask] for k, v in val.items()}
        elif isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == mask.shape[0]:
            result[key] = val[mask]
        else:
            result[key] = val
    return result

