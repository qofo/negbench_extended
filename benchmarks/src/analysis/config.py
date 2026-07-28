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
