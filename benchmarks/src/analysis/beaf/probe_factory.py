"""
Linear Probe Classifier Factory Module.

Provides a unified factory interface to create and tune various linear probing
classifiers (Logistic Regression, Linear SVM, Ridge Classifier, SGD Classifier).
"""

from typing import Any, List, Dict, Tuple
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC


SUPPORTED_PROBES = ["logistic", "svm_linear", "ridge", "sgd_log", "sgd_hinge"]


def get_c_candidates(probe_type: str) -> List[float]:
    """Return default candidate hyperparameter C/alpha values for inner CV tuning."""
    if probe_type in ["logistic", "svm_linear"]:
        return [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    elif probe_type == "ridge":
        # For Ridge, C corresponds to inverse regularization strength 1/alpha
        return [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    elif probe_type in ["sgd_log", "sgd_hinge"]:
        return [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    else:
        return [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def create_probe_classifier(probe_type: str, C: float = 1.0, seed: int = 42) -> Any:
    """
    Instantiate a linear classifier based on probe_type and hyperparameter C.

    Args:
        probe_type (str): Name of classifier algorithm.
                          Options: "logistic", "svm_linear", "ridge", "sgd_log", "sgd_hinge".
        C (float): Regularization parameter (higher = less regularization for logistic/svm).
        seed (int): Random seed for reproducibility.

    Returns:
        Scikit-learn classifier instance.
    """
    p_type = probe_type.lower().strip()

    if p_type == "logistic":
        return LogisticRegression(C=C, max_iter=1000, random_state=seed)

    elif p_type == "svm_linear":
        return SVC(kernel="linear", C=C, random_state=seed)

    elif p_type == "ridge":
        # Ridge uses alpha = 1 / (2 * C) or 1 / C
        alpha = 1.0 / max(C, 1e-6)
        return RidgeClassifier(alpha=alpha, random_state=seed)

    elif p_type == "sgd_log":
        # SGDClassifier with log_loss (logistic regression via SGD)
        # alpha = 1 / (N * C), using a default base scaling
        alpha = 1.0 / max(C * 1000.0, 1e-6)
        return SGDClassifier(loss="log_loss", alpha=alpha, max_iter=1000, random_state=seed)

    elif p_type == "sgd_hinge":
        # SGDClassifier with hinge loss (linear SVM via SGD)
        alpha = 1.0 / max(C * 1000.0, 1e-6)
        return SGDClassifier(loss="hinge", alpha=alpha, max_iter=1000, random_state=seed)

    else:
        raise ValueError(
            f"Unsupported probe_type '{probe_type}'. Supported probes: {SUPPORTED_PROBES}"
        )
