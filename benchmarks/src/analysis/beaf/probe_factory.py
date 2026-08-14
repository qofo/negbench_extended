"""
BEAF Multi-Classifier Probing Factory Module.

Provides a unified factory interface to create and tune various linear,
non-linear (MLP, RBF-SVM), and Bilinear (Full, Low-Rank) probing classifiers.
"""

from typing import Any, List, Dict
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC


SUPPORTED_PROBES = [
    "logistic",
    "svm_linear",
    "ridge",
    "sgd_log",
    "sgd_hinge",
    "svm_rbf",
    "mlp",
    "bilinear_lowrank",
    "bilinear_full",
]


# ==============================================================================
# PyTorch-based Bilinear and MLP Probe Modules
# ==============================================================================

class LowRankBilinearPyTorch(nn.Module):
    """Low-Rank Bilinear Probe: f(x) = sum_r (x U_r)(x V_r) + x w_lin + b."""

    def __init__(self, d_in: int, rank: int):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_in, rank) * (1.0 / np.sqrt(d_in)))
        self.V = nn.Parameter(torch.randn(d_in, rank) * (1.0 / np.sqrt(d_in)))
        self.w_lin = nn.Parameter(torch.zeros(d_in))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(x, self.U)
        h = torch.matmul(x, self.V)
        quad = torch.sum(z * h, dim=-1)
        lin = torch.matmul(x, self.w_lin)
        return quad + lin + self.bias.squeeze(-1)


class FullBilinearPyTorch(nn.Module):
    """Full Bilinear Quadratic Probe: f(x) = x^T W x + x w_lin + b."""

    def __init__(self, d_in: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_in, d_in) * (1.0 / np.sqrt(d_in)))
        self.w_lin = nn.Parameter(torch.zeros(d_in))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(x, self.W)  # (B, d_in)
        quad = torch.sum(z * x, dim=-1)
        lin = torch.matmul(x, self.w_lin)
        return quad + lin + self.bias.squeeze(-1)


class MLPVisionPyTorch(nn.Module):
    """MLP Vision Classifier: f_V(x) = fc2(GELU(fc1(x)))."""

    def __init__(self, d_in: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x))).squeeze(-1)


# ==============================================================================
# Scikit-Learn Compatible Estimator Wrappers
# ==============================================================================

class PyTorchProbeEstimator(BaseEstimator, ClassifierMixin):
    """Base Scikit-learn Estimator for PyTorch binary classification probes."""

    def __init__(
        self,
        model_type: str = "mlp",
        hidden_dim: int = 64,
        rank: int = 4,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        epochs: int = 200,
        seed: int = 42,
    ):
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.seed = seed
        self.classes_ = np.array([0, 1])
        self.model_ = None
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"

    def _build_model(self, d_in: int) -> nn.Module:
        if self.model_type == "mlp":
            return MLPVisionPyTorch(d_in, self.hidden_dim)
        elif self.model_type == "bilinear_lowrank":
            return LowRankBilinearPyTorch(d_in, self.rank)
        elif self.model_type == "bilinear_full":
            return FullBilinearPyTorch(d_in)
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'")

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        d_in = X.shape[1]
        self.model_ = self._build_model(d_in).to(self.device_)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device_)
        y_t = torch.tensor(y.astype(np.float32), dtype=torch.float32, device=self.device_)

        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model_(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device_)
            logits = self.model_(X_t)
            p1 = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
            p0 = 1.0 - p1
            return np.hstack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))


# ==============================================================================
# Hyperparameter Grid Definitions & Classifier Instantiation
# ==============================================================================

def get_param_candidates(probe_type: str) -> List[Dict[str, Any]]:
    """Return candidate hyperparameter dictionaries for inner CV tuning."""
    p_type = probe_type.lower().strip()

    if p_type in ["logistic", "svm_linear"]:
        return [{"C": c} for c in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]]

    elif p_type == "ridge":
        return [{"C": c} for c in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]]

    elif p_type in ["sgd_log", "sgd_hinge"]:
        return [{"C": c} for c in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]]

    elif p_type == "svm_rbf":
        candidates = []
        for c in [0.1, 1.0, 10.0, 100.0]:
            for gamma in ["scale", "auto", 0.01, 0.1]:
                candidates.append({"C": c, "gamma": gamma})
        return candidates

    elif p_type == "mlp":
        candidates = []
        for hidden in [8, 16, 32, 64, 128]:
            for wd in [1e-4, 1e-3, 1e-2, 1e-1]:
                candidates.append({"hidden_dim": hidden, "lr": 0.01, "weight_decay": wd, "epochs": 200})
        return candidates

    elif p_type == "bilinear_lowrank":
        candidates = []
        for rank in [2, 4, 8, 16, 32]:
            for wd in [1e-4, 1e-3, 1e-2, 1e-1]:
                candidates.append({"rank": rank, "lr": 0.01, "weight_decay": wd, "epochs": 200})
        return candidates

    elif p_type == "bilinear_full":
        candidates = []
        for lr in [0.005, 0.01, 0.02]:
            for wd in [1e-4, 1e-3, 1e-2, 1e-1]:
                candidates.append({"lr": lr, "weight_decay": wd, "epochs": 200})
        return candidates

    else:
        raise ValueError(
            f"Unsupported probe_type '{probe_type}'. Supported probes: {SUPPORTED_PROBES}"
        )


def format_params(params: Dict[str, Any]) -> str:
    """Format parameter dictionary into a clean string representation."""
    items = []
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            items.append(f"{k}={v:.4g}")
        else:
            items.append(f"{k}={v}")
    return ", ".join(items)


def create_probe_classifier(probe_type: str, seed: int = 42, **params) -> Any:
    """
    Instantiate classifier based on probe_type and hyperparameter dictionary.

    Args:
        probe_type (str): Probe algorithm identifier.
        seed (int): Random seed for reproducibility.
        **params: Hyperparameters for the classifier.

    Returns:
        Scikit-learn compatible classifier instance.
    """
    p_type = probe_type.lower().strip()

    if p_type == "logistic":
        c = params.get("C", 1.0)
        return LogisticRegression(C=c, max_iter=1000, random_state=seed)

    elif p_type == "svm_linear":
        c = params.get("C", 1.0)
        return SVC(kernel="linear", C=c, random_state=seed)

    elif p_type == "ridge":
        c = params.get("C", 1.0)
        alpha = 1.0 / max(c, 1e-6)
        return RidgeClassifier(alpha=alpha, random_state=seed)

    elif p_type == "sgd_log":
        c = params.get("C", 1.0)
        alpha = 1.0 / max(c * 1000.0, 1e-6)
        return SGDClassifier(loss="log_loss", alpha=alpha, max_iter=1000, random_state=seed)

    elif p_type == "sgd_hinge":
        c = params.get("C", 1.0)
        alpha = 1.0 / max(c * 1000.0, 1e-6)
        return SGDClassifier(loss="hinge", alpha=alpha, max_iter=1000, random_state=seed)

    elif p_type == "svm_rbf":
        c = params.get("C", 1.0)
        gamma = params.get("gamma", "scale")
        return SVC(kernel="rbf", C=c, gamma=gamma, random_state=seed)

    elif p_type in ["mlp", "bilinear_lowrank", "bilinear_full"]:
        hidden_dim = params.get("hidden_dim", 64)
        rank = params.get("rank", 4)
        lr = params.get("lr", 1e-2)
        weight_decay = params.get("weight_decay", 1e-4)
        epochs = params.get("epochs", 200)
        return PyTorchProbeEstimator(
            model_type=p_type,
            hidden_dim=hidden_dim,
            rank=rank,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            seed=seed,
        )

    else:
        raise ValueError(
            f"Unsupported probe_type '{probe_type}'. Supported probes: {SUPPORTED_PROBES}"
        )

