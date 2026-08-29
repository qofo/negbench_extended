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


# ==============================================================================
# Probe registry -- the single place a probe type is declared
# ==============================================================================
# A probe used to be declared in three places that had to agree by hand: this
# name list, an elif chain in get_param_candidates, and another in
# create_probe_classifier. Nothing enforced the agreement, so adding a probe to
# two of the three would raise "Unsupported probe_type" from whichever one was
# missed -- or, worse, tune a probe against another's grid.
#
# Now the grid lives in the registry and the name list is derived from it.
# ``build`` stays a small function per entry because each one constructs a
# genuinely different estimator; the registry is what keeps the set of names,
# the grids, and the builders from drifting apart.

_C_GRID = [{"C": c} for c in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]]

_TORCH_DEFAULTS = {"hidden_dim": 64, "rank": 4, "lr": 1e-2, "weight_decay": 1e-4, "epochs": 200}

# sklearn's SVC exposes no fit_intercept, so --no_bias cannot be honored for these.
_NO_INTERCEPT_CONTROL = ("svm_linear", "svm_rbf")


def _grid_svm_rbf() -> List[Dict[str, Any]]:
    return [{"C": c, "gamma": g}
            for c in [0.1, 1.0, 10.0, 100.0]
            for g in ["scale", "auto", 0.01, 0.1]]


def _build_torch(p_type: str, params: Dict[str, Any], seed: int, use_bias: bool) -> Any:
    kw = {k: params.get(k, v) for k, v in _TORCH_DEFAULTS.items()}
    return PyTorchProbeEstimator(model_type=p_type, seed=seed, use_bias=use_bias, **kw)


PROBE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "logistic": {
        "grid": lambda: list(_C_GRID),
        "build": lambda p, seed, use_bias: LogisticRegression(
            C=p.get("C", 1.0), max_iter=1000, random_state=seed, fit_intercept=use_bias),
    },
    "svm_linear": {
        "grid": lambda: list(_C_GRID),
        "build": lambda p, seed, use_bias: SVC(
            kernel="linear", C=p.get("C", 1.0), random_state=seed),
    },
    "ridge": {
        "grid": lambda: list(_C_GRID),
        "build": lambda p, seed, use_bias: RidgeClassifier(
            alpha=1.0 / max(p.get("C", 1.0), 1e-6), random_state=seed, fit_intercept=use_bias),
    },
    "sgd_log": {
        "grid": lambda: list(_C_GRID),
        "build": lambda p, seed, use_bias: SGDClassifier(
            loss="log_loss", alpha=1.0 / max(p.get("C", 1.0) * 1000.0, 1e-6),
            max_iter=1000, random_state=seed, fit_intercept=use_bias),
    },
    "sgd_hinge": {
        "grid": lambda: list(_C_GRID),
        "build": lambda p, seed, use_bias: SGDClassifier(
            loss="hinge", alpha=1.0 / max(p.get("C", 1.0) * 1000.0, 1e-6),
            max_iter=1000, random_state=seed, fit_intercept=use_bias),
    },
    "svm_rbf": {
        "grid": _grid_svm_rbf,
        "build": lambda p, seed, use_bias: SVC(
            kernel="rbf", C=p.get("C", 1.0), gamma=p.get("gamma", "scale"), random_state=seed),
    },
    "mlp": {
        "grid": lambda: [{"hidden_dim": h, "lr": 0.01, "weight_decay": wd, "epochs": 200}
                         for h in [8, 16, 32, 64, 128] for wd in [1e-4, 1e-3, 1e-2, 1e-1]],
        "build": lambda p, seed, use_bias: _build_torch("mlp", p, seed, use_bias),
    },
    "bilinear_lowrank": {
        "grid": lambda: [{"rank": r, "lr": 0.01, "weight_decay": wd, "epochs": 200}
                         for r in [2, 4, 8, 16, 32] for wd in [1e-4, 1e-3, 1e-2, 1e-1]],
        "build": lambda p, seed, use_bias: _build_torch("bilinear_lowrank", p, seed, use_bias),
    },
    "bilinear_full": {
        "grid": lambda: [{"lr": lr, "weight_decay": wd, "epochs": 200}
                         for lr in [0.005, 0.01, 0.02] for wd in [1e-4, 1e-3, 1e-2, 1e-1]],
        "build": lambda p, seed, use_bias: _build_torch("bilinear_full", p, seed, use_bias),
    },
    "elementwise": {
        "grid": lambda: [{"lr": 0.01, "weight_decay": wd, "epochs": 200}
                         for wd in [1e-4, 1e-3, 1e-2, 1e-1]],
        "build": lambda p, seed, use_bias: _build_torch("elementwise", p, seed, use_bias),
    },
}

SUPPORTED_PROBES = list(PROBE_REGISTRY)


def _lookup(probe_type: str) -> Dict[str, Any]:
    spec = PROBE_REGISTRY.get(probe_type.lower().strip())
    if spec is None:
        raise ValueError(
            f"Unsupported probe_type '{probe_type}'. Supported probes: {SUPPORTED_PROBES}")
    return spec


# ==============================================================================
# PyTorch-based Bilinear and MLP Probe Modules
# ==============================================================================

class LowRankBilinearPyTorch(nn.Module):
    """Low-Rank Bilinear Probe: f(x) = sum_r (x U_r)(x V_r) + x w_lin + b."""

    def __init__(self, d_in: int, rank: int, use_bias: bool = True):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_in, rank) * (1.0 / np.sqrt(d_in)))
        self.V = nn.Parameter(torch.randn(d_in, rank) * (1.0 / np.sqrt(d_in)))
        self.w_lin = nn.Parameter(torch.zeros(d_in))
        self.bias = nn.Parameter(torch.zeros(1)) if use_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(x, self.U)
        h = torch.matmul(x, self.V)
        quad = torch.sum(z * h, dim=-1)
        lin = torch.matmul(x, self.w_lin)
        out = quad + lin
        if self.bias is not None:
            out = out + self.bias.squeeze(-1)
        return out


class FullBilinearPyTorch(nn.Module):
    """Full Bilinear Quadratic Probe: f(x) = x^T W x + x w_lin + b."""

    def __init__(self, d_in: int, use_bias: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_in, d_in) * (1.0 / np.sqrt(d_in)))
        self.w_lin = nn.Parameter(torch.zeros(d_in))
        self.bias = nn.Parameter(torch.zeros(1)) if use_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(x, self.W)  # (B, d_in)
        quad = torch.sum(z * x, dim=-1)
        lin = torch.matmul(x, self.w_lin)
        out = quad + lin
        if self.bias is not None:
            out = out + self.bias.squeeze(-1)
        return out


class MLPVisionPyTorch(nn.Module):
    """MLP Vision Classifier: f_V(x) = fc2(GELU(fc1(x)))."""

    def __init__(self, d_in: int, hidden_dim: int = 64, use_bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden_dim, bias=use_bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x))).squeeze(-1)


class ElementWiseNonLinearPyTorch(nn.Module):
    """Element-wise Non-linear Probe: f(x) = sum_d (w_d * GELU(x_d)) + b.
    Guarantees 0% dimension mixing to isolate pure non-linearity from bilinear/MLP dimension cross-talk.
    """
    def __init__(self, d_in: int, use_bias: bool = True):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d_in))
        self.bias = nn.Parameter(torch.zeros(1)) if use_bias else None
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(x)
        out = torch.sum(h * self.w, dim=-1)
        if self.bias is not None:
            out = out + self.bias.squeeze(-1)
        return out


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
        use_bias: bool = True,
    ):
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.seed = seed
        self.use_bias = use_bias
        # classes_ is a *fitted* attribute in the sklearn contract, so it is set in
        # fit() from the data. Hardcoding [0, 1] here meant labels like {2, 5} were
        # cast straight into BCE and predict() still returned 0/1 -- scoring 0% on
        # perfectly separable data with no error raised anywhere.
        self.model_ = None
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"

    def _build_model(self, d_in: int) -> nn.Module:
        if self.model_type == "mlp":
            return MLPVisionPyTorch(d_in, self.hidden_dim, use_bias=self.use_bias)
        elif self.model_type == "bilinear_lowrank":
            return LowRankBilinearPyTorch(d_in, self.rank, use_bias=self.use_bias)
        elif self.model_type == "bilinear_full":
            return FullBilinearPyTorch(d_in, use_bias=self.use_bias)
        elif self.model_type == "elementwise":
            return ElementWiseNonLinearPyTorch(d_in, use_bias=self.use_bias)
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'")

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"{type(self).__name__} is a binary probe (BCEWithLogitsLoss), but y has "
                f"{len(self.classes_)} classes: {self.classes_.tolist()}"
            )
        # BCE needs 0/1 targets; remember which observed label is the positive one so
        # predict() can hand the caller back its own labels.
        y = (y == self.classes_[1]).astype(np.float32)

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
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device_)

        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model_(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        return self

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise ValueError(f"{type(self).__name__} is not fitted yet; call fit() first.")

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device_)
            logits = self.model_(X_t).cpu().numpy().flatten()
            return logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device_)
            logits = self.model_(X_t)
            p1 = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
            p0 = 1.0 - p1
            return np.hstack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return self.classes_[(proba >= 0.5).astype(int)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


# ==============================================================================
# Hyperparameter Grid Definitions & Classifier Instantiation
# ==============================================================================

def get_param_candidates(probe_type: str) -> List[Dict[str, Any]]:
    """Return candidate hyperparameter dictionaries for inner CV tuning."""
    return _lookup(probe_type)["grid"]()


def format_params(params: Dict[str, Any]) -> str:
    """Format parameter dictionary into a clean string representation."""
    items = []
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            items.append(f"{k}={v:.4g}")
        else:
            items.append(f"{k}={v}")
    return ", ".join(items)


_NO_BIAS_WARNED = set()


def create_probe_classifier(probe_type: str, seed: int = 42, fit_intercept: bool = True, **params) -> Any:
    """
    Instantiate classifier based on probe_type and hyperparameter dictionary.

    Args:
        probe_type (str): Probe algorithm identifier, one of ``SUPPORTED_PROBES``.
        seed (int): Random seed for reproducibility.
        fit_intercept (bool): Whether to include bias / intercept term (default: True).
        **params: Hyperparameters for the classifier.

    Returns:
        Scikit-learn compatible classifier instance.
    """
    p_type = probe_type.lower().strip()
    spec = _lookup(p_type)
    use_bias = params.get("use_bias", fit_intercept)

    # sklearn's SVC has no fit_intercept, so --no_bias cannot be honored for the two
    # SVM probes. Say so instead of reporting a "no-bias" number that still has one.
    if not use_bias and p_type in _NO_INTERCEPT_CONTROL and p_type not in _NO_BIAS_WARNED:
        _NO_BIAS_WARNED.add(p_type)
        print(f"[WARNING] probe '{p_type}' is backed by sklearn SVC, which always fits an "
              f"intercept; fit_intercept=False is NOT applied. Use 'logistic' or 'ridge' "
              f"for a genuinely intercept-free linear probe.")

    return spec["build"](params, seed, use_bias)
