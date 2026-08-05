"""
Expressive Scoring Heads for Multimodal Image-Text MCQ Matching.

This module provides 8 scoring models ranging from simple cosine similarity to
caching-preserving Non-Linear Bi-Encoders:
1. CosineScorer: Baseline CLIP dot product / cosine similarity.
2. WeightedCosineScorer: Feature-wise weighted cosine similarity (w * (v * t)).
3. BilinearScorer: Full bilinear interaction tensor (v^T W t).
4. LogisticRegressionScorer: Linear decision boundary over joint features [v, t, v*t, |v-t|].
5. ShallowMLPScorer: 2-layer neural network with GELU non-linearity.
6. DeepMLPScorer: 4-layer neural network with LayerNorm, GELU, and residual connections.
7. LowRankBilinearScorer: Low-rank bilinear (Av).(Bt), O(1) offline caching preserved.
8. NonLinearBiEncoderScorer: GELU(Av).GELU(Bt), O(1) offline caching + non-linearity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseScorer(nn.Module):
    """Abstract base class for image-text candidate scoring models."""

    def forward(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_emb: Tensor of shape (B, D) or (B, 1, D)
            text_emb: Tensor of shape (B, K, D)
        Returns:
            scores: Tensor of shape (B, K)
        """
        raise NotImplementedError


class CosineScorer(BaseScorer):
    """
    1. Cosine Similarity (Baseline CLIP default).
    Expressiveness: Very Low
    Hypothesis: Standard CLIP inner product space.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        # img_emb: (B, D), text_emb: (B, K, D)
        if img_emb.dim() == 2:
            img_emb = img_emb.unsqueeze(1)  # (B, 1, D)
        
        v_norm = F.normalize(img_emb, dim=-1)
        t_norm = F.normalize(text_emb, dim=-1)
        
        # Pairwise dot product along feature dimension
        return torch.sum(v_norm * t_norm, dim=-1)  # (B, K)


class WeightedCosineScorer(BaseScorer):
    """
    2. Weighted Cosine Similarity.
    Expressiveness: Low
    Hypothesis: Is simple feature-wise weighting sufficient?
    Formula: s(v, t) = sum_d w_d * (v_d * t_d) + b
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        # Initialize weights to 1.0 so initial prediction matches standard cosine
        self.weight = nn.Parameter(torch.ones(feature_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        if img_emb.dim() == 2:
            img_emb = img_emb.unsqueeze(1)
        
        v_norm = F.normalize(img_emb, dim=-1)
        t_norm = F.normalize(text_emb, dim=-1)
        
        # Elementwise product: (B, K, D)
        elem_prod = v_norm * t_norm
        scores = torch.sum(elem_prod * self.weight, dim=-1) + self.bias
        return scores  # (B, K)


class BilinearScorer(BaseScorer):
    """
    3. Bilinear Scoring Function.
    Expressiveness: Medium
    Hypothesis: Are dimension-to-dimension cross-modal interactions required?
    Formula: s(v, t) = v^T W t + b
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        # Initialize W near Identity for fast convergence to cosine similarity
        self.W = nn.Parameter(torch.eye(feature_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        if img_emb.dim() == 2:
            img_emb = img_emb.unsqueeze(1)
        
        v_norm = F.normalize(img_emb, dim=-1)  # (B, 1, D)
        t_norm = F.normalize(text_emb, dim=-1)  # (B, K, D)
        
        # v @ W -> (B, 1, D)
        v_W = torch.matmul(v_norm, self.W)
        
        # (v @ W) * t -> sum over D -> (B, K)
        scores = torch.sum(v_W * t_norm, dim=-1) + self.bias
        return scores


class LogisticRegressionScorer(BaseScorer):
    """
    4. Logistic Regression (Linear Decision Boundary).
    Expressiveness: Medium
    Hypothesis: Is a linear decision boundary over concatenated features sufficient?
    Input feature vector: x = [v, t, v * t, |v - t|] (4 * D)
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.in_dim = feature_dim * 4
        self.linear = nn.Linear(self.in_dim, 1)

    def _construct_features(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        if img_emb.dim() == 2:
            img_emb = img_emb.unsqueeze(1)
        
        v_norm = F.normalize(img_emb, dim=-1).expand_as(text_emb)  # (B, K, D)
        t_norm = F.normalize(text_emb, dim=-1)  # (B, K, D)
        
        mult = v_norm * t_norm
        diff = torch.abs(v_norm - t_norm)
        
        # Concatenate along feature dimension -> (B, K, 4D)
        feat = torch.cat([v_norm, t_norm, mult, diff], dim=-1)
        return feat

    def forward(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        feat = self._construct_features(img_emb, text_emb)  # (B, K, 4D)
        scores = self.linear(feat).squeeze(-1)  # (B, K)
        return scores


class ShallowMLPScorer(BaseScorer):
    """
    5. Shallow MLP (2-Layer Neural Network).
    Expressiveness: High
    Hypothesis: Is non-linearity required?
    Architecture: 4D -> 256 -> GELU -> Dropout -> 1
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.in_dim = feature_dim * 4
        
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def _construct_features(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        if img_emb.dim() == 2:
            img_emb = img_emb.unsqueeze(1)
        
        v_norm = F.normalize(img_emb, dim=-1).expand_as(text_emb)
        t_norm = F.normalize(text_emb, dim=-1)
        
        mult = v_norm * t_norm
        diff = torch.abs(v_norm - t_norm)
        
        return torch.cat([v_norm, t_norm, mult, diff], dim=-1)

    def forward(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        feat = self._construct_features(img_emb, text_emb)  # (B, K, 4D)
        scores = self.mlp(feat).squeeze(-1)  # (B, K)
        return scores


class DeepMLPScorer(BaseScorer):
    """
    6. Deep MLP (4-Layer Neural Network with LayerNorm & GELU).
    Expressiveness: Very High
    Hypothesis: Is the expressiveness in representation itself lacking?
    Architecture: 4D -> 512 -> 256 -> 128 -> 1
    """

    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.in_dim = feature_dim * 4
        
        self.layer1 = nn.Sequential(
            nn.Linear(self.in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.out_head = nn.Linear(128, 1)

    def _construct_features(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        if img_emb.dim() == 2:
            img_emb = img_emb.unsqueeze(1)
        
        v_norm = F.normalize(img_emb, dim=-1).expand_as(text_emb)
        t_norm = F.normalize(text_emb, dim=-1)
        
        mult = v_norm * t_norm
        diff = torch.abs(v_norm - t_norm)
        
        return torch.cat([v_norm, t_norm, mult, diff], dim=-1)

    def forward(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        feat = self._construct_features(img_emb, text_emb)  # (B, K, 4D)
        
        h1 = self.layer1(feat)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        scores = self.out_head(h3).squeeze(-1)  # (B, K)
        return scores


class LowRankBilinearScorer(BaseScorer):
    """
    7. Low-Rank Bilinear Scorer (Caching-Preserving).
    Expressiveness: Medium (constrained to rank-k subspace)
    Hypothesis: Does negation information live in a low-rank interaction subspace?
    Formula: s(v, t) = (A v) . (B t) + b  where A, B in R^(k x d)
    Key property: v' = Av and t' = Bt can be pre-computed independently
                  -> O(1) offline caching for Bi-Encoder retrieval preserved.
    """

    def __init__(self, feature_dim: int, rank: int = 32):
        super().__init__()
        self.feature_dim = feature_dim
        self.rank = rank
        # Initialize A, B near zero for stable early training
        self.proj_v = nn.Linear(feature_dim, rank, bias=False)
        self.proj_t = nn.Linear(feature_dim, rank, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.proj_v.weight, std=0.02)
        nn.init.normal_(self.proj_t.weight, std=0.02)

    def forward(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        if img_emb.dim() == 2:
            img_emb = img_emb.unsqueeze(1)  # (B, 1, D)

        v_norm = F.normalize(img_emb, dim=-1)   # (B, 1, D)
        t_norm = F.normalize(text_emb, dim=-1)  # (B, K, D)

        # Independent projections -> offline cacheable
        # (B, 1, D) @ (D, k) -> (B, 1, k)
        Av = torch.matmul(v_norm, self.proj_v.weight.T)
        # (B, K, D) @ (D, k) -> (B, K, k)
        Bt = torch.matmul(t_norm, self.proj_t.weight.T)

        # Inner product over rank dimension
        scores = torch.sum(Av * Bt, dim=-1) + self.bias  # (B, K)
        return scores


class NonLinearBiEncoderScorer(BaseScorer):
    """
    8. Non-Linear Bi-Encoder Scorer (Caching-Preserving).
    Expressiveness: High (non-linear projection, constrained to rank-k)
    Hypothesis: Does non-linearity in the projection recover negation accuracy
                while preserving O(1) Bi-Encoder retrieval caching?
    Formula: s(v, t) = GELU(A v) . GELU(B t) + b
    Key property: v' = GELU(Av) and t' = GELU(Bt) can be pre-computed independently
                  -> O(1) offline caching preserved despite non-linearity.
    """

    def __init__(self, feature_dim: int, rank: int = 32):
        super().__init__()
        self.feature_dim = feature_dim
        self.rank = rank
        self.proj_v = nn.Linear(feature_dim, rank, bias=True)
        self.proj_t = nn.Linear(feature_dim, rank, bias=True)
        self.act = nn.GELU()
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.proj_v.weight, std=0.02)
        nn.init.normal_(self.proj_t.weight, std=0.02)

    def forward(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        if img_emb.dim() == 2:
            img_emb = img_emb.unsqueeze(1)  # (B, 1, D)

        v_norm = F.normalize(img_emb, dim=-1)   # (B, 1, D)
        t_norm = F.normalize(text_emb, dim=-1)  # (B, K, D)

        # Non-linear independent projections -> offline cacheable
        # (B, 1, k)
        Av = self.act(torch.matmul(v_norm, self.proj_v.weight.T) + self.proj_v.bias)
        # (B, K, k)
        Bt = self.act(torch.matmul(t_norm, self.proj_t.weight.T) + self.proj_t.bias)

        scores = torch.sum(Av * Bt, dim=-1) + self.bias  # (B, K)
        return scores


def build_scorer(model_type: str, feature_dim: int, rank: int = 32) -> BaseScorer:
    """Factory function to build a scoring head model by name.

    Args:
        model_type: Name of the scoring head (e.g., 'cosine', 'bilinear', 'low_rank_bilinear').
        feature_dim: Dimensionality of CLIP embeddings (typically 512).
        rank: Rank k for LowRankBilinearScorer and NonLinearBiEncoderScorer (default 32).
    """
    name_lower = model_type.lower().replace(" ", "_").replace("-", "_")

    if name_lower in ["cosine", "cosine_similarity"]:
        return CosineScorer(feature_dim)
    elif name_lower in ["weighted_cosine", "weighted_cosine_similarity"]:
        return WeightedCosineScorer(feature_dim)
    elif name_lower in ["bilinear", "bilinear_matrix"]:
        return BilinearScorer(feature_dim)
    elif name_lower in ["logistic_regression", "log_reg", "linear"]:
        return LogisticRegressionScorer(feature_dim)
    elif name_lower in ["shallow_mlp", "mlp_shallow"]:
        return ShallowMLPScorer(feature_dim)
    elif name_lower in ["deep_mlp", "mlp_deep"]:
        return DeepMLPScorer(feature_dim)
    elif name_lower in ["low_rank_bilinear", "lr_bilinear", "low_rank"]:
        return LowRankBilinearScorer(feature_dim, rank=rank)
    elif name_lower in ["nonlinear_biencoder", "nl_biencoder", "nonlinear_bi", "nl_bi"]:
        return NonLinearBiEncoderScorer(feature_dim, rank=rank)
    else:
        raise ValueError(f"Unknown scoring model type: {model_type}. "
                         f"Available: cosine, weighted_cosine, bilinear, logistic_regression, "
                         f"shallow_mlp, deep_mlp, low_rank_bilinear, nonlinear_biencoder")
