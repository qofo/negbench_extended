"""
Negation-Aware OpenCLIP Model Wrapper for Hypothesis Verification.

This module wraps pre-trained OpenCLIP models to support 4 evaluation modes:
1. 'baseline': Standard OpenCLIP forward pass.
2. 'procrustes_orthogonal': (H1) Applies an Orthogonal Procrustes transformation Q (Q^T Q = I)
   to Layer 12 LN features to eliminate singular value distortion while preserving vector space isometry.
3. 'hyperplane_projection': (H2) Applies cross-modal hyperplane projection t' = L2Norm(t + lambda * (t @ w) * w).
4. 'subspace_bilinear': (H4) Applies Subspace-Constrained Bilinear Tensor M = I + alpha * U_neg^T U_neg
   via isometric matrix square root transformation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Any


class NegationAwareCLIPWrapper(nn.Module):
    """
    Wrapper around OpenCLIP model implementing 4 hypothesis test modes.
    """

    def __init__(
        self,
        base_model: nn.Module,
        negation_method: str = "baseline",
        subspace_basis_path: Optional[str] = None,
        hyperplane_weight_path: Optional[str] = None,
        hyperplane_lambda: float = 0.5,
        bilinear_alpha: float = 0.5
    ):
        super().__init__()
        self.base_model = base_model
        self.negation_method = negation_method.lower()
        self.hyperplane_lambda = hyperplane_lambda
        self.bilinear_alpha = bilinear_alpha

        # Register Procrustes Orthogonal Matrix Q for H1
        self.register_buffer("Q_ortho", None)
        self.register_buffer("w_hyperplane", None)
        self.register_buffer("b_hyperplane", torch.tensor(0.0))
        self.register_buffer("U_neg", None)

        self._initialize_resources(subspace_basis_path, hyperplane_weight_path)

    def _initialize_resources(self, subspace_path: Optional[str], probe_path: Optional[str]):
        """Load pre-computed basis matrices and probe weights if provided."""
        text_tower = getattr(self.base_model, 'text', self.base_model)
        text_proj = getattr(text_tower, 'text_projection', None)

        # Initialize H1 Procrustes Matrix Q if in procrustes mode
        if self.negation_method == "procrustes_orthogonal":
            if text_proj is not None:
                if isinstance(text_proj, nn.Linear):
                    W = text_proj.weight.T.detach().cpu().numpy()
                else:
                    W = text_proj.detach().cpu().numpy()
                
                # Compute Orthogonal Procrustes approximation Q = U V^T from W's SVD W = U S V^T
                U, _, Vh = np.linalg.svd(W, full_matrices=False)
                Q = U @ Vh # Orthogonal matrix Q where Q^T Q = I
                self.Q_ortho = torch.from_numpy(Q).float()
            else:
                self.Q_ortho = torch.eye(512).float()

        # Initialize H2 Hyperplane Weights
        if probe_path and os.path.exists(probe_path):
            data = np.load(probe_path)
            w = data["weight"]
            b = data.get("bias", 0.0)
            self.w_hyperplane = F.normalize(torch.from_numpy(w).float(), dim=-1)
            self.b_hyperplane = torch.tensor(float(b)).float()

        # Initialize H3/H4 Subspace Basis U_neg
        if subspace_path and os.path.exists(subspace_path):
            U_np = np.load(subspace_path) # Shape (k, D)
            self.U_neg = torch.from_numpy(U_np).float()

    def encode_image(self, image: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode image tensor using base model."""
        device = next(self.base_model.parameters()).device
        image = image.to(device)

        if hasattr(self.base_model, "encode_image"):
            img_feats = self.base_model.encode_image(image, normalize=normalize)
        else:
            img_feats = self.base_model(image)

        # If H4 subspace bilinear mode is active, transform image features with M^{1/2}
        if self.negation_method == "subspace_bilinear" and self.U_neg is not None:
            U = self.U_neg.to(img_feats.device, dtype=img_feats.dtype)
            coeff = np.sqrt(1.0 + self.bilinear_alpha) - 1.0
            # v' = v + coeff * (v @ U^T) @ U
            img_feats = img_feats + coeff * (img_feats @ U.T) @ U
            if normalize:
                img_feats = F.normalize(img_feats, dim=-1)

        return img_feats

    def encode_text(self, text: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode text tokens using specified hypothesis method mode."""
        device = next(self.base_model.parameters()).device
        text = text.to(device)

        text_tower = getattr(self.base_model, 'text', self.base_model)
        token_emb = text_tower.token_embedding
        pos_emb = text_tower.positional_embedding
        transformer = text_tower.transformer
        ln_final = text_tower.ln_final

        if self.negation_method == "procrustes_orthogonal":
            # H1 Mode: Extract Layer 12 LN features and apply Orthogonal Procrustes Q
            with torch.no_grad():
                cast_dtype = transformer.get_cast_dtype()
                x = token_emb(text).to(cast_dtype) + pos_emb[:text.shape[1]].to(cast_dtype)
                x_perm = x.permute(1, 0, 2)
                for block in transformer.resblocks:
                    x_perm = block(x_perm, attn_mask=getattr(text_tower, 'attn_mask', None))
                x_final = ln_final(x_perm.permute(1, 0, 2))
                
                eot_idx = text.argmax(dim=-1)
                batch_idx = torch.arange(text.shape[0])
                step2_ln = x_final[batch_idx, eot_idx].float()

                Q = self.Q_ortho.to(device=step2_ln.device, dtype=step2_ln.dtype)
                text_feats = step2_ln @ Q
                if normalize:
                    text_feats = F.normalize(text_feats, dim=-1)
                return text_feats

        # Baseline text encoding
        text_feats = self.base_model.encode_text(text, normalize=normalize)

        # H2 Mode: Hyperplane Projection-Guided Metric
        if self.negation_method == "hyperplane_projection":
            if self.w_hyperplane is not None:
                w = self.w_hyperplane.to(device=text_feats.device, dtype=text_feats.dtype)
                b = self.b_hyperplane.to(device=text_feats.device, dtype=text_feats.dtype)
                
                probe_score = (text_feats @ w) + b
                text_feats = text_feats + self.hyperplane_lambda * probe_score.unsqueeze(-1) * w
                if normalize:
                    text_feats = F.normalize(text_feats, dim=-1)

        # H4 Mode: Subspace-Constrained Bilinear Interface (M^{1/2})
        elif self.negation_method == "subspace_bilinear":
            if self.U_neg is not None:
                U = self.U_neg.to(device=text_feats.device, dtype=text_feats.dtype)
                coeff = np.sqrt(1.0 + self.bilinear_alpha) - 1.0
                text_feats = text_feats + coeff * (text_feats @ U.T) @ U
                if normalize:
                    text_feats = F.normalize(text_feats, dim=-1)

        return text_feats

    def forward(self, image=None, text=None):
        """Standard forward pass returning image and text features."""
        image_features = self.encode_image(image) if image is not None else None
        text_features = self.encode_text(text) if text is not None else None
        return image_features, text_features
