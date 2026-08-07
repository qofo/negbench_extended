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
    Wrapper around OpenCLIP model implementing 4 hypothesis test modes and trained Scoring Heads.
    """

    def __init__(
        self,
        base_model: nn.Module,
        negation_method: str = "baseline",
        subspace_basis_path: Optional[str] = None,
        hyperplane_weight_path: Optional[str] = None,
        hyperplane_lambda: float = 0.5,
        bilinear_alpha: float = 0.5,
        scorer_checkpoint: Optional[str] = None,
        scorer_type: str = "deep_mlp",
        feature_dim: int = 512
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

        self.scorer = None
        self._initialize_resources(subspace_basis_path, hyperplane_weight_path, scorer_checkpoint, scorer_type, feature_dim)

    def _initialize_resources(self, subspace_path: Optional[str], probe_path: Optional[str], scorer_ckpt: Optional[str], scorer_type: str, feature_dim: int):
        """Load pre-computed basis matrices, probe weights, and trained scoring head checkpoints."""
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

        # Load trained Scoring Head Checkpoint
        if self.negation_method in ["scoring_head", "trained_scorer", "dual_classifier_product", "product_probe"] or scorer_ckpt is not None:
            if scorer_ckpt and os.path.exists(scorer_ckpt):
                from evaluation.scoring_heads import build_scorer, DualClassifierProductScorer
                if scorer_ckpt.endswith(".npz"):
                    data = np.load(scorer_ckpt)
                    w_t = torch.from_numpy(data["w_t"]).float()
                    b_t = float(data["b_t"])
                    b_v = float(data.get("b_v", 0.0))
                    feat_dim = w_t.shape[0]

                    U_v = torch.from_numpy(data["U_v"]).float() if "U_v" in data else None
                    V_v = torch.from_numpy(data["V_v"]).float() if "V_v" in data else None
                    w_lin_v = torch.from_numpy(data["w_lin_v"]).float() if "w_lin_v" in data else None
                    w_v = torch.from_numpy(data["w_v"]).float() if "w_v" in data else None

                    v_rank = U_v.shape[1] if U_v is not None else 4
                    self.scorer = DualClassifierProductScorer(feature_dim=feat_dim, vision_rank=v_rank)
                    self.scorer.load_weights(w_t=w_t, b_t=b_t, w_v=w_v, b_v=b_v, U_v=U_v, V_v=V_v, w_lin_v=w_lin_v)
                    print(f"✅ Loaded trained DualClassifierProductScorer (Vision Low-Rank={U_v is not None}) from NPZ: {scorer_ckpt}")

                else:
                    checkpoint = torch.load(scorer_ckpt, map_location="cpu")
                    ckpt_type = checkpoint.get("model_name", scorer_type) if isinstance(checkpoint, dict) else scorer_type
                    ckpt_dim = checkpoint.get("feature_dim", feature_dim) if isinstance(checkpoint, dict) else feature_dim
                    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

                    self.scorer = build_scorer(ckpt_type, ckpt_dim)
                    self.scorer.load_state_dict(state_dict)
                    print(f"✅ Loaded trained Scoring Head '{ckpt_type}' from: {scorer_ckpt}")


    def compute_similarity(self, img_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores for MCQ options.
        img_feats: (B, D)
        text_feats: (B, K, D) or (K, B, D)
        """
        if self.scorer is not None:
            if text_feats.dim() == 3 and text_feats.shape[1] != img_feats.shape[0] and text_feats.shape[0] != img_feats.shape[0]:
                # Transpose if text_feats is (K, B, D)
                text_feats = text_feats.transpose(0, 1)
            
            device = next(self.parameters()).device
            self.scorer = self.scorer.to(device)
            return self.scorer(img_feats.to(device), text_feats.to(device))
        
        # Default cosine / einsum similarity
        if text_feats.dim() == 3 and text_feats.shape[0] != img_feats.shape[0]:
            # (K, B, D)
            return torch.einsum('bf,nbf->bn', img_feats, text_feats)
        else:
            # (B, K, D)
            return torch.sum(img_feats.unsqueeze(1) * text_feats, dim=-1)

    def compute_retrieval_similarity(self, texts_emb: torch.Tensor, images_emb: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        """
        Compute pairwise similarity matrix S (N_txt, N_img) for Retrieval using trained scorer.
        """
        if self.scorer is None:
            return texts_emb @ images_emb.T

        N_txt = texts_emb.shape[0]
        N_img = images_emb.shape[0]
        device = next(self.parameters()).device
        self.scorer = self.scorer.to(device)
        self.scorer.eval()

        scores = torch.zeros(N_txt, N_img, device="cpu")

        with torch.no_grad():
            for start_t in range(0, N_txt, batch_size):
                end_t = min(start_t + batch_size, N_txt)
                t_batch = texts_emb[start_t:end_t].to(device) # (B_t, D)

                for start_i in range(0, N_img, batch_size):
                    end_i = min(start_i + batch_size, N_img)
                    i_batch = images_emb[start_i:end_i].to(device) # (B_i, D)

                    B_t = t_batch.shape[0]
                    B_i = i_batch.shape[0]

                    i_exp = i_batch.unsqueeze(0).expand(B_t, B_i, -1) # (B_t, B_i, D)
                    t_exp = t_batch.unsqueeze(1).expand(B_t, B_i, -1) # (B_t, B_i, D)

                    sub_scores = self.scorer(i_exp, t_exp) # (B_t, B_i)
                    scores[start_t:end_t, start_i:end_i] = sub_scores.cpu()

        return scores

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

        if self.negation_method in ["procrustes_orthogonal", "layer12_raw"]:
            # H1 Mode & Zero-Projection Ablation: Extract Layer 12 LN features
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

                if self.negation_method == "layer12_raw":
                    # Pure Projection Removal (No W_proj, no Q)
                    text_feats = step2_ln
                else:
                    # Procrustes Orthogonal Alignment Q (Q^T Q = I)
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

