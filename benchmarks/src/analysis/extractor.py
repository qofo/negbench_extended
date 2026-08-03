"""
Single-Pass Unified Feature Extraction Engine.

This module implements a computationally efficient single-pass forward pass
extractor that captures hidden states across all Transformer residual blocks and 
tracks granular pipeline transformation steps (Embedding -> Transformer Layers -> LayerNorm -> Projection -> L2 Normalization).
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PipelineStep


def extract_all_features_unified(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: str = "cpu",
    target_token: str = "eot",
    batch_size: int = 256,
    custom_projection: Optional[Union[np.ndarray, str]] = None,
) -> Dict[str, Any]:
    """
    Extract intermediate representations across all Transformer layers and pipeline steps in a single forward pass.

    Args:
        model (nn.Module): Pre-trained CLIP model architecture.
        tokenizer (Any): Model tokenization engine.
        texts (List[str]): Input sequence strings.
        device (str): Compute device ('cuda' or 'cpu').
        target_token (str): Pooling strategy ('eot', 'mean', or 'all').
        batch_size (int): Mini-batch processing size.
        custom_projection (Optional[Union[np.ndarray, str]]): Optional custom projection matrix or 'identity'.

    Returns:
        Dict[str, Any]: Dictionary containing layer-wise and pipeline step feature matrices.
    """
    model.eval()
    all_tokens = tokenizer(texts).to(device)

    # Dissect model architecture attributes
    text_tower = getattr(model, 'text', model)
    token_embedding = text_tower.token_embedding
    positional_embedding = text_tower.positional_embedding
    transformer = text_tower.transformer
    ln_final = text_tower.ln_final
    text_projection = getattr(text_tower, 'text_projection', None)
    attn_mask = getattr(text_tower, 'attn_mask', None)

    resblocks = transformer.resblocks
    num_layers = 1 + len(resblocks)

    layer_batches = [[] for _ in range(num_layers)]
    pipeline_batches = {
        PipelineStep.EMBEDDING.value: [],
        PipelineStep.LAYER12_RAW.value: [],
        PipelineStep.LAYER12_LN.value: [],
        PipelineStep.PROJECTED_UNNORM.value: [],
        PipelineStep.FINAL_L2NORM.value: []
    }
    inter_layer_batches = {f"Layer{i}": [] for i in range(1, len(resblocks))}

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_tokens = all_tokens[start:end]

        with torch.no_grad():
            cast_dtype = transformer.get_cast_dtype()
            eot_indices = batch_tokens.argmax(dim=-1).cpu()
            batch_idx = torch.arange(batch_tokens.shape[0])

            # Step 0: Input Token + Positional Embedding
            x = token_embedding(batch_tokens).to(cast_dtype)
            seq_len = batch_tokens.shape[1]
            x = x + positional_embedding[:seq_len].to(cast_dtype)

            hidden_states = [x]

            # Forward pass through residual Transformer blocks
            x_perm = x.permute(1, 0, 2)
            for block in resblocks:
                x_perm = block(x_perm, attn_mask=attn_mask)
                hidden_states.append(x_perm.permute(1, 0, 2))

            # Pool token representations per layer
            for l_idx, hs in enumerate(hidden_states):
                hs_cpu = hs.float().cpu()
                if target_token == "eot":
                    feat = hs_cpu[batch_idx, eot_indices].numpy()
                elif target_token == "mean":
                    feat = hs_cpu.mean(dim=1).numpy()
                elif target_token == "all":
                    feat = hs_cpu.reshape(-1, hs_cpu.shape[-1]).numpy()
                else:
                    feat = hs_cpu[batch_idx, eot_indices].numpy()
                layer_batches[l_idx].append(feat)

                if 1 <= l_idx < len(resblocks):
                    inter_layer_batches[f"Layer{l_idx}"].append(feat)

            # Extract 5 pipeline transformation steps on compute device
            def extract_step_token(tensor_b_l_d):
                if target_token == "eot":
                    return tensor_b_l_d[batch_idx.to(tensor_b_l_d.device), eot_indices.to(tensor_b_l_d.device)]
                elif target_token == "mean":
                    return tensor_b_l_d.mean(dim=1)
                else:
                    return tensor_b_l_d[batch_idx.to(tensor_b_l_d.device), eot_indices.to(tensor_b_l_d.device)]

            step0_dev = extract_step_token(hidden_states[0])
            step1_dev = extract_step_token(hidden_states[-1])

            x_ln = ln_final(hidden_states[-1])
            step2_dev = extract_step_token(x_ln)

            # Apply Linear Projection (Step 3)
            if custom_projection is not None:
                if isinstance(custom_projection, str) and custom_projection == "identity":
                    step3_dev = step2_dev.clone()
                else:
                    W_custom = torch.from_numpy(custom_projection).to(device=step2_dev.device, dtype=step2_dev.dtype)
                    step3_dev = step2_dev @ W_custom
            elif text_projection is not None:
                if isinstance(text_projection, nn.Linear):
                    step3_dev = text_projection(step2_dev.to(text_projection.weight.dtype))
                else:
                    step3_dev = step2_dev.to(text_projection.dtype) @ text_projection
            else:
                step3_dev = step2_dev.clone()

            # Apply Unit Hyper-sphere Normalization (Step 4)
            step4_dev = F.normalize(step3_dev.float(), dim=-1)

            # Offload tensors to host CPU memory
            step0 = step0_dev.float().cpu()
            step1 = step1_dev.float().cpu()
            step2 = step2_dev.float().cpu()
            step3 = step3_dev.float().cpu()
            step4 = step4_dev.float().cpu()

            pipeline_batches[PipelineStep.EMBEDDING.value].append(step0.numpy())
            pipeline_batches[PipelineStep.LAYER12_RAW.value].append(step1.numpy())
            pipeline_batches[PipelineStep.LAYER12_LN.value].append(step2.numpy())
            pipeline_batches[PipelineStep.PROJECTED_UNNORM.value].append(step3.numpy())
            pipeline_batches[PipelineStep.FINAL_L2NORM.value].append(step4.numpy())

    layer_dict = {}
    for l_idx, feats in enumerate(layer_batches):
        name = "Embedding" if l_idx == 0 else f"Layer {l_idx}"
        layer_dict[name] = np.concatenate(feats, axis=0)

    pipeline_dict = {k: np.concatenate(v, axis=0) for k, v in pipeline_batches.items()}
    for k, v in inter_layer_batches.items():
        pipeline_dict[k] = np.concatenate(v, axis=0)

    return {
        "layers": layer_dict,
        "pipeline": pipeline_dict,
        "final_l2norm": pipeline_dict[PipelineStep.FINAL_L2NORM.value]
    }


def assert_embedding_consistency(
    model: nn.Module,
    tokenizer: Any,
    sample_texts: List[str],
    extracted_final_embs: np.ndarray,
    device: str = "cpu"
):
    """
    Validate equivalence between manual forward pass outputs and official model.encode_text().

    Args:
        model (nn.Module): Pre-trained CLIP model.
        tokenizer (Any): Model tokenization engine.
        sample_texts (List[str]): Sample text captions for validation.
        extracted_final_embs (np.ndarray): Extracted final normalized embeddings.
        device (str): Compute device.
    """
    if len(sample_texts) == 0:
        return

    n_check = min(10, len(sample_texts))
    check_tokens = tokenizer(sample_texts[:n_check]).to(device)

    with torch.no_grad():
        official_embs = model.encode_text(check_tokens, normalize=True).float().cpu().numpy()

    diff = np.abs(extracted_final_embs[:n_check] - official_embs)
    max_diff = float(np.max(diff))

    assert max_diff < 1e-3, f"Embedding consistency assertion failed! Max diff: {max_diff:.6f}"
    print(f"✅ Embedding Consistency Verified! (Max diff vs model.encode_text: {max_diff:.6e})")
