"""
E5-V Model Wrapper
===================
Wraps LlavaNextForConditionalGeneration to provide a CLIP-like interface
(encode_text / encode_image) for NegBench evaluation.

Key design decisions:
- Uses the LAST token's hidden state from the final layer as the embedding
  (matching E5-V's original implementation in retrieval.py).
- Supports optional bitsandbytes quantization (int4/int8) for constrained VRAM.
- Supports optional LoRA adapters via --lora-path.
- Batch processing with configurable internal batch size to avoid OOM.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
from PIL import Image

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from .utils import build_text_prompts, build_img_prompts


class E5VWrapper(torch.nn.Module):
    """
    Wrapper around E5-V (LlavaNext Llama-3-8B) that exposes
    encode_text() and encode_image() for compatibility with
    NegBench evaluation pipelines.

    Usage:
        wrapper = E5VWrapper("royokong/e5-v", device="cuda")
        text_embs = wrapper.encode_text(["A dog sitting in the grass."])
        img_embs = wrapper.encode_image([pil_img1, pil_img2])
    """

    def __init__(
        self,
        model_name: str = "royokong/e5-v",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        quantize: Optional[str] = None,
        lora_path: Optional[str] = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model_name = model_name

        # Load processor (tokenizer + image processor)
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        # Ensure left-padding for batched generation/embedding
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

        # Build quantization config if requested
        model_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }

        if quantize == "int4":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
        elif quantize == "int8":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            model_kwargs["device_map"] = "auto"
        elif device == "auto":
            model_kwargs["device_map"] = "auto"
        else:
            # No quantization: load to specified device
            model_kwargs["device_map"] = None

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs
        )

        # Apply LoRA adapter if provided
        if lora_path is not None:
            from peft import PeftModel
            self.model.language_model = PeftModel.from_pretrained(
                self.model.language_model, lora_path
            ).merge_and_unload()

        # Move to device if not using device_map="auto"
        if model_kwargs.get("device_map") is None:
            self.model = self.model.to(device)
            self.target_device = device
        else:
            self.target_device = self.model.device

        self.model.eval()

    @torch.no_grad()
    def encode_text(
        self,
        texts: List[str],
        batch_size: int = 8,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode a list of raw text strings into E5-V embeddings.

        Args:
            texts: List of caption strings (e.g. ["A dog", "No cat"]).
            batch_size: Internal batch size for processing.
            normalize: Whether to L2-normalize the output embeddings.

        Returns:
            Tensor of shape (len(texts), hidden_dim), on CPU.
        """
        prompts = build_text_prompts(texts)
        all_embs = []

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            inputs = self.processor(
                text=batch_prompts,
                return_tensors="pt",
                padding=True,
            ).to(self.target_device)

            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            # Last token of the last hidden layer
            embs = outputs.hidden_states[-1][:, -1, :]

            if normalize:
                embs = F.normalize(embs, dim=-1)

            all_embs.append(embs.cpu().float())

        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def encode_image(
        self,
        images: List[Image.Image],
        batch_size: int = 4,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode a list of PIL images into E5-V embeddings.

        Args:
            images: List of PIL.Image objects.
            batch_size: Internal batch size for processing.
            normalize: Whether to L2-normalize the output embeddings.

        Returns:
            Tensor of shape (len(images), hidden_dim), on CPU.
        """
        all_embs = []

        for start in range(0, len(images), batch_size):
            batch_images = images[start:start + batch_size]
            img_prompts = build_img_prompts(len(batch_images))

            inputs = self.processor(
                text=img_prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
            ).to(self.target_device)

            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            # Last token of the last hidden layer
            embs = outputs.hidden_states[-1][:, -1, :]

            if normalize:
                embs = F.normalize(embs, dim=-1)

            all_embs.append(embs.cpu().float())

        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def encode_text_layerwise(
        self,
        texts: List[str],
        batch_size: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode texts and return hidden states from ALL layers.

        Used for interpretability analysis (layer-wise linear probe, PCA, etc.).
        For each layer, extracts the last token's hidden state.

        Returns:
            Dict mapping layer name (e.g. "Layer 0", "Layer 31") to
            Tensor of shape (len(texts), hidden_dim) on CPU.
        """
        all_layer_feats = None  # Will be initialized on first batch

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            prompts = build_text_prompts(batch_texts)

            inputs = self.processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
            ).to(self.target_device)

            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states  # Tuple of (num_layers+1,) tensors

            if all_layer_feats is None:
                all_layer_feats = [[] for _ in range(len(hidden_states))]

            for layer_idx, hs in enumerate(hidden_states):
                # Last token hidden state
                feat = hs[:, -1, :].cpu().float()
                all_layer_feats[layer_idx].append(feat)

        # Concatenate and build named dict
        layer_dict = {}
        for layer_idx, feats in enumerate(all_layer_feats):
            name = f"Layer {layer_idx}"
            layer_dict[name] = torch.cat(feats, dim=0)

        return layer_dict
