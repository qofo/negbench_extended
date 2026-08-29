"""
One place that loads a CLIP backbone for feature extraction.

``open_clip.create_model_and_transforms`` returns three things in the order
``(model, preprocess_train, preprocess_val)``. The train transform is a
stochastic ``RandomResizedCrop`` plus flip; feeding it to an encoder at
extraction time makes every embedding a different sample of the same image, so
probe accuracies and cosine margins move between runs for no stated reason.
Four entrypoints unpacked it by mistake (``model, preprocess, _``), and the
remaining call sites each spelled the same three-line incantation by hand:

    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=tag)
    tokenizer = open_clip.get_tokenizer(name)
    model = model.to(device).eval()

Every repetition is another chance to take the wrong element. This module makes
the val transform the only thing a caller can get.
"""

from typing import Any, Optional, Tuple

import torch

import open_clip


def resolve_device(device: Optional[str] = None) -> str:
    """Return ``device`` when given, else cuda if it is available, else cpu."""
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_for_eval(
    model_name: str,
    pretrained: Optional[str],
    device: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    """
    Load a backbone ready for feature extraction, in eval mode on ``device``.

    Args:
        model_name: OpenCLIP architecture, e.g. ``"ViT-B-32"``.
        pretrained: Weights tag or checkpoint path. ``None`` gives random init,
            which the random-initialisation control runs rely on.
        device: Target device; defaults to cuda when available.

    Returns:
        tuple: ``(model, preprocess_val, tokenizer)``. The train transform is
        deliberately not returned -- see the module docstring.
    """
    dev = resolve_device(device)
    model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(dev).eval()
    return model, preprocess_val, tokenizer
