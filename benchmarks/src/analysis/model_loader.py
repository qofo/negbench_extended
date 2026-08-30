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


def get_embed_dim(model: Any) -> int:
    """
    Read the joint embedding width from the backbone instead of assuming it.

    Three entrypoints wrote ``embed_dim = 512`` immediately after loading the model.
    That is ViT-B/32's width, and ViT-B/16's, so it went unnoticed -- but ViT-L/14 is
    768, and the two failure modes differ:

    - ``eval_per_object_alignment_intervention`` builds ``np.eye(embed_dim)`` for the
      baseline and the random-rotation control, so a wrong width raises at the first
      matmul. Loud, but it blocks the run.
    - ``eval_4condition_decomposition`` and ``eval_unary_mechanistic_analysis`` pass it
      to ``encode_images_safely``, which only uses it to shape the zero row that stands
      in for an image that failed to load. With every image loading, the wrong value is
      never touched and the run *succeeds*; one unreadable file turns it into a confusing
      concatenation error partway through.

    Two model layouts are covered: the standard ``CLIP``, where ``visual.output_dim``
    holds the joint width, and ``CustomTextCLIP`` (SigLIP among others), where the towers
    are separate modules, ``visual`` carries no ``output_dim``, and the text tower states
    it instead. SigLIP is what surfaced the second case -- and it surfaced as the intended
    exception rather than as a silent 512.

    Args:
        model: An OpenCLIP model (or any object exposing the width; see below).

    Returns:
        int: Joint embedding dimension.

    Raises:
        AttributeError: When the width cannot be read, rather than guessing 512.
    """
    visual = getattr(model, "visual", None)
    width = getattr(visual, "output_dim", None)
    if isinstance(width, int):
        return width

    # SigLIP and friends are ``CustomTextCLIP``: the towers are separate modules and
    # ``visual`` carries no output_dim, but the text tower states the joint width.
    text_tower = getattr(model, "text", None)
    width = getattr(text_tower, "output_dim", None)
    if isinstance(width, int):
        return width

    proj = getattr(model, "text_projection", None)
    if proj is not None:
        if hasattr(proj, "weight"):        # nn.Linear
            return int(proj.weight.shape[0])
        if hasattr(proj, "shape"):         # nn.Parameter (d_model, embed_dim)
            return int(proj.shape[-1])

    raise AttributeError(
        f"cannot read the embedding width from {type(model).__name__}; "
        "pass it explicitly rather than assuming 512"
    )
