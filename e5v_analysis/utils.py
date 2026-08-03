"""
E5-V Analysis Utilities
========================
Common data loading, prompt templates, and helper functions
shared across E5-V NegBench evaluation and interpretability analysis.
"""

import os
import logging
from typing import List, Optional

# ---------------------------------------------------------------------------
# Llama-3 prompt templates (from E5-V README)
# ---------------------------------------------------------------------------

LLAMA3_TEMPLATE = (
    '<|start_header_id|>user<|end_header_id|>\n\n'
    '{}<|eot_id|>'
    '<|start_header_id|>assistant<|end_header_id|>\n\n \n'
)

IMG_PROMPT = LLAMA3_TEMPLATE.format('<image>\nSummary above image in one word: ')
TEXT_PROMPT_FMT = '<sent>\nSummary above sentence in one word: '


def build_text_prompt(text: str) -> str:
    """Wrap a raw caption into the E5-V text embedding prompt."""
    return LLAMA3_TEMPLATE.format(
        TEXT_PROMPT_FMT.replace('<sent>', text)
    )


def build_text_prompts(texts: List[str]) -> List[str]:
    """Wrap a list of captions into E5-V text embedding prompts."""
    return [build_text_prompt(t) for t in texts]


def build_img_prompts(n: int) -> List[str]:
    """Return a list of n identical image prompts."""
    return [IMG_PROMPT] * n


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def resolve_image_path(image_path: str, data_root: Optional[str] = None) -> str:
    """
    Resolve a potentially relative image path from the CSV.

    CSV paths are typically like 'data/coco/images/val2017/000000000139.jpg'.
    If data_root is provided, the path is joined with it.
    If the path already exists as-is, return it directly.
    """
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path

    if data_root:
        resolved = os.path.join(data_root, image_path)
        if os.path.exists(resolved):
            return resolved

    # Try common prefixes
    for prefix in ['benchmarks', '.']:
        candidate = os.path.join(prefix, image_path)
        if os.path.exists(candidate):
            return candidate

    # Return as-is; caller should handle FileNotFoundError
    return image_path


def setup_logging(log_dir: str, name: str = "e5v_analysis") -> logging.Logger:
    """Set up a logger that writes to both console and a file."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, "out.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    return logger
