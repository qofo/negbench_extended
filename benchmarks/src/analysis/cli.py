"""
Shared argparse groups for the analysis and evaluation entrypoints.

Thirty-nine entrypoints each re-declared the same flags: ``--model`` in 29 of
them, ``--seed`` in 25, ``--batch_size`` in 26. Nothing tied those copies
together, so a flag added later reached only the scripts someone remembered to
edit -- which is why ``--use_cache`` and ``--restrict_objects`` sat at 6 and 7
call sites while the older flags were everywhere, and why pinning a concept set
had to be re-plumbed script by script.

Each group here declares one concern. A script calls the groups matching the
flags it actually honors: an accepted flag that changes nothing is worse than an
absent one, so nothing here adds a flag a caller does not implement.

Defaults that genuinely differ between experiments (``--output_dir``,
``--min_pairs``, ``--batch_size``) are parameters of these functions rather than
constants, and ``--min_pairs`` has no default at all -- see ``add_concept_args``.
"""

import argparse
from typing import Optional

from .feature_cache import DEFAULT_CACHE_DIR

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "openai"
DEFAULT_SEED = 42
DEFAULT_IMAGE_ROOT = "benchmarks/data/images"
DEFAULT_COUNTERFACTUAL_CSV = "benchmarks/data/images/beaf_counterfactual_6col.csv"


def add_model_args(parser: argparse.ArgumentParser,
                   model: str = DEFAULT_MODEL,
                   pretrained: Optional[str] = DEFAULT_PRETRAINED) -> argparse.ArgumentParser:
    """Backbone selection. ``--pretrained ''`` gives random init for control runs."""
    parser.add_argument("--model", type=str, default=model)
    parser.add_argument("--pretrained", type=str, default=pretrained)
    return parser


def add_run_args(parser: argparse.ArgumentParser, output_dir: str,
                 seed: Optional[int] = DEFAULT_SEED,
                 batch_size: Optional[int] = 128) -> argparse.ArgumentParser:
    """
    Where the run writes, and the knobs that make it reproducible.

    Pass ``seed=None`` or ``batch_size=None`` to leave that flag out for a script
    that has no use for it. Skipping is explicit so a flag is never accepted by a
    script that ignores it.
    """
    parser.add_argument("--output_dir", type=str, default=output_dir)
    if seed is not None:
        parser.add_argument("--seed", type=int, default=seed)
    if batch_size is not None:
        parser.add_argument("--batch_size", type=int, default=batch_size)
    return parser


def add_data_args(parser: argparse.ArgumentParser,
                  csv_path: Optional[str] = DEFAULT_COUNTERFACTUAL_CSV,
                  image_root: Optional[str] = DEFAULT_IMAGE_ROOT) -> argparse.ArgumentParser:
    """
    Input CSV and the root that its relative image paths resolve against.

    Either may be ``None``: some scripts take two CSVs under their own names, and
    some never touch an image.
    """
    if csv_path is not None:
        parser.add_argument("--csv_path", type=str, default=csv_path)
    if image_root is not None:
        parser.add_argument("--image_root", type=str, default=image_root)
    return parser


def add_cache_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """On-disk encoder feature cache. Only add this where cached_encode is wired in."""
    parser.add_argument("--use_cache", action="store_true", default=False,
                        help="Reuse on-disk encoder features keyed by (model, pretrained, items)")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    return parser


def add_restriction_args(parser: argparse.ArgumentParser,
                         help_text: Optional[str] = None) -> argparse.ArgumentParser:
    """Pin the concept set so a run is comparable to another experiment's."""
    parser.add_argument(
        "--restrict_objects", type=str, default=None,
        help=help_text or ("Comma list, or path to txt/csv/json, limiting evaluation to an "
                           "exact concept set (use to share another experiment's set verbatim)"))
    return parser


def add_concept_args(parser: argparse.ArgumentParser, min_pairs: int,
                     help_text: Optional[str] = None) -> argparse.ArgumentParser:
    """
    Minimum counterfactual pairs a concept needs to be evaluated.

    ``min_pairs`` is required, with no fallback, because it silently selects the
    population: the defaults in this repo range over 6, 10 and 20, and the paper's
    33-concept set only appears at 20. Running the E2 decomposition on its own
    default of 10 yields 53 concepts instead -- a different experiment wearing the
    same name. Every caller therefore has to state its threshold, and provenance
    records it.
    """
    parser.add_argument(
        "--min_pairs", type=int, default=min_pairs,
        help=help_text or (f"Minimum counterfactual pairs per concept (default: {min_pairs}; "
                           f"the paper's 33-concept set requires 20)"))
    return parser


def add_bias_args(parser: argparse.ArgumentParser,
                  help_text: Optional[str] = None) -> argparse.ArgumentParser:
    """``--no_bias`` tests whether a result survives without an intercept absorbing class priors."""
    parser.add_argument("--no_bias", "--no-bias", action="store_true", default=False,
                        help=help_text or "Disable bias/intercept in linear probes (default: bias enabled)")
    return parser
