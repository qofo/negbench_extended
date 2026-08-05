"""
Legacy placeholder for pca_text_encoder (v3).

This module has been relocated to `analysis.pca_text_encoder` for domain organization.
Please use:
    python -m analysis.pca_text_encoder
"""

import warnings
from analysis.pca_text_encoder import main

warnings.warn(
    "evaluation.pca_text_encoder is deprecated and moved to analysis.pca_text_encoder.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    main()
