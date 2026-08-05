"""
Legacy placeholder for pca_text_encoder_v4.

This module has been unified and relocated to `analysis.pca_text_encoder`.
Please use:
    python -m analysis.pca_text_encoder
"""

import warnings
from analysis.pca_text_encoder import main

warnings.warn(
    "evaluation.pca_text_encoder_v4 is deprecated and moved to analysis.pca_text_encoder.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    main()
