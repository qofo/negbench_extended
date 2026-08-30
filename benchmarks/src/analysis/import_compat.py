"""
Guard for the dual-path imports that let a script run under two conventions.

Several entrypoints are importable two ways -- ``benchmarks.src.evaluation.x``
from the repo root, or ``evaluation.x`` via the editable install -- and they
carry the corresponding pair of import blocks::

    try:
        from benchmarks.src.analysis.feature_cache import cached_encode
    except ImportError:
        from analysis.feature_cache import cached_encode

A bare ``except ImportError`` catches far more than the case it exists for. A
typo in a module name, a function renamed on one side only, a genuine circular
import -- all of them fall through to the second block, which then either works
(hiding the mistake until someone runs the other way) or fails with an error
pointing at the wrong line. The two branches drifted apart exactly this way.

``reraise_unless_standalone`` narrows the catch to its actual purpose: the
fallback is for when the ``benchmarks`` package is not importable at all. If it
*is* importable, the ImportError came from something else and must propagate.
"""

import importlib.util


def reraise_unless_standalone() -> None:
    """
    Re-raise the in-flight ImportError unless ``benchmarks`` is genuinely absent.

    Call this as the first statement of an ``except ImportError`` block that
    guards a ``benchmarks.src.*`` import. It is a no-op in the case the fallback
    is meant for, and re-raises everything else.

    Raises:
        ImportError: The original error, when ``benchmarks`` is importable and
            the failure therefore was not about the invocation convention.
    """
    try:
        found = importlib.util.find_spec("benchmarks") is not None
    except (ImportError, ValueError):
        found = False
    if found:
        raise
