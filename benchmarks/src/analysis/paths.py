"""
Dataset path resolution shared by the mechanistic experiments.

Image paths inside the CSVs are relative (``data/coco/images/val2014/...``) and are
resolved against ``--image_root``. Four E1/E2 entrypoints each carried a private,
byte-identical copy of this function; keeping one copy means a change to the
resolution rule cannot apply to some experiments and not others.

Kept out of ``config.py``: that module is the pure config/geometry single source of
truth and performs no I/O.
"""

import os

__all__ = ["resolve_image_path"]


def resolve_image_path(p: str, root: str) -> str:
    """
    Resolve a CSV image path against ``root``.

    An absolute path is returned untouched. A relative path is joined onto ``root``
    when that lands on an existing file; otherwise the original relative path is
    returned, so the caller's own existence check reports the path as written in the
    CSV rather than a misleading joined one.

    Args:
        p: Path as written in the CSV.
        root: ``--image_root`` value.

    Returns:
        str: Resolved path.
    """
    if os.path.isabs(p):
        return p
    full = os.path.join(root, p)
    if os.path.exists(full):
        return full
    return p
