"""
Shared on-disk feature cache and run-provenance helpers.

Two concerns that every E1/E2 mechanistic script needs and none of them had:

1. **Feature cache.** The E1/E2 scripts each re-encode the same BEAF images and
   captions on every run. Beyond being slow, it means two experiments that claim
   to analyze "the same embeddings" have no guarantee that they do. `cached_encode`
   memoizes an encoder call keyed on ``(model, pretrained, kind, exact item list)``,
   so any two runs sharing that key provably share the arrays.

2. **Run provenance.** Summary JSONs recorded coefficients but not which checkpoint
   produced them, so a reported number could not be traced back to a backbone.
   `build_provenance` returns the block every summary should carry.

Kept out of ``config.py`` deliberately: that module is the pure config/geometry
single source of truth and performs no I/O.
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_CACHE_DIR = "logs/evaluation/cached_embeddings/feature_cache"


def make_cache_key(model: str, pretrained: str, kind: str, items: Sequence[str]) -> str:
    """
    Build a content hash identifying one encoder call.

    The key covers everything that changes the output: the architecture, the
    weights tag, what is being encoded (``kind``), and the exact ordered item
    list. Two calls collide only if they would produce identical arrays.

    ``kind`` must identify **what the closure returns**, not merely what is being
    encoded. Two call sites that encode the same images but return different tuples
    (say ``(norm, raw, flags)`` versus ``(norm, flags)``) would otherwise collide and
    hand each other the wrong arrays. The convention in this repo is
    ``"<slot>@<contents>"``, e.g. ``"image_pres@norm+raw+flags"`` and
    ``"image_pres@l2norm+flags"``.

    Args:
        model: OpenCLIP architecture name, e.g. "ViT-B-32".
        pretrained: Weights tag or checkpoint path.
        kind: Slot-and-contents label for the call site, e.g. "text_pos@norm+raw".
        items: Ordered image paths or caption strings.

    Returns:
        str: 32-hex-character digest.
    """
    h = hashlib.sha256()
    for part in (model, pretrained, kind, str(len(items))):
        h.update(str(part).encode("utf-8"))
        h.update(b"\x00")
    for item in items:
        h.update(str(item).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


def _cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"{key}.npz")


def cached_encode(
    compute_fn: Callable[[], Tuple[np.ndarray, ...]],
    *,
    model: str,
    pretrained: str,
    kind: str,
    items: Sequence[str],
    cache_dir: str = DEFAULT_CACHE_DIR,
    enabled: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Return ``compute_fn()``, reading from / writing to the on-disk cache.

    ``compute_fn`` must be a zero-argument closure returning a tuple of ndarrays
    (the shape returned by ``encode_images_unified`` / ``encode_texts_unified``).
    With ``enabled=False`` this is a transparent pass-through, so callers can
    wire the cache in unconditionally and gate it on a CLI flag.

    Args:
        compute_fn: Closure performing the actual encoder forward pass.
        model: Architecture name, part of the cache key.
        pretrained: Weights tag or checkpoint path, part of the cache key.
        kind: Slot-and-contents label (see :func:`make_cache_key`), part of the cache key.
        items: Ordered items being encoded, part of the cache key.
        cache_dir: Directory holding the ``.npz`` entries.
        enabled: When False, skip the cache entirely.
        verbose: Print hit/miss lines.

    Returns:
        tuple: The arrays ``compute_fn`` returned, in the same order.
    """
    if not enabled:
        return compute_fn()

    key = make_cache_key(model, pretrained, kind, items)
    path = _cache_path(cache_dir, key)

    if os.path.exists(path):
        try:
            with np.load(path, allow_pickle=False) as z:
                n = int(z["__n_arrays__"][0])
                arrays = tuple(z[f"arr_{i}"] for i in range(n))
            if verbose:
                print(f"      [cache hit ] {kind:12s} n={len(items):5d}  {key[:12]}")
            return arrays
        except Exception as exc:  # corrupt or partially written entry
            print(f"      [cache warn] unreadable entry {key[:12]} ({exc}); recomputing")

    arrays = compute_fn()
    if not isinstance(arrays, tuple):
        arrays = (arrays,)

    os.makedirs(cache_dir, exist_ok=True)
    payload = {f"arr_{i}": np.asarray(a) for i, a in enumerate(arrays)}
    payload["__n_arrays__"] = np.array([len(arrays)], dtype=np.int64)
    tmp = path + ".tmp.npz"  # np.savez appends .npz unless the name already ends with it
    np.savez(tmp, **payload)
    os.replace(tmp, path)  # atomic: a killed run never leaves a half-written entry

    if verbose:
        print(f"      [cache miss] {kind:12s} n={len(items):5d}  {key[:12]} -> stored")
    return arrays


def get_git_commit() -> Optional[str]:
    """Return the current short commit hash, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def build_provenance(args, **extra) -> dict:
    """
    Build the provenance block that every summary JSON should carry.

    Pulls the standard fields off an argparse namespace when present, so a script
    that lacks e.g. ``--min_pairs`` simply omits it. ``extra`` adds run-specific
    entries (dataset row counts, filter thresholds).

    Args:
        args: Parsed argparse namespace.
        **extra: Additional key/value pairs to record.

    Returns:
        dict: Provenance block, safe to ``json.dump``.
    """
    fields = [
        "model", "pretrained", "csv_path", "vision_csv", "text_csv",
        "per_pair_csv", "image_root", "min_pairs", "min_samples", "seed",
        "batch_size", "restrict_objects", "use_cache", "no_bias",
    ]
    prov = {f: getattr(args, f) for f in fields if hasattr(args, f)}
    prov["git_commit"] = get_git_commit()
    prov["run_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prov.update(extra)
    return prov


def inherit_upstream_provenance(input_path: str) -> Optional[dict]:
    """
    Recover the provenance of the run that produced ``input_path``.

    Scripts that only read another experiment's CSV (significance tests, sanity
    checks) never touch a model, so they cannot state which backbone their numbers
    came from. They can, however, point at the summary JSON sitting beside their
    input and carry its provenance forward, which is what makes a derived number
    traceable to a checkpoint.

    Args:
        input_path: Path to the upstream CSV being consumed.

    Returns:
        dict | None: The upstream ``provenance`` block, or None when the upstream
        run predates provenance recording (in which case the caller should mark
        the result as untraceable rather than assume defaults).
    """
    directory = os.path.dirname(os.path.abspath(input_path))
    if not os.path.isdir(directory):
        return None
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("provenance"), dict):
            prov = dict(data["provenance"])
            prov["_source_summary"] = name
            return prov
    return None


def load_object_restriction(spec: Optional[str]) -> Optional[List[str]]:
    """
    Resolve ``--restrict_objects`` into an explicit concept list.

    Accepts a comma-separated list, or a path to a newline- or comma-separated
    text file, or a JSON/CSV artifact carrying an ``object_name`` column so a
    prior experiment's exact concept set can be replayed verbatim.

    Args:
        spec: CLI value, or None to apply no restriction.

    Returns:
        list[str] | None: Sorted unique concept names, or None when unrestricted.
    """
    if not spec:
        return None

    if os.path.exists(spec):
        if spec.endswith(".csv"):
            import pandas as pd
            names = pd.read_csv(spec)["object_name"].astype(str).tolist()
        elif spec.endswith(".json"):
            with open(spec, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("analyzed_objects") or data.get("object_names") or []
            names = [str(x) for x in data]
        else:
            with open(spec, encoding="utf-8") as f:
                names = [t for t in f.read().replace(",", "\n").split("\n")]
    else:
        names = spec.split(",")

    names = sorted({n.strip() for n in names if n and n.strip()})
    return names or None


def resolve_upstream_artifact(
    path: str,
    *,
    produced_by: str,
    required: bool = True,
    label: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve a path holding another experiment's output.

    ``logs/`` is gitignored, so every cross-experiment default points at something
    a fresh clone does not have. The three call sites that read an upstream
    artifact each handled that differently: one raised a clear error, one let
    pandas raise from inside ``read_csv``, and one silently set its dataframe to
    ``None`` -- dropping a cross-check out of the report with nothing to say so.
    A reader of that report could not tell the check had been skipped.

    This makes the outcome one of exactly two things, chosen by the caller:
    a hard failure that names the command which produces the file, or a loud
    skip that the caller records in its own output.

    Args:
        path: The upstream file or directory.
        produced_by: The command that creates it, quoted in the error or warning
            so the reader can act without going to look for it.
        required: Fail when missing (default) rather than returning None.
        label: Human-readable name for the artifact; defaults to its basename.

    Returns:
        str | None: ``path`` when it exists; ``None`` when it is missing and
        ``required`` is False.

    Raises:
        FileNotFoundError: When it is missing and ``required`` is True.
    """
    name = label or os.path.basename(path.rstrip("/")) or path

    if os.path.exists(path):
        return path

    if required:
        raise FileNotFoundError(
            f"{name} not found at: {path}\n"
            f"  This is another experiment's output, and logs/ is gitignored, so a fresh\n"
            f"  clone will not have it. Produce it with:\n"
            f"      {produced_by}\n"
            f"  or pass an existing path explicitly."
        )

    print(f"  [skipped] {name} not found at {path}")
    print(f"            Produce it with: {produced_by}")
    print(f"            Continuing without it; the report will record the omission.")
    return None
