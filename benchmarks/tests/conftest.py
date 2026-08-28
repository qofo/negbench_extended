"""
Put the three coexisting import roots on sys.path.

`benchmarks/GUIDE.md` documents three invocation styles (`training.x`, `analysis.x`,
`benchmarks.src.analysis.x`) that all work from the repo root. The tests exercise
modules from more than one of them, so the same three roots go on the path here
rather than being duplicated into every test file.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "benchmarks"), os.path.join(REPO_ROOT, "benchmarks", "src")):
    if path not in sys.path:
        sys.path.insert(0, path)
