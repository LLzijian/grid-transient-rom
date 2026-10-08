"""Pytest bootstrap: put the ``src`` layout on ``sys.path`` so ``kiki`` imports.

The project uses a ``src/``-layout package that is not necessarily installed in
the active environment; this file makes ``python -m pytest`` work from a clean
checkout without an editable install.
"""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
