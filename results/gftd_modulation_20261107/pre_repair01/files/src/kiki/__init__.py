"""kiki: minimal rebuild of the kron-reduce scientific core.

Modular physics-structured SINDy for IEEE39 post-clear dynamics:
M0 swing backbone + prunable energy-structured modules + causal Pm closure,
with local energy-structure audits, heuristic candidate-energy screening, and
modal frequency identification. Formal stability certification requires a
verified global escape-energy lower bound that is not yet implemented.
"""

from .data import CanonicalStore, ModelDataRequest, adapt_event, load_split

__all__ = ["CanonicalStore", "ModelDataRequest", "adapt_event", "load_split"]
