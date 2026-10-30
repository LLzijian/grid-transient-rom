"""Small explicit CPU-structure-then-device constructor for CUDA closure models."""
from __future__ import annotations


def construct_terms_on_cpu(cpu_source, terms, device):
    """Apply the existing structure transform on CPU, then move once to device."""
    import torch

    if cpu_source.mass.device.type != "cpu":
        raise ValueError("source model must be on CPU")
    target = torch.device(device)
    if target.type != "cuda" or target.index is None:
        raise ValueError("an explicitly indexed CUDA device is required")
    if cpu_source.mass.dtype != torch.float64:
        raise ValueError("source model must use float64")
    return cpu_source.with_terms(terms).to(target)
