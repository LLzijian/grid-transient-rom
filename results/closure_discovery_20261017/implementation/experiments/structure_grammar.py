"""Explicit centered storage primitives for Loop 0076."""
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import torch

@dataclass(frozen=True)
class PrimitiveSpec:
    kind: str
    indices: tuple = ()
    coefficient: float = 1.0  # signed gamma initial value, external to primitive
    scale: tuple = ()
    phi: float = 0.0
    name: str = ""

def _spec(x): return x if isinstance(x, PrimitiveSpec) else PrimitiveSpec(**dict(x))

def primitive_value_and_gradient(spec: PrimitiveSpec | Mapping[str, Any], eta, s):
    """Analytic value, d/deta, d/ds with arbitrary leading batch dimensions."""
    spec = _spec(spec); k = spec.kind.lower().replace("-", "_"); idx = tuple(spec.indices)
    ge, gs = torch.zeros_like(eta), torch.zeros_like(s)
    if k in ("sin", "sin_eta", "centered_sin"):
        j=idx[0]; z=eta[...,j]; ph=torch.as_tensor(spec.phi,dtype=eta.dtype,device=eta.device)
        v=torch.sin(ph+z)-torch.sin(ph)-torch.cos(ph)*z; ge[...,j]=torch.cos(ph+z)-torch.cos(ph)
    elif k in ("cos", "cos_eta", "centered_cos"):
        j=idx[0]; z=eta[...,j]; ph=torch.as_tensor(spec.phi,dtype=eta.dtype,device=eta.device)
        v=torch.cos(ph+z)-torch.cos(ph)+torch.sin(ph)*z; ge[...,j]=-torch.sin(ph+z)+torch.sin(ph)
    elif k in ("quadratic", "state_quadratic", "sq"):
        i=idx[0]; sc=spec.scale[0] if spec.scale else 1.; z=s[...,i]/sc; v=.5*z*z; gs[...,i]=z/sc
    elif k in ("product", "state_product", "cross"):
        i,j=idx
        if i == j: raise ValueError("product requires two distinct state indices; use quadratic")
        si,sj=spec.scale or (1.,1.); v=(s[...,i]/si)*(s[...,j]/sj); gs[...,i]=s[...,j]/(si*sj); gs[...,j]=s[...,i]/(si*sj)
    elif k in ("state_sin", "state_sin_difference", "sin_state"):
        i,e=idx; sc=spec.scale[0] if spec.scale else 1.; z=eta[...,e]; ph=torch.as_tensor(spec.phi,dtype=eta.dtype,device=eta.device); ss=s[...,i]/sc
        diff=torch.sin(ph+z)-torch.sin(ph); v=ss*diff; gs[...,i]=diff/sc; ge[...,e]=ss*torch.cos(ph+z)
    elif k in ("state_cos", "state_cos_difference", "cos_state"):
        i,e=idx; sc=spec.scale[0] if spec.scale else 1.; z=eta[...,e]; ph=torch.as_tensor(spec.phi,dtype=eta.dtype,device=eta.device); ss=s[...,i]/sc
        diff=torch.cos(ph+z)-torch.cos(ph); v=ss*diff; gs[...,i]=diff/sc; ge[...,e]=-ss*torch.sin(ph+z)
    else: raise ValueError(f"unknown storage primitive kind: {spec.kind}")
    return v,ge,gs

def primitive_value(spec,eta,s): return primitive_value_and_gradient(spec,eta,s)[0]
def primitive_value_gradient(spec,eta,s,create_graph=True): return primitive_value_and_gradient(spec,eta,s)
def primitive_sum(specs: Sequence[PrimitiveSpec|Mapping[str,Any]],eta,s):
    out=eta.sum(dim=-1)*0.
    for raw in specs:
        sp=_spec(raw); out=out+sp.coefficient*primitive_value(sp,eta,s)
    return out
def as_specs(specs): return tuple(_spec(x) for x in (specs or ()))
