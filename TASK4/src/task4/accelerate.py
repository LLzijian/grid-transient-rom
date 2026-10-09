"""Optional TorchScript execution of exactly the registered power graph.

No approximation, changed precision, external compiler, or additional package.
The eager path remains the diagnostic reference and state_dict remains unchanged.
"""
import time
import warnings
import torch
from .dynamics import PowerGraph

class _Field(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):return self.model.field(x,create_graph=True)

class _FieldWithOrigins(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x,node0,edge_angle0,edge_left0,edge_right0,port0):
        # Explicit inputs preserve the graph back to the eager origin builder;
        # no parameter-dependent origin is captured as a trace constant.
        origins=(node0,edge_angle0,edge_left0,edge_right0,port0)
        return self.model.field(x,create_graph=True,origins=origins)

def trace_field(model):
    reuse=bool(getattr(model,'reuse_origin_terms',False))
    if reuse and type(model) is not PowerGraph:
        raise ValueError('Origin reuse supports only the exact PowerGraph')
    if getattr(model,'_compiled_field',None) is not None:
        if bool(getattr(model,'_compiled_uses_origins',False))!=reuse:
            raise ValueError('Origin reuse mode changed after tracing; rebuild a fresh execution trace')
        return
    started=time.monotonic()
    example=torch.zeros(2,model.dim,dtype=model.mass.dtype,device=model.mass.device)
    wrapper=_FieldWithOrigins(model) if reuse else _Field(model)
    with torch.enable_grad():examples=(example,*model.origin_terms(example)) if reuse else example
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=torch.jit.TracerWarning)
        traced=torch.jit.trace(wrapper,examples,check_trace=False,strict=False)
    original=dict(model.named_parameters())
    for name,p in traced.named_parameters():
        key=name.removeprefix('model.')
        if key not in original or p.data_ptr()!=original[key].data_ptr():
            raise RuntimeError('Traced parameters do not alias the optimized model')
    # Do not register a recursive child module or serialize the execution cache.
    object.__setattr__(model,'_compiled_field',traced)
    object.__setattr__(model,'_compiled_uses_origins',reuse)
    model.trace_build_seconds=time.monotonic()-started
