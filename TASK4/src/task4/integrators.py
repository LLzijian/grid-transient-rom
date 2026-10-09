import math,time
import torch
from torch.utils.checkpoint import checkpoint
from .dynamics import PowerGraph

def rk4(model,z,dt,training=True,origins=None):
    if dt<=0:raise ValueError('dt must be positive')
    if origins is None:fn=lambda x:model.field(x,create_graph=training)
    else:fn=lambda x:model.field(x,create_graph=training,origins=origins)
    a=fn(z);b=fn(z+dt*a/2);c=fn(z+dt*b/2);d=fn(z+dt*c)
    out=z+dt*(a+2*b+2*c+d)/6
    if not torch.isfinite(out).all():raise FloatingPointError('Nonfinite ODE state; not a physical instability label')
    return out

def rollout(model,z0,steps,sample_dt=.01,max_step=.01,training=True):
    reuse=bool(getattr(model,'reuse_origin_terms',False))
    if reuse and type(model) is not PowerGraph:
        raise ValueError('Origin reuse supports only the exact PowerGraph')
    if reuse and training and getattr(model,'checkpoint_steps',0)>0:
        raise ValueError('Origin reuse and training checkpoint recomputation are incompatible')
    if (reuse and getattr(model,'_compiled_field',None) is not None
            and not getattr(model,'_compiled_uses_origins',False)):
        raise ValueError('Origin reuse requires a trace built with explicit origin inputs')
    if training and getattr(model,'checkpoint_steps',0)>0 and getattr(model,'_compiled_field',None) is not None:
        raise ValueError('TorchScript and checkpoint recomputation are incompatible in this runtime; use eager with checkpoint or trace without checkpoint')
    # Local to this call: origin graphs are shared by all RK stages and belong
    # to the same backward; no cache survives on the model across updates.
    with torch.set_grad_enabled(training):origins=model.origin_terms(z0) if reuse else None
    sub=max(1,int(math.ceil(sample_dt/max_step-1e-10)));dt=sample_dt/sub
    z=z0;out=[z]
    width=max(1,int(getattr(model,'checkpoint_steps',0))) if training else 1
    def chunk(start,count):
        vals=[];zz=start
        for _ in range(count):
            if time.time()>=getattr(model,'save_deadline',float('inf')):
                raise TimeoutError('Save deadline reached during integration/recomputation')
            for __ in range(sub):zz=rk4(model,zz,dt,training,origins)
            vals.append(zz)
        return torch.stack(vals,dim=1)
    for start in range(0,steps,width):
        if time.time()>=getattr(model,'save_deadline',float('inf')):
            raise TimeoutError('Save deadline reached inside ODE integration')
        count=min(width,steps-start)
        if training and getattr(model,'checkpoint_steps',0)>0:
            part=checkpoint(lambda zz,count=count:chunk(zz,count),z,use_reentrant=False)
        else:part=chunk(z,count)
        out.extend(part.unbind(dim=1));z=part[:,-1]
    return torch.stack(out,dim=1)
