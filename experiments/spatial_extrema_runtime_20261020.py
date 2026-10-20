"""Same-trajectory physical spatial-extrema supervision for fixed L10-S ROMs."""
from __future__ import annotations
import math
import torch
from closure_training_runtime import rollout_errors
from train_local_memory_0098 import graph, risk
from train_structured_rom import METRIC, _point_error


def _physical_inputs(pred, mass, equilibrium_delta):
    if pred.ndim < 1 or pred.shape[-1] < 30 or not pred.is_floating_point():
        raise ValueError('observations must contain30 floating point coordinates')
    if not torch.is_tensor(equilibrium_delta):
        raise ValueError('explicit bound equilibrium_delta tensor required')
    mass = torch.as_tensor(mass,dtype=pred.dtype,device=pred.device)
    eq = equilibrium_delta.to(dtype=pred.dtype,device=pred.device)
    if mass.shape != (10,) or not bool((torch.isfinite(mass) & (mass > 0)).all()):
        raise ValueError('ten finite positive machine masses required')
    if eq.shape != (10,) and eq.shape != pred.shape[:-1]+(10,):
        raise ValueError('equilibrium_delta must have shape10 or match the observation batch')
    if not bool(torch.isfinite(eq).all()):
        raise ValueError('finite physical equilibrium angles required')
    return mass,eq


def _extrema_error(pred,truth,mass,eq):
    # No host reads: this kernel is also used within every genuine RK4 step.
    dp,dt = pred[...,:10]+eq,truth[...,:10]+eq
    ap,at = dp.amax(-1)-dp.amin(-1),dt.amax(-1)-dt.amin(-1)
    wp,wt = pred[...,10:20],truth[...,10:20]
    rp = wp-(wp*mass).sum(-1,keepdim=True)/mass.sum()
    rt = wt-(wt*mass).sum(-1,keepdim=True)/mass.sum()
    sp,st = rp.abs().amax(-1),rt.abs().amax(-1)
    angle = ((ap-at)/(2*METRIC[0])).square()
    speed = ((sp-st)/METRIC[1]).square()
    return .5*(angle+speed),angle,speed


def extrema_error(pred,truth,mass,equilibrium_delta):
    """Return(s, normalized angle-range squared error, normalized W squared error).

    Subterms exclude the factor0.5. Angles are physical unwrapped radians;
    omega is rad/s and its common component is removed using machine masses.
    """
    if pred.ndim < 1 or truth.ndim < 1 or pred.shape[:-1] != truth.shape[:-1] or truth.shape[-1] < 30:
        raise ValueError('prediction/truth observation batches must match')
    if pred.device != truth.device or pred.dtype != truth.dtype:
        raise ValueError('prediction/truth dtype and device must match')
    mass,eq = _physical_inputs(pred,mass,equilibrium_delta)
    return _extrema_error(pred,truth,mass,eq)


def spatial_rollout_errors(model,prepared,equilibrium_delta,initial=None):
    """Per-event(original point mean, s mean, angle mean, speed mean), one encode."""
    h,n = prepared.history,prepared.counts
    if h.ndim != 3 or h.shape[1:] != (11,30) or not len(h):
        raise ValueError('nonempty11x30 causal histories required')
    mass,eq = _physical_inputs(h[:,-1],model.mass,equilibrium_delta)
    if n.ndim != 1 or len(n) != len(h) or n.device != h.device or not bool((torch.isfinite(n)&(n>=1)&(n==n.floor())).all()):
        raise ValueError('positive integral true counts matching histories required')
    segments = []; coverage = torch.zeros_like(n); start = 0
    for ix,truth in prepared.segments:
        if ix.ndim != 1 or ix.dtype != torch.long or ix.device != h.device or truth.ndim != 3 or truth.shape[1:] != (len(ix),30) or not len(truth) or truth.device != h.device:
            raise ValueError('invalid genuine segment')
        expected = torch.nonzero(n>=start+len(truth),as_tuple=False).flatten()
        if not torch.equal(ix,expected):
            raise ValueError('segments require exactly ascending genuine active rows')
        coverage = coverage.index_add(0,ix,torch.full_like(ix,len(truth),dtype=coverage.dtype))
        segments.append((ix,truth,eq if eq.ndim==1 else eq.index_select(0,ix)))
        start += len(truth)
    if not segments or not torch.equal(coverage,n):
        raise ValueError('segment coverage must equal genuine event counts')
    state = torch.cat((h[:,-1],model.encode(h)),-1) if initial is None else initial
    sums = [h.new_zeros(len(h)) for _ in range(4)]
    for ix,truth,segment_eq in segments:
        current = state.index_select(0,ix)
        partial = [h.new_zeros(len(ix)) for _ in range(4)]
        for target in truth.unbind(0):
            current = model.rk4(current,.01)
            values = (_point_error(current,target,model),*_extrema_error(current,target,mass,segment_eq))
            partial = [s+v for s,v in zip(partial,values)]
        state = state.index_copy(0,ix,current)
        sums = [s.index_add(0,ix,v) for s,v in zip(sums,partial)]
    return tuple(s/n for s in sums)


def objective(model,store,samples,horizon,force,equilibrium_delta,spatial_weighted=False):
    if not isinstance(spatial_weighted,bool):
        raise ValueError('spatial_weighted must be a frozen bool')
    if not math.isfinite(horizon) or horizon<=0 or not math.isfinite(force) or force<=0:
        raise ValueError('finite positive horizon and force required')
    long_batch = store.prepare([(e,10) for e,_ in samples],horizon)
    local_batch = store.prepare(samples,.2)
    long,spatial,angle,speed = spatial_rollout_errors(model,long_batch,equilibrium_delta)
    history = local_batch.history
    state = torch.cat((history[:,-1],model.encode(history)),-1)
    local = rollout_errors(model,local_batch,initial=state).mean()
    adjacent = store.histories(samples,adjacent=True)
    hidden = ((model.rk4(state,.01)[:,30:]-model.encode(adjacent))/.1).square().mean()
    predicted = state[:,20:30]-model.mass*model.field(state)[:,10:20]
    target = torch.stack([store.events[e['id']][1][i] for e,i in samples])
    torque = ((predicted-target)/force).square().mean()
    terms = {'risk':risk(long),'mean':long.mean(),'local':local,'latent_consistency':hidden,
        'torque':torque,'complexity':model.complexity(state),'spatial_risk':risk(spatial),
        'spatial_mean':spatial.mean(),'angle_range_mse':angle.mean(),'relative_speed_max_mse':speed.mean()}
    total = 10*terms['risk']+local+.1*hidden+.1*torque
    return (total+10*terms['spatial_risk'] if spatial_weighted else total),terms


def _tensors(value):
    if torch.is_tensor(value): yield value
    elif isinstance(value,dict):
        for v in value.values(): yield from _tensors(v)
    elif isinstance(value,(tuple,list)):
        for v in value: yield from _tensors(v)


def _finite(value):
    return all(bool(torch.isfinite(t).all()) for t in _tensors(value))


def one_update(model,opt,store,samples,horizon,force,equilibrium_delta,spatial_weighted=False):
    opt.zero_grad(set_to_none=True)
    try:
        with graph(model):
            value,components = objective(model,store,samples,horizon,force,equilibrium_delta,spatial_weighted)
            if not _finite([value,components]): raise FloatingPointError('nonfinite complete objective')
            value.backward()
        params = [p for p in model.parameters() if p.requires_grad]
        if not _finite([p.grad for p in params]): raise FloatingPointError('nonfinite complete gradient')
        norm = torch.nn.utils.clip_grad_norm_(params,100.)
        if not _finite(norm): raise FloatingPointError('nonfinite clipped gradient norm')
        opt.step(); model.end_graph_cache()
        if not _finite([model.state_dict(),opt.state_dict()]): raise FloatingPointError('nonfinite Adam state')
        return {'loss':float(value.detach()),'gradient_norm':float(norm.detach()),
                'components':{k:float(v.detach()) for k,v in components.items()}}
    finally:
        model.end_graph_cache()
