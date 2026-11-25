"""Matched fixed-budget training; no data are loaded by this module."""
import math
import numpy as np
import torch
from experiments.osgkd_v2_p2a_equivariant_graph_20261124 import fit as previous
from kiki.models.osgkd import OSGKDConfig
from kiki.models.osgkd_graph import OSGKDGraph
from kiki.models.osgkd_matched_flat import OSGKDMatchedFlat
from kiki.models.osgkd_spectral import reference_geometry, spectral_penalty
old=previous.old
require=previous.require
ATTEMPTS,CAP=previous.ATTEMPTS,previous.CAP
GUARD_CAP=previous.GUARD_CAP
horizon,multiplier=previous.horizon,previous.multiplier
new_optimizer,components,objective,guard=previous.new_optimizer,previous.components,previous.objective,previous.guard
gradient_compare=previous.gradient_compare


def plan(events):
    sequence = previous.plan(events)
    del sequence['seeds']['model']
    return sequence

def new_model(constants,kind='graph',seed=24111):
    torch.manual_seed(seed)
    if constants.get('candidate_graph')=='undirected_complete_K10':
        require(len(constants['theta_ref'])==10,'K10 requires ten machines')
        edges=np.stack(np.triu_indices(10,k=1))
        require('edge_index' not in constants or np.array_equal(edges,constants['edge_index']),'K10 edge ordering differs')
    else: edges=constants['edge_index']
    cls=OSGKDGraph if kind=='graph' else OSGKDMatchedFlat
    require(kind in ('graph','flat'),'unknown initializer')
    model=cls(OSGKDConfig('C1',hidden_size=20 if kind=='graph' else 32,p_init_bound=4.,a_init_bound=2.,e_init_bound=4.,c_init=float(np.median(constants['Se']))/10,beta_fraction_init=.1),edge_index=edges,**{key:constants[key] for key in old.p1a.old.MODEL_CONSTANTS}).cpu().double()
    with torch.no_grad():
        for coeff in (model.mechanical_coeff,model.amplitude_coeff,model.memory_coeff):
            coeff.zero_(); coeff[0].fill_(-1)
    require(encoder_parameters(model)==(1928 if kind=='graph' else 6144) or model.n!=10,'encoder size differs')
    return model

def encoder_parameters(model):
    return sum(p.numel() for p in model.encoder.parameters())+sum(p.numel() for p in model.initial_head.parameters()) if isinstance(model,OSGKDMatchedFlat) else sum(p.numel() for p in model.encoder.parameters())

def attempt(model,opt,batch,k,constants,work,h):
    for group,base in zip(opt.param_groups,(.001,.0003),strict=True): group['lr']=base*multiplier(k)
    opt.zero_grad(set_to_none=True); paths={}; comp=components(model,batch,constants,work,h,horizon(k),paths); data_loss=objective(comp); loss=data_loss
    spectral=dict(alpha=None,spec=0.,hinge_active=False)
    if k>=896:
        try:
            penalty,alpha=spectral_penalty(model,reference_geometry(model))
            if not bool(torch.isfinite(alpha)) or not bool(torch.isfinite(penalty)):
                return dict(attempt=k+1,event_ids=[e['id'] for e in batch],horizon_s=8,internal_step=h,accepted=False,rejection='nonfinite_spectrum',alpha=None,spec=None,hinge_active=None)
            loss=loss+penalty
            spectral=dict(alpha=float(alpha.detach()),spec=float(penalty.detach()),hinge_active=bool(alpha.detach()>-.05))
        except (RuntimeError,ValueError) as error:
            return dict(attempt=k+1,event_ids=[e['id'] for e in batch],horizon_s=8,internal_step=h,accepted=False,rejection='nonfinite_spectrum',spectral_error=str(error),alpha=None,spec=None,hinge_active=None)
    row=dict(**spectral,data_loss=float(data_loss.detach()) if bool(torch.isfinite(data_loss)) else None,attempt=k+1,event_ids=[e['id'] for e in batch],horizon_s=horizon(k)/100,internal_step=h,lr_multiplier=multiplier(k),accepted=False,loss=float(loss.detach()) if bool(torch.isfinite(loss)) else None)
    if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(p).all()) for p in paths.values()): return {**row,'rejection':'nonfinite_forward'}
    loss.backward(); norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
    if not math.isfinite(norm): return {**row,'rejection':'nonfinite_gradient'}
    opt.step(); require(old.prior.finite_tree(model.state_dict()) and old.prior.finite_tree(opt.state_dict()),'nonfinite parameter or Adam: immediate stop')
    with torch.no_grad(): model.project_parameters_()
    require(old.prior.finite_tree(model.state_dict()),'nonfinite projection')
    return {**row,'accepted':True,'gradient_norm':norm,'clipped':norm>5.}

def guard(model,batch,constants,work,limits=(200,800)):
    rows=[]; encoder={id(p) for p in list(model.encoder.parameters())+list(model.initial_head.parameters())}
    groups=[[p for p in model.parameters() if (id(p) in encoder)==flag] for flag in (True,False)]
    for limit in limits:
        output=[]
        for h in (.01,.005):
            model.zero_grad(set_to_none=True); paths={}
            loss=objective(components(model,batch,constants,work,h,limit,paths))
            if limit==800: loss=loss+spectral_penalty(model,reference_geometry(model))[0]
            finite=bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p).all()) for p in paths.values()); gradients=None
            if finite:
                loss.backward()
                gradients=[torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten() for p in group]).detach().clone() for group in groups]
                finite=all(bool(torch.isfinite(g).all()) for g in gradients)
            output.append((loss.detach(),{k:p.detach() for k,p in paths.items()},gradients,finite))
        main,fine=output; require(fine[3],'nonfinite fine qualification: blocked')
        item=dict(horizon_s=limit/100,main_finite=main[3],fine_finite=True,passed=False)
        if main[3]:
            scale=torch.cat((torch.ones(model.n),model.omega_scale,model.init_bounds)).double()
            delta=torch.cat([((main[1][i]-fine[1][i])/scale).flatten() for i in range(len(batch))])
            rms=float(delta.square().mean().sqrt()); maximum=float(delta.abs().max()); relative=float((main[0]-fine[0]).abs()/fine[0].abs().clamp_min(1e-12))
            grads=[gradient_compare(a,b) for a,b in zip(main[2],fine[2],strict=True)]
            item.update(loss_relative=relative,state_normalized_rms=rms,state_normalized_max=maximum,encoder_gradient=grads[0],ode_gradient=grads[1],passed=relative<=.02 and rms<=1e-4 and maximum<=1e-3 and all(g['passed'] for g in grads))
        rows.append(item)
    model.zero_grad(set_to_none=True); require(work.rhs['forward']<=GUARD_CAP,'guard cap exceeded')
    return dict(passed=all(r['passed'] for r in rows),checks=rows,work=work.snapshot(),spectral_jacobian_calls=2*sum(limit==800 for limit in limits))

