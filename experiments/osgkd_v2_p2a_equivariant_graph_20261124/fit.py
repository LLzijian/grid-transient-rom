"""Fixed graph training recipe and numerical guards."""
from collections import Counter, defaultdict
import math
import numpy as np
import torch
from experiments.osgkd_v2_fast_zero_sum_20261124 import run as old
from kiki.models.osgkd import OSGKDConfig
from kiki.models.osgkd_graph import OSGKDGraph
from kiki.models.osgkd_fitting import IntegrationWork, integrate as fine_integrate, tensor_events
require=old.require
ATTEMPTS,CAP,GUARD_CAP=2048,75_366_400,600_000

def plan(events):
    s=sorted(e['id'] for e in events if e['label']=='stable')
    u=sorted(e['id'] for e in events if e['label']=='unstable')
    require((len(s),len(u))==(342,43) and len(set(s+u))==385,'train population differs')
    s=np.random.default_rng(24112).permutation(s).tolist(); u=np.random.default_rng(24113).permutation(u).tolist()
    batches=[[s[(7*k+j)%342] for j in range(7)]+[u[k%43]] for k in range(ATTEMPTS)]
    require(all(len(set(b))==8 for b in batches) and len(set(sum(batches,[])))==385,'batch coverage differs')
    return dict(seeds=dict(model=24111,stable=24112,unstable=24113),batches=batches,coverage=dict(Counter(sum(batches,[]))))

def horizon(k):
    require(type(k) is int and 0<=k<ATTEMPTS,'invalid attempt')
    return 200 if k<512 else 400 if k<896 else 800

def multiplier(k):
    horizon(k)
    return .15+.425*(1+math.cos(math.pi*k/2047))

def new_model(constants):
    torch.manual_seed(24111)
    if constants.get('candidate_graph')=='undirected_complete_K10':
        require(len(constants['theta_ref'])==10,'audited K10 graph requires ten machines')
        edges=np.stack(np.triu_indices(10,k=1))
        require('edge_index' not in constants or np.array_equal(edges,constants['edge_index']),'audited edge ordering differs from complete K10 declaration')
    else:
        edges=constants['edge_index']
    model=OSGKDGraph(OSGKDConfig('C1',hidden_size=20,p_init_bound=4.,a_init_bound=2.,e_init_bound=4.,c_init=float(np.median(constants['Se']))/10,beta_fraction_init=.1),edge_index=edges,**{key:constants[key] for key in old.p1a.old.MODEL_CONSTANTS}).cpu().double()
    with torch.no_grad():
        for coeff in (model.mechanical_coeff,model.amplitude_coeff,model.memory_coeff):
            coeff.zero_(); coeff[0].fill_(-1)
    require(sum(p.numel() for p in model.encoder.parameters())==1928,'encoder size differs')
    return model

def new_optimizer(model): return old.p1a.old.new_optimizer(model)

def integrate(model,state,steps,h,work):
    if h==.005: return fine_integrate(model,state,steps,work=work)
    require(h==.01,'unsupported training mesh')
    output=[state]; batch=state.numel()//(5*model.n)
    for _ in range(steps):
        state=model.rk4_step(state,.01); work.rhs[work.phase]+=4*batch; output.append(state)
    return torch.stack(output,dim=-2)

def components(model,events,constants,work,h,limit,paths=None):
    theta,omega=[torch.stack([e['history'][key] for e in events]) for key in ('theta','omega')]
    initial=model.initialize(theta,omega); work.encoder_events+=len(events)
    groups=defaultdict(list); rows=[None]*len(events)
    for i,e in enumerate(events):
        count=min(limit,len(e['targets']['theta'])-1); require(count>0,'missing future'); groups[count].append(i)
    scale=torch.as_tensor(constants['S_m'],dtype=torch.float64)
    for count,indices in groups.items():
        path=integrate(model,initial[indices],count,h,work); ports=model.ports(path[:,1:])
        for local,index in enumerate(indices):
            target=events[index]['targets']
            if paths is not None: paths[index]=path[local]
            motion=old.prior.segmented_motion(path[local],target,model.mass,model.omega_scale,constants['s_coi'])
            terms=[]
            for key,pred,s in (('tau_m',ports.tau_m[local],scale),('tau_e',ports.tau_e[local],model.Se)):
                mask=events[index]['masks'][key][1:count+1]
                if mask.any(): terms.append(((pred-target[key][1:count+1])/s).square()[mask].mean())
            port=torch.stack(terms).mean() if terms else motion.new_zeros(())
            consistency=motion.new_zeros(())
            if count>=10:
                encoded=model.initialize(target['theta'][:11],target['omega'][:11]); work.encoder_events+=1
                consistency=((path[local,10,2*model.n:]-encoded[2*model.n:])/model.init_bounds).square().mean()
            rows[index]=torch.stack((motion,port,consistency))
    return torch.stack(rows)

def objective(comp): return (comp[:,0]+.1*comp[:,1]+.01*comp[:,2]).mean()

def gradient_compare(a,b):
    na,nb=float(a.norm()),float(b.norm()); difference=float((a-b).norm())
    if max(na,nb)<=1e-9: return dict(passed=difference<=1e-9,norm_main=na,norm_fine=nb,difference=difference,relative=None,cosine=None)
    rel=difference/max(nb,1e-30); cosine=float(torch.dot(a,b)/(a.norm()*b.norm())) if min(na,nb)>0 else -1.
    return dict(passed=rel<=.05 and cosine>=.995,norm_main=na,norm_fine=nb,difference=difference,relative=rel,cosine=cosine)

def guard(model,batch,constants,work,limits=(200,800)):
    rows=[]; encoder={id(p) for p in model.encoder.parameters()}
    groups=[[p for p in model.parameters() if (id(p) in encoder)==flag] for flag in (True,False)]
    for limit in limits:
        output=[]
        for h in (.01,.005):
            model.zero_grad(set_to_none=True); paths={}
            loss=objective(components(model,batch,constants,work,h,limit,paths))
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
    return dict(passed=all(r['passed'] for r in rows),checks=rows,work=work.snapshot())

def attempt(model,opt,batch,k,constants,work,h):
    for group,base in zip(opt.param_groups,(.001,.0003),strict=True): group['lr']=base*multiplier(k)
    opt.zero_grad(set_to_none=True); paths={}; comp=components(model,batch,constants,work,h,horizon(k),paths); loss=objective(comp)
    row=dict(attempt=k+1,event_ids=[e['id'] for e in batch],horizon_s=horizon(k)/100,internal_step=h,lr_multiplier=multiplier(k),accepted=False,loss=float(loss.detach()) if bool(torch.isfinite(loss)) else None)
    if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(p).all()) for p in paths.values()): return {**row,'rejection':'nonfinite_forward'}
    loss.backward(); norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
    if not math.isfinite(norm): return {**row,'rejection':'nonfinite_gradient'}
    opt.step(); require(old.prior.finite_tree(model.state_dict()) and old.prior.finite_tree(opt.state_dict()),'nonfinite parameter or Adam: immediate stop')
    with torch.no_grad(): model.project_parameters_()
    require(old.prior.finite_tree(model.state_dict()),'nonfinite projection')
    return {**row,'accepted':True,'gradient_norm':norm,'clipped':norm>5.}
