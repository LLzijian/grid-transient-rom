"""Small controlled compiler/discovery checks, never a replacement for real fits.

Observed-coordinate cases use linear-in-coefficient field identification followed
by autonomous validation. The partial-observation case fits coefficients/GRU
against trajectories. Their different information privileges are explicit.
"""
from __future__ import annotations
import copy
import itertools
from pathlib import Path
import time
import numpy as np
import torch


def v(name):return {'op':'var','name':name}
def mul(a,b):return {'op':'mul','args':[a,b]}
def power(a,n):return a if n==1 else {'op':f'pow{n}','arg':a}


def toy_source():
    from structured_rom import StructuredROM
    from local_memory_rom_0098 import LocalMemoryROM,_inv_softplus
    from symbolic_closure_rom import SymbolicClosureROM
    D=np.zeros((10,10))
    for i in range(10):D[i,i]=1;D[i,(i+1)%10]=-1
    base=StructuredROM(mass=np.ones(10),D=D,coupling=np.full(10,.2),
          delta_star=np.linspace(-.05,.05,10),damping=np.full(10,.15),gain=np.ones(10),
          jpi_p=np.full(10,.3),rpp=np.full(10,.4),n_q=0)
    base.rate=torch.full((10,),.4,dtype=torch.float64)
    old=LocalMemoryROM(base,input_scale=np.ones(20),p_scale=np.ones(10),seed=4801)
    with torch.no_grad():
        old.template_mask[:]=0;old.template_mask[3]=old.template_mask[5]=1
        old._template_scalars[3].fill_(.3)
        old._template_scalars[5].copy_(_inv_softplus(torch.tensor(.4,dtype=torch.float64)))
    return SymbolicClosureROM.from_source(old).with_terms([])


def pool():
    """Use the production expression generator, including one parent-tree expansion.

    Candidate generation sees syntax only, never fixture targets or fitted errors.
    The component test enumerates proposals before the budgeted beam screening.
    """
    from closure_discovery_support import _candidate_terms,fixed_library,term_key
    first=_candidate_terms(fixed_library())
    node_parents=[term for term in first if term['channel']=='node']
    second=_candidate_terms(node_parents)
    unique={term_key(t):t for t in first+second}
    return [unique[k] for k in sorted(unique)]


def path(model,initial,steps=100):
    from train_local_memory_0098 import graph
    state=initial;rows=[state]
    with graph(model):
        for _ in range(steps):state=model.rk4(state,.01);rows.append(state)
    return torch.stack(rows,1)


def identify(base,truth,states,noise,rng,candidates):
    """Only observed-coordinate fixtures may supply the complete state/RHS."""
    with torch.no_grad():
        y=truth.field(states)-base.field(states)
        if noise:y=y+torch.as_tensor(rng.normal(size=y.shape))*noise*y.std(0).clamp_min(1e-8)
    scores=[]
    for term in candidates:
        m=base.with_terms([term])
        with torch.no_grad():
            m.coefficients.fill_(1.);b=m.field(states)-base.field(states)
            c=(b*y).sum()/b.square().sum().clamp_min(1e-20)
            error=float((y-c*b).square().mean())
        scores.append((error,term,float(c)))
    best=min(scores,key=lambda r:r[0]);recovered=base.with_terms([best[1]])
    with torch.no_grad():recovered.coefficients.fill_(best[2])
    return recovered,best


def controlled_benchmarks(output,deadline=None):
    from discover_closure import write
    from symbolic_closure_rom import term_key
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(4801);base=toy_source();candidates=pool()
    node={'channel':'node','expression':mul(power(v('p'),2),power(v('q'),2))}
    edge={'channel':'edge','expression':mul({'op':'add','args':[v('qi'),mul({'op':'const','value':-1.},v('qj'))]},v('s'))}
    saturating={'channel':'port','expression':{'op':'tanh','scale':2.,'arg':v('q')}}
    write(output/'truths_before_fits.json',{'seed':4801,'cases':[node,edge,saturating],
            'outside':'port=q/(1+abs(q)); excluded from candidate syntax',
            'pool':candidates,'scope':'controlled implementation evidence'})
    rows=[]
    for name,term,coefficient in (('node',node,.2),('edge',edge,.15)):
        truth=base.with_terms([term])
        with torch.no_grad():truth.coefficients.fill_(coefficient)
        truth.export(output/f'truth_{name}.json')
        states=torch.as_tensor(rng.normal(0,.3,size=(128,40)),dtype=torch.float64)
        initial=torch.as_tensor(rng.normal(0,.2,size=(8,40)),dtype=torch.float64)
        with torch.no_grad():target=path(truth,initial)
        for noise in (0.,.01):
            if deadline and time.perf_counter()>deadline-60:raise TimeoutError('controlled cases incomplete')
            recovered,best=identify(base,truth,states,noise,rng,candidates)
            with torch.no_grad():prediction=path(recovered,initial)
            exact=(term_key(term)==term_key(best[1]))
            error=float((prediction-target).square().mean())
            # Symmetrically equivalent edge terms can differ by a scalar factor.
            field_error=float((recovered.field(states)-truth.field(states)).square().mean().detach())
            rows.append({'case':name,'noise':noise,'recovered_term':best[1],'coefficient':best[2],
                'same_canonical_term':exact,'field_mse':field_error,'autonomous_mse':error,
                'information':'complete hidden coordinates and field available in this controlled fixture'})
    # Partial observations: q0 comes only from the frozen synthetic observer;
    # the candidate fits its own observer and coefficient using physical paths.
    truth=base.with_terms([saturating])
    with torch.no_grad():truth.coefficients.fill_(.6)
    history=torch.as_tensor(rng.normal(0,.3,size=(8,11,30)),dtype=torch.float64)
    with torch.no_grad():initial=torch.cat((history[:,-1],truth.encode(history)),-1);target=path(truth,initial,20)[...,:30]
    for noise in (0.,.01):
        target_noisy=target.clone()
        if noise:target_noisy+=torch.as_tensor(rng.normal(size=target.shape))*noise*target.std((0,1)).clamp_min(1e-8)
        partial_pool=[t for t in candidates if t['channel']=='port' and
                      ((t['expression']['op']=='var' and t['expression']['name'] in ('p','w','q')) or
                       (t['expression']['op']=='tanh' and t['expression']['arg']==v('q')))]
        fitted=[]
        for proposal in partial_pool:
            m=base.with_terms([proposal]);opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001)
            loss0=None
            for i in range(128):
                if deadline and time.perf_counter()>deadline-60:raise TimeoutError('partial observation controlled case incomplete')
                opt.zero_grad(set_to_none=True)
                initial=torch.cat((history[:,-1],m.encode(history)),-1)
                pred=path(m,initial,20)[...,:30];loss=(pred-target_noisy).square().mean()
                if loss0 is None:loss0=float(loss.detach())
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),100.);opt.step()
            fitted.append({'proposal':proposal,'initial_mse':loss0,'final_mse':float(loss.detach())})
        best=min(fitted,key=lambda r:r['final_mse'])
        rows.append({'case':'partial_saturation','noise':noise,**best,'all_candidates':fitted,
                     'same_canonical_term':term_key(best['proposal'])==term_key(saturating),
                     'updates_per_candidate':128,'information':'100ms physical history; q not a fit target',
                     'scope':'finite blind candidate comparison; small hidden amplitudes can leave saturation scales indistinguishable'})
    # Out-of-grammar probe uses a deliberately omitted rational law. We preserve
    # approximation error instead of falsely reporting exact symbolic recovery.
    states=torch.as_tensor(rng.normal(0,1.5,size=(128,40)),dtype=torch.float64)
    target=states[:,30:]/(1+states[:,30:].abs())
    errors=[]
    for term in candidates:
        if term['channel']!='port':continue
        candidate=base.with_terms([term])
        with torch.no_grad():
            candidate.coefficients.fill_(1.);b=candidate.port_values(candidate.interface_to_state(states))
            c=(b*target).sum()/b.square().sum().clamp_min(1e-20)
            errors.append(float((target-c*b).square().mean()))
    rows.append({'case':'outside_grammar','best_field_mse':min(errors),'exact_recovery_claim':False,
                 'scope':'approximation diagnostic; outside grammar is known to evaluator only'})
    implementation_pass=all(np.isfinite(r.get('autonomous_mse',r.get('final_mse',r.get('best_field_mse')))) for r in rows)
    clean=[r for r in rows if r.get('noise')==0 and r['case'] in ('node','edge')]
    implementation_pass=implementation_pass and all(r['field_mse']<1e-16 and r['autonomous_mse']<1e-16 for r in clean)
    result={'implementation_pass':bool(implementation_pass),'cases':rows,
        'scope':'observed-state symbolic identification plus partial-observer component test; full search recovery is evaluated separately',
        'test_read':False}
    write(output/'results.json',result);return result
