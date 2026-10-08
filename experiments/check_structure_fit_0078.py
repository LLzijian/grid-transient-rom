"""Post-fit integrity audit for 0078 checkpoints; never reads development data."""
from __future__ import annotations
import argparse, json, math
import numpy as np
from pathlib import Path
import torch

FROZEN_COMMON=('mass','input_scale','q_transform','z_scale','voltage_star','load_positions','load_g','load_star','load_scale','p_scale','frozen_incidence','frozen_coupling','frozen_damping','frozen_delta_star','force_incidence','force_coupling','force_damping','force_delta_star')
G_FROZEN=('structured.mass','structured.D','structured.delta_star','structured.phi','structured.scales','structured.gain','structured.coupling_mask','structured.damping_mask','structured.jpi_p_mask','structured.rpp_mask','structured.storage_base_mask')

def _load(path): return torch.load(Path(path)/'model.pt',map_location='cpu',weights_only=True)
def _same(a,b,key): return key in a and key in b and torch.equal(a[key],b[key])
def _horizon(step):
    if step<=100:return 1.
    if step<=300:return (1.,2.)[(step-101)%2]
    return (1.,2.,5.)[(step-301)%3]

def _novel_auto_dissipation(state):
    represented={i for i,v in enumerate(state['structured.damping_mask']) if v!=0}
    represented.update(10+i for i,v in enumerate(state['structured.rpp_mask']) if v!=0)
    for key in sorted(k for k in state if k.startswith('structured.dissipation_a.')):
        direction=state[key]
        if int(torch.count_nonzero(direction))!=1 or float(direction.sum())!=1.: return False
        coordinate=int(direction.argmax())
        if coordinate in represented: return False
        represented.add(coordinate)
    return True
def _monitor_ok(m):
    if not isinstance(m,dict) or not isinstance(m.get('per_event'),list) or len(m['per_event'])!=1038:return False
    hs=m.get('horizons',{})
    if set(hs)!={'1','2','5'}: return False
    ids=None; risks=[]
    for h in hs:
        rows=[r for r in m['per_event'] if r.get('horizon_s')==float(h)]
        current=[r.get('event_id') for r in rows]
        if len(rows)!=346 or None in current or len(set(current))!=346 or (ids is not None and set(current)!=ids): return False
        ids=set(current); values=np.asarray([r.get('error',np.nan) for r in rows],dtype=float)
        if not np.isfinite(values).all() or (values<0).any(): return False
        mean=float(values.mean()); tail=float(np.sort(values)[-35:].mean()); risk=.8*mean+.2*tail
        if hs[h].get('n_events')!=346 or any(not math.isclose(float(hs[h].get(k,np.nan)),v,rel_tol=1e-12,abs_tol=1e-12) for k,v in [('mean',mean),('tail35',tail),('risk',risk)]): return False
        risks.append(risk)
    return math.isclose(float(m.get('average_risk',np.nan)),float(np.mean(risks)),rel_tol=1e-12,abs_tol=1e-12)

def audit(initial: Path, final: Path, evidence: Path, *, loader=None):
    initial,final,evidence=map(Path,(initial,final,evidence)); meta0=json.loads((initial/'checkpoint.json').read_text()); metaf=json.loads((final/'checkpoint.json').read_text()); ev=json.loads(evidence.read_text())
    arm=metaf['arm']; s0,sf=_load(initial),_load(final); checks={}
    checks['arm_match']=meta0['arm']==arm==ev.get('arm'); checks['step1200']=metaf.get('step')==1200 and final.name=='step1200' and Path(ev.get('final_checkpoint','')).resolve()==final.resolve()
    checks['initial_frozen_parameters_unchanged']=all(_same(s0,sf,k) for k,v in meta0.get('requires_grad',{}).items() if not v)
    checks['frozen_common']=all(_same(s0,sf,k) for k in FROZEN_COMMON)
    checks['head_bias_value_unchanged']=_same(s0,sf,'head.bias')
    if arm=='C-free':
        checks['no_g_structured']=not any(k.startswith('structured.') for k in sf)
        core_buffers=('core.mass','core.incidence','core.delta_star','core.coupling','core.damping','core.rate','core.gain','core.input_scale','core.z_scale','core.voltage_star','core.load_positions','core.load_g','core.load_star','core.load_scale')
        checks['c_core_buffers_frozen']=all(_same(s0,sf,k) for k in core_buffers)
        for k in ('core.support','core.alpha_support'):
            checks[k+'_subset_only']=k in s0 and k in sf and bool(torch.all(sf[k] <= s0[k]))
    else:
        checks['no_xi_g']=not any('xi' in k for k in sf)
        checks['g_frozen']=all(_same(s0,sf,k) for k in G_FROZEN)
        direction=[k for k in s0 if 'exchange_a' in k or 'exchange_b' in k or 'dissipation_a' in k]
        checks['fixed_directions']=all(_same(s0,sf,k) for k in direction) and all(not metaf['requires_grad'].get(k,True) for k in direction)
        new_direction=[k for k in sf if ('exchange_a' in k or 'exchange_b' in k or 'dissipation_a' in k) and k not in s0]
        checks['new_auto_coordinate_directions']=arm!='G-auto' or all(torch.count_nonzero(sf[k])==1 and sf[k].sum()==1 and not metaf['requires_grad'].get(k,True) for k in new_direction)
        checks['no_duplicate_base_or_extra_R']=arm!='G-auto' or _novel_auto_dissipation(sf)
    checks['head_bias_frozen']=not bool(metaf.get('requires_grad',{}).get('head.bias',False))
    traces=ev.get('traces',[]); steps=[r.get('step') for r in traces if 'horizon_s' in r]
    checks['trace_steps_1200']=steps==list(range(1,1201))
    checks['trace_horizons']=all(r.get('horizon_s')==_horizon(r['step']) for r in traces if 'horizon_s' in r)
    checks['finite_trace']=all(math.isfinite(float(r.get('loss',float('nan')))) and math.isfinite(float(r.get('gradient_norm_before_cap',float('nan')))) for r in traces if 'horizon_s' in r)
    growth=[r for r in traces if 'growth' in r]; prune=[r for r in traces if 'pruning' in r]
    checks['growth_rule']=(len(growth)==2 and [r.get('step') for r in growth]==[400,800] and all(len(r['growth'].get('proposals',[]))==8 for r in growth) and len(metaf.get('extra_groups',[]))<=16) if arm=='G-auto' else len(growth)==0
    checks['prune_step1000']=len(prune)==1 and prune[0].get('step')==1000
    checks['ports_bound']=metaf.get('counts',{}).get('active_groups',{}).get('ports',0)<=2
    before={k:v for k,v in ev.get('hashes',{}).items() if k!='_source_path'}
    after={k:v for k,v in ev.get('hashes_after',{}).items() if k!='_source_path'}
    checks['source_hashes_unchanged']=bool(before) and before==after
    checks['monitors_346']=set(ev.get('monitors',{}))=={'0','800_before_growth','1200'} and all(_monitor_ok(ev['monitors'][k]) for k in ev['monitors'])
    checks['provenance_sampling_gap']=True  # traces intentionally lack per-step event IDs.
    if loader is not None:
        model=loader(final); probe=torch.linspace(-.02,.02,66,dtype=torch.float64).reshape(2,33)
        with torch.no_grad():
            f=model.field(probe); shifted=probe.clone(); shifted[:,:10]+=.37
            checks['strictload_finite']=bool(torch.isfinite(f).all())
            checks['origin']=bool(model.field(torch.zeros_like(probe)).abs().max()<=1e-10)
            checks['kinematic']=bool((f[:,:10]-probe[:,10:20]).abs().max()<=1e-12)
            checks['angle_gauge']=bool(torch.allclose(f,model.field(shifted),atol=1e-10,rtol=1e-10))
        if arm!='C-free':
            ports=json.loads((final/'structured_numeric_export.json').read_text())['ports']
            checks['port_terms_bound']=len(ports)<=2 and all(len(p['terms'])<=8 for p in ports)
    return {'arm':arm,'checks':checks,'passed':all(v for k,v in checks.items() if k!='provenance_sampling_gap'),'provenance_gap':'per-step event IDs are not saved; matched sampling cannot be independently verified from traces'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--initial',type=Path,required=True); p.add_argument('--final',type=Path,required=True); p.add_argument('--evidence',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from train_structured_rom import load_model
    out=audit(a.initial,a.final,a.evidence,loader=load_model); a.output.write_text(json.dumps(out,indent=2)); print(json.dumps(out)); return 0
if __name__=='__main__': raise SystemExit(main())
