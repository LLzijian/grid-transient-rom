"""Loop0080 frozen qF-at-100ms initialization diagnostic; TRAIN-only CLI."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from train_joint_latent import BASE, NETWORK, DT, ROOTS, read_development
from train_structured_rom import load_model, METRIC
from kiki.models.private_pm import MechanicalROM

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hashes(checkpoint,force):
    ps={'model':checkpoint/'model.pt','meta':checkpoint/'checkpoint.json','norm':checkpoint/'normalization.npz','export':checkpoint/'structured_numeric_export.json','base':BASE,'network':NETWORK,'force_npz':force/'projection_arrays.npz','force_evidence':force/'evidence.json','audit':Path(__file__),'trainer':ROOT/'experiments'/'train_structured_rom.py','reader':ROOT/'experiments'/'train_joint_latent.py'}
    ps.update(core=ROOT/'experiments/structured_rom.py',grammar=ROOT/'experiments/structure_grammar.py',extension_evidence=checkpoint.parent/'evidence.json')
    for r in ROOTS:
        ps[f'{r.name}_csv']=r/'dataset_index.csv'; ps[f'{r.name}_splits']=r/'splits.json'
    return {k:sha(v) for k,v in ps.items()}
def point(pred,truth,mass,p_scale):
    if len(pred)==0 or pred.shape[0]!=truth.shape[0] or not all(np.isfinite(v).all() for v in (pred,truth,mass,p_scale)) or np.any(mass<=0) or np.any(p_scale<=0): raise FloatingPointError('invalid scored point input')
    e=pred[:,:30]-truth[:,:30]; d=e[:,:10]-(e[:,:10]*mass).sum(1,keepdims=True)/mass.sum(); w=e[:,10:20]-(e[:,10:20]*mass).sum(1,keepdims=True)/mass.sum(); c=(e[:,10:20]*mass).sum(1)/mass.sum()
    return .5*((d/METRIC[0])**2).mean(1)+.5*((w/METRIC[1])**2).mean(1)+(c/METRIC[1])**2+((e[:,20:30]/p_scale)**2).mean(1)
def risk(values):
    values=np.asarray(values)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all(): raise FloatingPointError('invalid event risk')
    return {'n_events':len(values),'mean':float(values.mean()),'tail35':float(np.sort(values)[-35:].mean()),'risk':float(.8*values.mean()+.2*np.sort(values)[-35:].mean())}
def rollout(model,states,lengths,dt):
    """Autonomous masked RK4: no row advances beyond its genuine endpoint."""
    out=np.full((len(states),int(lengths.max())+1,33),np.nan); out[:,0]=states.detach().cpu().numpy(); cur=states
    with torch.no_grad():
      if hasattr(model,'begin_graph_cache'):model.begin_graph_cache()
      try:
       for k in range(1,out.shape[1]):
        ix=np.flatnonzero(lengths>=k)
        if not len(ix):break
        t=torch.as_tensor(ix); nxt=model.rk4(cur.index_select(0,t),dt); cur=cur.index_copy(0,t,nxt); out[ix,k]=nxt.cpu().numpy()
      finally:
       if hasattr(model,'end_graph_cache'):model.end_graph_cache()
    return out
def rollout32(model,states,lengths,dt):
    out=np.full((len(states),int(lengths.max())+1,33),np.nan)
    for start in range(0,len(states),32):
        ix=np.arange(start,min(start+32,len(states))); part=rollout(model,states[ix],lengths[ix],dt)
        out[ix,:part.shape[1]]=part
    return out
def numeric(fine,coarse,lengths,mass,p_scale):
    rows=[]
    for i,n in enumerate(lengths):
      a=fine[i,:2*n+1]; b=coarse[i,:n+1]
      if not(np.isfinite(a).all() and np.isfinite(b).all()):raise FloatingPointError(f'nonfinite numeric {i}')
      x=a[::2]-b; d=x[:,:10]-(x[:,:10]*mass).sum(1,keepdims=True)/mass.sum(); raw_w=x[:,10:20]; w=raw_w-(raw_w*mass).sum(1,keepdims=True)/mass.sum(); c=(raw_w*mass).sum(1)/mass.sum()
      rows.append({'coi':float(np.sqrt(.5*np.mean((d/METRIC[0])**2+(w/METRIC[1])**2))),'common':float(np.sqrt(np.mean((c/METRIC[1])**2))),'p':float(np.sqrt(np.mean((x[:,20:30]/p_scale)**2)))})
    return rows
def run(checkpoint,force,output):
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1); started=time.perf_counter()
    output.mkdir(); before=hashes(checkpoint,force); current={'stage':'provenance'}
    try:
     meta=json.loads((checkpoint/'checkpoint.json').read_text())
     if meta.get('arm')!='G-fixed' or meta.get('step')!=3000:raise AssertionError('frozen G-fixed3000 required')
     base=MechanicalROM.load(BASE); net=dict(np.load(NETWORK,allow_pickle=False)); net['observed_buses']=np.unique(np.r_[net['load_buses'],net['generator_buses']]);net['load_positions']=np.searchsorted(net['observed_buses'],net['load_buses'])
     events,excluded,identities=read_development(base,net,full_train=True); train=events['train']
     if len(train)!=346 or sum(e['label']=='stable' for e in train)!=311 or sum(e['label']=='unstable' for e in train)!=35:raise AssertionError('TRAIN identity')
     extension=json.loads((checkpoint.parent/'evidence.json').read_text())
     if identities!=extension['identities'] or sum(v=='train' for v in excluded.values())!=23 or len({e['group'] for e in train})!=24: raise AssertionError('0078 split/count mismatch')
     fe=json.loads((force/'evidence.json').read_text()); arr=np.load(force/'projection_arrays.npz',allow_pickle=False); ids=fe['train_event_ids']; ix=arr['G-fixed_event_index']; ep=arr['G-fixed_endpoint']; qf=arr['G-fixed_qF']; qe=arr['G-fixed_qenc']
     if fe['identities']!=identities or fe['hashes_before']['G']['model']!=sha(checkpoint/'model.pt') or fe['hashes_before']['G']['normalization']!=sha(checkpoint/'normalization.npz') or fe['hashes_before']['G']['base']!=sha(BASE) or fe['hashes_before']['G']['network']!=sha(NETWORK):raise AssertionError('0079 provenance mismatch')
     saved=np.full((346,3),np.nan)
     for i in range(346):
      hit=np.flatnonzero((ix==i)&(ep==10))
      if len(hit)!=1:raise AssertionError('0079 endpoint10 mapping')
      saved[i]=qf[hit[0]]
     if ids!=[e['id'] for e in train]:raise AssertionError('0079 event ordering')
     model=load_model(checkpoint,base=base);model.eval(); hist=np.stack([e['x'][:11] for e in train]);
     with torch.no_grad(): qenc=model.encode(torch.as_tensor(hist,dtype=torch.float64)).cpu().numpy()
    # exact endpoint selections rather than relying on an array ordering
     qsaved=np.stack([qf[np.flatnonzero((ix==i)&(ep==10))[0]] for i in range(346)])
     if np.max(abs(qenc-np.stack([qe[np.flatnonzero((ix==i)&(ep==10))[0]] for i in range(346)])))>1e-10:raise AssertionError('encoder/qenc mismatch')
     if not all(np.isfinite(v).all() for v in (hist,qenc,qsaved)): raise FloatingPointError('nonfinite initial state')
     lengths=np.array([min(500,len(e['x'])-11) for e in train]);
     if np.any(lengths<=0):raise AssertionError('no future')
     states=torch.as_tensor(np.c_[hist[:,-1],qenc],dtype=torch.float64); states_f=torch.as_tensor(np.c_[hist[:,-1],qsaved],dtype=torch.float64)
     if not torch.equal(states[:,:30],states_f[:,:30]): raise AssertionError('physical initial state changed')
     frozen={k:v.detach().clone() for k,v in model.state_dict().items()}
     current={'stage':'autonomous_rollouts','event_ids':ids}
     coarse_b=rollout32(model,states,lengths,.01); coarse_f=rollout32(model,states_f,lengths,.01); fine_b=rollout32(model,states,lengths*2,.005); fine_f=rollout32(model,states_f,lengths*2,.005)
     truth=np.full((346,int(lengths.max())+1,30),np.nan)
     for i,e in enumerate(train):truth[i,:lengths[i]+1]=e['x'][10:10+lengths[i]+1]
     pscale=model.p_scale.detach().cpu().numpy(); mass=model.mass.detach().cpu().numpy(); rows=[]; summaries={}; coarse_monitor={}
     # Persist even numerically invalid genuine forecasts before any scoring gate.
     np.savez_compressed(output/'trajectories.npz',event_ids=np.array(ids),labels=np.array([e['label'] for e in train]),mass=mass,p_scale=pscale,lengths=lengths,qenc=qenc,qF=qsaved,truth=truth,coarse_baseline=coarse_b,coarse_qF=coarse_f,fine_baseline=fine_b,fine_qF=fine_f)
     current={'stage':'numerical_QA'}
     num={'baseline':numeric(fine_b,coarse_b,lengths,mass,pscale),'qF':numeric(fine_f,coarse_f,lengths,mass,pscale)}
     if any(not np.isfinite(v) or v>.001 for arm in num.values() for r in arm for v in r.values()): raise FloatingPointError('numerical comparison unresolved')
     current={'stage':'scoring_and_monitor_reproduction'}; coarse_rows=[]
     for h in (1,2,5):
      vb=[];vf=[];vc=[]
      for i,n in enumerate(lengths):
       k=min(round(h/DT),n); vc.append(float(point(coarse_b[i,1:k+1],truth[i,1:k+1],mass,pscale).mean()));vb.append(float(point(fine_b[i,2:2*k+1:2],truth[i,1:k+1],mass,pscale).mean()));vf.append(float(point(fine_f[i,2:2*k+1:2],truth[i,1:k+1],mass,pscale).mean()));rows.append({'event_id':train[i]['id'],'label':train[i]['label'],'horizon':h,'available_steps':int(k),'baseline':vb[-1],'qF':vf[-1]})
      summaries[str(h)]={'baseline':risk(vb),'qF':risk(vf)};coarse_monitor[str(h)]=risk(vc)
      coarse_rows.extend({'event_id':e['id'],'horizon_s':float(h),'error':v} for e,v in zip(train,vc,strict=True))
     saved_monitor=extension['monitor3000']; expected={(r['event_id'],r['horizon_s']):r['error'] for r in saved_monitor['per_event']}
     actual={(r['event_id'],r['horizon_s']):r['error'] for r in coarse_rows}
     if len(expected)!=1038 or actual.keys()!=expected.keys(): raise AssertionError('monitor1038 identity mismatch')
     reproduction=max(abs(actual[k]-expected[k]) for k in expected)/max(1.,max(abs(v) for v in expected.values()))
     if reproduction>1e-9 or any(abs(coarse_monitor[h][k]-saved_monitor['horizons'][h][k])/max(1.,abs(saved_monitor['horizons'][h][k]))>1e-9 for h in coarse_monitor for k in ('mean','tail35','risk','n_events')):raise AssertionError('coarse monitor reproduction')
     prefixes=[r for r in rows if r['horizon']==5 and r['label']=='unstable']; prefix={'baseline_mean':float(np.mean([r['baseline'] for r in prefixes])),'qF_mean':float(np.mean([r['qF'] for r in prefixes])),'baseline_tail4':float(np.mean(sorted(r['baseline'] for r in prefixes)[-4:])),'qF_tail4':float(np.mean(sorted(r['qF'] for r in prefixes)[-4:]))}
     average={'baseline':float(np.mean([summaries[x]['baseline']['risk'] for x in summaries])),'qF':float(np.mean([summaries[x]['qF']['risk'] for x in summaries]))}
     after=hashes(checkpoint,force)
     if before!=after:raise RuntimeError('provenance changed')
     if any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()): raise AssertionError('in-memory model changed')
     ev={'test_read':False,'validation_materialized_by_reader':True,'identities':identities,'excluded':excluded,'hashes_before':before,'hashes_after':after,'n_train':346,'per_event':rows,'horizons':summaries,'coarse_monitor_reproduction':coarse_monitor,'average_risk':average,'unstable_prefix':prefix,'numerics':num}
     ev.update(coarse_event_errors=coarse_rows,coarse_monitor_max_relative_error=reproduction,seconds=time.perf_counter()-started,forecast_reference='.005s fixed',model_tensors_unchanged=True)
     (output/'evidence.json').write_text(json.dumps(ev,indent=2,allow_nan=False));return ev
    except Exception as error:
     (output/'failure_evidence.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error),'test_read':False,'hashes_before':before,'current':current},indent=2));raise
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--force',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.checkpoint,a.force,a.output)
if __name__=='__main__':main()
