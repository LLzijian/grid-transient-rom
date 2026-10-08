"""Loop0079 frozen force-span diagnostic; execution is TRAIN-only by CLI."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from train_joint_latent import BASE, NETWORK, DT, ROOTS, read_development
from train_structured_rom import load_model
from kiki.models.private_pm import MechanicalROM

QFIX=np.array([.7,-.4,.2]); SIM=np.array([[1,.2,0],[0,2,.3],[.1,0,.5]])

def sha(path: Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _relative(value, scale):
    if not (np.isfinite(value).all() and np.isfinite(scale).all()): raise FloatingPointError('nonfinite QA output')
    return float(np.max(np.abs(value))/max(1.,float(np.max(np.abs(scale)))))

def project(u: np.ndarray, L: np.ndarray, qenc: np.ndarray) -> dict[str,np.ndarray]:
    """Independent per-endpoint SVD projection with rtol=1e-12."""
    if u.ndim!=2 or L.shape!=(len(u),u.shape[1],3) or qenc.shape!=(len(u),3): raise ValueError('projection shape')
    if not (np.isfinite(u).all() and np.isfinite(L).all() and np.isfinite(qenc).all()): raise FloatingPointError('nonfinite projection input')
    qf=np.empty((len(u),3)); ranks=np.empty(len(u),int); singular=np.empty((len(u),3))
    for i,(ui,li) in enumerate(zip(u,L,strict=True)):
        a,s,b=np.linalg.svd(li,full_matrices=False); singular[i]=s; cutoff=(s[0]*1e-12 if len(s) else 0.); inv=np.zeros_like(s); np.divide(1.,s,out=inv,where=s>cutoff)
        qf[i]=(b.T*inv)@a.T@ui; ranks[i]=int(np.sum(s>cutoff))
    ep=u-np.einsum('nij,nj->ni',L,qf); ea=np.einsum('nij,nj->ni',L,qf-qenc); et=u-np.einsum('nij,nj->ni',L,qenc)
    return {'qF':qf,'e_perp':ep,'e_parallel':ea,'e_total':et,'ranks':ranks,'singular_values':singular}

def affine_map(model: Any,x: np.ndarray,q: np.ndarray) -> np.ndarray:
    """h=p-M*f_omega at fixed physical x and supplied q, batched."""
    z=torch.as_tensor(np.c_[x,q],dtype=torch.float64)
    with torch.no_grad(): f=model.field(z)[:,10:20].detach().cpu().numpy()
    return x[:,20:30]-np.asarray(model.mass.detach().cpu())*f

def target(base: MechanicalROM,event: dict, endpoints: np.ndarray) -> np.ndarray:
    x=event['x'][endpoints]; inc=np.asarray(base.incidence); phase=np.asarray(base.delta_star)@inc
    e0=((np.sin(x[:,:10]@inc+phase)-np.sin(phase))*np.asarray(base.coupling))@inc.T
    return np.asarray(event['force'])[endpoints]+e0+np.asarray(base.damping)*x[:,10:20]

def endpoint_block(model: Any,base: MechanicalROM,event: dict,endpoints: np.ndarray) -> dict[str,np.ndarray]:
    x=np.asarray(event['x'])[endpoints]; histories=np.stack([event['x'][i-10:i+1] for i in endpoints])
    with torch.no_grad(): q=model.encode(torch.as_tensor(histories,dtype=torch.float64)).detach().cpu().numpy()
    h0=affine_map(model,x,np.zeros_like(q)); L=np.stack([affine_map(model,x,np.broadcast_to(np.eye(3)[j],q.shape))-h0 for j in range(3)],-1)
    hq=affine_map(model,x,q); hfixed=affine_map(model,x,np.broadcast_to(QFIX,q.shape)); truth=target(base,event,endpoints); out=project(truth-h0,L,q)
    out.update({'T':truth,'h0':h0,'L':L,'qenc':q,'h_qenc':hq,'h_fixed':hfixed})
    return out

def qa(block: dict[str,np.ndarray]) -> dict[str,float|bool]:
    u=block['T']-block['h0']; L=block['L']; q=block['qenc']; p=block
    affine_q=_relative(p['h_qenc']-(p['h0']+np.einsum('nij,nj->ni',L,q)),p['h_qenc'])
    affine_fixed=_relative(p['h_fixed']-(p['h0']+np.einsum('nij,j->ni',L,QFIX)),p['h_fixed'])
    decomposition=_relative(p['e_total']-(p['e_perp']+p['e_parallel']),p['e_total'])
    orthogonal=_relative(np.einsum('nji,nj->ni',L,p['e_perp']),u)
    pythagoras=_relative(np.sum(p['e_total']**2,1)-np.sum(p['e_perp']**2,1)-np.sum(p['e_parallel']**2,1),np.sum(p['e_total']**2,1))
    ip=np.linalg.inv(SIM); pp=project(u,np.einsum('nij,jk->nik',L,ip),q@SIM.T)
    similarity=max(_relative(pp['e_perp']-p['e_perp'],p['e_perp']),_relative(pp['e_parallel']-p['e_parallel'],p['e_parallel']))
    rows={'affine_qenc':affine_q,'affine_fixed':affine_fixed,'decomposition':decomposition,'orthogonality':orthogonal,'pythagoras':pythagoras,'similarity':similarity}
    return {**rows,'passed':bool(max(rows.values())<=1e-9)}

def _event_summary(block: dict[str,np.ndarray], mass: np.ndarray) -> dict[str,float]:
    out={'raw_force_sq':float(np.mean(block['T']*block['T']))}
    for key in ('e_perp','e_parallel','e_total'):
        e=block[key]; out[key]=float(np.mean(e*e)); out['net_'+key]=float(np.mean(np.sum(e,1)**2)); out['common_accel_'+key]=float(np.mean((np.sum(e,1)/mass.sum())**2))
    return out

def _aggregate(rows: list[dict[str,Any]]) -> dict[str,Any]:
    keys=[k for k in rows[0] if k not in ('event_id','group','label','n_endpoints')]
    out={'n_events':len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in keys}}
    if 'n_endpoints' in rows[0]: out['n_endpoints']=sum(r['n_endpoints'] for r in rows)
    return out

def _report(rows: list[dict[str,Any]]) -> dict[str,Any]:
    out=_aggregate(rows)
    for key,value in list(out.items()):
        if key!='n_events' and (key=='raw_force_sq' or key.startswith(('e_','net_','common_'))): out[key+'_rms']=float(np.sqrt(value))
    return out

def _hashes(checkpoint:Path)->dict[str,str]:
    paths={'model':checkpoint/'model.pt','checkpoint':checkpoint/'checkpoint.json','normalization':checkpoint/'normalization.npz','base':Path(BASE),'network':Path(NETWORK),'execution':Path(__file__),'trainer':ROOT/'experiments'/'train_structured_rom.py','reader':ROOT/'experiments'/'train_joint_latent.py','core':ROOT/'experiments'/'structured_rom.py','grammar':ROOT/'experiments'/'structure_grammar.py'}
    if (checkpoint/'structured_numeric_export.json').exists():paths['export']=checkpoint/'structured_numeric_export.json'
    return {k:sha(v) for k,v in paths.items()}

def _dataset_hashes(): return {str(root):{name:sha(root/name) for name in ('dataset_index.csv','splits.json')} for root in ROOTS}

def _evaluation_hash_match(checkpoint: Path) -> dict[str,bool]:
    path=checkpoint.parent.with_name(checkpoint.parent.name+'_eval')/'evidence.json'
    meta=json.loads(path.read_text())['metadata']['source_hashes_before']
    return {'evaluation_evidence_present':True,'model':meta.get('model')==sha(checkpoint/'model.pt'),'normalization':meta.get('normalization')==sha(checkpoint/'normalization.npz'),'base':meta.get('base')==sha(Path(BASE)),'network':meta.get('network')==sha(Path(NETWORK))}

def _original_identities_match(checkpoint: Path, identities: dict) -> bool:
    extension=json.loads((checkpoint.parent/'evidence.json').read_text())
    evaluation=json.loads((checkpoint.parent.with_name(checkpoint.parent.name+'_eval')/'evidence.json').read_text())
    return identities==extension['identities']==evaluation['metadata']['identities']

def run(g_checkpoint:Path,c_checkpoint:Path,output:Path)->dict[str,Any]:
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(1); before={'G':_hashes(g_checkpoint),'C':_hashes(c_checkpoint)}; dataset_before=_dataset_hashes(); evaluation_hashes={'G':_evaluation_hash_match(g_checkpoint),'C':_evaluation_hash_match(c_checkpoint)}
    if not all(all(v.values()) for v in evaluation_hashes.values()): raise AssertionError('checkpoint does not match extension evaluation provenance')
    for checkpoint,arm in ((g_checkpoint,'G-fixed'),(c_checkpoint,'C-free')):
        meta=json.loads((checkpoint/'checkpoint.json').read_text())
        if meta['arm']!=arm or meta['step']!=3000: raise AssertionError('requires fixed3000 arm')
        if not _original_identities_match(checkpoint,dataset_before): raise AssertionError('original0078 dataset identity changed')
    base=MechanicalROM.load(BASE); network=dict(np.load(NETWORK,allow_pickle=False)); network['observed_buses']=np.unique(np.r_[network['load_buses'],network['generator_buses']]); network['load_positions']=np.searchsorted(network['observed_buses'],network['load_buses'])
    events,excluded,identities=read_development(base,network,full_train=True); train=events['train']
    if len(train)!=346 or sum(e['label']=='stable' for e in train)!=311 or sum(e['label']=='unstable' for e in train)!=35 or sum(v=='train' for v in excluded.values())!=23:raise AssertionError('unexpected frozen TRAIN counts')
    if identities!=dataset_before: raise AssertionError('reader identities mismatch direct CSV/split hashes')
    output.mkdir()
    models={'G-fixed':load_model(g_checkpoint,base=base),'C-free':load_model(c_checkpoint,base=base)}; arrays={}; result={'loop':'0079','test_read':False,'validation_materialized_by_reader':True,'n_train':len(train),'identities':identities,'excluded':excluded,'hashes_before':before,'extension_evaluation_hash_match':evaluation_hashes,'train_event_ids':[e['id'] for e in train],'mass':np.asarray(base.mass).tolist(),'models':{}}
    for model in models.values(): model.eval()
    current={}
    try:
      for name,model in models.items():
        event_rows=[]; initial_rows=[]; bin_rows={}; qa_rows=[]; pieces={k:[] for k in ('T','h0','L','qenc','qF','e_perp','e_parallel','e_total','singular_values','ranks','event_index','endpoint')}
        for ei,event in enumerate(train):
            endpoints=np.arange(10,len(event['x']),dtype=int); per=[]
            for start in range(0,len(endpoints),256):
                ep=endpoints[start:start+256]; current={'model':name,'event_id':event['id'],'endpoint':int(ep[0]),'x':event['x'][ep].tolist(),'history':event['x'][ep[0]-10:ep[0]+1].tolist()}; block=endpoint_block(model,base,event,ep); current.update({k:block[k].tolist() for k in ('T','h0','L','qenc')}); check=qa(block); qa_rows.append(check)
                if not check['passed']:raise AssertionError(f'QA failed {name} {event["id"]} {int(ep[0])}')
                for k in ('T','h0','L','qenc','qF','e_perp','e_parallel','e_total','singular_values','ranks'):pieces[k].append(block[k])
                pieces['event_index'].append(np.full(len(ep),ei)); pieces['endpoint'].append(ep)
                per.append({k:block[k] for k in ('T','e_perp','e_parallel','e_total')})
            joined={k:np.concatenate([x[k] for x in per]) for k in per[0]}; row={'event_id':event['id'],'group':str(event['group']),'label':event['label'],'n_endpoints':len(endpoints),**_event_summary(joined,np.asarray(base.mass))}; event_rows.append(row)
            initial_rows.append({'event_id':event['id'],'group':str(event['group']),'label':event['label'],'n_endpoints':1,**_event_summary({k:joined[k][:1] for k in joined},np.asarray(base.mass))})
            for label,lo,hi in (('[.1,1)',.1,1.),('[1,2)',1.,2.),('[2,5)',2.,5.),('[5,8)',5.,8.),('[8,inf)',8.,np.inf)):
                mask=(endpoints*DT>=lo)&(endpoints*DT<hi)
                if mask.any():
                    blockbin={k:v[mask] for k,v in joined.items()}; bin_rows.setdefault(label,[]).append({'event_id':event['id'],'group':str(event['group']),'label':event['label'],'n_endpoints':int(mask.sum()),**_event_summary(blockbin,np.asarray(base.mass))})
        overall=_report(event_rows); overall['raw_force_rms']=float(np.sqrt(overall['raw_force_sq'])); denom=overall['e_perp']+overall['e_parallel']; overall['share_perp']=None if denom==0 else float(overall['e_perp']/denom)
        groups={g:_report([r for r in event_rows if r['group']==g]) for g in sorted({r['group'] for r in event_rows})}
        if len(groups)!=24: raise AssertionError('expected all 24 TRAIN groups')
        subsets={s:_report([r for r in event_rows if r['label']==s]) for s in ('stable','unstable')}
        bins={key:(_report(value) | {'n_endpoints':int(sum(r['n_endpoints'] for r in value))} if value else {'n_events':0,'n_endpoints':0}) for key,value in bin_rows.items()}
        singular=np.concatenate(pieces['singular_values'])
        result['models'][name]={'overall':overall,'event_metrics':event_rows,'groups':groups,'subsets':subsets,'time_bins':bins,'rank_counts':{str(k):int(np.sum(np.concatenate(pieces['ranks'])==k)) for k in range(4)},'singular_values_summary':{'min':singular.min(0).tolist(),'median':np.median(singular,axis=0).tolist(),'max':singular.max(0).tolist()},'qa_maxima':{k:float(max(x[k] for x in qa_rows)) for k in qa_rows[0] if k!='passed'},'initial_endpoint':_report(initial_rows)}
        arrays[name]={k:np.concatenate(v) for k,v in pieces.items()}
    except Exception as error:
      (output/'failure_evidence.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error),'test_read':False,'hashes_before':before,'dataset_hashes_before':dataset_before,'offending':current},indent=2)); raise
    after={'G':_hashes(g_checkpoint),'C':_hashes(c_checkpoint)}
    dataset_after=_dataset_hashes()
    if before!=after or dataset_before!=dataset_after:raise RuntimeError('source or dataset identity changed')
    result['hashes_after']=after; result['dataset_hashes_before']=dataset_before; result['dataset_hashes_after']=dataset_after
    np.savez_compressed(output/'projection_arrays.npz',**{f'{n}_{k}':v for n,a in arrays.items() for k,v in a.items()}); (output/'evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False)); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--g',type=Path,required=True);p.add_argument('--c',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(); run(a.g,a.c,a.output)
if __name__=='__main__':main()
