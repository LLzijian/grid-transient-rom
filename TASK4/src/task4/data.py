"""KIKI schema adapter, derived from the provided Loop0101 source.
Never imports/runs KIKI scripts. HDF5 is opened in 'r'; only explicit split IDs
are indexed. TEST payloads are unsupported by this starter implementation.
"""
from dataclasses import dataclass
from pathlib import Path
import csv, hashlib, json
import h5py
import numpy as np
from .paths import ROOT, source_root, source_file

@dataclass
class Event:
    id: str
    group: str
    label: str
    t: np.ndarray
    y: np.ndarray             # [time, machine, delta / rad/s / p(system pu)]
    torque: np.ndarray | None # raw te - Pm_star, same as KIKI's training target
    source: str
    metadata: dict | None = None

@dataclass
class Physics:
    mass: np.ndarray
    incidence: np.ndarray    # N x E, +/-1 columns
    delta_star: np.ndarray
    pm_star: np.ndarray
    coupling: np.ndarray
    damping: np.ndarray
    rate: np.ndarray
    gain: np.ndarray
    omega_s: float = 1.0
    @property
    def n(self): return len(self.mass)
    def as_dict(self):
        return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(self).items()}
    @classmethod
    def from_dict(cls,obj):
        return cls(**{k:(float(v) if k=='omega_s' else np.asarray(v,dtype=float)) for k,v in obj.items()})
    def check(self):
        n=self.n
        if np.shape(self.incidence)[0]!=n or self.incidence.ndim!=2: raise ValueError('Bad incidence shape')
        e=self.incidence.shape[1]
        for col in self.incidence.T:
            if not (np.count_nonzero(col==1)==1 and np.count_nonzero(col==-1)==1 and np.count_nonzero(col)==2):
                raise ValueError('Expected signed two-endpoint incidence')
        for k in ['mass','delta_star','pm_star','damping','rate','gain']:
            a=getattr(self,k)
            if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError(f'Bad physics: {k}')
        if self.coupling.shape!=(e,) or not np.isfinite(self.coupling).all():raise ValueError('Bad coupling')
        if np.any(self.mass<=0) or np.any(self.rate<=0) or np.any(self.gain<=0) or np.any(self.damping<0):
            raise ValueError('Invalid M3 initialization parameters')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load_physics(config,root=ROOT):
    kiki=source_root(config,root)
    base=source_file(kiki,config['baseline'])
    if not base.exists():raise FileNotFoundError(f'Missing KIKI baseline: {base}; no fabricated fallback')
    with np.load(base,allow_pickle=False) as z:
        keys=['mass','incidence','delta_star','pm_star','coupling','damping','rate','gain']
        p=Physics(**{k:np.array(z[k],dtype=float,copy=True) for k in keys})
    p.check();return p

def split_metadata(config,root=ROOT):
    kiki=source_root(config,root); out=[]; group_sets={'train':set(),'validation':set()}
    for relative in config['datasets']:
        directory=source_file(kiki,relative)
        sp=directory/'splits.json'; ix=directory/'dataset_index.csv'; hp=directory/'dataset.h5'
        for path in [sp,ix,hp]:
            if not path.is_file():raise FileNotFoundError(f'Missing required KIKI asset: {path}')
        splits=json.loads(sp.read_text(encoding='utf-8'))
        sets={name:set(splits[name]) for name in ['train','validation','test']}
        if any(len(splits[k])!=len(sets[k]) for k in sets):raise ValueError('Duplicate IDs in split metadata')
        if sets['train']&sets['validation'] or (sets['train']|sets['validation'])&sets['test']:
            raise ValueError('Split overlap; refuse reading any event')
        with ix.open(encoding='utf-8-sig',newline='') as f:
            items=list(csv.DictReader(f));rows={r['event_id']:r for r in items}
        if len(rows)!=len(items):raise ValueError('Duplicate dataset_index IDs')
        for name in ['train','validation']:
            for eid in splits[name]:
                if eid not in rows:raise ValueError(f'Missing metadata for {eid}')
                group_sets[name].add(str(rows[eid]['fault_bus']))
        out.append((relative,hp,splits,rows))
    if group_sets['train']&group_sets['validation']:raise ValueError('Fault groups overlap across TRAIN/development')
    return out

def load_events(config,split='train',root=ROOT,check_counts=True):
    if split not in ('train','validation'):raise PermissionError('TEST payload access is not implemented/authorized')
    phys=load_physics(config,root);events=[];excluded=[];all_ids=set();raw_count=0
    dt=float(config['sample_dt']);history=float(config['history_ms'])/1000
    if abs(history-.1)>1e-9 or abs(dt-.01)>1e-9:raise ValueError('TASK4 contract: 100 ms / 11 samples / 10 ms grid')
    for relative,hp,splits,rows in split_metadata(config,root):
        # Access list is validated before the file is opened. Never enumerate events.
        with h5py.File(hp,'r') as f:
            inertia=np.asarray(f['inertia'],dtype=float);ws=float(f.attrs['omega_s'])
            eq_delta=np.asarray(f['equilibrium/delta'],dtype=float)
            eq_pm=np.asarray(f['equilibrium/Pm'],dtype=float)
            if not np.allclose(inertia/ws,phys.mass,rtol=1e-8,atol=1e-10):raise ValueError('Mass/unit mismatch')
            if not np.allclose(eq_pm,phys.pm_star,rtol=1e-8,atol=1e-10):raise ValueError('Pm equilibrium mismatch')
            # M3 delta_star may be COI-centered; only edge phases must match.
            if not np.allclose(eq_delta@phys.incidence,phys.delta_star@phys.incidence,atol=1e-8):
                raise ValueError('Equilibrium angle differences disagree with M3')
            if phys.omega_s!=1.0 and not np.isclose(phys.omega_s,ws):raise ValueError('Mixed omega_s')
            phys.omega_s=ws
            for eid in splits[split]:
                if eid in all_ids:raise ValueError(f'Duplicate cross-dataset ID: {eid}')
                all_ids.add(eid);row=rows[eid];raw_count+=1
                if float(row['trajectory_end_s'])<=history+1e-9:
                    excluded.append({'id':eid,'group':str(row['fault_bus']),'label':row['label'],'reason':'latency_uncovered','source':relative,'metadata':dict(row)});continue
                g=f['events'][eid]
                t=np.asarray(g['t'],float);d=np.asarray(g['delta'],float)-eq_delta
                w=(np.asarray(g['omega'],float)-1.0)*ws;p=np.asarray(g['Pm'],float)-phys.pm_star
                if len(t)<=11 or abs(t[0])>1e-9 or not np.allclose(np.diff(t),dt,rtol=0,atol=1e-9):
                    raise ValueError(f'Invalid time/history for {eid}')
                if any(a.shape!=(len(t),phys.n) or not np.isfinite(a).all() for a in [d,w,p]):
                    raise ValueError(f'Bad states: {eid}')
                if row['label'] not in ['stable','unstable']:raise ValueError(f'Unrecognized original label {row["label"]}')
                if 'te' not in g:raise ValueError('Missing te; cannot silently substitute terminal Pe')
                torque=np.asarray(g['te'],float)-phys.pm_star
                if torque.shape!=d.shape or not np.isfinite(torque).all():raise ValueError('Bad te')
                events.append(Event(eid,str(row['fault_bus']),row['label'],t,np.stack([d,w,p],-1),torque,relative,dict(row)))
    facts={'split':split,'raw_count':raw_count,'eligible_count':len(events),'unstable_count':sum(e.label=='unstable' for e in events),'excluded':excluded,'test_payload_read':False,'schema':'KIKI-Loop0101'}
    if check_counts:
        for metric,value in [('raw',raw_count),('eligible',len(events)),('unstable',facts['unstable_count'])]:
            expected=config.get('expected',{}).get(f'{split}_{metric}')
            if expected is not None and expected!=value:raise ValueError(f'Source dataset changed: {split}_{metric}={value}, expected {expected}; reconcile metadata, do not discard events')
    return events,phys,facts

def train_scales(events,phys):
    # Event-balanced RMS, TRAIN only. Full omega is retained; COI is separate in metrics.
    squares=[]
    for e in events:
        a=e.y[10:].copy();a[:,:,0]-=(a[:,:,0]*phys.mass).sum(-1,keepdims=True)/phys.mass.sum()
        squares.append(np.mean(a*a,axis=(0,1)))
    scale=np.maximum(np.sqrt(np.mean(squares,axis=0)),[1e-4,1e-4,1e-5])
    return scale

def metadata_preflight(config,root=ROOT):
    p=load_physics(config,root);infos=split_metadata(config,root)
    counts={s:sum(len(sp[s]) for _,_,sp,_ in infos) for s in ['train','validation','test']}
    return {'source_root':str(source_root(config,root)),'machines':p.n,'counts_metadata_only':counts,
            'source_access':'read_only','test_payload_read':False,
            'note':'Dataset index and split metadata checked; training payload/physics numerical parity still require first run.'}
