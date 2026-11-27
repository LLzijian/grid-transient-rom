"""Frozen confirmation entry. A separate, explicitly authorized release is required."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/'src')]

import h5py
import numpy as np
import torch

from experiments.osgkd_v2_frozen_confirmation_20261127 import data, freeze_spec as spec, score
from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io
from kiki.evaluation.osgkd import aggregate_metrics
from kiki.evaluation.osgkd_autonomous import autonomous_outcome
from kiki.models.osgkd_baselines import ClosedPassiveKuramoto, GraphNeuralODE
from kiki.models.osgkd_graph import OSGKDGraph
from kiki.models.osgkd_matched_flat import OSGKDMatchedFlat
from kiki.models.osgkd_restricted import RestrictedOSGKDGraph


LOCAL = ROOT/'artifacts'/spec.RUN
MODEL_TYPES = dict(graph=OSGKDGraph,flat=OSGKDMatchedFlat,neural_ode=GraphNeuralODE,
                   closed_passive=ClosedPassiveKuramoto,restricted=RestrictedOSGKDGraph)


def data_identity(metadata):
    sources = {name:{k:v for k,v in source.items() if k!='test_ids'}
               for name,source in metadata['sources'].items()}
    for name,source in sources.items():
        stamp=(ROOT/'private_data'/name/'dataset.h5').stat()
        source['container_file_metadata']=dict(bytes=stamp.st_size,modified_ns=stamp.st_mtime_ns)
    ids = json.dumps(metadata['event_ids'],separators=(',',':')).encode('utf-8')
    historical=io.identity(ROOT/'private_data/ieee39_holdout_supplement/coverage_evidence.json')
    return dict(sources=sources,whitelist_sha256=hashlib.sha256(ids).hexdigest(),historical_coverage=historical)


def numerics():
    return dict(python=list(sys.version_info[:3]),torch=str(torch.__version__),numpy=np.__version__,h5py=h5py.__version__,
                device='cpu',threads=1,dtype='float64',history_samples=11,history_s=.1,
                output_step=.01,internal_steps=[.005,.0025],fixed_batch_cap=16)


def validate_frozen(path):
    io.require(path==spec.PUBLIC/'freeze.json', 'canonical frozen authority required')
    frozen = io.read(path)
    io.require(frozen['run_id']==spec.RUN and frozen['status']=='frozen',
               'canonical reviewed freeze required')
    io.require(frozen['test_trajectory_access'] is False and frozen['new_fom']==0 and
               frozen['default_model']=='none' and frozen['official_updates']==0,'frozen scope differs')
    io.require(frozen['independent_freeze_passed'] is True,'independent freeze review did not pass')
    io.verified(frozen['independent_freeze_review'])
    optional=io.read(io.verified(frozen['wp3_2_decision']))
    io.require(optional['status']=='closed' and optional['decision'] in {'deferred','completed'},
               'optional WP3.2 decision is not closed')
    io.verified(frozen['claim_ledger'])
    io.require(set(frozen['source_hashes'])==spec.evidence_sources(),'frozen source inventory differs')
    for name,sha in frozen['source_hashes'].items():
        local=(ROOT/name).resolve()
        io.require(local.is_relative_to(ROOT) and io.identity(local)['sha256']==sha,'frozen source differs: '+name)
    expected,alias=spec.fixed_endpoints()
    io.require(frozen['prerequisites']==spec.closed_prerequisites() and frozen['endpoints']==expected
               and frozen['alias']==alias and frozen['comparisons']==spec.fixed_comparisons(), 'fixed endpoint or lineage differs')
    io.require(frozen['constants_source']==io.identity(ROOT/spec.CONSTANTS), 'frozen constants source differs')
    io.require(frozen['numerics']==numerics(), 'frozen numerical environment differs')
    return frozen


def load_models(book):
    constants = io.read(io.verified(book['constants_source']))['constants']
    models = {}
    for entry in book['endpoints']:
        path=io.verified(entry['checkpoint'])
        saved=torch.load(path, map_location='cpu', weights_only=True)
        io.require(saved['attempt']==entry['attempt'], 'frozen attempt differs: '+entry['id'])
        model=MODEL_TYPES[entry['kind']].from_checkpoint(saved['model']).eval().requires_grad_(False)
        io.require(model.n==10 and model.history_points==11 and model.contract=='motion_only_v1',
                   'frozen model contract differs')
        for key in ('mass','damping','theta_ref','omega_scale','F','Sp','Se'):
            actual=getattr(model,key)
            expected=torch.as_tensor(constants[key],dtype=torch.float64)
            if actual.ndim==2 and expected.ndim==1:
                expected=expected.diag()
            io.require(torch.equal(actual,expected), 'model constant differs: '+entry['id']+'/'+key)
        models[entry['id']]=model
    return models,constants


def pack(value):
    if isinstance(value,np.ndarray):
        return torch.from_numpy(value.copy(order='C'))
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,dict):
        return {key:pack(child) for key,child in value.items()}
    if isinstance(value,list):
        return [pack(child) for child in value]
    return value


def unpack(value):
    if isinstance(value,torch.Tensor):
        return value.numpy()
    if isinstance(value,dict):
        return {key:unpack(child) for key,child in value.items()}
    if isinstance(value,list):
        return [unpack(child) for child in value]
    return value


def approved_data(metadata, constants, authority):
    exposure=LOCAL/'test_access_receipt.json'
    completed=LOCAL/'data_complete.json'
    if completed.exists():
        receipt=io.read(completed)
        io.require(receipt['authorization']==authority,'cached test authority differs')
        stored=torch.load(io.verified(receipt['cache']),map_location='cpu',weights_only=True)
        io.require(stored['authorization']==authority,'cached packet identity differs')
        io.verified(receipt['exposure'])
        return unpack(stored['events']),receipt['cache']
    io.require(not exposure.exists(), 'original test already entered without a complete private cache')
    observed=dict(authorization=authority,opened_utc=io.utc(),reads=[],official_updates=0)
    io.write(exposure,observed)

    def on_read(note):
        observed['reads'].append({**note,'utc':io.utc()})
        io.write(exposure,observed)

    events=data.load_allowed_events(ROOT, metadata, constants, on_read)
    io.require(len(events)==86 and Counter(e['label'] for e in events)=={'stable':67,'unstable':19},
               'frozen raw test population differs')
    opened=[r for r in observed['reads'] if r['phase']=='opened_shard']
    read=[r for r in observed['reads'] if r['phase']=='read_event']
    io.require(len(opened)==2 and len(read)==86, 'test read activity differs')
    cache=LOCAL/'fixed_input_packets.pt'
    temp=cache.with_suffix('.pt.tmp')
    with temp.open('xb') as stream:
        torch.save(dict(events=pack(events),authorization=authority),stream)
        stream.flush()
        os.fsync(stream.fileno())
    io.replace_with_retry(temp,cache)
    ref=io.identity(cache)
    observed.update(closed_utc=io.utc(),event_trajectory_reads=86,physical_shard_opens=2,cache=ref)
    io.write(exposure,observed)
    io.write(completed,dict(authorization=authority,cache=ref,exposure=io.identity(exposure)))
    return events,ref


def paired_continuous(pairs):
    result={}
    scopes=[(f'stable_{span}s','stable',span) for span in ('1','2','5','8')]
    scopes.append(('unstable_actual','unstable','actual'))
    for name,label,scope in scopes:
        population=[(a,b) for a,b in pairs if a['label']==label]
        comparable=[(a,b) for a,b in population if all(row['scopes'][scope]['metrics'] is not None and
                    (label=='unstable' or row['scopes'][scope]['full_horizon_truth']) for row in (a,b))]
        record=dict(raw_count=len(population),paired_valid_count=len(comparable),difference=None)
        if comparable:
            ref,comparison=[aggregate_metrics(rows,scope)['official'] for rows in zip(*comparable)]
            port_deltas={key:(comparison[key]['normalized_rmse']-ref[key]['normalized_rmse'])
                         if comparison[key]['normalized_rmse'] is not None and ref[key]['normalized_rmse'] is not None
                         else None for key in ('tau_m','tau_e')}
            record.update(reference=ref,candidate=comparison,difference=dict(
                motion_risk_v1=comparison['motion_risk_v1']-ref['motion_risk_v1'],
                common_speed_rmse_rad_s=comparison['common_speed_rmse_rad_s']-ref['common_speed_rmse_rad_s'],
                normalized_port_rmse=port_deltas))
        result[name]=record
    return result


def paired_rows(reference, candidate, source_by_id):
    a={row['event_id']:row for row in reference}
    b={row['event_id']:row for row in candidate}
    io.require(a.keys()==b.keys(), 'paired test event populations differ')
    counts=Counter()
    by_source,by_group={},{}
    paired_by_source,paired_by_group={},{}
    disagreements=[]
    lag_delta=[]
    for event in a:
        left,right=a[event],b[event]
        io.require(left['label']==right['label'] and left['group']==right['group'],'paired event labels differ')
        x,y=autonomous_outcome(left),autonomous_outcome(right)
        outcome = (left['label'],x,y)
        source = source_by_id[event]
        counts[outcome] += 1
        by_source.setdefault(source,Counter())[outcome] += 1
        by_group.setdefault(left['group'],Counter())[outcome] += 1
        paired_by_source.setdefault(source,[]).append((left,right))
        paired_by_group.setdefault(left['group'],[]).append((left,right))
        if x!=y:
            disagreements.append(dict(event_id=event,source=source,group=left['group'],
                                      label=left['label'],reference=x,candidate=y))
        if left['label']=='unstable' and x==y=='autonomous_trigger':
            l,r=(row['classification_8s']['predicted_time_record_s'] for row in (left,right))
            lag_delta.append(r-l)
    def table(values):
        return [dict(label=k[0],reference=k[1],candidate=k[2],count=v) for k,v in sorted(values.items())]
    return dict(outcomes=table(counts),by_source={k:table(v) for k,v in by_source.items()},
                by_group={k:table(v) for k,v in by_group.items()},disagreements=disagreements,
                continuous=paired_continuous(list(zip(a.values(),(b[event] for event in a),strict=True))),
                continuous_by_source={k:paired_continuous(v) for k,v in paired_by_source.items()},
                continuous_by_group={k:paired_continuous(v) for k,v in paired_by_group.items()},
                common_unstable_trigger_count=len(lag_delta),
                common_predicted_time_delta_quantiles=np.quantile(lag_delta,[.25,.5,.75]).tolist() if lag_delta else None)


def execute(path):
    # This record is written by root only after an explicit user grant in the active chat.
    io.require(path==spec.PUBLIC/'authorized_release.json','canonical authorized release required')
    grant=io.read(path)
    io.require(grant['run_id']==spec.RUN
               and grant['user_authorized_once'] is True and bool(grant['user_authorization_reference']),
               'explicit authorization for this one frozen evaluation required')
    io.require(Path(sys.executable).resolve()==(ROOT.parent/'.venv/Scripts/python.exe').resolve(),
               'shared Python environment required')
    io.require(grant['freeze']['path']==(spec.PUBLIC/'freeze.json').relative_to(ROOT).as_posix(),
               'authorized freeze path differs')
    frozen=validate_frozen(io.verified(grant['freeze']))
    metadata=data.read_source_metadata(ROOT)
    io.require(grant['frozen_evaluation_id']==frozen['evaluation_id'] and
               frozen['data_identity']==data_identity(metadata),'approved dataset or evaluation identity differs')
    torch.set_num_threads(1)
    models,constants=load_models(frozen)
    authority=io.identity(path)
    LOCAL.mkdir(parents=True,exist_ok=True)
    io.require(not (LOCAL/'scientific_stop.json').exists(), 'previous scientific stop requires a separate decision')
    lease=LOCAL/'score_writer.lock'
    with lease.open('x') as stream:
        stream.write(str(os.getpid()))
    start,cpu=perf_counter(),process_time()
    try:
        out=spec.PUBLIC/'summary.json'
        io.require(not out.exists(),'one frozen test score has already finished')
        events,cache=approved_data(metadata,constants,authority)
        results,public_refs={},{}
        for entry in frozen['endpoints']:
            name=entry['id']
            root=LOCAL/'endpoints'/name
            root.mkdir(parents=True,exist_ok=True)
            canonical=root/'terminal.json'
            if not canonical.exists():
                existing=sorted(root.glob('work*/endpoint_summary.json'))
                if existing:
                    io.require(len(existing)==1,'duplicate completed scores for fixed endpoint')
                    measured=io.read(existing[0])
                else:
                    measured=score.score_endpoint(models[name],events,constants,root/f'work_{os.getpid()}')
                io.write(canonical,dict(entry=entry,authorization=authority,cache=cache,result=measured))
            row=io.read(canonical)
            io.require(row['entry']==entry and row['cache']==cache and row['authorization']==authority,
                       'frozen endpoint score identity differs')
            io.verified(row['result']['rows_ref'])
            results[name]=row['result']
            public=spec.PUBLIC/'endpoints'/f'{name}.json'
            public.parent.mkdir(parents=True,exist_ok=True)
            public_value=dict(endpoint=name,result=results[name],test_events=len(events))
            io.require(len(json.dumps(public_value,allow_nan=False).encode('utf-8'))<=512*1024,
                       'public endpoint exceeds aggregate budget')
            io.write(public,public_value)
            public_refs[name]=io.identity(public)
            print(name,'sealed',flush=True)
        pairwise=[]
        sources={event['id']:event['metadata']['source_root'] for event in events}
        for pair in frozen['comparisons']:
            a,b=[io.read(io.verified(results[pair[k]]['rows_ref'])) for k in ('reference','candidate')]
            outcome=paired_rows(a,b,sources)
            file=LOCAL/f"paired_{pair['reference']}_vs_{pair['candidate']}.json"
            io.write(file,outcome.pop('disagreements'))
            public=spec.PUBLIC/'pairs'/file.name
            public.parent.mkdir(parents=True,exist_ok=True)
            io.write(public,{**pair,**outcome,'disagreements_ref':io.identity(file)})
            pairwise.append({**pair,'paired_result':io.identity(public)})
        io.write(out,dict(run_id=spec.RUN,status='completed',evaluation_id=frozen['evaluation_id'],
                          release=authority,cache=cache,access_receipt=io.identity(LOCAL/'test_access_receipt.json'),
                          endpoint_results=public_refs,alias=frozen['alias'],paired_comparisons=pairwise,
                          raw_events=len(events),new_fom=0,official_updates=0,default_model='none',
                          wall_s=perf_counter()-start,cpu_s=process_time()-cpu,completed_utc=io.utc()))
    except BaseException as error:
        technical=isinstance(error,(OSError,MemoryError,KeyboardInterrupt))
        failure=dict(error=repr(error),authorization=authority,technical=technical,utc=io.utc())
        io.write(LOCAL/f'failed_{os.getpid()}.json',failure)
        if not technical:
            io.write(LOCAL/'scientific_stop.json',failure)
        raise
    finally:
        lease.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',required=True,type=Path)
    execute(parser.parse_args().release.resolve())
