"""Pure-JSON, read-only 0101 evidence extraction; no model or data loading."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from summarize_joint_numerical_pair_0100 import Reader, finite, statistics, monitor_summary

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'private_data/task2_shared_closure_FS_0101'
ARMS=('S23','F23')
START=1167
MILESTONES=(0,200,500,1000)
IDENTITY=('step','new_step','sample_ids','endpoints','horizon','coverage_before',
          'coverage_after','numpy_rng_before_sha256','numpy_rng_after_sha256','torch_rng_before_sha256')


def pair_identity(raw,planned_steps):
    indices,issues,live_skew={},[],[]
    for arm,items in raw.items():
        indices[arm]={}
        for kind in ('samples','trace'):
            rows=items[kind]
            counts=Counter(r.get('step') for r in rows)
            idx={r['step']:r for r in rows if isinstance(r.get('step'),int)}
            indices[arm][kind]=idx
            if any(n>1 for n in counts.values()) or [r.get('step') for r in rows]!=sorted(idx):
                issues.append({'arm':arm,'kind':kind,'reason':'duplicate, malformed or unordered steps'})
            for step,row in idx.items():
                needed=IDENTITY+('torch_rng_after_sha256',) if kind=='trace' else IDENTITY
                missing=[k for k in needed if row.get(k) is None]
                if missing:issues.append({'arm':arm,'kind':kind,'step':step,'missing':missing})
                if row.get('new_step')!=step-START or step<=START or row.get('training_dt')!=.01 or row.get('horizon')!=(1,2,5)[(step-301)%3]:
                    issues.append({'arm':arm,'kind':kind,'step':step,'reason':'step/grid/horizon differs from registration'})
                if planned_steps is not None and step>START+planned_steps:
                    issues.append({'arm':arm,'kind':kind,'step':step,'reason':'past launched plan'})
                prev=idx.get(step-1)
                if prev:
                    checks=[('coverage_after','coverage_before'),('numpy_rng_after_sha256','numpy_rng_before_sha256')]
                    if kind=='trace':checks.append(('torch_rng_after_sha256','torch_rng_before_sha256'))
                    bad=[f'{a}->{b}' for a,b in checks if prev.get(a)!=row.get(b)]
                    if bad:issues.append({'arm':arm,'kind':kind,'step':step,'broken_chains':bad})
            if idx and sorted(idx)!=list(range(START+1,max(idx)+1)):
                issues.append({'arm':arm,'kind':kind,'reason':'missing interior/initial steps'})
        for step,row in indices[arm]['trace'].items():
            sample=indices[arm]['samples'].get(step,{})
            if not sample and step>max(indices[arm]['samples'],default=START):
                live_skew.append({'arm':arm,'step':step,'reason':'trace tail read after earlier sample snapshot'})
                continue
            bad=[k for k in IDENTITY if k not in sample or k not in row or sample[k]!=row[k]]
            if bad:issues.append({'arm':arm,'step':step,'reason':'sample/committed trace differ','fields':bad})
    paired={}
    for kind in ('samples','trace'):
        left,right=indices['S23'][kind],indices['F23'][kind]
        common=sorted(left.keys() & right.keys())
        fields=IDENTITY+('torch_rng_after_sha256',) if kind=='trace' else IDENTITY
        mismatches=[]
        for step in common:
            bad=[k for k in fields if k not in left[step] or k not in right[step] or left[step][k]!=right[step][k]]
            if bad:mismatches.append({'step':step,'fields':bad})
        issues.extend({'kind':kind,'reason':'paired identity mismatch',**m} for m in mismatches)
        prefix=0
        while START+prefix+1 in common and not any(m['step']==START+prefix+1 for m in mismatches):prefix+=1
        paired[kind]={'S23_rows':len(left),'F23_rows':len(right),'common_rows':len(common),
                      'matched_contiguous_prefix_steps':prefix,'last_common_absolute_step':max(common) if common else None}
    errors={a:data.get('parse_errors',{}) for a,data in raw.items() if any(data.get('parse_errors',{}).values())}
    complete=(planned_steps is not None and all(p['matched_contiguous_prefix_steps']==planned_steps for p in paired.values())
              and not issues and not errors and not live_skew)
    return {'planned_steps':planned_steps,'paired':paired,'issues':issues,'parse_errors':errors,
            'all_launched_steps_match':complete,
            'live_snapshot_skew':live_skew,
            'status':'mismatch' if issues else 'complete_match' if complete else 'partial_match' if not errors else 'partial_parse',
            'fields_compared':list(IDENTITY)+['torch_rng_after_sha256'],
            'scope':'Recorded sampling identity only; does not establish completion of monitors/evaluation or choose an endpoint.'}


def coverage_summary(rows,pool):
    out={}
    for h in (1,2,5):
        selected=[r for r in rows if r.get('horizon')==h]
        counts=Counter(e for row in selected for e in row.get('sample_ids',[]))
        out[str(h)]={'updates':len(selected),'sample_draws':sum(counts.values()),'unique_events':len(counts),
                     'draws_per_event':dict(sorted(counts.items())),
                     'unsampled_ids':[eid for eid in pool if not counts[eid]]}
    return out


def complexity_summary(row):
    if not isinstance(row,dict):return None
    result={}
    for grid,data in (row.get('batch_terms') or {}).items():
        names=data.get('term_names') or []
        if 'complexity' not in names:continue
        index=names.index('complexity')
        values=[h[index] for batch in data.get('terms',[]) for h in batch if len(h)>index]
        stats=statistics(values)
        stats['constant_across_fixed_batches']=bool(values) and stats['finite_rows']==len(values) and max(values)-min(values)<=1e-12
        result[grid]=stats
    return result or None


def evaluation(reader,base):
    parts={name:reader.read(base/path) for name,path in {
        'root':'evidence.json','validation':'validation/evidence.json','train':'train/evidence.json',
        'modal':'modal/evidence.json','train_per_event':'train/per_event_rows.json','progress':'progress.json'}.items()}
    val,train,modal,outer=(parts[k] or {} for k in ('validation','train','modal','root'))
    harmful={'false_window_settled':[],'false_los':[]}
    categories={}
    for row in val.get('classifications',[]):
        classification=row.get('classification') or {}
        categories.setdefault(classification.get('category','missing'),[]).append(row.get('event_id'))
        if row.get('label')=='unstable' and classification.get('window_settled'):
            harmful['false_window_settled'].append(row)
        if row.get('label')=='stable' and classification.get('predicted_los'):
            harmful['false_los'].append(row)
    return {'path':str(base),'available':any(p is not None for p in parts.values()),
        'missing_components':[k for k in ('root','validation','train','modal') if parts[k] is None],
        'supervisor_seconds':outer.get('supervisor_seconds'),'status':outer.get('status'),
        'checkpoint':outer.get('checkpoint'),'family':outer.get('family'),
        'summary':{'validation_complete':val.get('summary'),
            'validation_severe_prefix':val.get('unstable_available_prefix'),
            'common_frequency_event_rows':val.get('common_frequency'),
            'validation_l1_counts_original_denominators':val.get('l1_numerically_gated_counts'),
            'train':train.get('summary'),'train_denominators':train.get('denominators'),
            'train_complete_counts':train.get('complete_counts'),
            'modal_matches':modal.get('matched_count'),'modal_qa':modal.get('qa'),
            'local_alpha':modal.get('spectral_alpha'),'power_qa':outer.get('energy_power_qa')},
        'adverse_classifications':{k:{'all_ids':[r.get('event_id') for r in rows],
             'numerically_admissible_ids':[r.get('event_id') for r in rows if r.get('numerically_admissible')],
             'rows':rows} for k,rows in harmful.items()},
        'classification_ids_by_stored_category':categories,'full_metric_json':parts,
        'charge_scope':'Outer supervisor_seconds only; nested worker/component times overlap.'}


def parameter_summary(metadata):
    if not metadata:return None
    grad=metadata.get('requires_grad') or {}
    sizes=metadata.get('parameter_numel') or {}
    return {'all_parameter_tensors':len(sizes),'all_parameter_scalars':sum(sizes.values()),
        'active_parameter_tensors':sum(v is True for v in grad.values()),
        'active_parameter_scalars':sum(sizes.get(n,0) for n,v in grad.items() if v),
        'fixed_parameter_names':[n for n,v in grad.items() if not v],
        'optimizer_layout':metadata.get('optimizer_parameter_names'),
        'fixed_buffers_sha256':metadata.get('fixed_buffers_sha256')}


def build_report(data_root=DATA):
    began=time.perf_counter();data_root=Path(data_root);reader=Reader()
    launch=reader.read(data_root/'formal_launch.json')
    planned=(launch or {}).get('new_steps')
    plan_valid=planned in (500,1000)
    if not plan_valid:planned=None
    raw,arms={},{}
    for arm in ARMS:
        base=data_root/f'formal_{arm}'
        formal=reader.read(base/'evidence.json')
        samples=reader.read(base/'samples.jsonl',lines=True)
        trace=reader.read(base/'trace.jsonl',lines=True)
        parse={kind:reader.inputs[str(base/f'{kind}.jsonl')].get('parse_error_lines',[]) for kind in ('samples','trace')}
        raw[arm]={'samples':samples,'trace':trace,'parse_errors':parse}
        manifest=reader.read(base/'manifest.json') or reader.read(data_root/f'qa_monitor_{arm}/manifest.json')
        qa_context=reader.read(data_root/f'qa_monitor_{arm}/context_0101.json')
        monitors={}
        for n in MILESTONES:
            path=base/f'monitor_step{n:04d}/evidence.json'
            row=((formal or {}).get('monitors') or {}).get(str(n))
            origin='embedded formal evidence' if row is not None else 'formal monitor file'
            if row is not None:path=base/'evidence.json'
            else:row=reader.read(path)
            if row is None and n==0:
                path=data_root/f'qa_monitor_{arm}/evidence.json';row=reader.read(path);origin='0101 QA monitor'
            summary=monitor_summary(row,path,n)
            summary['origin']=origin
            summary['C23_complexity']=complexity_summary(row)
            summary['full_metric_json']=row
            if n==0 and origin=='0101 QA monitor':
                summary['context_0101']=qa_context
                summary['context_consistent']=bool(row and qa_context and qa_context.get('loop')=='0101'
                    and qa_context.get('objective')=='C23' and qa_context.get('arm')==arm
                    and qa_context.get('monitor_evidence_sha256')==reader.inputs[str(path)].get('sha256')
                    and qa_context.get('model_state_sha256')==row.get('model_state_sha256_before')==row.get('model_state_sha256_after'))
            monitors[str(n)]=summary
        checkpoints={str(n):reader.read(base/('step0000' if n==0 else f'step{START+n:04d}')/'checkpoint.json') for n in MILESTONES}
        params={n:parameter_summary(c) for n,c in checkpoints.items()}
        initial=params['0']
        invariant={n:None if p is None or initial is None else p==initial for n,p in params.items()}
        arms[arm]={'formal_run_metadata':None if formal is None else {k:v for k,v in formal.items() if k!='monitors'},
            'formal_evidence_available':formal is not None,
            'state_note':None if formal else 'Formal evidence absent; running/interrupted/not started cannot be distinguished from this file alone.',
            'trace':{'rows':len(trace),'last_step':trace[-1].get('step') if trace else None,
                'last_elapsed_seconds':trace[-1].get('elapsed_seconds') if trace else None,
                'loss':statistics([r.get('loss') for r in trace]),'gradient_norm':statistics([r.get('gradient_norm') for r in trace]),
                'recorded_clipped_count':sum(r.get('gradient_clipped') is True for r in trace),
                'finite_gradient_tensor_counts':sorted({r['finite_gradient_parameter_tensors'] for r in trace if 'finite_gradient_parameter_tensors' in r}),
                'finite_gradient_scalar_counts':sorted({r['finite_gradient_scalar_count'] for r in trace if 'finite_gradient_scalar_count' in r}),
                'step_seconds':statistics([r.get('seconds') for r in trace]),
                'sampling_by_horizon':coverage_summary(trace,(manifest or {}).get('event_ids',[])),
                'first_preupdate_complexity':(trace[0].get('terms') or {}).get('complexity') if trace else None,
                'last_preupdate_complexity':(trace[-1].get('terms') or {}).get('complexity') if trace else None,
                'parse_errors':parse},
            'monitors':monitors,'milestone_metadata':checkpoints,'parameter_counts':params,
            'parameter_layout_and_fixed_buffers_unchanged':invariant,
            'evaluation':evaluation(reader,data_root/f'eval_{arm}')}
    report={'loop':'0101','scope':'Factual JSON extraction; no scientific decision, endpoint selection, or model/data loading.',
        'generated_utc':datetime.now(timezone.utc).isoformat(),'data_root':str(data_root),
        'formal_launch':launch,'planned_steps':planned,'launch_plan_recognized':plan_valid,
        'pair_identity':pair_identity(raw,planned),'arms':arms,
        'common_saved_milestones':[n for n in MILESTONES if all(arms[a]['milestone_metadata'][str(n)] is not None for a in ARMS)],
        'execution_costs':reader.read(data_root/'execution_costs.json'),
        'input_snapshots':list(reader.inputs.values()),'extractor_seconds':time.perf_counter()-began,
        'test_read':False,'raw_data_read':False,'model_loaded':False,'integration_run':False,
        'snapshot_note':'Sequential live-file byte snapshots; a pending sample can precede its committed trace. Missing future milestones remain missing.'}
    return report


def write_report(data_root=DATA,output=None):
    output=Path(output or DATA/'derived'/f'capability_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}.json').resolve()
    if not output.is_relative_to(DATA.resolve()/'derived'):raise ValueError('output must stay under 0101/derived')
    if output.exists():raise FileExistsError(output)
    report=build_report(data_root)
    report['extractor_code']={'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return output,report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=DATA)
    p.add_argument('--output',type=Path)
    out,value=write_report(**vars(p.parse_args()))
    print(json.dumps({'output':str(out),'planned_steps':value['planned_steps'],
        'pair_status':value['pair_identity']['status'],
        'committed_rows':{a:value['arms'][a]['trace']['rows'] for a in ARMS},
        'extractor_seconds':value['extractor_seconds']}))
