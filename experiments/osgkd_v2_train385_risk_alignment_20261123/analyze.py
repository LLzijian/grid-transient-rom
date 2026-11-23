"""Analyze completed scored archives only; never load a model or scientific packets."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments.osgkd_v2_train385_risk_alignment_20261123 import run
from experiments.osgkd_v2_train385_baseline_20261123 import run as baseline
require,read,identity=run.require,run.read,run.identity


def finite(value):
    if isinstance(value,float): return math.isfinite(value)
    if isinstance(value,dict): return all(finite(v) for v in value.values())
    if isinstance(value,list): return all(finite(v) for v in value)
    return True


def hits(rows):
    return {r['event_id'] for r in rows if r['label']=='unstable' and
            (c:=r['classification_actual']) is not None and c['los_pair_pass'] and
            c['truth_supported'] and c['classification']['predicted_los']}


def qualification(rows,split):
    expected={'train':(385,342,43,0),'val':(117,95,22,10)}[split]
    missing=[r for r in rows if r['actual_future_points'] is None]
    eligible=[r for r in rows if r['actual_future_points'] is not None]
    population=(len(rows),sum(r['label']=='stable' for r in rows),sum(r['label']=='unstable' for r in rows),len(missing))
    if population != expected or any(r['label'] != 'unstable' for r in missing):
        return False
    for row in eligible:
        stable = row['label'] == 'stable'
        horizons = ('1', '2', '5', '8') if stable else ('actual',)
        category = row['classification_8s' if stable else 'classification_actual']
        if category is None or not category['los_pair_pass']:
            return False
        if any(not row['scopes'][h]['numerical']['passed'] or row['scopes'][h]['metrics'] is None for h in horizons):
            return False
    return True


def read_score(ref,split):
    rows,metrics=read(ref['rows_ref']),read(ref['metrics_ref'])
    done=read(ref['completed_ref'])
    require(done=={'status':'completed','rows_ref':ref['rows_ref'],'metrics_ref':ref['metrics_ref']},'split completion differs')
    require(baseline.public_summary(metrics)==ref['summary'],'public/private metrics differ')
    require(len({r['event_id'] for r in rows})==len(rows),'duplicate scored ID')
    require(set(metrics['classification'][split]['actual']['detected_unstable_ids'])==hits(rows),'actual hit tally differs')
    require(metrics['coverage']['denominator']==len(rows),'score denominator differs')
    stable=[r for r in rows if r['label']=='stable']
    for h in ('1','2','5','8'):
        values=[r['scopes'][h]['metrics']['mean_motion_error'] for r in stable if r['scopes'][h]['metrics'] is not None]
        official=metrics['splits'][split][f'stable_{h}s']['official']
        if official is not None:
            tail=sorted(values)[-math.ceil(.1*len(values)):]
            risk=.8*sum(values)/len(values)+.2*sum(tail)/len(tail)
            require(len(values)==len(stable) and math.isclose(risk,official['motion_risk_v1'],rel_tol=1e-12,abs_tol=1e-12),'stable risk differs')
    return rows,metrics


def check_short(short,rows):
    require(len(rows)==27 and len({r['event_id'] for r in rows})==27 and
            sum(r['selected_machine'] is not None for r in rows)==22,'short cohort identity counts differ')
    for name in ('control','candidate'):
        for w in ('0.03','capped_0.25'):
            for cohort in ('parent_misses','persistent_22'):
                selected=[r for r in rows if cohort=='parent_misses' or r['selected_machine'] is not None]
                good=[r for r in selected if r['models'][name]['windows']['fine_'+w]['complete'] and
                      r['models'][name]['windows']['fine_'+w]['numerical_gate']['passed']]
                aggregate=short['aggregates'][name][w][cohort]
                require(len(selected)==aggregate['raw_denominator'] and len(good)==aggregate['cross_precision_qualified_events'],'short coverage differs')
                for key,extract in (('event_equal_electrical_normalized_mse',lambda r:r['models'][name]['windows']['fine_'+w]['electrical_normalized_mse']),
                                    ('selected_signed_mean',lambda r:r['models'][name]['windows']['fine_'+w]['geometry']['selected_total'])):
                    use=good if key.startswith('event_equal') else [r for r in good if r['selected_machine'] is not None]
                    value=sum(extract(r) for r in use)/len(use) if use else None
                    require(value is None and aggregate[key] is None or value is not None and aggregate[key] is not None and
                            math.isclose(value,aggregate[key],rel_tol=1e-12,abs_tol=1e-12),'short aggregate differs')


def reduction(candidate,reference):
    return 1-candidate/reference if candidate is not None and reference is not None and reference>1e-12 else None


def diagnostic(short):
    result={}
    for window in ('0.03','capped_0.25'):
        c=short['aggregates']['candidate'][window]['persistent_22']
        a=short['aggregates']['control'][window]['persistent_22']
        p=short['frozen_parent_aggregates']['persistent_22'][window]['parent_1860']
        complete=all(x['raw_denominator']==22 and x['cross_precision_qualified_events']==22 and x['selected_complete']==22 for x in (a,c)) and p['raw_denominator']==p['complete_events']==22
        result[window]={'complete_22_both_endpoints':complete,
            'electrical_mse_reduction_vs_control':reduction(c['event_equal_electrical_normalized_mse'],a['event_equal_electrical_normalized_mse']),
            'selected_signed_bias_reduction_vs_control':reduction(c['selected_signed_mean'],a['selected_signed_mean']),
            'selected_signed_bias_reduction_vs_parent':reduction(c['selected_signed_mean'],p['selected_signed_mean']),
            'candidate':c,'control':a,'parent_1860':p}
    positive=all(x['complete_22_both_endpoints'] and x['electrical_mse_reduction_vs_control'] is not None and
                 x['electrical_mse_reduction_vs_control']>=.05 for x in result.values())
    primary=result['capped_0.25']
    insufficient=(primary['complete_22_both_endpoints'] and all(primary[key] is not None and primary[key]<.05 for key in
        ('selected_signed_bias_reduction_vs_control','selected_signed_bias_reduction_vs_parent')))
    return {'windows':result,'positive_electrical_diagnostic':positive,'primary_signed_reductions_both_below_5pct':insufficient,
            'parent_misses_27':{name:{w:short['aggregates'][name][w]['parent_misses'] for w in result} for name in ('control','candidate')},
            'private_rows_ref':short['rows_ref'],'scope':'fixed short cohorts only; does not identify a structural mechanism'}


def decisions(result,rows,metrics,parent_rows):
    t=result['treatments']; counts={name:t[name]['accepted'] for name in t}
    attempts=(result['paired_attempts']==200 and result['fixed_endpoint']==2060 and not result['stopped_early'] and
              all(t[n]['attempts']==200 and t[n]['accepted']+t[n]['rejected']==200 for n in t))
    health=attempts and min(counts.values())>=196 and abs(counts['candidate']-counts['control'])<=2 and all(
        t[n]['work']['rhs_forward']<=10_240_000 and t[n]['work']['rhs_recompute']==0 and
        t[n]['actual_event_exposures']==1600 for n in t)
    numerical=all(qualification(rows[n][s],s) for n in t for s in ('train','val'))
    val=metrics['candidate']['val']; stable=[r for r in rows['candidate']['val'] if r['label']=='stable']
    official=val['splits']['val']['stable_8s']['official']
    classified=all(r['scopes']['8']['full_horizon_truth'] and (c:=r['classification_8s']) is not None and
                   c['los_pair_pass'] and c['truth_supported'] for r in stable)
    fp=sum(bool(r['classification_8s'] and r['classification_8s']['classification']['predicted_los']) for r in stable)
    protection=(len(stable)==95 and classified and fp==0 and official is not None and official['motion_risk_v1']<1.)
    detected={n:{s:hits(rows[n][s]) for s in ('train','val')} for n in t}
    inherited={s:hits(parent_rows[s]) for s in ('train','val')}
    gates={'all_200_matched_attempts':attempts,'optimization_health':health,'matched_numerical_coverage':numerical,
           'candidate_val_stable_protection':protection,
           'candidate_val_actual_at_least_6_of_12':len(detected['candidate']['val'])>=6,
           'candidate_val_actual_at_least_control_plus_2':len(detected['candidate']['val'])>=len(detected['control']['val'])+2,
           'candidate_train_actual_at_least_15_of_43':len(detected['candidate']['train'])>=15,
           'retains_parent_train_actual_hits':inherited['train']<=detected['candidate']['train'],
           'retains_parent_val_actual_hits':inherited['val']<=detected['candidate']['val']}
    gates['all_working_parent_gates']=all(gates.values())
    changes={s:{'candidate_gained_vs_control':len(detected['candidate'][s]-detected['control'][s]),
                'candidate_lost_vs_control':len(detected['control'][s]-detected['candidate'][s]),
                'parent_hits':len(inherited[s]),'parent_hits_retained':len(inherited[s]&detected['candidate'][s])} for s in ('train','val')}
    return gates,changes


def timing(rows):
    eligible={r['event_id']:r['classification_actual'] for r in rows if r['label']=='unstable' and
              r['classification_actual'] is not None and r['classification_actual']['los_pair_pass'] and
              r['classification_actual']['timing_comparable'] and r['classification_actual']['truth_supported']}
    return {eid:c['trigger_time_error_s'] for eid,c in eligible.items() if c['trigger_time_error_s'] is not None}


def describe(rows,metrics,split):
    classes=metrics['classification'][split]
    return {'stable_risks':{str(h):{'raw_denominator':metrics['splits'][split][f'stable_{h}s']['denominator'],
                'covered':metrics['splits'][split][f'stable_{h}s']['covered'],
                'official':metrics['splits'][split][f'stable_{h}s']['official']} for h in (1,2,5,8)},
            'classification':baseline.public_summary(classes),'true_actual_hits':len(hits(rows)),
            'input_uncovered':sum(r['actual_future_points'] is None for r in rows),'work':metrics['work'],'cost':metrics['cost']}


def execute(path):
    path=path.resolve(); require(path==ROOT/'results'/run.RUN/'pair01'/'summary.json','canonical pair01 summary required')
    summary_ref=identity(path); summary=read(summary_ref)
    require(summary['run_id']==run.RUN and summary['job_id']=='pair01' and summary['status']=='completed','completed pair01 required')
    require(read(summary['release_ref'])==summary['release'],'native release identity differs')
    inputs=run.validate(summary['release'])
    require(read(identity(ROOT/'artifacts'/run.RUN/'pair01'/'completed.json'))=={'status':'completed','summary_ref':summary_ref},'native completion differs')
    result=summary['result']; rows={}; metrics={}; private={}
    plan=read(result['plan_ref'])
    require(set(result['treatments']) == {'control', 'candidate'}, 'exact pair required')
    for name,treatment in result['treatments'].items():
        log=read(treatment['attempts_ref'])
        require(len(log)==treatment['attempts'] and sum(r['accepted'] for r in log)==treatment['accepted'],'attempt tally differs')
        require([r['event_ids'] for r in log]==plan[name][:len(log)],'immutable plan order differs')
        require(all(r['attempt']==1861+i and r['index']==i and r['lr_multiplier']==run.multiplier(i) and
                    len(set(r['event_ids']))==8 and (not r['accepted'] or
                    (r['loss'] is not None and r['gradient_norm'] is not None and finite(r))) for i,r in enumerate(log)),'attempt health or sequence differs')
        require(sum(r.get('clipped',False) for r in log)==treatment['clipped'] and
                dict(Counter(r['rejection'] for r in log if not r['accepted']))==treatment['nonfinite_rejections'], 'optimization health aggregates differ')
        if result['stopped_early']:
            require('evaluation' not in treatment, 'stopped pair unexpectedly has official scores')
            continue
        rows[name]={};metrics[name]={};private[name]={}
        for split in ('train','val'):
            ref=treatment['evaluation'][split]
            rows[name][split],metrics[name][split]=read_score(ref,split)
            require(ref['data_identity']==inputs['baseline_summary']['result'][split]['data_identity'],'baseline packet digest differs')
            private[name][split]={'rows_ref':ref['rows_ref'],'metrics_ref':ref['metrics_ref']}
    if result['stopped_early']:
        output = {'run_id':run.RUN, 'job_id':'pair01', 'summary_ref':summary_ref,
                  'analysis_source_ref':identity(Path(__file__).resolve()),
                  'gates':{'all_200_matched_attempts':False, 'all_working_parent_gates':False},
                  'provisional_action':'retain_existing_working_parent', 'automatic_adoption':False,
                  'paired_attempts':result['paired_attempts'], 'endpoint_metrics':None}
        ref = baseline.atomic_write(ROOT/'results'/run.RUN/'analysis.json', output, public=True)
        print(json.dumps({'status':'stopped_early', 'analysis_ref':ref}))
        return
    parent={};parent_metrics={}
    for split in ('train','val'):
        parent[split],parent_metrics[split]=read_score(inputs['baseline_summary']['result'][split],split)
        for name in rows:
            require([(r['event_id'],r['label'],r['actual_future_points']) for r in rows[name][split]]==
                    [(r['event_id'],r['label'],r['actual_future_points']) for r in parent[split]],'paired endpoint identities differ')
    gates,changes=decisions(result,rows,metrics,parent)
    short=diagnostic(result['short_diagnostic'])
    check_short(result['short_diagnostic'],read(short['private_rows_ref']))
    detections=all(gates[k] for k in ('candidate_val_actual_at_least_6_of_12','candidate_val_actual_at_least_control_plus_2',
        'candidate_train_actual_at_least_15_of_43'))
    action=('root_review_working_parent' if gates['all_working_parent_gates'] else
            'consider_direct_zero_sum_fast_dynamics_design' if gates['optimization_health'] and gates['matched_numerical_coverage'] and
             gates['candidate_val_stable_protection'] and not detections and short['primary_signed_reductions_both_below_5pct'] else 'retain_existing_working_parent')
    paired={}
    for split in ('train','val'):
        a,b=timing(rows['control'][split]),timing(rows['candidate'][split]); common=sorted(set(a)&set(b))
        paired[split]={'common_comparable_count':len(common),'mean_candidate_minus_control_signed_lag_s':
            sum(b[e]-a[e] for e in common)/len(common) if common else None,
            'mean_candidate_minus_control_absolute_lag_s':sum(abs(b[e])-abs(a[e]) for e in common)/len(common) if common else None,
            'identity_sources':private}
    output={'run_id':run.RUN,'job_id':'pair01','summary_ref':summary_ref,'baseline_summary_ref':summary['release']['input_refs']['baseline_summary'],
        'analysis_source_ref':identity(Path(__file__).resolve()),'gates':gates,'provisional_action':action,'automatic_adoption':False,
        'scope':'single fixed-budget matched strategy comparison; root review and final decision required',
        'finite_health_basis':'completed frozen runner enforces finite updated model and Adam states; accepted attempt logs independently checked; no model loaded by analysis',
        'changes':changes,'private_identity_sources':private,'short_diagnostic':short,'paired_actual_timing':paired,
        'endpoints':{name:{s:{**describe(rows[name][s],metrics[name][s],s),'autonomous_timing':result['treatments'][name]['evaluation'][s]['timing']}
                         for s in ('train','val')} for name in rows},
        'frozen_parent':{s:describe(parent[s],parent_metrics[s],s) for s in ('train','val')},
        'training':{n:{k:t[k] for k in ('attempts','accepted','rejected','clipped','maximum_finite_gradient_norm','nonfinite_rejections','actual_event_exposures','work','train_cost','attempts_ref')} for n,t in result['treatments'].items()},
        'loss_definitions':summary['loss_definitions'],'cost':summary['cost']}
    require(finite(output),'nonfinite analysis output')
    ref=baseline.atomic_write(ROOT/'results'/run.RUN/'analysis.json',output,public=True)
    print(json.dumps({'status':'completed','analysis_ref':ref,'gates':gates,'provisional_action':action}))


def self_check():
    def synthetic(split,detected):
        stable,unstable,missing=(342,43,0) if split=='train' else (95,22,10)
        rows=[]
        for i in range(stable+unstable):
            absent=i>=stable+unstable-missing
            category={'los_pair_pass':True,'truth_supported':True,'classification':{'predicted_los':stable<=i<stable+detected}}
            rows.append({'event_id':split+str(i),'label':'stable' if i<stable else 'unstable',
                'actual_future_points':None if absent else 800,
                'scopes':{h:{'numerical':{'passed':not absent},'metrics':None if absent else {},'full_horizon_truth':not absent} for h in ('1','2','5','8','actual')},
                'classification_8s':None if absent else category,'classification_actual':None if absent or i<stable else category})
        return rows
    rows={n:{s:synthetic(s,15 if s=='train' else 6 if n=='candidate' else 4) for s in ('train','val')} for n in ('control','candidate')}
    result={'paired_attempts':200,'fixed_endpoint':2060,'stopped_early':False,'treatments':{
        n:{'attempts':200,'accepted':198,'rejected':2,'work':{'rhs_forward':100,'rhs_recompute':0},'actual_event_exposures':1600} for n in rows}}
    metrics={'candidate':{'val':{'splits':{'val':{'stable_8s':{'official':{'motion_risk_v1':.9}}}}}}}
    parent={s:synthetic(s,3 if s=='train' else 0) for s in ('train','val')}
    gates,_=decisions(result,rows,metrics,parent)
    assert gates['all_working_parent_gates']
    result['treatments']['candidate']['accepted']=195;result['treatments']['candidate']['rejected']=5
    assert not decisions(result,rows,metrics,parent)[0]['optimization_health']
    result['treatments']['candidate'].update(accepted=198,rejected=2)
    rows['candidate']['val'][0]['scopes']['8']['numerical']['passed']=False
    assert not decisions(result,rows,metrics,parent)[0]['matched_numerical_coverage']
    rows['candidate']['val'][0]['scopes']['8']['numerical']['passed']=True
    extra=copy.deepcopy(parent['train'][-1]); extra['classification_actual']['classification']['predicted_los']=True
    parent['train'][-1]=extra
    assert not decisions(result,rows,metrics,parent)[0]['retains_parent_train_actual_hits']
    metrics['candidate']['val']['splits']['val']['stable_8s']['official']['motion_risk_v1']=1.
    assert not decisions(result,rows,metrics,parent)[0]['candidate_val_stable_protection']
    item={'raw_denominator':22,'cross_precision_qualified_events':22,'selected_complete':22,'complete_events':22,
          'event_equal_electrical_normalized_mse':1.,'selected_signed_mean':1.}
    short={'aggregates':{n:{w:{'persistent_22':copy.deepcopy(item),'parent_misses':{**item,'raw_denominator':27}}
        for w in ('0.03','capped_0.25')} for n in ('control','candidate')},
        'frozen_parent_aggregates':{'persistent_22':{w:{'parent_1860':copy.deepcopy(item)} for w in ('0.03','capped_0.25')}},'rows_ref':{}}
    assert not diagnostic(short)['positive_electrical_diagnostic']
    assert diagnostic(short)['primary_signed_reductions_both_below_5pct']
    for w in ('0.03','capped_0.25'): short['aggregates']['candidate'][w]['persistent_22']['event_equal_electrical_normalized_mse']=.9
    assert diagnostic(short)['positive_electrical_diagnostic']
    short['aggregates']['candidate']['0.03']['persistent_22']['cross_precision_qualified_events']=21
    assert not diagnostic(short)['positive_electrical_diagnostic']
    assert reduction(1.,0.) is None and reduction(1.,None) is None
    c={'los_pair_pass':True,'truth_supported':True,'classification':{'predicted_los':True},'timing_comparable':True,'trigger_time_error_s':.1}
    assert hits([{'event_id':'a','label':'unstable','classification_actual':c}])=={'a'}
    assert not hits([{'event_id':'a','label':'unstable','classification_actual':{**c,'truth_supported':False}}])
    assert not finite({'bad':float('nan')})
    print(json.dumps({'self_check':'passed','scientific_payloads_opened':False,'outputs_written':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('self-check');sub.add_parser('analyze').add_argument('--summary',required=True,type=Path)
    args=parser.parse_args();self_check() if args.command=='self-check' else execute(args.summary)
