"""Recover compact public accounting from the exact saved failed-probe JSON only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter, process_time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.osgkd_v2_electrical_decomposition_20261123 import run

LOCAL = run.ROOT/'artifacts'/run.RUN/'probe01'
PUBLIC = run.ROOT/'results'/run.RUN/'probe01'
EXPECTED = {
 'started.json':'85a21140c9458897699103d840c802f0e578443333f5e7f6b41535dc1dc0a2ad',
 'failed.json':'4af109ccffc1e124c42c70b1e08906777c48712a34114df827eb702614385d64',
 'rows.json':'aad6fb8c3d78fa72a439628e938ad1dccebff71c92d4a34e64c46e315f35d28e'}
RELEASE_SHA = 'c7e68e1ff61a0ae38d574cf0b7ad1d46a02e5ab8da614157cacde6a0ed640285'


def compact(item):
    """Drop repeated event distributions, retaining event-equal scales, signs and coverage."""
    keys = ('denominator','covered_events','complete_events','point_count','event_equal_means','fractions','amplitude','edges')
    result = {k:item[k] for k in keys}
    result['components'] = {component:{metric:{'count':d['count'],'mean':d.get('mean')}
                                      for metric,d in values.items()} for component,values in item['components'].items()}
    return result


def build():
    refs = {name:run.identity(LOCAL/name) for name in EXPECTED}
    run.require(all(refs[name]['sha256']==sha for name,sha in EXPECTED.items()), 'native receipt/rows hash differs')
    started,failed,saved = (run.read(refs[name]) for name in ('started.json','failed.json','rows.json'))
    run.require(started['run_id']==run.RUN and started['job_id']=='probe01' and started['status']=='started', 'native started receipt differs')
    run.require(failed['status']=='failed' and failed['error']=='ValueError: public summary exceeds 512 KiB', 'not the authorized presentation-only failure')
    release_ref = started['release_ref']
    run.require(release_ref['sha256']==RELEASE_SHA, 'native release hash differs')
    release = run.read(release_ref)
    run.require(release==started['release'], 'native embedded release differs')
    for name,sha in release['sources'].items():
        run.require(run.identity(run.repo_path(name))['sha256']==sha, 'native source differs: '+name)
    rows,groups,detail = saved['events'],saved['group_event_ids'],saved['spectrum_detail']
    run.require(len(rows)==297 and len({r['event_id'] for r in rows})==297 and
                {k:len(v) for k,v in groups.items()}=={'parent_misses':27,'persistent_22':22,'worsened':24,'remaining':243}, 'saved cohort differs')
    gate = run.gate(rows,groups)
    for check in gate['checks'].values():
        for key in ('missed','selected'): check[key] = compact(check[key])
    values = detail['eigenvalues']
    run.require(len(values)==49 and len(detail['normalized_jacobian'])==50, 'saved spectrum dimensions differ')
    spectrum = {'dimension':49,'max_real':max(v['real'] for v in values),'eigenvalues':values,
      'autodiff_unstable_mode_count_above_1e_7':sum(v['real']>1e-7 for v in values),
      'native_numerical_qualification_passed':True,
      'qualification_evidence':'Exact native source writes rows only after qualified 49-dimensional AD/two-step FD checks pass.',
      'finite_difference':None,'max_real_fd_difference':None,'unstable_count_real_tolerance_per_s':None,
      'unstable_mode_count':None,'direction':None,
      'unavailable_reason':'Native public receipt was not saved; exact FD residuals, FD maxima and original tolerance-adjusted sign/count were not persisted. No FD recomputation.',
      'historical_max_real_per_s':{'v1':1.475,'epoch1732':.997}}
    points = sum(r['models'][name]['windows']['capped_0.25']['requested_point_count']+1 for r in rows for name in run.ALIASES)
    run.require(points<=23166 and all(set(r['models'])==set(run.ALIASES) for r in rows), 'saved readout work differs')
    work = {'port_state_points':points,'port_calls':len(rows)*len(run.ALIASES),'reference_spectra':1,
      'spectrum_rhs_calls':202,'spectrum_autodiff_jacobians':1,'spectrum_finite_difference_directions':100,
      'parameter_updates':0,'integrations':0,'encoder_calls':0,'new_fom':0,
      'count_evidence':'Saved capped-window requested points plus t=0 per model/event; frozen P0 spectrum source: 1 equilibrium RHS + 1 AD function call + 2*50*2 central-difference RHS calls.'}
    return {'run_id':run.RUN,'job_id':'probe01','status':'completed','completion_kind':'technical_summary_recovery',
      'native_execution_status':'failed','native_failure':failed,'native_started_ref':refs['started.json'],
      'native_failed_ref':refs['failed.json'],'release_ref':release_ref,'native_sources':release['sources'],
      'recovery_code_ref':run.identity(Path(__file__).resolve()),
      'result':{'rows_ref':refs['rows.json'],'group_event_ids':groups,
        'groups':{g:{w:{m:compact(run.aggregate(rows,set(ids),m,w)) for m in run.ALIASES} for w in run.WINDOWS} for g,ids in groups.items()},
        'gate':gate,'spectrum':spectrum,'resource_counts':work,
        'sensitivity_removed_generator':{'index':9,'name':'GENROU_10','bus':39},
        'component_scope':'Output absolute normalized mean square and signed mean only; no separate observed component labels.',
        'window_mean':'t>0; complete ten-node whole-window label coverage; time mean per event then event-equal group mean.'},
      'recovery_work':{'checkpoint_loads':0,'trajectory_loads':0,'model_calls':0,'rhs_calls':0,
                       'integrations':0,'optimizer_updates':0,'new_fom':0},
      'native_cost':{'wall_s':failed['wall_s'],'cpu_s':failed['cpu_s']}}


def main(write_outputs=False):
    wall,cpu = perf_counter(),process_time()
    summary = build()
    summary['recovery_cost'] = {'wall_s':perf_counter()-wall,'cpu_s':process_time()-cpu}
    summary['recovered_utc'] = datetime.now(timezone.utc).isoformat()
    raw = (json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    run.require(len(raw)<512*1024, 'compact public summary exceeds 512 KiB')
    if write_outputs:
        run.require(not (PUBLIC/'summary.json').exists() and not (LOCAL/'recovery_completed.json').exists(), 'recovery output already exists')
        ref = run.write(PUBLIC/'summary.json',summary)
        run.write(LOCAL/'recovery_completed.json',{'status':'completed','completion_kind':'technical_summary_recovery','summary_ref':ref,
                                                'native_failed_ref':summary['native_failed_ref'],'recovery_code_ref':summary['recovery_code_ref']})
    print(json.dumps({'mode':'write' if write_outputs else 'check','summary_bytes':len(raw),'gate':summary['result']['gate']['selected'],
                      'recovery_code_ref':summary['recovery_code_ref']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('check','write'))
    args=parser.parse_args()
    main(args.command=='write')
