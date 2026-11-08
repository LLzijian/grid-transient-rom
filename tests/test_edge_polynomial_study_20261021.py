"""Train-only selector and accounting contracts; no private observations."""
import inspect,json
import numpy as np
import pytest
from experiments import edge_polynomial_study_20261021 as s

def row(key,mean,B=0,valid=True):
 return {'key':key,'score':{'valid':valid,'mean':mean if valid else None,'B':B if valid else None}}
def test_counts_and_selection_input_excludes_teacher_val():
 c=s.counts()
 assert c['total_updates']==2304 and c['total_checkpoints']==45 and c['total_restores']==9 and c['total_rankings']==27
 assert set(inspect.signature(s.search_seed).parameters)=={'terms','keys','seed','coarse','fine','sc','output','commit','deadline'}
def test_invalid_candidate_cannot_be_restored_by_tie():
 selected,amb=s.choose([row('invalid',0,valid=False),row('valid',1)],2)
 assert selected==[] and amb is False
def test_transitive_interval_chain_is_one_group():
 rows=[row('z',1,.6),row('a',2,.6),row('m',3,.6)]
 selected,amb=s.choose(rows,2)
 assert [x['key'] for x in selected]==['a','m'] and amb
 assert all(set(x['overlap_group'])=={'a','m','z'} for x in rows)
def test_failed_event_keeps_full16_denominator():
 x=np.zeros((16,500,30));bad=x.copy();bad[3,0,0]=np.inf
 r=s.score(bad,x,x,x,np.ones(30))
 assert r['count']==16 and not r['valid'] and r['mean'] is None
 assert len(r['eligible'])==16 and r['eligible'][3] is False
def test_val_barrier_checks_all_sources_before_future(tmp_path):
 f=tmp_path/'selection.json';f.write_text(json.dumps({'all_three_frozen':False,'source_commit':'a'*40,'rows':[]}),encoding='utf-8')
 b={'selection_manifest':{'path':str(f),'sha256':s.sha(f)}}
 with pytest.raises(ValueError,match='all three'):s.frozen_selection(b,'a'*40)
def timing_rows(rank_new=.8,update_new=1.):
 return [{'index':i,'path':p,'operation':op,'rep':rep,'warmup':False,'seconds':(rank_new if op=='rank' else update_new) if p=='new' else 1.}
         for i in range(6) for p in ('old','new') for op in ('update','rank') for rep in range(1,4)]
def test_engineering_all_six_constraints():
 assert s.engineering_gate(timing_rows())['pass']
 rows=timing_rows()
 for r in rows:
  if r['index']==0 and r['path']=='new' and r['operation']=='rank':r['seconds']=1.01
 assert not s.engineering_gate(rows)['pass']
 assert not s.engineering_gate(timing_rows(update_new=1.051))['pass']
def test_fixed_timing_count_plan():
 c=s.cost_plan()
 assert c['checkpoint_saves']==45 and c['restores']==9
 assert c['constructions']+c['final_loads']==30
 assert c['json_writes']==64 and c['raw_writes']==9 and c['startup_multiplier']==2
 assert c['T_cap']*c['science_margin']+c['support']==c['complete_cap']

