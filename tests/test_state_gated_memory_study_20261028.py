"""Synthetic-only study tests; no optimizer calls or real event payloads."""
import copy,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
import state_gated_memory_study_20261028 as s

@pytest.fixture(autouse=True)
def no_optimizer(monkeypatch):
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**k:(_ for _ in ()).throw(AssertionError('no optimizer calls')))

def p(miss=5,risk=1.,r5=1.,r8=1.,timing=0.,uncertain=False):
    cl={'missed_resolved_unstable':miss,'false_los_resolved_stable':0,'numerical_unresolved':0}
    return {'horizon_risks':{h:{'value':v,'numerical_score_perturbation':.00001} for h,v in [('5',r5),('8',r8)]},'classification_actual15s':cl.copy(),'classification_8s':cl.copy(),'unstable_interval_risk':99.,'unstable_actual15_risk':risk,'unstable_actual15_numerical_score_perturbation':.00001,'unstable_actual15_unresolved_ids':[], 'common_frequency_rmse_8s':1.,'normalized_p_rmse_8s':1.,'all_8s_rows_qualified':True,'all_required_rows_qualified':not uncertain,'timing_actual15':{'e':{'numerically_qualified':not uncertain,'truth':{'time_s':0.},'fine':{'time_s':timing},'coarse':{'time_s':timing}}}}

def trace():return [{'gradient_norm':1.,'block_updates':{'a_bar':{'relative_l2':.01}}} for _ in range(48)]

def test_confirmation_is_five_point_endpoint_with_history_offset_and_NA():
    assert s.confirmation({'predicted_los':True,'trigger_index':14},True)['time_s']==.04
    assert s.confirmation({'predicted_los':True,'trigger_index':14},False)['reason']=='numerical_unresolved'
    assert s.confirmation({'category':'latency_uncovered','predicted_los':True,'trigger_index':14},True)['time_s'] is None
    assert s.confirmation({'predicted_los':False,'trigger_index':None},True)['time_s'] is None

def test_original_detector_restores_physics_and_confirms_fifth_sample():
    from evaluate_structured_rom import _classification
    history=np.zeros((11,30));history[10,0]=4.;history[10,10]=.03*2*np.pi*60
    path=np.repeat(history[10:11],5,axis=0)
    meta={'equilibrium_delta':np.zeros(10),'omega_s':2*np.pi*60,'inertia':np.ones(10)}
    result=_classification(path,history,meta,np.ones(10))
    assert result['predicted_los'] and result['trigger_index']==14
    assert s.confirmation(result,True)['time_s']==.04
    assert not _classification(path[:4],history,meta,np.ones(10))['predicted_los']
    uncovered=_classification(path,history,{**meta,'event_time_s':.05},np.ones(10))
    assert s.confirmation(uncovered,True)['reason']=='latency_uncovered'

def test_actual15_risk_not_old_up_to8_and_new_metric_guards():
    before,after=p(),p(risk=.98)
    assert s.extension_decision(before,after,trace())['extend']
    after['unstable_actual15_risk']=1.;after['unstable_interval_risk']=.01
    assert not s.extension_decision(before,after,trace())['extend']
    for key in ('common_frequency_rmse_8s','normalized_p_rmse_8s'):
        after=p(risk=.98);after[key]=1.051;assert not s.extension_decision(before,after,trace())['extend']
    after=p(miss=4);after['classification_8s']['missed_resolved_unstable']=6
    assert not s.extension_decision(before,after,trace())['extend']

def test_optimization_limits_and_perturbation_not_update_count():
    t=trace();t[:13]=[{'gradient_norm':101.,'block_updates':{}}]*13
    assert not s.extension_decision(p(),p(risk=.9),t)['extend']
    a,b=p(),p(risk=.98);a['unstable_actual15_numerical_score_perturbation']=.02
    assert not s.extension_decision(a,b,trace())['extend']

def test_common_T_requires_same_events_and_all_R_N_A_B_C():
    profiles={k:p() for k in ('R_N','A','B','C')};assert s.timing_comparison(profiles)['event_count']==1
    profiles['C']['timing_actual15']['e']['fine']['time_s']=None
    assert s.timing_comparison(profiles)['status']=='UNRESOLVED'
    profiles.pop('B');assert s.timing_comparison(profiles)['status']=='UNRESOLVED'

def test_retention_protects_R_N_timing_and_original_denominators():
    profiles={k:p() for k in ('R_N','A','B')};profiles['C']=p(miss=4,risk=.9)
    assert s.retention(profiles)['second_seed']
    profiles['R_N']=p(miss=4);assert not s.retention(profiles)['second_seed']
    profiles['R_N']=p();profiles['C']['timing_actual15']['e']['fine']['time_s']=.04;profiles['C']['timing_actual15']['e']['coarse']['time_s']=.04
    assert not s.retention(profiles)['second_seed']
    profiles.pop('B');assert s.retention(profiles)['locality']=='UNRESOLVED'

def test_paired_history_steps_explicit_and_restore_on_exception(monkeypatch):
    calls=[]
    class Fake:
        history_dt=.0025
        _cache=None
        fail=False
        def begin_graph_cache(self):self._cache={}
        def end_graph_cache(self):self._cache=None
        def rollout(self,h,n,dt,*,history_dt):
            calls.append((history_dt,n,dt))
            if self.fail:raise ValueError("MOCK")
            return torch.zeros((len(h),n+1,40),dtype=torch.float64)
    m=Fake();h=np.zeros((1,11,30))
    s._paired_rollouts(m,h,1,.01);s._paired_rollouts(m,h,2,.005)
    assert calls==[(s.HISTORY_DT_MAIN,1,.01),(s.HISTORY_DT_REFERENCE,2,.005)] and calls[0][0]!=calls[1][0]
    assert m.history_dt==.0025 and m._cache is None
    m.fail=True
    with pytest.raises(ValueError):s._paired_rollouts(m,h,1,.005)
    assert m.history_dt==.0025 and m._cache is None
    m._cache={}
    with pytest.raises(RuntimeError):s._paired_rollouts(m,h,1,.005)

def test_B_degenerate_completed_update_saved_before_stop(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'build_model',lambda *a:SimpleNamespace(mass=SimpleNamespace(device='cpu')))
    monkeypatch.setattr(s,'make_optimizer',lambda m:object());monkeypatch.setattr(s.fg,'_seed',lambda *a:None)
    schedule={'initial':{'rng':np.random.default_rng(1).bit_generator.state,'coverage':{}},'rows':[{'step':1,'sample_ids':['e'],'horizon_s':5.,'rng_after':np.random.default_rng(1).bit_generator.state,'coverage_after':{}}]}
    s.write(tmp_path/'schedule.json',{'28311':schedule});s.write(tmp_path/'prepare.json',{'MOCK_ONLY':True});(tmp_path/'study').mkdir()
    monkeypatch.setattr(s,'EventStore',lambda *a:None);monkeypatch.setattr(s,'score_panel',lambda *a:p())
    monkeypatch.setattr(s,'one_update',lambda *a,**k:{'control_degenerate':True})
    saved=[];monkeypatch.setattr(s,'save_checkpoint',lambda *a:saved.append(a[5]))
    monkeypatch.setattr(s.old,'quarantine_failure',lambda *a:{'MOCK_ONLY':True})
    r=s.formal_run(tmp_path,'B',28311,{'source':{},'code':{},'schedule_sha256':'MOCK'},[{'id':'e'}],[],'cpu')
    assert r['status']=='CONTROL_DEGENERATE' and r['updates']==1 and saved==[0,1]

def test_paired_evaluator_encode_initialize_history_keyword_chain(tmp_path,monkeypatch):
    """Exercise the real evaluator wrapper; all physics/metrics are MOCK_ONLY."""
    import evaluate_structured_rom as evaluation
    import summarize_train_local_memory_0098 as metrics
    import fixed_edge_pair_20261019 as qualification
    import active_fom_pair_fit_20261020 as intervals
    calls=[]
    class InterfaceFixture(torch.nn.Module):
        def __init__(self):
            super().__init__();self.history_dt=.0025;self._cache=None
            self.register_buffer('mass',torch.ones(10,dtype=torch.float64))
            self.register_buffer('p_scale',torch.ones(10,dtype=torch.float64))
        def encode(self,history,*,history_dt):
            calls.append({'history_dt':history_dt,'persistent_history_dt':self.history_dt})
            return torch.zeros((len(history),40),dtype=torch.float64)
        def initialize(self,history,*,history_dt):
            return self.encode(history,history_dt=history_dt)
        def rollout(self,history,steps,dt,*,history_dt):
            return self.initialize(history,history_dt=history_dt)[:,None,:].repeat(1,steps+1,1)
        def begin_graph_cache(self):self._cache={}
        def end_graph_cache(self):self._cache=None
        def complexity_counts(self):return {'MOCK_ONLY':True,'raw_trainable_scalar_count':47}
    def pair(fine,coarse,*args,**kwargs):
        assert np.isfinite(fine).all() and np.isfinite(coarse).all()
        return {'MOCK_ONLY':True,'finite_to_endpoint':True,'los_pair_pass':True}
    monkeypatch.setattr(evaluation,'_pair_row',pair)
    monkeypatch.setattr(metrics,'event_metrics',lambda *a:{k:[] for k in ('records','common_frequency','normalized_p','objective_point_error','numerical')})
    monkeypatch.setattr(qualification,'scientific_qualification',lambda *a:True)
    monkeypatch.setattr(qualification,'physical_metrics',lambda *a:{'MOCK_ONLY':True})
    monkeypatch.setattr(intervals,'actual_interval_predictions',lambda event,f,c:(f[:9],c[:5],4))
    monkeypatch.setattr(intervals,'interval15',lambda *a:{'MOCK_ONLY':True})
    monkeypatch.setattr(s.old,'group_summary',lambda *a:{'MOCK_ONLY':True})
    model=InterfaceFixture();event={'id':'MOCK_EVENT','label':'stable','group':'MOCK_GROUP','x':np.zeros((15,30)),'classification_meta':{}}
    result=s.paired_evaluate_model(model,[event],tmp_path/'MOCK_EVALUATION',{'fit_ids':['MOCK_EVENT'],'no_update_ids':[],'fresh19_ids':[]})
    assert calls==[{'history_dt':s.HISTORY_DT_REFERENCE,'persistent_history_dt':s.HISTORY_DT_MAIN},
                   {'history_dt':s.HISTORY_DT_MAIN,'persistent_history_dt':s.HISTORY_DT_MAIN}]
    assert result['timing']['initialization_calls']==2
    assert all(row['exception'] is None for row in result['extras'])
    assert model.history_dt==s.HISTORY_DT_MAIN and model._cache is None
    assert 'encode' not in model.__dict__
