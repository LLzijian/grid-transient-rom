"""API/runner checks with existing algebraic fixtures and mocks only."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import gftd_fit297_20261107 as run
from kiki.models.base import RunContext
from kiki.models.gftd_kuramoto import GFTDKuramotoModel
from kiki.models import gftd_fitting as engine
import kiki.models as models
from test_gftd_model_fit import packet


def test_uniform_subset_uses_id_order_and_never_labels():
    assert run.adaptation_indices(297)==[j*297//24 for j in range(24)]
    assert run.adaptation_indices(3)==[0,1,2]
    for n in (0,-1,True,2.0):
        with pytest.raises(ValueError):run.adaptation_indices(n)
    events=[dict(id=str(i),group='stub',label='stable' if i<267 else 'unstable',
        x=np.zeros((11,30),dtype=np.float64),classification_meta=dict(equilibrium_delta=np.full(10,.2),omega_s=2*np.pi*60)) for i in range(297)]
    events[1]['x'][10,20:30]=np.arange(10)
    _,packets,receipt=run.prepare_packets(list(reversed(events)),[e['id'] for e in events],np.zeros(10))
    assert receipt['legal_ids']==sorted(str(i) for i in range(297))
    assert receipt['adaptation_ids']==[receipt['legal_ids'][i] for i in receipt['adaptation_indices']]
    assert all(set(p)=={'theta_history','omega_history','p0','history_times','sigma'} for p in packets)
    np.testing.assert_array_equal(packets[1]['p0'],np.arange(10))
    events[1]['x'][10,20:30]=9
    np.testing.assert_array_equal(packets[1]['p0'],np.arange(10))


def test_invalid_history_retains_id_and_denominator():
    events=[dict(id=str(i),group='stub',label='stable' if i<267 else 'unstable',x=np.zeros((11,30)),
        classification_meta=dict(equilibrium_delta=np.zeros(10))) for i in range(297)]
    events[7]['x'][0,0]=np.nan
    _,packets,receipt=run.prepare_packets(events,[e['id'] for e in events],np.zeros(10))
    assert len(packets)==296 and receipt['invalid'][0]['event_id']=='7'
    assert receipt['denominators']['initializations']==594
    with pytest.raises(ValueError,match='identity'):
        run.prepare_packets(events,['absent']*297,np.zeros(10))


def test_fit297_dispatch_is_explicit_atomic_and_keeps_private_details(monkeypatch,tmp_path):
    stub=ModuleType('kiki.models.gftd_fit297');calls=[]
    def fit(f,packets,*,mode,config):
        calls.append((packets,mode,config));config['mutable']='engine-owned'
        f.k_edge[:]=.3
        return dict(functional=f,config={'stub':True,'sigma':0.},fit_state={'support':None},
                    private_details={'private_matrix':[[1.,2.]]},status='not_identified')
    stub.fit_shared=fit
    monkeypatch.setitem(sys.modules,stub.__name__,stub);monkeypatch.setattr(models,'gftd_fit297',stub,raising=False)
    monkeypatch.setattr(engine,'validate_config',lambda f,c:c)
    request={'stub':True}
    model=GFTDKuramotoModel(n_nodes=4)
    result=model.fit([packet()],[],RunContext(gftd_fit297_v1=request))
    assert request=={'stub':True} and result.best_validation_loss is None
    assert set(calls[0][0][0])=={'theta_history','omega_history','p_end','history_times','sigma'}
    assert model.fit_private_details=={'private_matrix':[[1.,2.]]}
    model.save(tmp_path/'model.json')
    assert 'private_matrix' not in (tmp_path/'model.json').read_text()
    before=model.functional.k_edge.copy()
    def fail(f,*args,**kwargs):
        f.k_edge[:]=999
        raise engine.FitFailure('stub rejected',{'status':'no_output'})
    stub.fit_shared=fail
    with pytest.raises(engine.FitFailure):model.fit([packet()],[],RunContext(gftd_fit297_v1=request))
    np.testing.assert_array_equal(model.functional.k_edge,before)


def test_atomic_file_and_stale_pending_are_not_overwritten(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'ROOT',tmp_path)
    result=run.write_json(tmp_path/'record.json',{'value':np.float64(2.)})
    assert result['sha256']==hashlib.sha256((tmp_path/'record.json').read_bytes()).hexdigest()
    (tmp_path/'record.json.tmp').write_bytes(b'unknown write')
    with pytest.raises(FileExistsError):run.write_json(tmp_path/'record.json',{'value':3})
    assert json.loads((tmp_path/'record.json').read_text())=={'value':2.}


def test_missing_release_never_enters_dataset_or_claim(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'OUTPUT',tmp_path)
    run.write_json(tmp_path/'execution_release.json',{'released':False})
    with pytest.raises(PermissionError):run.run()
    assert not (tmp_path/'started.json').exists()


def test_claim_precedes_payload_and_unknown_job_cannot_replay(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run,'OUTPUT',tmp_path/'results')
    monkeypatch.setattr(run,'PRIVATE',tmp_path/'private')
    run.OUTPUT.mkdir()
    run.write_json(run.OUTPUT/'execution_release.json',dict(action_id='stub',source_sha256={},asset_sha256={}))
    monkeypatch.setattr(run,'verify_release',lambda r:True)
    monkeypatch.setattr(run,'environment',lambda : {'stub':True})
    def stop():
        assert (run.OUTPUT/'started.json').exists()
        raise RuntimeError('engineering stop before event read')
    monkeypatch.setattr(run,'frozen_model',stop)
    with pytest.raises(RuntimeError,match='engineering stop'):run.run()
    receipt=run.read_json(run.OUTPUT/'execution.json')
    assert receipt['counts']['fits_attempted']==0
    assert receipt['event_payload_read'] is False
    assert receipt['status']=='INTERRUPTED_RUNTIME'
    with pytest.raises(FileExistsError):run.run()


def test_integrator_counts_and_nan_placeholder_after_first_failure(monkeypatch):
    calls=[]
    def rk4(f,z,step):
        calls.append(z.copy())
        if len(calls)==2:raise FloatingPointError('stub failure')
        return z+1
    monkeypatch.setattr(run.core,'rk4',rk4)
    counts=dict(rk4_attempted=0,rk4_completed=0)
    path,error=run.integrate(None,np.zeros(50),.01,counts)
    assert path.shape==(1501,50) and error['first_failed_step']==2
    assert counts==dict(rk4_attempted=2,rk4_completed=1)
    assert np.isnan(path[2:]).all()


def test_mock_run_freezes_all_initializers_before_any_future_and_preserves_failure(monkeypatch,tmp_path):
    import closure_probe_20261030 as loader
    import gftd_fit297_analysis_20261107 as analysis
    from kiki.models import gftd_fit297 as fit_engine
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run,'OUTPUT',tmp_path/'results')
    monkeypatch.setattr(run,'PRIVATE',tmp_path/'private');run.OUTPUT.mkdir()
    run.write_json(run.OUTPUT/'execution_release.json',dict(action_id='stub',source_sha256={},asset_sha256={}))
    run.write_json(run.OUTPUT/'static_identity_audit.json',dict(shards=[dict(equilibrium_arrays=dict(delta=dict(sha256='unused')))]))
    monkeypatch.setattr(run,'verify_release',lambda r:True);monkeypatch.setattr(run,'environment',lambda : {'stub':True})
    trace=[]
    class Model:
        mass=np.ones(10);delta_star=np.zeros(10);functional=None;last_initialization=None
        def fit(self,packets,val,ctx):
            trace.append(('fit',self.fit_mode));self.fit_state={'counts':{},'preparation_key':'stub'}
            self.inference_config={};self.fit_private_details={'stub':True}
        def save(self,path):Path(path).write_text('{}')
        def initialize(self,packet):
            trace.append(('initialize',self.fit_mode));self.last_initialization={'counts':{'calls':1}}
            if self.fit_mode=='modulated':raise engine.InitializationFailure('stub failure',self.last_initialization)
            return np.zeros(50)
    monkeypatch.setattr(run,'frozen_model',lambda :(Model(),{k:[1.]*10 for k in ('p_scale','rstar','input_scale_delta','input_scale_omega')},{'k_ref':1.}))
    event=dict(id='stub',group='fixture',label='unstable',x=np.zeros((1511,30)),classification_meta=dict(equilibrium_delta=np.zeros(10)))
    pkt=packet(10)
    monkeypatch.setattr(loader,'load_frozen_events',lambda :({'fit_ids':['stub']},[event]))
    monkeypatch.setattr(run,'prepare_packets',lambda *args,**kwargs:({'stub':event},[pkt],dict(legal_ids=['stub'],ordered_all_ids=['stub'],invalid=[],adaptation_indices=[0])))
    monkeypatch.setattr(fit_engine,'_CACHE',{})
    monkeypatch.setattr(fit_engine,'coverage_information',lambda *args: {'status':'mock'})
    monkeypatch.setattr(fit_engine,'analysis_budget_snapshot',lambda key:{'status':'mock'})
    def integrate(f,state,step,counts):
        assert (run.PRIVATE/'all_initializations_frozen.json').exists()
        assert trace==[('fit','static'),('fit','modulated'),('initialize','static'),('initialize','modulated')]
        return np.zeros((int(round(15/step))+1,50)),None
    monkeypatch.setattr(run,'integrate',integrate)
    for name in ('l2_certificate','l2_point','rom_modes','kron_reference','fom_modes','match_modes','graph_correspondence'):
        monkeypatch.setattr(analysis,name,lambda *args,**kwargs: {'status':'mock'})
    result=run.run()
    assert result['status']=='COMPLETED'
    assert result['counts']['fits_attempted']==2
    assert result['counts']['initializations_attempted']==2 and result['counts']['initializations_completed']==1
    assert result['counts']['future_pairs_attempted']==1
    failed=run.read_json(run.OUTPUT/'events/modulated_000.json')
    assert failed['failure']['stage']=='initialization'
    summary=run.read_json(run.OUTPUT/'l1_summary.json')
    assert summary['static']['scopes']['unstable_actual15']['denominator']==30
    assert summary['modulated']['scopes']['unstable_actual15']['official_fine'] is None


def test_execution_budget_charges_each_scope_once_and_preserves_limit_failure():
    counts=dict(fit={'static':{'rk4_steps_attempted':100,'information_history_steps':20}},
        initialization={'rk4_steps_attempted':50},coverage={'static':{'rk4_steps_attempted':120,'information_history_steps':120}},
        rk4_attempted=200,rk4_completed=190)
    summary=run.execution_budget(counts)
    assert summary['actual']['rk4_steps_attempted']==470
    assert summary['actual']['information_history_steps']==140
    assert summary['within_limits']
    counts['rk4_attempted']=49229641
    assert not run.execution_budget(counts)['within_limits']
