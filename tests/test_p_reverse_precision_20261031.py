"""Only artificial scalar/matrix/history fixtures, never private assets or rollout."""
import importlib.util
import ast
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import p_reverse_precision_20261031 as p


def test_high_precision_reference_80_120_and_analytic_agree():
    q=p.generator_q([.7,.3],[0.,.6],[1.,1.2])
    reference,report=p.reference_set(q)
    assert report['qualified']
    np.testing.assert_allclose(reference,expm(q)[:2],atol=1e-12,rtol=1e-12)
    assert reference[0,0]==1
    assert reference[0,2]==pytest.approx(.007)
    assert reference[0,4]==pytest.approx(-.000035)


def test_registered_artificial_scalar_qualification():
    assert p.qualify_artificial()['status']=='PASS'


def test_primary_input_is_binary64_Q_and_secondary_time_is_same_float(monkeypatch):
    import mpmath as mp
    q=p.generator_q([.7],[.6],[1.2])
    assert q.dtype==np.float64
    ref,_=p.reference_set(q)
    seen=[];original=mp.mpf
    def track(value,*args,**kwargs):
        if isinstance(value,float):seen.append(value)
        return original(value,*args,**kwargs)
    monkeypatch.setattr(mp,'mpf',track)
    p.precise_assembly_effect(np.array([.7]),np.array([.6]),np.array([1.2]),ref)
    assert .01 in seen


def test_error_table_preserves_zero_reference_relative_NA_and_ulp():
    reference=np.array([[1.,0.,-.1]])
    value=reference.copy();value[0,0]=np.nextafter(1.,2.);value[0,1]=1e-9
    result=p.comparison(value,reference)
    assert result['element_ULP'][0,0]==1
    assert np.isnan(result['element_relative_error_nonzero_reference'][0,1])
    assert result['old_1e12_max_standardized_violation']>1
    assert p.safe(result)['element_relative_error_nonzero_reference'][0][1] is None


class ToyModel(torch.nn.Module):
    """Causal initialization fixture with the same independent p reverse block."""
    def __init__(self):
        super().__init__()
        q=p.generator_q(np.full(10,.7),np.full(10,.6),np.ones(10))
        self.register_buffer('p_backward',torch.as_tensor(expm(q)[:10],dtype=torch.float64))
        for name in ('mass','p_scale','rstar','input_scale_omega'):
            self.register_buffer(name,torch.ones(10,dtype=torch.float64))
        self.dummy=torch.nn.Parameter(torch.ones(1,dtype=torch.float64))
        self.calls=0;self.fields=0;self._cache=None
    def clear_cache(self):self._cache=None
    def history_system(self,history,*,history_dt=None):
        h=torch.as_tensor(history,dtype=torch.float64)
        omega=h[...,10:20];slope=(omega[:,1:]-omega[:,:-1])/.01
        states=[h[:,-1,20:30]]
        for i in range(9,-1,-1):
            states.append(torch.cat((states[-1],omega[:,i+1],slope[:,i]),-1)@self.p_backward.T)
        ph=torch.stack(states[::-1],dim=1)
        bscaled=torch.zeros((len(h),50),dtype=torch.float64);bscaled[:,:10]=ph.mean(dim=1)
        design=torch.zeros((len(h),60,10),dtype=torch.float64)
        design[:,:10]=torch.eye(10,dtype=torch.float64);design[:,50:]=.01*torch.eye(10,dtype=torch.float64)
        r0=bscaled[:,:10]/1.0001
        return dict(p_history=ph,r0=r0,design=design,bscaled=bscaled,F=torch.eye(10,dtype=torch.float64).repeat(len(h),1,1),q=torch.zeros((len(h),10),dtype=torch.float64))
    def encode(self,history,*,history_dt=None):
        return self.history_system(history,history_dt=history_dt)['r0']
    def initialize(self,history,*,history_dt=None):
        self.calls+=1
        return torch.cat((history[:,-1,:30],self.encode(history,history_dt=history_dt)),-1)
    def field(self,state):
        self.fields+=1
        result=torch.zeros_like(state);result[:,10:20]=-state[:,30:40]/self.mass
        result[:,20:30]=-.7*state[:,10:20]-.6*state[:,20:30]
        return result
    def rollout(self,*args,**kwargs):
        pytest.fail('probe may never call rollout, even for zero steps')
    @classmethod
    def from_export(cls,*args):return cls()


def histories():
    rng=np.random.default_rng(17)
    h=np.zeros((4,11,30));h[:,:,:20]=rng.normal(size=(4,11,20))
    h[:,-1,20:30]=rng.normal(size=(4,10))
    return h


def test_actual_call_trace_hits_buffer_ten_times_without_extra_initialize():
    model=ToyModel().eval();h=histories()
    before=p.state_identity(model)
    result,trace=p.capture_initialize(model,h,.0025,trace=True)
    assert trace['status']=='active_buffer'
    assert len(trace['p_backward_line_hits'])==10
    assert set(trace['calls'])=={'initialize','encode','history_system'}
    assert model.calls==1 and model.fields==1
    assert p.state_identity(model)==before
    assert 'history_system' not in model.__dict__
    np.testing.assert_array_equal(result['initial'][:,:30],h[:,-1])


def test_replace_only_buffer_preserves_original_flags_and_other_state():
    model=ToyModel().eval();before=p.state_identity(model)
    ref=model.p_backward.numpy().copy();ref[0,10]+=1e-10
    clone=p.replace_copy(model,ref)
    assert p.state_identity(model)==before
    assert not clone.training
    assert clone.dummy.requires_grad
    assert p.state_identity(clone)['dummy']==before['dummy']
    assert p.state_identity(clone)['p_backward']!=before['p_backward']


def test_propagation_recurrence_QR_and_initial_field_identities():
    model=ToyModel().eval();h=histories()
    ref=model.p_backward.numpy().copy();ref[np.arange(10),10+np.arange(10)]+=1e-10
    clone=p.replace_copy(model,ref)
    old,_=p.capture_initialize(model,h,.0025)
    new,_=p.capture_initialize(clone,h,.0025)
    rows,checks=p.sensitivity_rows(old,new,h,model,ref)
    assert all(checks.values())
    assert len(rows)==4
    assert max(row['delta_p_maxabs'] for row in rows)>0
    assert all(row['delta_p_dot_maxabs']==0 for row in rows)
    assert model.calls==clone.calls==1


def test_zero_r0_reference_relative_error_is_NA():
    model=ToyModel().eval();h=np.zeros((1,11,30))
    old,_=p.capture_initialize(model,h,.00125)
    ref=model.p_backward.numpy().copy()
    rows,checks=p.sensitivity_rows(old,old,h,model,ref)
    assert all(checks.values())
    assert rows[0]['delta_r0_relative_L2'] is None


def test_import_is_inert_and_does_not_open_model_or_data(monkeypatch):
    monkeypatch.setattr(Path,'read_text',lambda *a,**k:pytest.fail('import read payload'))
    spec=importlib.util.spec_from_file_location('reverse_inert',p.__file__)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.MODELS==('A96','step000016')


def test_direct_cli_help_needs_no_private_payloads():
    result=subprocess.run([sys.executable,p.__file__,'--help'],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert '--protocol' in result.stdout and '--design' in result.stdout


def test_cli_main_guard_follows_every_function_definition():
    tree=ast.parse(Path(p.__file__).read_text(encoding='utf-8'))
    guard=tree.body[-1]
    assert isinstance(guard,ast.If)
    assert ast.unparse(guard.test)=="__name__ == '__main__'"
    assert all(node.lineno<guard.lineno for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)))


@pytest.mark.parametrize('change',[{}, {'local_commit':'wrong'}, {'remote_commit':'wrong'}, {'ordinary_push_verified':False}, {'ordinary_push_verified':1}])
def test_archival_receipt_requires_exact_parent_both_sides_and_verified_boolean(change):
    receipt={'local_commit':p.PARENT_COMMIT,'remote_commit':p.PARENT_COMMIT,'ordinary_push_verified':True}
    receipt.update(change)
    if not change:
        p.validate_archival_receipt(receipt)
    else:
        with pytest.raises(ValueError,match='parent archival receipt'):
            p.validate_archival_receipt(receipt)


def test_assembly_deduplicates_exact_raw_factors_not_just_rounded_Q(monkeypatch):
    calls=[]
    def compute(g,rpp,storage,reference):
        calls.append((g.copy(),rpp.copy(),storage.copy()))
        return {'max_abs':0.}
    monkeypatch.setattr(p,'precise_assembly_effect',compute)
    cache={};g=np.array([.7]);rpp=np.array([.5]);storage=np.array([2.]);reference=np.zeros((1,3))
    first=p.cached_precise_assembly(cache,g,rpp,storage,reference)
    second=p.cached_precise_assembly(cache,g.copy(),rpp.copy(),storage.copy(),reference)
    assert first is second and len(calls)==1
    np.testing.assert_array_equal(p.generator_q(g,rpp,storage),p.generator_q(g,np.array([1.]),np.array([1.])))
    third=p.cached_precise_assembly(cache,g,np.array([1.]),np.array([1.]),reference)
    assert len(calls)==2
    assert third['raw_constant_identity_sha256']!=first['raw_constant_identity_sha256']
