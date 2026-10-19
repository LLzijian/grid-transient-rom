import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import numpy as np
import torch
import pytest
import json
import closure_oracle_q0_20261019 as oracle
from closure_oracle_q0_20261019 import reconstruct,oracle_prediction
from closure_causal_recovery_20261019 import fixture,truths,toy_source,copy_runtime

def test_reconstruction_and_prediction_at_forecast_origin():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source,clean,noisy,scale,meta=fixture('saturation',5101,0.,device,time.perf_counter()+60)
    qs,digests=reconstruct('saturation',5101,device,time.perf_counter()+60)
    assert digests==meta['clean_sha256']
    term,coefficient=truths()['saturation'];truth=copy_runtime(source,[term],device)
    with torch.no_grad():truth.coefficients.fill_(coefficient)
    truth.encode=lambda *args: (_ for _ in ()).throw(AssertionError('oracle must never call encoder'))
    obs=torch.tensor(noisy[0],dtype=torch.float64,device=device)
    with torch.no_grad():prediction=oracle_prediction(truth,obs,qs[0],50,.01,time.perf_counter()+60)
    np.testing.assert_allclose(prediction.cpu().numpy(),clean[0][:,11:],atol=2e-13,rtol=2e-13)
    assert not qs[0].requires_grad

def test_expired_reconstruction_cannot_proceed():
    with pytest.raises(TimeoutError):reconstruct('edge',5101,torch.device('cpu'),time.perf_counter()-1)


class ScalarModel(torch.nn.Module):
    terms=['fixed']
    def __init__(self):
        super().__init__();self.value=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64))
    @classmethod
    def from_export(cls,path):
        model=cls();model.value.data.fill_(json.loads(path.read_text())['value']);return model


def test_step32_restores_exact_adam_and_numpy_torch_rng(tmp_path):
    model=ScalarModel();opt=torch.optim.Adam(model.parameters(),lr=.03)
    rng=np.random.default_rng(42);torch.manual_seed(123)
    def update(model,opt,rng):
        opt.zero_grad();loss=(model.value-rng.normal()-torch.rand((),dtype=torch.float64))**2
        loss.backward();opt.step();return float(loss.detach())
    trace=[update(model,opt,rng) for _ in range(32)]
    checkpoint=tmp_path/'checkpoint_00032';checkpoint.mkdir()
    (checkpoint/'model.json').write_text(json.dumps({'value':float(model.value.detach())}))
    torch.save({'optimizer':opt.state_dict(),'fit_numpy_rng':rng.bit_generator.state,
                'torch_rng':torch.get_rng_state(),'cuda_rng':[], 'completed_updates':32,'trace':trace},checkpoint/'training.pt')
    (tmp_path/'latest.json').write_text(json.dumps({'updates':32,'checkpoint':checkpoint.name}))
    expected=update(model,opt,rng);expected_value=model.value.detach().clone()
    restored=ScalarModel();restored_opt=torch.optim.Adam(restored.parameters(),lr=.8);restored_rng=np.random.default_rng(1)
    step,restored_trace=oracle.restore_checkpoint(tmp_path,restored,restored_opt,restored_rng,'cpu')
    assert step==32 and restored_trace==trace
    assert update(restored,restored_opt,restored_rng)==expected
    assert torch.equal(restored.value,expected_value)
    for key in ('exp_avg','exp_avg_sq','step'):
        assert torch.equal(restored_opt.state[restored.value][key],opt.state[model.value][key])
    (tmp_path/'latest.json').write_text(json.dumps({'updates':33,'checkpoint':'checkpoint_00033'}))
    with pytest.raises(ValueError,match='updates'):oracle.restore_checkpoint(tmp_path,restored,restored_opt,restored_rng,'cpu')


def test_step512_only_evaluates_and_writes_native_bool_result(tmp_path,monkeypatch):
    monkeypatch.setattr(oracle,'ROOT',tmp_path)
    original=tmp_path/'private_data/closure_mechanism_20261019/P1/known_controls/node_noise0_seed5101'
    (original/'attempt').mkdir(parents=True)
    meta={'clean_sha256':['a','b'],'scales':[1.]*30,'dual_step_normalized_max_abs':[0.,0.],
          'data_streams':[(0,),(1,),(2,),(3,)]}
    (original/'latest.json').write_text(json.dumps({'attempt':'attempt'}))
    (original/'attempt/fixture.json').write_text(json.dumps(meta))
    arrays=[np.zeros((16,61,30)),np.zeros((16,111,30))]
    monkeypatch.setattr(oracle,'fixture',lambda *a:(None,arrays,arrays,np.ones(30),meta))
    monkeypatch.setattr(oracle,'reconstruct',lambda *a:([torch.zeros((16,10))]*2,['a','b']))
    monkeypatch.setattr(oracle,'runtime_model',lambda *a:ScalarModel())
    calls=[]
    monkeypatch.setattr(oracle,'restore_checkpoint',lambda *a:(512,[0.]*512))
    monkeypatch.setattr(oracle,'save_checkpoint',lambda *a:pytest.fail('must not rewrite checkpoint512'))
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a:pytest.fail('must not fit completed checkpoint'))
    def predict(model,obs,q,steps,dt,deadline):
        calls.append(steps);return torch.zeros((16,steps,30),dtype=torch.float64)
    monkeypatch.setattr(oracle,'oracle_prediction',predict)
    folder=tmp_path/'node_5101';folder.mkdir();(folder/'latest.json').write_text('{}')
    failure=folder/'failure.json';failure.write_text('original failure')
    result=oracle.one('node',5101,'cpu',folder,time.perf_counter()+10,resume=True)
    assert calls==[50,100,100,200]
    assert result['prediction_gate'] is True
    assert json.loads((folder/'result.json').read_text())['prediction_gate'] is True
    assert failure.read_text()=='original failure'
    calls.clear()
    fixture_bytes=(folder/'fixture.json').read_bytes()
    assert oracle.one('node',5101,'cpu',folder,time.perf_counter()+10,resume=True)==result
    assert calls==[]
    assert (folder/'fixture.json').read_bytes()==fixture_bytes
    meta['scales'][0]=2.
    with pytest.raises(ValueError,match='observations/scales differ'):
        oracle.one('node',5101,'cpu',folder,time.perf_counter()+10,resume=True)


def test_recovery_budget_never_resets_and_rejects_unknown_identity(tmp_path):
    previous={'status':'RUNNING','pid':123,'current':'node_5102','code_sha256':oracle.ORIGINAL_CODE_SHA,
              'dependencies':{'x':'hash'},'max_seconds':4500.}
    (tmp_path/'runtime_interruption.json').write_text(json.dumps({'pid':123,'current':'node_5102','elapsed_seconds':1444.3631872}),encoding='utf-8-sig')
    with pytest.raises(ValueError,match='manifest'):oracle.recovery_spend(tmp_path,previous,'repair',{'x':'hash'},4500.,0.)
    (tmp_path/'recovery_checkpoint_manifest.json').write_text(json.dumps({'original_code_sha256':oracle.ORIGINAL_CODE_SHA,'rows':[]}))
    assert oracle.recovery_spend(tmp_path,previous,'repair',{'x':'hash'},4500.,15.)==1459.3631872
    with pytest.raises(ValueError):oracle.recovery_spend(tmp_path,previous,'repair',{'x':'other'},4500.,0.)
    with pytest.raises(ValueError):oracle.recovery_spend(tmp_path,previous,'repair',{'x':'hash'},4500.,4000.)
    previous.update(code_sha256='repair',prior_elapsed_seconds=1459.3631872,pid=456)
    with pytest.raises(ValueError,match='interruption'):oracle.recovery_spend(tmp_path,previous,'repair',{'x':'hash'},4500.,0.)
    previous.update(status='INTERRUPTED',elapsed_seconds=1500.)
    assert oracle.recovery_spend(tmp_path,previous,'repair',{'x':'hash'},4500.,2.)==1502.


def test_recovery_manifest_must_cover_both_files_and_every_checkpoint(tmp_path):
    previous={'status':'INTERRUPTED','elapsed_seconds':100.,'code_sha256':oracle.ORIGINAL_CODE_SHA,
              'dependencies':{},'max_seconds':4500.}
    folder=tmp_path/'node_5101';folder.mkdir()
    (folder/'latest.json').write_text(json.dumps({'checkpoint':'checkpoint_00512','updates':512}))
    manifest={'original_code_sha256':oracle.ORIGINAL_CODE_SHA,'rows':[]}
    path=tmp_path/'recovery_checkpoint_manifest.json';path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='coverage'):oracle.recovery_spend(tmp_path,previous,'repair',{},4500.,0.)
    manifest['rows']=[{'folder':'node_5101','files':{},'checkpoint':'checkpoint_00512','updates':512}]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='both'):oracle.recovery_spend(tmp_path,previous,'repair',{},4500.,0.)
