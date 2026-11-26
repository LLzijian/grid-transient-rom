"""Synthetic baseline preflight; never opens scientific payloads."""
import numpy as np
import torch
from experiments.osgkd_v2_sprint2b_baselines_20261126 import fit
from experiments.osgkd_v2_discovery import prepare
from kiki.models.osgkd_fitting import IntegrationWork,tensor_events


def constants(n):
    value={key:np.ones(n) for key in ('mass','damping','Sp','Se','omega_scale','S_m')}
    value.update(F=np.eye(n),theta_ref=np.zeros(n),edge_index=np.stack(np.triu_indices(n,1)),s_coi=2.)
    return value


def test_shared_encoder_and_rng_course():
    torch.set_num_threads(1)
    c=constants(10)
    graph=fit.train.new_model(c,'graph',24111)
    original=graph.encoder.state_dict()
    optimizer_defaults=fit.train.new_optimizer(graph).defaults
    for job,name in fit.JOBS.items():
        reference=prepare.new_baseline(name,c,24111)
        expected_rng=torch.get_rng_state().clone()
        model=fit.new_model(c,job)
        assert torch.equal(torch.get_rng_state(),expected_rng)
        assert fit.run.old.prior.p1a.tree_equal(reference.checkpoint(),model.checkpoint())
        assert all(torch.equal(value,model.encoder.state_dict()[key]) for key,value in original.items())
        metadata=fit.counts(model,job)
        assert metadata['ode_parameters']==(45 if job==fit.ORDER[0] else 671)
        optimizer=fit.new_optimizer(model)
        assert optimizer.defaults==optimizer_defaults
        assert [group['lr'] for group in optimizer.param_groups]==[.001,.0003]
        assert [sum(parameter.numel() for parameter in group['params'])
                for group in optimizer.param_groups]==[1928,metadata['ode_parameters']]


def test_original_recipe_short_synthetic_batch():
    torch.set_num_threads(1)
    c=constants(2);rng=np.random.default_rng(22)
    event=dict(id='synthetic',history={key:rng.normal(0,.001,(11,2)) for key in ('theta','omega')},
               t=np.arange(12)*.01,targets={key:rng.normal(0,.001,(12,2)) for key in ('theta','omega','tau_m','tau_e')},
               masks={key:np.ones((12,2),bool) for key in ('theta','omega','tau_m','tau_e')})
    batch=tensor_events([event],'cpu',max_steps=800)
    for job in fit.ORDER:
        model=fit.new_model(c,job);opt=fit.new_optimizer(model);work=IntegrationWork()
        row=fit.train.attempt(model,opt,batch,896,c,work,.01)
        assert row['accepted'] and row['internal_step']==.01 and row['horizon_s']==8
        assert work.rhs['forward']==44
        assert row['alpha'] is not None and row['spec'] is not None
        saved=fit.payload(model,opt,[row],[],work,IntegrationWork(),{}, {})
        restored,adam=fit.restore(saved,c,job)
        assert fit.run.old.prior.p1a.tree_equal(restored.checkpoint(),model.checkpoint())
        assert fit.run.old.prior.p1a.tree_equal(adam.state_dict(),opt.state_dict())


def test_guard_only_converts_declared_numerical_failure(monkeypatch):
    import pytest
    work=IntegrationWork()
    def fail(*args,**kwargs):
        raise RuntimeError('nonfinite fine qualification: blocked')
    monkeypatch.setattr(fit.train,'guard',fail)
    assert fit.guarded(None,[],{},work,800)['passed'] is False
    def bug(*args,**kwargs):
        raise ValueError('unexpected programming failure')
    monkeypatch.setattr(fit.train,'guard',bug)
    with pytest.raises(ValueError,match='unexpected programming failure'):
        fit.guarded(None,[],{},work,800)


def test_effective_passive_spectrum_matches_embedded():
    from experiments.osgkd_v2_p0_20261123.run import spectrum
    from experiments.osgkd_v2_sprint2b_baselines_20261126.score import spectrum_effective_passive

    model = fit.new_model(constants(10), fit.ORDER[0])
    embedded, detail = spectrum(model)
    active = spectrum_effective_passive(model, embedded, detail)
    assert active['qualified']
    assert (active['state_dimension'], active['quotient_dimension']) == (20, 19)
    assert active['neutral_common_angle_mode_count'] == 1
    assert active['embedded_dummy_modes_at_minus_one'] == 30
    assert np.isclose(active['max_real'], embedded['max_real'], rtol=1e-10)
