"""API-only checks: deterministic arrays and a stub fitting engine, no fitting runs."""
import copy
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import kiki.models as models
from kiki.models.base import RunContext
from kiki.models.gftd_kuramoto import GFTDKuramotoModel


def packet(n=4):
    return dict(theta_history=np.zeros((11,n)), omega_history=np.zeros((11,n)),
                history_times=np.linspace(-.1,0,11), p0=np.arange(n)*.01, sigma=0.)


@pytest.fixture
def engine(monkeypatch):
    stub = ModuleType('kiki.models.gftd_fitting')
    class Failure(RuntimeError):
        def __init__(self, message, diagnostics):
            super().__init__(message)
            self.diagnostics = diagnostics
    stub.FitFailure = type('FitFailure', (Failure,), {})
    stub.InitializationFailure = type('InitializationFailure', (Failure,), {})
    stub.calls = []
    def validate_config(functional, config):
        if not isinstance(config, dict) or set(config) != {'sigma', 'mode', 'n_nodes'} or config['n_nodes'] != functional.n_nodes:
            raise stub.InitializationFailure('missing configuration', {'status':'invalid_config'})
    def fit_shared(functional, packets, *, mode):
        stub.calls.append((copy.deepcopy(packets), mode))
        functional.k_edge[:] = .2
        return dict(functional=functional, config=dict(sigma=float(packets[0]['sigma']), mode=mode,
                    n_nodes=functional.n_nodes), fit_state={'support':None}, status='not_identified')
    def initialize_complete(functional, pkt, config):
        stub.calls.append(copy.deepcopy(pkt))
        n = functional.n_nodes
        return functional.pack_state(pkt['theta_history'][-1],pkt['omega_history'][-1],
                    pkt['p_end'], np.zeros(n), np.zeros(n)), {'status':'completed','calls':2}
    stub.validate_config, stub.fit_shared, stub.initialize_complete = validate_config, fit_shared, initialize_complete
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    monkeypatch.setattr(models, 'gftd_fitting', stub, raising=False)
    return stub


@pytest.mark.parametrize('n', [4,10])
@pytest.mark.parametrize('mode', ['static','modulated'])
def test_fit_dispatch_null_and_roundtrip(engine, tmp_path, n, mode):
    model = GFTDKuramotoModel(n_nodes=n, fit_mode=mode)
    data = [packet(n) for _ in range(3)]
    result = model.fit(data, [], RunContext())
    assert result.best_validation_loss is None and result.val_case_ids == ()
    assert result.status == 'not_identified'
    assert model.initialization_mode == 'causal_two_stage'
    assert engine.calls[0][1] == mode
    assert set(engine.calls[0][0][0]) == {'theta_history','omega_history','history_times','p_end','sigma'}
    assert 'p0' in data[0]
    initial = {k:v for k,v in data[0].items() if k != 'sigma'}
    state = model.initialize(initial)
    assert state.shape == (5*n,)
    np.testing.assert_array_equal(state[2*n:3*n],initial['p0'])
    dest = tmp_path/'fit.json'
    model.save(dest)
    restored = model.load(dest)
    assert restored.fit_state == model.fit_state
    assert restored.last_initialization == model.last_initialization
    np.testing.assert_array_equal(restored.initialize(initial),state)
    np.testing.assert_array_equal(restored.functional.rhs(state),model.functional.rhs(state))
    assert restored.fit_mode == mode


@pytest.mark.parametrize('bad', ['val','extra','sigma','shape','nonfinite','time'])
def test_fit_boundary_rejection(engine, bad):
    model = GFTDKuramotoModel(n_nodes=4)
    data = [packet() for _ in range(3)]
    val = []
    if bad == 'val': val = [packet()]
    if bad == 'extra': data[0]['future'] = 1
    if bad == 'sigma': data[1]['sigma'] = 1e-4
    if bad == 'shape': data[1]['omega_history'] = np.zeros((11,3))
    if bad == 'nonfinite': data[0]['p0'][0] = np.nan
    if bad == 'time': data[0]['history_times'] *= 2
    with pytest.raises(ValueError): model.fit(data,val,RunContext())
    assert engine.calls == []
    assert model.initialization_mode == 'legacy_static'


def test_failed_fit_is_atomic(engine, monkeypatch):
    model = GFTDKuramotoModel(n_nodes=4)
    before = copy.deepcopy(vars(model))
    def fail(functional, packets, *, mode):
        functional.k_edge[:] = 999
        raise engine.FitFailure('no candidate', {'status':'no_output'})
    monkeypatch.setattr(engine, 'fit_shared', fail)
    with pytest.raises(engine.FitFailure): model.fit([packet()]*3,[],RunContext())
    np.testing.assert_array_equal(model.functional.k_edge,before['functional'].k_edge)
    assert model.fit_state is None and model.inference_config is None


def test_initialization_failure_preserves_diagnostics(engine, monkeypatch):
    model = GFTDKuramotoModel(n_nodes=4)
    model.fit([packet()]*3,[],RunContext())
    initial = {k:v for k,v in packet().items() if k != 'sigma'}
    def fail(*args):
        raise engine.InitializationFailure('blocked', {'status':'blocked_by_stage1','calls':17})
    monkeypatch.setattr(engine, 'initialize_complete', fail)
    with pytest.raises(engine.InitializationFailure): model.initialize(initial)
    assert model.last_initialization == {'status':'blocked_by_stage1','calls':17}
    model.inference_config = None
    with pytest.raises(engine.InitializationFailure): model.initialize(initial)
    assert model.last_initialization['status'] == 'invalid_config'


def test_old_v1_dispatch_stays_legacy(engine, tmp_path):
    model = GFTDKuramotoModel(n_nodes=4)
    initial = {k:v for k,v in packet().items() if k != 'sigma'}
    expected = model.initialize(initial)
    path = tmp_path/'legacy.json'
    model.save(path)
    content = json.loads(path.read_text())
    content['format'] = 'gftd_kuramoto_5n_v1'
    for key in ('initialization_mode','fit_mode','inference_config','fit_state','last_initialization'):
        del content[key]
    path.write_text(json.dumps(content))
    restored = model.load(path)
    assert restored.initialization_mode == 'legacy_static'
    np.testing.assert_array_equal(restored.initialize(initial),expected)
    assert engine.calls == []


def test_public_fit_accepts_nonempty_batch_sizes(engine):
    for size in (1, 3, 5):
        model = GFTDKuramotoModel(n_nodes=4)
        model.fit([packet() for _ in range(size)], [], RunContext())
        assert len(engine.calls[-1][0]) == size


@pytest.mark.parametrize('bad', ['shape', 'p_end'])
def test_invalid_engine_output_is_typed(engine, monkeypatch, bad):
    model = GFTDKuramotoModel(n_nodes=4)
    model.fit([packet()]*3, [], RunContext())
    def invalid(*args):
        state = np.zeros(20 if bad == 'p_end' else 19)
        return state, {'status':'completed'}
    monkeypatch.setattr(engine, 'initialize_complete', invalid)
    initial = {k:v for k,v in packet().items() if k != 'sigma'}
    with pytest.raises(engine.InitializationFailure):
        model.initialize(initial)
    assert model.last_initialization['status'] == 'invalid_output'


def test_missing_new_checkpoint_config_is_rejected(engine, tmp_path):
    model = GFTDKuramotoModel(n_nodes=4)
    model.fit([packet()]*3, [], RunContext())
    path = tmp_path/'new.json'
    model.save(path)
    content = json.loads(path.read_text())
    content['inference_config'] = None
    path.write_text(json.dumps(content))
    with pytest.raises(engine.InitializationFailure):
        model.load(path)
    model.inference_config = None
    with pytest.raises(engine.InitializationFailure):
        model.save(path)


def test_successful_fit_drops_legacy_parent_dependency(engine, monkeypatch, tmp_path):
    from test_gftd_kuramoto_architecture import parent_fixture, legal_history
    model = GFTDKuramotoModel.from_parent(parent_fixture())
    assert model.parent_initializer is not None
    expected_legacy = model.initialize(legal_history())
    assert expected_legacy.shape == (50,)
    model.fit([packet(10)]*3, [], RunContext())
    assert model.parent_initializer is None
    path = tmp_path/'refitted_parent.json'
    model.save(path)
    assert json.loads(path.read_text())['parent_initializer'] is None
    monkeypatch.setitem(sys.modules, 'r_memory_model_20261025', None)
    restored = GFTDKuramotoModel.load(path)
    assert restored.parent_initializer is None
    assert restored.initialization_mode == 'causal_two_stage'
    initial = {key:value for key,value in packet(10).items() if key != 'sigma'}
    np.testing.assert_array_equal(model.initialize(initial),restored.initialize(initial))
