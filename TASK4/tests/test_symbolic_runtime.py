"""Implementation parity only; these fixtures provide no prediction evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from task4.data import Physics
from task4.dynamics import PowerGraph, UnstructuredGraph
from task4.encoder import HistoryEncoder
from task4.symbolic import PolynomialFunction
from task4.symbolic_runtime import SymbolicPowerGraph, export_symbolic


def polynomial_model(n=3, rank=2, hidden=32, strength=0.025):
    """Connected nonuniform fixtures with repeated endpoints and signed phases."""
    rng = np.random.default_rng(730 + n + rank)
    torch.manual_seed(913 + n)
    edges = [(i, i+1) for i in range(n-1)]
    if n > 2:
        edges += [(n-1, 0)]
    incidence = np.zeros((n, len(edges)))
    for k, (i, j) in enumerate(edges):
        incidence[i, k], incidence[j, k] = 1, -1
    physics = Physics(
        mass=rng.uniform(0.7, 2.0, n), incidence=incidence,
        delta_star=rng.uniform(-0.5, 0.5, n), pm_star=rng.uniform(0.1, 0.8, n),
        coupling=rng.uniform(0.2, 1.0, len(edges)), damping=rng.uniform(0.0, 0.3, n),
        rate=rng.uniform(0.2, 1.0, n), gain=rng.uniform(0.3, 1.3, n), omega_s=2*np.pi*50,
    )
    model = PowerGraph(physics, [0.25, 0.3, 0.17], rank=rank, hidden=8, strength=strength)
    for name, module in list(model.function_modules().items()):
        # Include every input linearly, every square, and cross/cubic terms.
        powers = [np.zeros(module.inputs, dtype=np.int64)]
        for d in range(module.inputs):
            for degree in (1, 2):
                p = np.zeros(module.inputs, dtype=np.int64)
                p[d] = degree
                powers.append(p)
        for _ in range(12):
            p = np.zeros(module.inputs, dtype=np.int64)
            for d in rng.integers(module.inputs, size=3):
                p[d] += 1
            powers.append(p)
        coef = rng.normal(0, 0.03, (len(powers), module.outputs))
        setattr(model, name, PolynomialFunction(np.stack(powers), coef))
    with torch.no_grad():
        model.log_coupling.copy_(torch.as_tensor(rng.normal(0, 0.1, len(edges))))
    encoder = HistoryEncoder(model, hidden)
    return model, encoder, physics


@pytest.mark.parametrize('n', [2, 3, 7])
@pytest.mark.parametrize('leading', [(), (4,), (2, 3)])
def test_polynomial_complete_field_matches_torch(n, leading):
    model, encoder, physics = polynomial_model(n=n)
    spec = export_symbolic(model, encoder, omega_s=physics.omega_s)
    runtime = SymbolicPowerGraph(json.loads(json.dumps(spec)))
    x = np.random.default_rng(n).normal(0, 0.12, (*leading, model.dim))
    # Exact zero coordinates exercise analytic derivatives without division by x.
    x[..., ::5] = 0
    expected, detail_torch = model.field(torch.as_tensor(x), create_graph=False, details=True)
    actual, detail_numpy = runtime.field(x, details=True)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-11, atol=2e-12)
    assert set(detail_numpy) == set(detail_torch)
    for key, value in detail_torch.items():
        np.testing.assert_allclose(detail_numpy[key], value.numpy(), rtol=2e-11, atol=2e-12,
                                   err_msg=key)
    np.testing.assert_allclose(runtime.physical(x), model.physical(torch.as_tensor(x)).numpy(),
                               rtol=0, atol=0)
    np.testing.assert_allclose(detail_numpy['Hdot'], detail_numpy['expected_Hdot'],
                               rtol=2e-11, atol=2e-12)
    assert np.all(detail_numpy['dissipation'] >= 0)


@pytest.mark.parametrize('n,hidden', [(2, 32), (3, 5), (7, 32)])
def test_shared_gru_11_point_initialization_matches_torch(n, hidden):
    model, encoder, physics = polynomial_model(n=n, hidden=hidden)
    runtime = SymbolicPowerGraph(export_symbolic(model, encoder, omega_s=physics.omega_s))
    history = np.random.default_rng(805+n).normal(0, 0.3, (3, 11, n, 3))
    ht = torch.as_tensor(history)
    np.testing.assert_allclose(runtime.encode_history(history), encoder(ht).detach().numpy(),
                               rtol=2e-11, atol=2e-13)
    np.testing.assert_allclose(runtime.initial(history), model.initial(ht, encoder).detach().numpy(),
                               rtol=2e-11, atol=2e-13)
    changed = history.copy()
    changed[..., 0] += 3.2
    changed[..., 2] += 400
    np.testing.assert_allclose(runtime.encode_history(changed), runtime.encode_history(history),
                               rtol=2e-11, atol=2e-13)
    # Preserve the exact centered origin across the matrix batch sizes that can
    # select different BLAS paths; do not relax this to a floating tolerance.
    for batch in (1, 3, 5):
        zero = np.zeros((batch, 11, n, 3))
        np.testing.assert_array_equal(runtime.encode_history(zero), np.zeros((batch, n, 2)))
    with pytest.raises(ValueError, match='history'):
        runtime.initial(history[:, :-1])


def test_analytic_gradient_and_centered_origin():
    model, encoder, _ = polynomial_model()
    runtime = SymbolicPowerGraph(export_symbolic(model, encoder))
    x = np.random.default_rng(504).normal(0, 0.07, (2, model.dim))
    _, gradient = runtime.energy_gradient(x)
    numeric = np.empty_like(x)
    step = 2e-6
    for k in range(model.dim):
        plus, minus = x.copy(), x.copy()
        plus[:, k] += step
        minus[:, k] -= step
        numeric[:, k] = (runtime.energy(plus) - runtime.energy(minus))/(2*step)
    np.testing.assert_allclose(gradient, numeric, rtol=2e-7, atol=2e-9)
    zero = np.zeros((2, model.dim))
    h0, g0 = runtime.energy_gradient(zero)
    np.testing.assert_allclose(h0, 0, atol=1e-15)
    np.testing.assert_allclose(g0, 0, atol=1e-15)
    np.testing.assert_allclose(runtime.field(zero), 0, atol=1e-15)
    shifted = x.copy()
    shifted[:, :model.n] += 5.0
    np.testing.assert_allclose(runtime.energy(shifted), runtime.energy(x), atol=1e-12)
    np.testing.assert_allclose(runtime.field(shifted), runtime.field(x), atol=1e-12)


def test_fixed_physical_ports_and_motion_are_retained():
    model, encoder, physics = polynomial_model(strength=0)
    runtime = SymbolicPowerGraph(export_symbolic(model, encoder))
    x = np.random.default_rng(944).normal(0, 0.1, (2, model.dim))
    d, pi, p, q = runtime.split(x)
    w = pi/physics.mass
    k = physics.coupling*np.exp(model.log_coupling.detach().numpy())
    electric = (k*(np.sin(runtime.phase+d@physics.incidence)-np.sin(runtime.phase)))@physics.incidence.T
    expected = runtime.assemble(w, p-electric-physics.damping*w,
                                -physics.rate*p-physics.gain*w, -0.25*q)
    np.testing.assert_allclose(runtime.field(x), expected, rtol=1e-12, atol=1e-12)


def test_export_rejects_uncompressed_and_unstructured_models():
    model, encoder, physics = polynomial_model()
    raw = PowerGraph(physics, [0.25, 0.3, 0.17], hidden=8)
    model.port = raw.port
    with pytest.raises(ValueError, match='Uncompressed.*port'):
        export_symbolic(model, encoder)
    unstructured = UnstructuredGraph(physics, [0.25, 0.3, 0.17], hidden=8)
    with pytest.raises(ValueError, match='exact structured'):
        export_symbolic(unstructured, HistoryEncoder(unstructured))


@pytest.mark.parametrize('corruption', ['negative_power', 'fractional_power', 'nan_coefficient',
                                         'neural_function', 'gate_order', 'missing_function'])
def test_runtime_rejects_invalid_export(corruption):
    model, encoder, _ = polynomial_model()
    spec = copy.deepcopy(export_symbolic(model, encoder))
    fn = spec['functions']['node_energy']
    if corruption == 'negative_power':
        fn['powers'][0][0] = -1
    elif corruption == 'fractional_power':
        fn['powers'][0][0] = 0.5
    elif corruption == 'nan_coefficient':
        fn['coefficients'][0][0] = float('nan')
    elif corruption == 'neural_function':
        fn['type'] = 'mlp'
    elif corruption == 'gate_order':
        spec['encoder']['gate_order'] = ['update', 'reset', 'new']
    else:
        del spec['functions']['port']
    with pytest.raises(ValueError):
        SymbolicPowerGraph(spec)


def test_saved_runtime_has_no_torch_or_training_dependency(tmp_path):
    model, encoder, physics = polynomial_model()
    path = tmp_path/'polynomial.json'
    spec = export_symbolic(model, encoder, path, omega_s=physics.omega_s,
                           metadata={'purpose': 'synthetic implementation regression only'})
    assert spec['omega_s'] == physics.omega_s
    assert spec['coordinates']['time_unit'] == 's'
    assert spec['metadata']['purpose'].startswith('synthetic')
    runtime = SymbolicPowerGraph.from_json(path)
    assert runtime.n == model.n
    source = Path(sys.modules['task4.symbolic_runtime'].__file__)
    # Load just this source file in a fresh interpreter while explicitly denying
    # Torch and all training-package imports; evaluate initialization and field.
    code = '''
import builtins, runpy, sys
real_import = builtins.__import__
def independent_import(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'task4', 'h5py'):
        raise ImportError('Training dependency denied: '+name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = independent_import
import numpy as np
runtime = runpy.run_path(sys.argv[1])['SymbolicPowerGraph'].from_json(sys.argv[2])
x = runtime.initial(np.full((2,11,runtime.n,3),0.04))
f, details = runtime.field(x, details=True)
assert f.shape == x.shape and np.isfinite(f).all()
np.testing.assert_allclose(details['Hdot'],details['expected_Hdot'],atol=1e-12)
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(source), str(path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
