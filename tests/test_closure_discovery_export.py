import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))

from local_memory_rom_0098 import LocalMemoryROM
from structured_rom import StructuredROM
from symbolic_closure_rom import SymbolicClosureROM
from closure_discovery_export import export_numpy_package


def _source():
    eye = torch.eye(10, dtype=torch.float64)
    D = torch.zeros((12, 10), dtype=torch.float64)
    D[:9] = eye[:9] - eye[1:]
    D[9] = eye[0] - eye[9]
    D[10] = D[2].clone()
    D[11] = -D[6].clone()
    base = StructuredROM(
        mass=torch.linspace(1.0, 2.0, 10), D=D,
        coupling=torch.linspace(.4, 1.2, 12),
        delta_star=torch.linspace(-.17, .23, 10),
        damping=torch.full((10,), .08), gain=torch.linspace(.8, 1.2, 10),
        jpi_p=torch.full((10,), .2), rpp=torch.full((10,), .15),
        n_q=0, storage=(),
    ).double()
    source = LocalMemoryROM(
        base, input_scale=torch.linspace(.7, 1.4, 20),
        p_scale=torch.linspace(.8, 1.2, 10), arm="L10-S", seed=712,
    )
    return source


def _model():
    terms = [
        {"channel": "node", "expression": {"op": "pow2", "arg": {"op": "var", "name": "q"}}},
        {"channel": "edge", "expression": {"op": "mul", "args": [
            {"op": "pow2", "arg": {"op": "add", "args": [
                {"op": "var", "name": "qi"}, {"op": "var", "name": "qj"}]}},
            {"op": "var", "name": "c"}]}},
        {"channel": "port", "expression": {"op": "tanh", "scale": .5,
            "arg": {"op": "add", "args": [
                {"op": "var", "name": "w"}, {"op": "var", "name": "q"}]}}},
    ]
    model = SymbolicClosureROM.from_source(_source(), terms=terms)
    with torch.no_grad():
        model.coefficients.copy_(torch.tensor([.17, -.23, .31], dtype=torch.float64))
    return model


def _load_runtime(path):
    spec = importlib.util.spec_from_file_location("exported_numpy_simulate", path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    return runtime


def test_numpy_package_field_encoder_and_rollout_parity(tmp_path):
    model = _model()
    paths = export_numpy_package(model, tmp_path / "package")
    assert set(paths) == {"model", "simulator", "equations", "readme"}
    assert all(path.exists() for path in paths.values())
    payload = json.loads(paths["model"].read_text(encoding="utf-8"))
    runtime = _load_runtime(paths["simulator"])

    history = torch.randn(2, 11, 30, dtype=torch.float64) * .025
    encoded = model.encode(history)
    np.testing.assert_allclose(runtime.encode(payload, history.numpy()),
                               encoded.detach().numpy(), atol=2e-11, rtol=0)

    initial = torch.cat((history[:, -1], encoded), dim=-1)
    np.testing.assert_allclose(runtime.field(payload, initial.detach().numpy()),
                               model.field(initial).detach().numpy(), atol=4e-10, rtol=0)
    got = runtime.rollout(payload, history.numpy(), steps=3, dt=.001)
    expected = [initial]
    state = initial
    for _ in range(3):
        state = model.rk4(state, .001)
        expected.append(state)
    expected_array = torch.stack(expected, dim=1).detach().numpy()
    np.testing.assert_allclose(got, expected_array, atol=8e-10, rtol=0)
    np.testing.assert_allclose(got[:, 0, :30], history[:, -1].numpy(), atol=0, rtol=0)

    equations = paths["equations"].read_text(encoding="utf-8")
    assert "not SI units" in equations
    assert "J`, `R`, and `B" in equations
    assert "centered in physical coordinates" in equations
    assert "g_c0 * (c + sin(phi) * eta)" in equations
    assert "AST names `pi` and `pj`" in equations
    assert "0.17" in equations and "-0.23000000000000001" in equations
    assert "does not infer or invent" in equations


def test_exported_simulator_runs_with_torch_and_project_imports_blocked(tmp_path):
    paths = export_numpy_package(_model(), tmp_path / "package")
    history_path = tmp_path / "history.npy"
    output_path = tmp_path / "trajectory.npy"
    np.save(history_path, np.zeros((1, 11, 30), dtype=np.float64))
    script = r'''
import builtins, json, runpy, sys
import numpy as np
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch.") or name in {
        "symbolic_closure_rom", "local_memory_rom_0098", "structured_rom"
    }:
        raise AssertionError("forbidden runtime dependency: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
runtime = runpy.run_path(sys.argv[1])
with open(sys.argv[2], encoding="utf-8") as stream:
    payload = json.load(stream)
history = np.zeros((1, 11, 30), dtype=np.float64)
trajectory = runtime["rollout"](payload, history, steps=1, dt=0.001)
assert trajectory.shape == (1, 2, 40)
assert "torch" not in sys.modules
print(trajectory.shape)
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(paths["simulator"]), str(paths["model"])],
        check=True, capture_output=True, text=True,
    )
    assert result.stdout.strip() == "(1, 2, 40)"

    subprocess.run(
        [sys.executable, str(paths["simulator"]), str(paths["model"]), str(history_path),
         "--steps", "1", "--dt", "0.001", "--output", str(output_path)],
        check=True, capture_output=True, text=True,
    )
    assert np.load(output_path).shape == (1, 2, 40)
