import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))

from local_memory_rom_0098 import LocalMemoryROM
from structured_rom import StructuredROM
from symbolic_closure_rom import (
    SymbolicClosureROM,
    evaluate_expression_numpy,
    evaluate_expression_torch,
    numpy_field,
    validate_expression,
)


def _source():
    eye = torch.eye(10, dtype=torch.float64)
    D = torch.zeros((7, 10), dtype=torch.float64)
    D[:6] = eye[:6] - eye[1:7]
    D[6] = eye[0] - eye[9]
    base = StructuredROM(
        mass=torch.linspace(1.0, 2.0, 10), D=D,
        coupling=torch.linspace(.4, 1.2, 7),
        delta_star=torch.linspace(-.17, .23, 10),
        damping=torch.full((10,), .08), gain=torch.linspace(.8, 1.2, 10),
        jpi_p=torch.full((10,), .2), rpp=torch.full((10,), .15),
        n_q=0, storage=(),
    ).double()
    source = LocalMemoryROM(
        base, input_scale=torch.linspace(.7, 1.4, 20),
        p_scale=torch.linspace(.8, 1.2, 10), arm="L10-S", seed=98,
    )
    with torch.no_grad():
        source.template_mask.copy_(torch.tensor([1., 1., 1., 1., 1., 1., 0., 1.]))
        source._template_scalars[0].copy_(torch.tensor(.12))
        source._template_scalars[1].copy_(torch.tensor(.08))
        source._template_scalars[2].copy_(torch.tensor(.05))
        source._template_scalars[3].copy_(torch.tensor(.04))
        source._template_scalars[4].copy_(torch.tensor(.03))
        source._template_scalars[5].copy_(torch.tensor(-1.5))
        source._q_quadratic_raw.copy_(torch.tensor(1.1))
        source.port_raw.copy_(torch.linspace(.01, .06, 6))
    return source


NODE_P2Q2 = {"channel": "node", "expression": {"op": "mul", "args": [
    {"op": "pow2", "arg": {"op": "var", "name": "p"}},
    {"op": "pow2", "arg": {"op": "var", "name": "q"}},
]}}
EDGE_GENERIC = {"channel": "edge", "expression": {"op": "mul", "args": [
    {"op": "pow2", "arg": {"op": "add", "args": [
        {"op": "var", "name": "qi"}, {"op": "var", "name": "qj"},
    ]}},
    {"op": "add", "args": [{"op": "const", "value": 1},
                               {"op": "var", "name": "c"}]},
]}}


def test_generic_ast_field_gradient_and_autograd_match_independent_numpy():
    source_model = SymbolicClosureROM.from_source(_source())
    model = source_model.with_terms(source_model.terms + [NODE_P2Q2, EDGE_GENERIC])
    with torch.no_grad():
        model.coefficients[-2:].copy_(torch.tensor([.17, -.12], dtype=torch.float64))

    for term in (NODE_P2Q2, EDGE_GENERIC):
        channel = term["channel"]
        width = 2 if channel == "node" else 6
        features = torch.randn(2, 7, width, dtype=torch.float64, requires_grad=True)
        value, gradient = evaluate_expression_torch(
            term["expression"], features, channel)
        np_value, np_gradient = evaluate_expression_numpy(term["expression"], features.detach().numpy(), channel)
        auto = torch.autograd.grad(value.sum(), features)[0]
        assert torch.allclose(gradient, auto, atol=2e-12, rtol=0)
        np.testing.assert_allclose(value.detach().numpy(), np_value, atol=2e-12, rtol=0)
        np.testing.assert_allclose(gradient.detach().numpy(), np_gradient, atol=2e-12, rtol=0)

    x = (torch.randn(3, 40, dtype=torch.float64) * .05).requires_grad_(True)
    H, analytic = model.energy_and_gradient(x)
    autodiff = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
    assert torch.allclose(analytic, autodiff, atol=3e-11, rtol=0)

    z = torch.randn(3, 40, dtype=torch.float64) * .04
    field = model.field(z)
    np.testing.assert_allclose(numpy_field(model, z.numpy()), field.detach().numpy(), atol=3e-10, rtol=0)

    loss = H.sum() + .2 * field.square().sum()
    grads = torch.autograd.grad(loss, [model.coefficients, x], create_graph=True)
    for actual in grads:
        assert torch.isfinite(actual).all()
    second = torch.autograd.grad(sum(g.square().sum() for g in grads),
                                 [model.coefficients, x])
    for actual in second:
        assert torch.isfinite(actual).all()


def test_each_port_expression_may_have_one_tanh_but_not_two():
    source_model = SymbolicClosureROM.from_source(_source())
    port_a = {"channel": "port", "expression": {"op": "tanh", "scale": .5,
        "arg": {"op": "add", "args": [{"op": "var", "name": "w"},
                                         {"op": "var", "name": "p"}]}}}
    port_b = {"channel": "port", "expression": {"op": "tanh", "scale": 2,
        "arg": {"op": "add", "args": [{"op": "var", "name": "ec"},
                                         {"op": "var", "name": "nq"}]}}}
    model = source_model.with_terms(source_model.terms + [port_a, port_b])
    assert len(model.terms) == len(source_model.terms) + 2
    assert validate_expression(port_a["expression"], "port")["op"] == "tanh"
    invalid = {"op": "tanh", "scale": 1,
        "arg": {"op": "tanh", "scale": .5, "arg": {"op": "var", "name": "w"}}}
    with pytest.raises(ValueError, match="one tanh"):
        validate_expression(invalid, "port")
