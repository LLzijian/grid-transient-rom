import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))

from local_memory_rom_0098 import LocalMemoryROM
from shared_closure_rom_0101 import S23
from structured_rom import StructuredROM
from symbolic_closure_rom import (
    SymbolicClosureROM,
    evaluate_expression_numpy,
    evaluate_expression_torch,
    expression_key,
    numpy_field,
    term_key,
    validate_expression,
)


def _base(flip_edge=None):
    eye = torch.eye(10, dtype=torch.float64)
    D = torch.zeros((7, 10), dtype=torch.float64)
    D[:6] = eye[:6] - eye[1:7]
    D[6] = eye[0] - eye[9]
    if flip_edge is not None:
        D[flip_edge] *= -1
    return StructuredROM(
        mass=torch.linspace(1.0, 2.0, 10), D=D,
        coupling=torch.linspace(.4, 1.2, 7),
        delta_star=torch.linspace(-.17, .23, 10),
        damping=torch.full((10,), .08), gain=torch.linspace(.8, 1.2, 10),
        jpi_p=torch.full((10,), .2), rpp=torch.full((10,), .15),
        n_q=0, storage=(),
    ).double()


def _source(as_s23=False):
    source_type = S23 if as_s23 else LocalMemoryROM
    source = source_type(
        _base(), input_scale=torch.linspace(.7, 1.4, 20),
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


def test_from_source_replaces_h_and_port_and_preserves_the_full_field():
    source = _source(as_s23=True)
    model = SymbolicClosureROM.from_source(source)
    z = torch.randn(5, 40, dtype=torch.float64) * .06
    x = source.interface_to_state(z)
    h0, e0 = source.energy_and_gradient(x)
    h1, e1 = model.energy_and_gradient(x)
    assert torch.allclose(h1, h0, atol=2e-12, rtol=0)
    assert torch.allclose(e1, e0, atol=2e-12, rtol=0)
    assert torch.allclose(model.port_values(x), source.port_values(x), atol=2e-12, rtol=0)
    history = torch.randn(2, 11, 30, dtype=torch.float64)
    assert torch.equal(model.encode(history), source.encode(history))
    assert torch.allclose(model.field(z), source.field(z), atol=3e-12, rtol=0)
    assert len(model.terms) == 9
    assert torch.allclose(model.coefficients[:3], source.template_coefficients[:3])
    assert torch.allclose(model.coefficients[3:], source.port_raw * source.template_mask[7])
    assert not model.template_mask[:3].any() and model.template_mask[7] == 0
    assert all(not p.requires_grad for p in list(model._template_scalars[:3]) + [model.port_raw])


def test_analytic_gradient_ad_and_numpy_field_include_nonzero_phase_edges():
    source = _source()
    model = SymbolicClosureROM.from_source(source)
    x = torch.randn(4, 40, dtype=torch.float64, requires_grad=True) * .05
    H, gradient = model.energy_and_gradient(x)
    autodiff = torch.autograd.grad(H.sum(), x)[0]
    assert torch.allclose(gradient, autodiff, atol=2e-11, rtol=0)
    assert torch.max(torch.abs(model.phi)) > 0
    z = torch.randn(4, 40, dtype=torch.float64) * .05
    np.testing.assert_allclose(numpy_field(model, z), model.field(z).detach().numpy(), atol=3e-10, rtol=0)
    external = torch.randn(4, 10, dtype=torch.float64) * .02
    np.testing.assert_allclose(numpy_field(model, z, external.numpy()),
                               model.field(z, external).detach().numpy(), atol=3e-10, rtol=0)


def test_edge_reversal_and_power_identity():
    a = SymbolicClosureROM.from_source(_source())
    flipped_source = _source()
    flipped_source.base_D = flipped_source.base_D.clone()
    flipped_source.base_D[0] *= -1
    flipped_source.phi = flipped_source.base_D @ flipped_source.delta_star
    flipped_source.structured.D = flipped_source.structured.D.clone()
    flipped_source.structured.D[0] *= -1
    flipped_source.structured.phi = flipped_source.structured.D @ flipped_source.structured.delta_star
    flipped_source.port_s_weight = flipped_source.port_s_weight.clone()
    flipped_source.port_s_weight[0] *= -1
    b = SymbolicClosureROM.from_source(flipped_source)
    z = torch.randn(3, 40, dtype=torch.float64) * .04
    assert torch.allclose(a.field(z), b.field(z), atol=2e-10, rtol=0)

    # A non-baseline AST uses endpoint features nonlinearly; the evaluator's
    # reverse-edge average must preserve the same physical field.
    custom_edge = {"channel": "edge", "expression": {"op": "mul", "args": [
        {"op": "pow2", "arg": {"op": "add", "args": [
            {"op": "var", "name": "qi"}, {"op": "var", "name": "qj"}]}},
        {"op": "var", "name": "c"}]}}
    custom_a = SymbolicClosureROM.from_source(_source(), terms=[custom_edge])
    custom_b = SymbolicClosureROM.from_source(flipped_source, terms=[custom_edge])
    with torch.no_grad():
        custom_a.coefficients.fill_(.6)
        custom_b.coefficients.fill_(.6)
    assert torch.allclose(custom_a.field(z), custom_b.field(z), atol=2e-10, rtol=0)
    xx = custom_a.interface_to_state(z).requires_grad_(True)
    hh, gg = custom_a.energy_and_gradient(xx)
    assert torch.allclose(gg, torch.autograd.grad(hh.sum(), xx)[0], atol=2e-11, rtol=0)

    x = a.interface_to_state(z)
    _, e = a.energy_and_gradient(x)
    f = a.internal_field(x)
    u = a.port_values(x)
    B = a.port_matrix()
    _, R = a.matrices()
    balance = (e * f).sum(-1) + (e @ R * e).sum(-1) - (u * (e @ B)).sum(-1)
    assert balance.abs().max() < 2e-10


def test_general_energy_and_port_terms_are_centered_at_the_origin():
    node_term = {"channel": "node", "expression": {"op": "add", "args": [
        {"op": "pow2", "arg": {"op": "var", "name": "q"}},
        {"op": "var", "name": "q"}]}}
    port_term = {"channel": "port", "expression": {"op": "add", "args": [
        {"op": "var", "name": "w"}, {"op": "const", "value": 1}]}}
    model = SymbolicClosureROM.from_source(_source(), terms=[node_term, port_term])
    with torch.no_grad():
        model.coefficients.copy_(torch.tensor([.7, .4], dtype=torch.float64))
    origin = torch.zeros(40, dtype=torch.float64)
    H, gradient = model.energy_and_gradient(origin)
    assert H.abs() < 1e-14
    assert gradient.abs().max() < 1e-13
    assert model.port_values(origin).abs().max() < 1e-14


def test_edge_symmetrization_precedes_origin_tangent_subtraction():
    features = torch.randn(3, 5, 6, dtype=torch.float64)
    reversed_features = torch.stack((features[..., 1], features[..., 0],
                                     features[..., 3], features[..., 2],
                                     -features[..., 4], features[..., 5]), -1)
    linear_expressions = [
        {"op": "var", "name": "qi"},
        {"op": "var", "name": "s"},
        {"op": "add", "args": [
            {"op": "var", "name": "qi"}, {"op": "var", "name": "qj"}]},
    ]
    for expression in linear_expressions:
        value, gradient = evaluate_expression_torch(expression, features, "edge")
        np_value, np_gradient = evaluate_expression_numpy(expression, features.numpy(), "edge")
        assert value.abs().max() < 1e-14
        assert gradient.abs().max() < 1e-14
        np.testing.assert_allclose(np_value, 0., atol=1e-14, rtol=0)
        np.testing.assert_allclose(np_gradient, 0., atol=1e-14, rtol=0)

    nonlinear = {"op": "pow2", "arg": {"op": "var", "name": "qi"}}
    value, gradient = evaluate_expression_torch(nonlinear, features, "edge")
    reverse_value, reverse_gradient = evaluate_expression_torch(nonlinear, reversed_features, "edge")
    expected_reverse_gradient = torch.stack((gradient[..., 1], gradient[..., 0],
                                             gradient[..., 3], gradient[..., 2],
                                             -gradient[..., 4], gradient[..., 5]), -1)
    assert torch.allclose(value, reverse_value, atol=1e-14, rtol=0)
    assert torch.allclose(reverse_gradient, expected_reverse_gradient, atol=1e-14, rtol=0)


def test_torch_and_numpy_ast_value_gradient_match_for_port_tanh():
    expr = {"op": "tanh", "scale": .5, "arg": {"op": "add", "args": [
        {"op": "var", "name": "w"},
        {"op": "mul", "args": [{"op": "const", "value": -1}, {"op": "var", "name": "p"}]}]}}
    values = torch.randn(2, 10, 6, dtype=torch.float64, requires_grad=True)
    torch_value, torch_gradient = evaluate_expression_torch(expr, values, "port")
    auto = torch.autograd.grad(torch_value.sum(), values)[0]
    numpy_value, numpy_gradient = evaluate_expression_numpy(expr, values.detach().numpy(), "port")
    assert torch.allclose(torch_gradient, auto, atol=2e-12, rtol=0)
    np.testing.assert_allclose(torch_value.detach().numpy(), numpy_value, atol=2e-12, rtol=0)
    np.testing.assert_allclose(torch_gradient.detach().numpy(), numpy_gradient, atol=2e-12, rtol=0)


def test_terms_mutation_validation_gradient_flow_and_self_contained_roundtrip(tmp_path):
    model = SymbolicClosureROM.from_source(_source())
    original = copy.deepcopy(model.coefficients.detach())
    extra = {"channel": "node", "expression": {
        "op": "pow2", "arg": {"op": "var", "name": "q"}}}
    changed = model.with_terms(model.terms + [extra])
    assert len(changed.terms) == 10
    assert torch.equal(changed.coefficients[:-1], original)
    assert changed.coefficients[-1] == 0
    assert torch.equal(model.coefficients, original)

    loss_state = torch.randn(3, 40, dtype=torch.float64) * .04
    history = torch.randn(2, 11, 30, dtype=torch.float64) * .03
    loss = changed.field(loss_state).square().mean() + changed.encode(history).square().mean() + changed.complexity()
    trainable = [(name, p) for name, p in changed.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, [p for _, p in trainable], allow_unused=True)
    grad_map = dict(zip((name for name, _ in trainable), grads))
    assert grad_map["coefficients"] is not None
    assert torch.isfinite(grad_map["coefficients"]).all()
    assert torch.any(grad_map["coefficients"].abs() > 0)
    assert any("local_gru" in name and grad is not None and torch.any(grad.abs() > 0)
               for name, grad in grad_map.items())
    assert all(name not in grad_map for name in ("port_raw", "_template_scalars.0",
                                                  "_template_scalars.1", "_template_scalars.2"))

    path = tmp_path / "symbolic.json"
    payload = changed.export(path)
    restored = SymbolicClosureROM.from_export(path)
    np.testing.assert_allclose(numpy_field(payload, loss_state), changed.field(loss_state).detach().numpy(),
                               atol=3e-10, rtol=0)
    assert restored.terms == changed.terms
    assert torch.equal(restored.coefficients, changed.coefficients)
    assert torch.allclose(restored.field(loss_state), changed.field(loss_state), atol=1e-12, rtol=0)
    assert term_key(changed.terms[1]) == term_key({
        "channel": "edge", "expression": {"op": "mul", "args": [
            {"op": "var", "name": "s"}, {"op": "add", "args": [
                {"op": "mul", "args": [{"op": "var", "name": "qj"}, {"op": "const", "value": -1}]},
                {"op": "var", "name": "qi"}] } ]}})
    assert expression_key({"op": "mul", "args": [
        {"op": "var", "name": "q"}, {"op": "var", "name": "p"}]}) == expression_key({
            "op": "mul", "args": [{"op": "var", "name": "p"}, {"op": "var", "name": "q"}]})


def test_grammar_enforces_channel_depth_operator_degree_and_tanh_limits():
    assert validate_expression({"op": "tanh", "scale": .5,
                                "arg": {"op": "var", "name": "q"}}, "port")["op"] == "tanh"
    with pytest.raises(ValueError, match="not allowed"):
        validate_expression({"op": "var", "name": "w"}, "node")
    deep = {"op": "pow2", "arg": {"op": "pow2", "arg": {"op": "pow2", "arg": {
        "op": "pow2", "arg": {"op": "var", "name": "q"}}}}}
    with pytest.raises(ValueError, match="depth"):
        validate_expression(deep, "node")
    high_degree_port = {"op": "pow4", "arg": {"op": "var", "name": "q"}}
    with pytest.raises(ValueError, match="degree"):
        validate_expression(high_degree_port, "port")
    many_ops = {"op": "add", "args": [
        {"op": "mul", "args": [{"op": "var", "name": "p"}, {"op": "var", "name": "q"}]}
        for _ in range(13)]}
    with pytest.raises(ValueError, match="operator nodes"):
        validate_expression(many_ops, "node")
    with pytest.raises(ValueError, match="only in port"):
        validate_expression({"op": "tanh", "scale": 1,
                             "arg": {"op": "var", "name": "p"}}, "node")
    two_tanh_terms = [
        {"channel": "port", "expression": {"op": "tanh", "scale": .5,
         "arg": {"op": "var", "name": "w"}}},
        {"channel": "port", "expression": {"op": "tanh", "scale": 2,
         "arg": {"op": "var", "name": "q"}}},
    ]
    saturated=SymbolicClosureROM.from_source(_source(), terms=two_tanh_terms)
    assert len(saturated.terms)==2
    with pytest.raises(ValueError, match="at most one tanh"):
        validate_expression({'op':'tanh','scale':1.,'arg':two_tanh_terms[0]['expression']},'port')
    with pytest.raises(ValueError, match="at most 12"):
        SymbolicClosureROM.from_source(_source(), terms=SymbolicClosureROM.from_source(_source()).terms +
                                       [{"channel": "node", "expression": {"op": "var", "name": "p"}},
                                        {"channel": "node", "expression": {"op": "var", "name": "q"}},
                                        {"channel": "node", "expression": {"op": "add", "args": [
                                            {"op": "var", "name": "p"}, {"op": "var", "name": "q"}]}},
                                        {"channel": "node", "expression": {"op": "pow2", "arg": {
                                            "op": "var", "name": "p"}}}])
