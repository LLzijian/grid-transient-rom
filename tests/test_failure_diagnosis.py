import numpy as np

from experiments.diagnose_refit_failures import sine_cut_audit


def test_two_node_capacity_detects_impossible_load_both_directions():
    incidence = np.array([[1.0], [-1.0]])
    for sign in (1.0, -1.0):
        audit = sine_cut_audit(incidence, [1.0], sign * np.array([2.0, -2.0]))
        assert audit["violated"]
        assert audit["maximum_excess"] == 1.0


def test_feasible_two_node_phase_passes_necessary_test():
    incidence = np.array([[1.0], [-1.0]])
    demand = incidence[:, 0] * np.sin(0.7)
    audit = sine_cut_audit(incidence, [1.0], demand)
    assert not audit["violated"]
    assert audit["passing_does_not_prove_equilibrium_exists"]


def test_multi_node_cut_detects_bottleneck_when_singletons_pass():
    incidence = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 1], [0, 0, -1]])
    coupling = np.array([10.0, 1.0, 10.0])
    demand = np.array([0.75, 0.75, -0.75, -0.75])
    assert np.all(np.abs(demand) <= np.abs(incidence) @ coupling)
    audit = sine_cut_audit(incidence, coupling, demand)
    assert audit["violated"]
    assert audit["worst_cut_nodes_zero_based"] == [0, 1]
    assert audit["maximum_excess"] == 0.5
