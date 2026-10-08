import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from compare_physical_modes import (OMEGA_S, S_DELTA, S_OMEGA, assignment, mac, physical_output)


def test_mac_is_phase_and_scale_invariant():
    rng = np.random.default_rng(71)
    x = rng.normal(size=19) + 1j * rng.normal(size=19)
    assert abs(mac(x, x * (3.2 * np.exp(1j * .73))) - 1) < 1e-12


def test_assignment_is_one_to_one_and_prefers_eligible_edges():
    values = np.array([-0.1 + 2j, -0.2 + 2.2j, -0.3 + 5j])
    vectors = np.eye(3, dtype=complex)
    mass = np.ones(10)
    embed = np.zeros((20, 3), float); embed[0, 0] = 1; embed[1, 1] = 1; embed[10, 2] = 1
    targets = [{"index": 7, "frequency_hz": 2 / (2*np.pi), "damping_ratio": .05},
               {"index": 8, "frequency_hz": 2.2 / (2*np.pi), "damping_ratio": .1}]
    fv = [physical_output(vectors[:, 0], mass, embed), physical_output(vectors[:, 1], mass, embed)]
    out = assignment(targets, fv, values, vectors, mass, embed)
    assert len(out["matched"]) == 2
    assert len({x["rom_index"] for x in out["matched"]}) == 2


def test_assignment_max_cardinality_beats_greedy_best_edge():
    mass = np.ones(10)
    a = np.zeros(20, complex); a[0], a[1] = 1, -1
    u = np.zeros(20, complex); u[2], u[3] = 1, -1
    target0 = a
    target1 = np.cos(np.deg2rad(20)) * a + np.sin(np.deg2rad(20)) * u
    candidate0 = a
    candidate1 = np.cos(np.deg2rad(-10)) * a + np.sin(np.deg2rad(-10)) * u
    values = np.array([-.2 + 2j, -.2 + 2.1j])
    vectors = np.column_stack((candidate0, candidate1))
    targets = [{"index": 10, "frequency_hz": 2 / (2*np.pi), "damping_ratio": .1},
               {"index": 11, "frequency_hz": 1.8 / (2*np.pi), "damping_ratio": .1}]
    out = assignment(targets, [physical_output(target0, mass), physical_output(target1, mass)],
                     values, vectors, mass, None)
    eligible = [r for r in out["all_pair_metrics"] if r["eligible"]]
    assert {(r["fom_index"], r["rom_index"]) for r in eligible} == {(10, 0), (10, 1), (11, 0)}
    greedy = sorted(eligible, key=lambda r: r["cost"])
    greedy_pairs, used_fom, used_rom = [], set(), set()
    for row in greedy:
        if row["fom_index"] not in used_fom and row["rom_index"] not in used_rom:
            greedy_pairs.append(row); used_fom.add(row["fom_index"]); used_rom.add(row["rom_index"])
    assert len(greedy_pairs) == 1
    assert len(out["matched"]) == 2
    assert {(r["fom_index"], r["rom_index"]) for r in out["matched"]} == {(10, 1), (11, 0)}


def test_physical_output_applies_omega_units_and_coi_gauge():
    mass = np.array([1., 2.] + [1.] * 8)
    v = np.zeros(20, complex); v[0] = 2; v[1] = 5; v[10] = 3
    y = physical_output(v, mass, omega_scale=OMEGA_S)
    expected_coi = np.array([2 - (2+10)/11, 5 - (2+10)/11]) / S_DELTA
    assert np.allclose(y[:2], expected_coi)
    assert y[10] == 3 * OMEGA_S / S_OMEGA
    assert S_DELTA > 0 and S_OMEGA > 0


def test_fom_pu_and_rom_rad_per_second_outputs_are_identical():
    mass = np.arange(1., 11.)
    fom = np.zeros(20, complex); fom[:10] = np.linspace(1, 2, 10); fom[10:20] = .01j
    rom = fom.copy(); rom[10:20] *= OMEGA_S
    assert np.allclose(physical_output(fom, mass, omega_scale=OMEGA_S),
                       physical_output(rom, mass, omega_scale=1.))
