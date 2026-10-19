"""Synthetic-only P3 arithmetic and boundary QA; no real data/model import."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import numpy as np
from closure_p3_disagreement_20261020 import (KEYS, EPSILON, S_DELTA, S_OMEGA,
                                              decide, disagreement, numerical, observable,
                                              summary, verify_manifest)


def states(angle=0., frequency=0., length=2):
    a = np.zeros((length, 40), dtype=np.float64)
    a[:, :10] = angle
    a[:, 10:20] = frequency
    return a


class SyntheticQA(unittest.TestCase):
    def setUp(self):
        self.mass = np.arange(1., 11.)

    def test_observables(self):
        zero = observable(states(), self.mass)
        np.testing.assert_allclose(observable(states(angle=4.), self.mass), zero, atol=1e-14)
        z = observable(states(frequency=S_OMEGA), self.mass)
        self.assertAlmostEqual(float(np.sum(z[0]**2)), .5)
        a = states()
        a[:, 0] = S_DELTA
        self.assertGreater(float(np.sum(observable(a, self.mass)[0]**2)), 0.)
        a[:, 30] = np.nan
        with self.assertRaises(ValueError):
            observable(a, self.mass)

    def test_b_over_four_and_seed_sample_variance(self):
        paths = {key: states() for key in KEYS}
        for key in KEYS:
            if key.endswith("delete7"):
                paths[key] = states(frequency=S_OMEGA)
        result = disagreement(paths, self.mass)["total"]
        self.assertAlmostEqual(result["B"], .5 / 4)
        self.assertAlmostEqual(result["W"], 0.)
        self.assertAlmostEqual(result["D"], result["B"] / EPSILON)
        # Three seed values -1,0,+1 on a unit-norm z coordinate: sample variance is 1.
        for seed, value in zip((6101, 6102, 6103), (-1., 0., 1.)):
            paths[f"{seed}_full8"] = states(frequency=value * S_OMEGA * math.sqrt(2))
        result = disagreement(paths, self.mass)["total"]
        self.assertAlmostEqual(result["W"], .5)
        permuted = {key: paths[key] for key in KEYS}
        permuted["6101_full8"], permuted["6103_full8"] = permuted["6103_full8"], permuted["6101_full8"]
        self.assertEqual(result, disagreement(permuted, self.mass)["total"])

    def test_time_equal_weight_and_rejection(self):
        paths = {key: states(length=1) for key in KEYS}
        self.assertEqual(disagreement(paths, self.mass)["total"]["B"], 0.)
        paths["6101_full8"] = states(length=0)
        with self.assertRaises(ValueError):
            disagreement(paths, self.mass)
        short = {"label": "stable", "fault_bus": "1", "fine": {"total": {"B": 1., "W": 0., "D": 1.}},
                 "coarse": {"total": {"B": 1., "W": 0., "D": 1.}}}
        long = {"label": "stable", "fault_bus": "2", "fine": {"total": {"B": 3., "W": 0., "D": 3.}},
                "coarse": {"total": {"B": 3., "W": 0., "D": 3.}}}
        self.assertEqual(summary([short, long], [])["all"]["fine_B"]["event_mean"], 2.)

    def test_short_toy_rollout_common_window(self):
        def toy(steps, dt):
            a = states(length=steps+1)
            a[:, 10:20] = np.arange(steps+1)[:, None] * dt
            return a
        fine, coarse = toy(4, .005), toy(2, .01)
        np.testing.assert_array_equal(fine[::2], coarse)
        self.assertEqual(len(fine[2:2*2+1:2]), len(coarse[1:2+1]))
        self.assertEqual(len(fine[2:2*2+1:2]), 2)
        paths = {key: states(length=1) for key in KEYS}
        paths["6101_full8"] = states(length=1)
        paths["6101_full8"][0, 39] = np.inf
        with self.assertRaises(ValueError):
            disagreement(paths, self.mass)

    def test_actual_window_numerical_and_classification_origin(self):
        from evaluate_structured_rom import _pair_row
        fine, coarse = states(length=1001), states(length=501)
        history = np.zeros((11, 30))
        history[:10, 0] = .02
        event = {"id": "toy", "group": "1", "label": "stable"}
        meta = {"equilibrium_delta": np.zeros(10), "omega_s": 2*np.pi*60,
                "inertia": self.mass, "event_time_s": None}
        seen = []
        def classify(raw, *_args, **_kwargs):
            seen.append(raw.copy())
            return {"category": "stable", "predicted_los": False, "trigger_index": None}
        with patch("evaluate_structured_rom.classify_prediction", side_effect=classify):
            pair = _pair_row(fine[:5], coarse[:3], history, event, meta, self.mass)
        self.assertTrue(pair["los_pair_pass"])
        self.assertEqual([len(raw) for raw in seen], [13, 13])
        self.assertNotEqual(float(seen[0][0, 0]), float(seen[0][10, 0]))
        pair_ok = {"finite_to_endpoint": True, "full_fine_finite": True,
                   "full_coarse_finite": True, "los_pair_pass": True}
        with patch("evaluate_structured_rom._pair_row", return_value=pair_ok):
            row = numerical(fine, coarse, history, event, meta, self.mass, 2)
            self.assertEqual(row["actual_window_N"], 2)
            fine[5, 39] = np.nan  # outside the actual N=2 window
            numerical(fine, coarse, history, event, meta, self.mass, 2)
            fine[3, 39] = np.nan  # hidden fine substep inside that window
            with self.assertRaises(ValueError):
                numerical(fine, coarse, history, event, meta, self.mass, 2)
            fine[3, 39] = 0.
            coarse[2, 10:20] = .01  # late discrepancy after any earlier LOS comparison cutoff
            with self.assertRaises(ValueError):
                numerical(fine, coarse, history, event, meta, self.mass, 2)

    def test_manifest_identity_rejection(self):
        with patch("closure_p3_disagreement_20261020.frozen_inputs", return_value={"sha": "expected"}):
            with self.assertRaises(ValueError):
                verify_manifest({"sha": "damaged"})

    def test_decision_full_denominator_and_inclusive_boundaries(self):
        def row(i, bus):
            return {"event_id": str(i), "N": 1, "label": "stable", "fault_bus": str(bus),
                    "numerical_pass": True,
                    "fine": {"total": {"B": 25*EPSILON, "D": 1.}},
                    "coarse": {"total": {"B": 25*EPSILON, "D": 1.}}}
        rows = [row(i, i % 3) for i in range(366)]
        self.assertEqual(decide(rows)[0], "ACTIVE_DESIGN_ADMISSIBLE")
        rows[0]["coarse"]["total"]["D"] = 1. - 1e-12
        self.assertEqual(len(decide(rows)[1]), 365)
        for r in rows[12:]:
            r["fine"]["total"]["B"] = 0.
        self.assertEqual(decide(rows)[0], "NO_ACTIVE_DESIGN_ADMISSION")
        rows[0]["coarse"]["total"]["D"] = 1.
        self.assertEqual(decide(rows)[0], "ACTIVE_DESIGN_ADMISSIBLE")
        rows[0]["numerical_pass"] = False
        self.assertEqual(decide(rows)[0], "INCONCLUSIVE_NUMERICAL_OR_IMPLEMENTATION")
        self.assertEqual(decide(rows[:-1])[0], "INCONCLUSIVE_NUMERICAL_OR_IMPLEMENTATION")
        rows[0]["numerical_pass"] = True
        rows[0]["N"] = 0
        self.assertEqual(decide(rows)[0], "NO_ACTIVE_DESIGN_ADMISSION")


def run_qa():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticQA)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests": result.testsRun,
            "failures": [str(err) for _, err in result.failures],
            "errors": [str(err) for _, err in result.errors], "synthetic_only": True}


if __name__ == "__main__":
    outcome = run_qa()
    if outcome["status"] != "PASS":
        raise SystemExit(1)
