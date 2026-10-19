"""Changed-entry QA only: no real fixture, optimizer update, rollout or dataset."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments')]
import numpy as np
import torch
import closure_saturation_pair_20261019 as entry


class ChangedEntry(unittest.TestCase):
    def setUp(self):
        self.source = entry.toy_source()
        self.reference = entry.copy_runtime(self.source, [entry.TERM], 'cpu')

    def test_registered_order_caps_and_binding(self):
        self.assertEqual(entry.ORDER, [(s, 'saturation', g) for s in (5101, 5102, 5103) for g in ('control', 'residual')])
        self.assertEqual(entry.CAPS, dict(startup=50., fit=535., evaluation=120., item=655., items=3930., total=4000.))
        identity = entry.source_identity()
        for name, sha in entry.PINNED.items():
            self.assertEqual(identity['files']['experiments/' + name], sha)
        changed = copy.deepcopy(identity)
        changed['files']['experiments/closure_training_runtime.py'] = 'changed'
        state = dict(status='INTERRUPTED_RUNTIME', resumes=0, identity=identity,
                     device='cpu', phase='fit', elapsed_seconds=10.)
        entry.resume_allowed(state, identity, 'cpu')
        for key, value in [('status', 'RUNNING'), ('resumes', 1), ('phase', 'startup'), ('elapsed_seconds', 4000.)]:
            bad = dict(state, **{key: value})
            with self.assertRaises(ValueError):
                entry.resume_allowed(bad, identity, 'cpu')
        with self.assertRaises(ValueError):
            entry.resume_allowed(state, changed, 'cpu')

    def test_admission_rejects_unregistered_structures(self):
        entry.fast_model(self.reference, self.reference)
        mutations = [lambda m: setattr(m.structured, 'D', torch.empty((0, 10), dtype=torch.float64)),
                     lambda m: setattr(m, 'terms', []),
                     lambda m: setattr(m, 'terms', [dict(channel='port', expression=dict(op='var', name='q'))]),
                     lambda m: setattr(m.structured, 'storage_specs', ('extra',)),
                     lambda m: m.mass.add_(.1),
                     lambda m: next(m.parameters()).requires_grad_(False),
                     lambda m: setattr(m, 'arm', 'G3-S'),
                     lambda m: setattr(m, 'extra', torch.nn.Linear(1, 1).double())]
        for mutation in mutations:
            m = copy.deepcopy(self.reference)
            with torch.no_grad():
                mutation(m)
            with self.assertRaises((ValueError, TypeError)):
                entry.fast_model(m, self.reference)
        wrapper = entry.core.CausalHistoryResidual(self.reference, np.ones(6))
        with self.assertRaises(TypeError):
            entry.fast_model(wrapper, self.reference)

    def test_learned_export_and_wrapper_preservation(self):
        # Perturb every trainable parameter without taking an optimizer step.
        base = copy.deepcopy(self.reference)
        with torch.no_grad():
            for p in base.parameters():
                if p.requires_grad:
                    p.add_(.007)
        entry.strict_admit(base, self.reference)
        wrapper = entry.core.CausalHistoryResidual(base, np.arange(1, 7.), np.ones(10))
        with torch.no_grad():
            wrapper.history_coefficients.copy_(torch.arange(6, dtype=torch.float64) * .003)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'wrapper.json'
            wrapper.export(path)
            loaded = entry.evaluation_model(path, 'residual', 'cpu', self.reference)
            self.assertIs(type(loaded.base), entry.TanhPortTrainingROM)
            for name, tensor in wrapper.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]), name)
            path = Path(tmp) / 'control.json'
            base.export(path)
            loaded = entry.evaluation_model(path, 'control', 'cpu', self.reference)
            self.assertIs(type(loaded), entry.TanhPortTrainingROM)
            for name, tensor in base.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]), name)

    @staticmethod
    def fake_prepare(output, device, deadline):
        fixtures = output / 'fixtures'
        fixtures.mkdir()
        entry.toy_source().export(fixtures / 'source.json')
        rows = {}
        for seed in (5101, 5102, 5103):
            folder = fixtures / str(seed)
            folder.mkdir()
            obs = [np.zeros((16, 61, 30)), np.zeros((16, 111, 30))]
            scale = np.ones(30)
            meta = dict(clean_sha256=entry.cache_digest(obs, scale)[:2], scales=scale.tolist(), dual_step_normalized_max_abs=[0., 0.])
            entry.core.atomic_json(folder / 'fixture.json', meta)
            np.savez(folder / 'observations.npz', train=obs[0], val=obs[1], scales=scale)
            rows[str(seed)] = dict(meta=meta, arrays=entry.cache_digest(obs, scale), files={n: entry.core.sha(folder / n) for n in ('fixture.json', 'observations.npz')})
        manifest = dict(source_sha256=entry.core.sha(fixtures / 'source.json'), seeds=rows)
        entry.core.atomic_json(fixtures / 'manifest.json', manifest)
        return manifest

    @staticmethod
    def fake_checkpoint(folder, model, opt, rng, step, trace, identity):
        endpoint = folder / f'checkpoint_{step:05d}'
        endpoint.mkdir()
        if step == 512:
            model.export(endpoint / 'model.json')
        (endpoint / 'training.pt').write_text('MOCK, no optimizer updates')
        entry.core.atomic_json(folder / 'latest.json', dict(updates=step, checkpoint=endpoint.name,
                               sha256={p.name: entry.core.sha(p) for p in endpoint.iterdir()}))

    def test_fakecache_binding_tamper_and_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            manifest = self.fake_prepare(output, 'cpu', 1.)
            for seed in (5101, 5102, 5103):
                entry.load_cache(output, seed, manifest['seeds'])
            path = output / 'fixtures/5101/observations.npz'
            path.write_bytes(path.read_bytes() + b'tamper')
            with self.assertRaises(ValueError):
                entry.load_cache(output, 5101, manifest['seeds'])

    def test_phase_caps_and_failure_accounting(self):
        item = dict(fit_seconds=534., eval_seconds=0.)
        state = dict(items=[item])
        self.assertEqual(entry.phase_budget(state, item, 'fit', 1.), 1.)
        self.assertEqual(entry.phase_budget(state, item, 'evaluation', 1.), 120.)
        self.assertEqual(entry.phase_budget(state, item, 'evaluation', 3999.5), .5)
        self.assertEqual(entry.phase_budget(state, item, 'fit', 4001.), 0.)
        state['items'].append(dict(fit_seconds=3396., eval_seconds=0.))
        self.assertEqual(entry.phase_budget(state, item, 'evaluation', 1.), 0.)

    def test_mockrun_freeze_six_items_and_startup_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            evaluated = []
            def fake_evaluate(model, obs, scales, deadline):
                frozen = json.loads((output / 'fits_frozen.json').read_text())
                self.assertEqual([i['fit_status'] for i in frozen], ['complete'] * 6)
                self.assertIs(type(model.base if isinstance(model, entry.core.CausalHistoryResidual) else model), entry.TanhPortTrainingROM)
                evaluated.append(True)
                row = dict(mse=.001, weak_mse=.001, strong_mse=.001,
                           event_mse=[.001]*16, dual_step_max_normalized=0.)
                return dict(train=row.copy(), heldout=row.copy(), numerical_qualified=True, prediction_gate=True)
            with patch.object(entry, 'PROCESS_START', time.perf_counter()), patch.object(entry, 'prepare', side_effect=self.fake_prepare), patch.object(entry.core, 'checked_fixture', side_effect=AssertionError('forbidden real fixture')), patch.object(entry.core, 'train_update', return_value=(0., {})) as updates, patch.object(entry.core, 'loss', return_value=0.), patch.object(entry.core, 'checkpoint', side_effect=self.fake_checkpoint), patch.object(entry.core, 'evaluate', side_effect=fake_evaluate):
                result = entry.run(output, 'cpu')
            self.assertEqual(updates.call_count, 6 * 512)
            self.assertEqual(len(evaluated), 6)
            self.assertEqual(result['complete_pairs'], 3)
            self.assertLessEqual(sum(i['fit_seconds'] + i['eval_seconds'] for i in result['items']), result['elapsed_seconds'])
            output2 = Path(tmp) / 'startup_fail'
            with patch.object(entry, 'PROCESS_START', time.perf_counter()), patch.object(entry, 'prepare', side_effect=TimeoutError('startup cap')), patch.object(entry.core, 'train_update') as forbidden:
                with self.assertRaises(TimeoutError):
                    entry.run(output2, 'cpu')
                forbidden.assert_not_called()
            state = json.loads((output2 / 'state.json').read_text())
            self.assertEqual(state['status'], 'UNRESOLVED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attempt', type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ChangedEntry)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    entry.core.atomic_json(args.output / 'checks.json', dict(attempt=args.attempt, successful=result.wasSuccessful(),
                           tests=result.testsRun, errors=len(result.errors), failures=len(result.failures),
                           in_process_seconds=time.perf_counter()-started,
                           scope='CPU fake caches and mocked updates/evaluation only; no generated fixture or rollout'))
    sys.exit(0 if result.wasSuccessful() else 1)
