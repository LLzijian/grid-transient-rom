"""Changed-entry QA only: no real fixture, optimizer update, rollout or dataset."""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager, ExitStack
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
                     device='cpu', phase='fit', elapsed_seconds=10., failure_import=dict(binding_sha256=entry.BINDING_SHA))
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
                     lambda m: m._template_scalars.append('unregistered object'),
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

    def test_actual_cuda_to_cpu_canonical_admission(self):
        self.assertTrue(torch.cuda.is_available(), 'registered QA2 requires CUDA')
        base = entry.core.runtime_model(self.source, [entry.TERM], 5101, 'cuda:0')
        self.assertNotEqual(base._template_scalars.extra_repr(), self.reference._template_scalars.extra_repr())
        selected = entry.fast_model(base, self.reference)
        self.assertIs(type(selected), entry.TanhPortTrainingROM)
        self.assertEqual(entry.structure(selected), entry.structure(self.reference))
        with torch.no_grad():
            for parameter in selected.parameters():
                if parameter.requires_grad:
                    parameter.add_(.007)
        wrapper = entry.core.CausalHistoryResidual(selected, np.arange(1, 7.), np.ones(10))
        with torch.no_grad():
            wrapper.history_coefficients.copy_(torch.arange(6, dtype=torch.float64, device='cuda:0')*.003)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cuda_wrapper.json'
            wrapper.export(path)
            reloaded = entry.evaluation_model(path, 'residual', 'cuda:0', self.reference)
            for name, tensor in wrapper.state_dict().items():
                self.assertTrue(torch.equal(tensor, reloaded.state_dict()[name]), name)
            path = Path(tmp) / 'control.json'
            selected.export(path)
            loaded = entry.evaluation_model(path, 'control', 'cuda:0', self.reference)
            self.assertIs(type(loaded), entry.TanhPortTrainingROM)
            for name, tensor in selected.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]), name)
        with self.fake_origin() as (origin, target, binding):
            target.mkdir()
            _, manifest = entry.import_zero_update_failure(target, json.loads(json.dumps(entry.source_identity())), 'cpu')
            models = entry.startup_models(self.source, self.reference, target, manifest, 'cuda:0')
            self.assertEqual(len(models), 6)
            self.assertTrue(all(m.mass.is_cuda for m in models.values()))
            for seed in (5101, 5102, 5103):
                control_rng = models[(seed, 'control')]._registered_start_rng
                residual_rng = models[(seed, 'residual')]._registered_start_rng
                self.assertTrue(torch.equal(control_rng[0], residual_rng[0]))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(control_rng[1], residual_rng[1])))
        torch.cuda.synchronize()

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

    @contextmanager
    def fake_origin(self):
        # Fake observation cache only; lives under ROOT so path identity rules are
        # exercised unchanged. The real private fixture is never loaded by QA.
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            temp = Path(tmp)
            origin, target = temp / 'run01', temp / 'run02'
            origin.mkdir()
            manifest = self.fake_prepare(origin, 'cpu', 1.)
            costs = [dict(seed=s, case=c, group=g, fit_seconds=.03 + i*.001, eval_seconds=0.)
                     for i, (s, c, g) in enumerate(entry.ORDER)]
            identity = json.loads(json.dumps(entry.source_identity()))
            old_files = identity['files'].copy()
            old_files[str(Path(entry.__file__).relative_to(ROOT))] = entry.ORIGIN_ENTRY_SHA
            error = "ValueError('registered structure/order/trainability changed')"
            items = []
            for cost in costs:
                item = dict(cost, fit_status='unresolved', error=error)
                items.append(item)
                folder = origin / f"saturation_{item['seed']}_{item['group']}"
                folder.mkdir()
                entry.core.atomic_json(folder / 'failure_mock.json', dict(error=error, traceback='fast_model(core.runtime_model( BEFORE OPTIMIZER'))
            entry.core.atomic_json(origin / 'fits_frozen.json', items)
            state = dict(status='CLOSED', device='cpu', complete_pairs=0, identity=dict(files=old_files),
                         startup_seconds=entry.PRIOR_STARTUP, items=items, manifest=manifest,
                         freeze_sha256=entry.core.sha(origin / 'fits_frozen.json'))
            entry.core.atomic_json(origin / 'state.json', state)
            entry.core.atomic_json(origin / 'receipt.json', dict(status='CLOSED', this_invocation_seconds=entry.PRIOR_TOTAL, prior_seconds=0.))
            filenames = ['state.json', 'receipt.json', 'fits_frozen.json', 'fixtures/manifest.json', 'fixtures/source.json']
            filenames += [f"saturation_{s}_{g}/failure_mock.json" for s, _, g in entry.ORDER]
            files = {str((origin / name).relative_to(ROOT)).replace('\\', '/'): entry.core.sha(origin / name) for name in filenames}
            binding = dict(origin=str(origin.relative_to(ROOT)), target=str(target.relative_to(ROOT)),
                           origin_entry_sha256=entry.ORIGIN_ENTRY_SHA, files=files,
                           prior_total_seconds=entry.PRIOR_TOTAL, prior_startup_seconds=entry.PRIOR_STARTUP,
                           prior_items=costs, error=error, prior_dependency_files=old_files)
            path = temp / 'binding.json'
            entry.core.atomic_json(path, binding)
            with ExitStack() as stack:
                for key, value in dict(BINDING=path, BINDING_SHA=entry.core.sha(path), ORIGIN=origin, TARGET=target).items():
                    stack.enter_context(patch.object(entry, key, value))
                yield origin, target, binding

    def test_import_rejects_tamper_checkpoint_wrongroute(self):
        for problem in ('origin', 'cache', 'checkpoint', 'route', 'binding'):
            with self.fake_origin() as (origin, target, binding):
                target.mkdir()
                identity = json.loads(json.dumps(entry.source_identity()))
                if problem == 'origin':
                    (origin / 'state.json').write_text('{}')
                elif problem == 'cache':
                    p = origin / 'fixtures/5101/observations.npz'
                    p.write_bytes(p.read_bytes()+b'tamper')
                elif problem == 'checkpoint':
                    (origin / 'saturation_5101_control/checkpoint_00000').mkdir()
                elif problem == 'route':
                    target = target.with_name('unregistered')
                else:
                    entry.BINDING.write_text('{}')
                with self.assertRaises(ValueError):
                    entry.import_zero_update_failure(target, identity, 'cpu')

    def test_startup_failure_zero_updates_and_repeated_import_rejected(self):
        with self.fake_origin() as (origin, target, binding):
            with patch.object(entry, 'PROCESS_START', time.perf_counter()-30.), patch.object(entry, 'startup_models', return_value={}), patch.object(entry.core, 'train_update') as updates, patch.object(entry.core, 'make_optimizer') as optimizers:
                with self.assertRaises(TimeoutError):
                    entry.run(target, 'cpu', import_failure=True)
                updates.assert_not_called()
                optimizers.assert_not_called()
            state = json.loads((target / 'state.json').read_text())
            self.assertGreater(state['startup_seconds'], 50.)
            self.assertGreater(state['elapsed_seconds'], entry.PRIOR_TOTAL+30.)
            for item, old in zip(state['items'], binding['prior_items']):
                self.assertEqual(item['fit_seconds'], old['fit_seconds'])
            with self.assertRaises(FileExistsError):
                entry.run(target, 'cpu', import_failure=True)
            with self.assertRaises(ValueError):
                entry.run(target, 'cpu')

    def test_mockrun_freeze_six_items_inherited_costs(self):
        with self.fake_origin() as (origin, output, binding):
            evaluated = []
            def fake_evaluate(model, obs, scales, deadline):
                frozen = json.loads((output / 'fits_frozen.json').read_text())
                self.assertEqual([i['fit_status'] for i in frozen], ['complete'] * 6)
                self.assertIs(type(model.base if isinstance(model, entry.core.CausalHistoryResidual) else model), entry.TanhPortTrainingROM)
                evaluated.append(True)
                row = dict(mse=.001, weak_mse=.001, strong_mse=.001,
                           event_mse=[.001]*16, dual_step_max_normalized=0.)
                return dict(train=row.copy(), heldout=row.copy(), numerical_qualified=True, prediction_gate=True)
            with patch.object(entry, 'PROCESS_START', time.perf_counter()), patch.object(entry, 'startup_models', return_value={}), patch.object(entry.core, 'checked_fixture', side_effect=AssertionError('forbidden real fixture')), patch.object(entry.core, 'train_update', return_value=(0., {})) as updates, patch.object(entry.core, 'loss', return_value=0.), patch.object(entry.core, 'checkpoint', side_effect=self.fake_checkpoint), patch.object(entry.core, 'evaluate', side_effect=fake_evaluate):
                result = entry.run(output, 'cpu', import_failure=True)
            self.assertEqual(updates.call_count, 6 * 512)
            self.assertEqual(len(evaluated), 6)
            self.assertEqual(result['complete_pairs'], 3)
            self.assertLessEqual(sum(i['fit_seconds'] + i['eval_seconds'] for i in result['items']), result['elapsed_seconds'])
            receipt = json.loads((output / 'receipt.json').read_text())
            self.assertEqual(receipt['prior_seconds'], entry.PRIOR_TOTAL)
            self.assertAlmostEqual(receipt['cumulative_seconds'], entry.PRIOR_TOTAL + receipt['this_invocation_seconds'], places=2)
            for item, old in zip(result['items'], binding['prior_items']):
                self.assertGreater(item['fit_seconds'], old['fit_seconds'])
            first = dict(binding['prior_items'][0])
            budget = entry.phase_budget(dict(items=binding['prior_items']), first, 'fit', entry.PRIOR_TOTAL)
            self.assertEqual(budget, 535. - first['fit_seconds'])

    def test_mock_caught_resume_once_retains_identity_and_costs(self):
        with self.fake_origin() as (origin, output, binding):
            with patch.object(entry, 'PROCESS_START', time.perf_counter()), patch.object(entry, 'startup_models', return_value={}), patch.object(entry.core, 'train_update', side_effect=KeyboardInterrupt), patch.object(entry.core, 'checkpoint', side_effect=self.fake_checkpoint):
                with self.assertRaises(KeyboardInterrupt):
                    entry.run(output, 'cpu', import_failure=True)
            interrupted = json.loads((output / 'state.json').read_text())
            self.assertEqual(interrupted['status'], 'INTERRUPTED_RUNTIME')
            prior = interrupted['elapsed_seconds']
            with patch.object(entry, 'PROCESS_START', time.perf_counter()), patch.object(entry.core, 'restore', return_value=(0, [])), patch.object(entry.core, 'train_update', side_effect=FloatingPointError('mock unresolved; no real update')), patch.object(entry.core, 'checkpoint', side_effect=self.fake_checkpoint):
                result = entry.run(output, 'cpu', resume=True)
            self.assertEqual(result['resumes'], 1)
            self.assertGreater(result['elapsed_seconds'], prior)
            self.assertEqual(json.loads((output / 'receipt.json').read_text())['prior_seconds'], prior)
            with self.assertRaises(ValueError):
                entry.resume_allowed(result, result['identity'], 'cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attempt', type=int, choices=(2,), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ChangedEntry)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    entry.core.atomic_json(args.output / 'checks.json', dict(attempt=args.attempt, successful=result.wasSuccessful(),
                           tests=result.testsRun, errors=len(result.errors), failures=len(result.failures),
                           in_process_seconds=time.perf_counter()-started,
                           scope='CUDA/CPU admission and export; fake caches/mocked updates and evaluation; no optimizer updates, fixture generation or rollout'))
    sys.exit(0 if result.wasSuccessful() else 1)
