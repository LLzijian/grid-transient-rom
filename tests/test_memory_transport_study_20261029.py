"""Pure guard definitions only; no private payload, numerical suite or Adam.

Execution requires a separate root release. All profiles, traces and ledgers
below are synthetic, including the apparent 108/1848 call records.
"""
import copy
import json
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import memory_transport_study_20261029 as study


def profile(misses=21, risk=2., stable=1., timing=.1):
    return dict(valid=True,
        classification_actual15s=dict(missed_resolved_unstable=misses, false_los_resolved_stable=0),
        classification_8s=dict(missed_resolved_unstable=3, false_los_resolved_stable=0),
        unstable_actual15_risk=risk, unstable_actual15_numerical_score_perturbation=.001,
        horizon_risks={'5': dict(value=stable), '8': dict(value=stable)},
        common_frequency_rmse_8s=.3, normalized_p_rmse_8s=.4,
        timing_actual15={'shared': dict(numerically_qualified=True,
            truth=dict(time_s=1.), fine=dict(time_s=1.+timing), coarse=dict(time_s=1.+timing+.001))})


def good_comparison():
    return dict(A=profile(), P=profile(), S=profile(misses=20, risk=1.8), R_N=profile(risk=3.))


def trace(n=64, clipped=0, jumps=0, block='s_raw'):
    return [dict(gradient_norm=101. if i < clipped else 100.,
        block_updates={block: dict(relative_l2=.251 if i < jumps else .25)}) for i in range(n)]


class StudyGuards(unittest.TestCase):
    def setUp(self):
        self.sgm = types.SimpleNamespace(valid=lambda p: p.get('valid', False))
        self.dependencies = patch.object(study, 'deps', return_value=(None, None, self.sgm, None))
        self.dependencies.start()
        self.addCleanup(self.dependencies.stop)

    def test_S_requires_a_separate_improvement_against_each_control(self):
        p = good_comparison()
        self.assertTrue(study.retention(p)['second_seed'])
        # S improves A, but cannot borrow that gain to cover its lack of gain over P.
        p['P'] = profile(misses=20, risk=1.8)
        result = study.retention(p)
        self.assertTrue(result['improvements']['A']['passed'])
        self.assertFalse(result['improvements']['P']['passed'])
        self.assertFalse(result['second_seed'])

    def test_risk_OR_branch_preserves_miss_name_and_cannot_bypass_R_N(self):
        p = dict(A=profile(), P=profile(), S=profile(risk=1.8), R_N=profile(risk=3.))
        result = study.retention(p)
        self.assertTrue(result['second_seed'])
        for control in ('A', 'P'):
            self.assertTrue(result['improvements'][control]['risk_gain'])
            self.assertFalse(result['improvements'][control]['fewer_misses'])
        p['R_N']['unstable_actual15_risk'] = 1.7
        self.assertFalse(study.retention(p)['second_seed'])
        p['R_N']['unstable_actual15_risk'] = 3.
        p['S']['unstable_actual15_numerical_score_perturbation'] = .3
        self.assertFalse(study.retention(p)['second_seed'])

    def test_every_accuracy_event_and_frozen_A96_guard_is_necessary(self):
        for metric in ('common_frequency_rmse_8s', 'normalized_p_rmse_8s'):
            for control in ('A', 'P', 'R_N'):
                p = good_comparison(); p[control][metric] = p['S'][metric]/1.051
                self.assertFalse(study.retention(p)['second_seed'], (control, metric))
        for horizon in ('5', '8'):
            p = good_comparison(); p['R_N']['horizon_risks'][horizon]['value'] = .9
            self.assertFalse(study.retention(p)['second_seed'])
        for key in ('classification_actual15s', 'classification_8s'):
            p = good_comparison(); p['S'][key]['false_los_resolved_stable'] = 1
            self.assertFalse(study.retention(p)['second_seed'])
        p = good_comparison(); p['R_N']['classification_actual15s']['missed_resolved_unstable'] = 19
        self.assertFalse(study.retention(p)['second_seed'])
        p = good_comparison()
        for item in p.values(): item['classification_8s']['missed_resolved_unstable'] = 4
        self.assertFalse(study.retention(p)['second_seed'])  # Frozen A96 still has only 3 misses.

    def test_common_T_requires_real_localizable_qualified_rows_and_timing_protection(self):
        for edit in ('empty', 'truth_NA', 'unqualified', 'late'):
            p = good_comparison()
            if edit == 'empty': p['P']['timing_actual15'] = {}
            elif edit == 'truth_NA': p['S']['timing_actual15']['shared']['truth']['time_s'] = None
            elif edit == 'unqualified': p['R_N']['timing_actual15']['shared']['numerically_qualified'] = False
            else:
                p['S']['timing_actual15']['shared']['fine']['time_s'] = 1.4
                p['S']['timing_actual15']['shared']['coarse']['time_s'] = 1.401
            self.assertFalse(study.retention(p)['second_seed'], edit)
        p = good_comparison(); del p['P']
        self.assertEqual(study.retention(p)['mechanism'], 'UNRESOLVED')
        p = good_comparison(); p['A']['valid'] = False
        self.assertFalse(study.retention(p)['second_seed'])
        self.assertFalse(study.retention(good_comparison(), optimization_unresolved=True)['second_seed'])

    def test_absolute_cost_target_R_N_risk_is_outside_both_improvement_branches(self):
        A96, reference = profile(risk=3.), profile(risk=1.9)
        candidate = profile(misses=20, risk=2.)
        result = study.efficiency_target(candidate, A96, reference)
        self.assertTrue(result['fewer_misses']); self.assertTrue(result['risk_gain'])
        self.assertFalse(result['passed'])
        reference['unstable_actual15_risk'] = 2.1
        self.assertTrue(study.efficiency_target(candidate, A96, reference)['passed'])
        candidate['classification_actual15s']['missed_resolved_unstable'] = 22
        self.assertFalse(study.efficiency_target(candidate, A96, reference)['passed'])

    def test_last64_optimization_boundaries_and_older_history(self):
        self.assertFalse(study.optimization(trace(clipped=16))['optimization_unresolved'])
        self.assertTrue(study.optimization(trace(clipped=17))['optimization_unresolved'])
        for block in ('a_bar', 's_raw', 'd_raw'):
            self.assertFalse(study.optimization(trace(jumps=15, block=block))['optimization_unresolved'])
            self.assertTrue(study.optimization(trace(jumps=16, block=block))['optimization_unresolved'])
        self.assertFalse(study.optimization(trace(clipped=64)+trace())['optimization_unresolved'])
        self.assertTrue(study.optimization(trace(n=63))['optimization_unresolved'])

    def test_extension_needs_actual15_gain_full_protections_and_resolved_optimization(self):
        before, after = profile(risk=2.), profile(misses=20, risk=1.97)
        self.assertTrue(study.extension_decision(before, after, trace())['extend'])
        after['normalized_p_rmse_8s'] = before['normalized_p_rmse_8s']*1.051
        self.assertFalse(study.extension_decision(before, after, trace())['extend'])
        after = profile(risk=1.875)
        before['unstable_actual15_numerical_score_perturbation'] = .0625
        after['unstable_actual15_numerical_score_perturbation'] = .0625
        self.assertFalse(study.extension_decision(before, after, trace())['extend'])  # Difference equals uncertainty sum.
        self.assertFalse(study.extension_decision(profile(), profile(misses=20), trace(clipped=17))['extend'])

    def test_pending_is_charged_and_quarantined_run_does_not_stop_healthy_run(self):
        with tempfile.TemporaryDirectory() as folder:
            study.journal(folder, 'optimizer_step_attempt', 'formal', 'S_30311', 1, 'pending')
            self.assertEqual(study.counts(folder)['attempted_total'], 1)
            with self.assertRaises(ValueError):
                study.journal(folder, 'optimizer_step_attempt', 'formal', 'S_30311', 2, 'retry')
            study.journal(folder, 'optimizer_step_attempt', 'formal', 'A_30311', 1, 'healthy')
            study.journal(folder, 'optimizer_step_completed', 'formal', 'A_30311', 1, 'healthy')
            count = study.counts(folder)
            self.assertEqual((count['attempted_total'], count['completed_total']), (2, 1))
            self.assertEqual([r['call_id'] for r in count['pending']], ['pending'])
            for category in ('repair', 'replay'):
                with self.assertRaises(ValueError):
                    study.journal(folder, 'optimizer_step_attempt', category, 'S_30311', 1, 'reserved')

    def test_total1848_and_formal_run_bound_do_not_reset_with_seed(self):
        self.assertEqual(study.LIMITS, dict(formal=1536, qa=108, repair=108, replay=96))
        self.assertEqual(study.SEEDS, (30311, 30312))
        with tempfile.TemporaryDirectory() as folder:
            records = []
            for category, n in study.LIMITS.items():
                for i in range(n):
                    identity = dict(category=category, run_id='synthetic_prior', step=i+1, call_id=f'{category}_{i}')
                    records.extend(dict(identity, event=e) for e in ('optimizer_step_attempt', 'optimizer_step_completed'))
            (Path(folder)/'optimizer_calls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
            self.assertEqual(study.counts(folder)['attempted_total'], 1848)
            with self.assertRaises(ValueError):
                study.journal(folder, 'optimizer_step_attempt', 'formal', 'P_30312', 1, 'new_seed')
        with tempfile.TemporaryDirectory() as folder:
            for name, step in (('A_30311', 257), ('A_30313', 1), ('D_29311', 1)):
                with self.assertRaises(ValueError): study.journal(folder, 'optimizer_step_attempt', 'formal', name, step, name)

    def test_shared_schedule_reaches256_with_fixed_sampling_and_horizons(self):
        import numpy as np
        from experiments import force_graph_study_20261028 as old
        events = [dict(id=f'artificial_{i}') for i in range(297)]
        helper = types.SimpleNamespace(fg=types.SimpleNamespace(sample_step=old.sample_step), horizon_for=old.horizon_for)
        with patch.object(study, 'deps', return_value=(np,None,helper,None)):
            first = study.make_schedule(events,30311)
            repeated = study.make_schedule(events,30311)
            self.assertEqual(first,repeated)
            self.assertEqual(len(first['rows']),256)
            self.assertEqual([r['horizon_s'] for r in first['rows']], [5. if i%2 else 8. for i in range(1,257)])
            self.assertTrue(all(len(r['random_indices'])==8 and len(set(r['coverage_indices']))==4 and len(r['sample_ids'])==12 for r in first['rows']))
            self.assertEqual({i for r in first['rows'][:75] for i in r['coverage_indices']},set(range(297)))
            self.assertNotEqual(first,study.make_schedule(events,30312))
            with self.assertRaises(ValueError): study.make_schedule(events,30313)
            with self.assertRaises(ValueError): study.make_schedule(events[:296],30311)

    def test_common128_failure_cannot_be_rescued_by_a_good256_endpoint(self):
        bad, good = good_comparison(), good_comparison()
        bad['S']['classification_actual15s']['missed_resolved_unstable'] = 21
        bad['S']['unstable_actual15_risk'] = 2.
        prepared = dict(panel_ids=['fit'], no_update_ids=['held'])
        fit, all_events = [dict(id='fit')], [dict(id='fit'), dict(id='held')]
        writes = {}
        fake_sgm = types.SimpleNamespace(valid=lambda p: p.get('valid', False))
        fake_sgm.read = lambda path: (dict(eligible_variants=list(study.VARIANTS)) if str(path).endswith('qa/summary.json')
            else dict(R_N_profile=good['R_N'], A96_profile=profile()))
        fake_sgm.write = lambda path, value, **kw: writes.__setitem__(Path(path).name, value)
        fake_sgm.evaluate = Mock()
        loader = Mock(return_value=object())
        fake_mt = types.SimpleNamespace(MemoryTransportROM=types.SimpleNamespace(from_export=loader))
        def fake_run(base, variant, seed, *args):
            return dict(status='COMPLETED', run_id=f'{variant}_{seed}', variant=variant, seed=seed,
                updates=128,
                fit_endpoints={'0': profile(), '128': bad[variant], '256': good[variant]},
                cost_milestones=[], run_total_seconds=1., endpoint='synthetic_endpoint', optimization_unresolved_at128=False)
        def fake_prepare(base, *, fit_only=True):
            if not fit_only: self.assertIn('fit_decisions_closed.json', writes)
            return prepared, fit if fit_only else all_events
        with tempfile.TemporaryDirectory() as folder, patch.object(study, 'admitted'), \
                patch.object(study, 'deps', return_value=(None, None, fake_sgm, fake_mt)), \
                patch.object(study, 'checked_preparation', side_effect=fake_prepare), \
                patch.object(study, 'formal_run', side_effect=fake_run) as formal:
            result = study.study(Path(folder), device='cpu')
        self.assertEqual([c.args[2] for c in formal.call_args_list], [30311]*3)
        self.assertFalse(result['decisions'][0]['second_seed'])
        self.assertEqual(loader.call_count, 3)  # Reporting also uses the new transport loader.

    def test_restore_uses_transport_loader_preserving_plane_and_extra_parameter(self):
        for variant, extra, value in (('S','s_raw',-.4),('P','d_raw',.3)):
            with self.subTest(variant=variant):
                self.restore_case(variant,extra,value)

    def restore_case(self,variant,extra,value):
        identity = dict(variant=variant)
        frozen20 = tuple((i+1)/100 for i in range(20))
        loaded = Mock(history_dt=.0025, variant=variant, frozen20=frozen20)
        setattr(loaded,extra,value)
        loaded.to.return_value = loaded; loaded.double.return_value = loaded
        opt = Mock(state={}); device = types.SimpleNamespace(type='cpu')
        runtime = dict(fake_runtime=True); layout = ['synthetic_s_raw_layout']
        meta = dict(identity=identity, resumable=True, runtime=runtime, model_sha256='model', training_sha256='training', step=0, parameter_layout=layout)
        state = dict(step=0, optimizer={}, numpy_rng={}, numpy_global_rng='np', python_rng='py', torch_rng='torch', coverage={'cursor': 5})
        def read(path): return dict(directory='step000000', checkpoint_sha256='meta') if Path(path).name=='latest.json' else meta
        def digest(path): return {'checkpoint.json':'meta','model.json':'model','training.pt':'training'}[Path(path).name]
        old = types.SimpleNamespace(runtime_identity=lambda d: runtime, _parameter_layout=lambda m,o: layout, _adam_states_finite=lambda x: True)
        sgm = types.SimpleNamespace(read=read, sha=digest, resolve_device=lambda d: device, old=old, StateGatedMemoryROM=Mock())
        rng = types.SimpleNamespace(bit_generator=types.SimpleNamespace(state=None))
        np = types.SimpleNamespace(random=types.SimpleNamespace(default_rng=lambda: rng, set_state=Mock()))
        torch = types.SimpleNamespace(load=Mock(return_value=state), set_rng_state=Mock())
        transport = types.SimpleNamespace(MemoryTransportROM=types.SimpleNamespace(from_export=Mock(return_value=loaded)))
        with tempfile.TemporaryDirectory() as folder, patch.object(study, 'deps', return_value=(np,torch,sgm,transport)), \
                patch.object(study, 'make_optimizer', return_value=opt), patch.object(study.random, 'setstate'):
            restored = study.restore_checkpoint(folder, 'cpu', identity)
            self.assertIs(restored[0], loaded)
            self.assertEqual((restored[0].frozen20, getattr(restored[0],extra)), (frozen20,value))
            transport.MemoryTransportROM.from_export.assert_called_once_with(Path(folder)/'step000000/model.json')
            sgm.StateGatedMemoryROM.from_export.assert_not_called()
            transport.MemoryTransportROM.from_export.side_effect = ValueError('old SGM format rejected')
            torch.load.reset_mock()
            with self.assertRaises(ValueError): study.restore_checkpoint(folder, 'cpu', identity)
            torch.load.assert_not_called()


class Scalar:
    """Tiny fake arithmetic object; no tensor, autograd or optimizer runtime."""
    requires_grad = True
    def __init__(self, value): self.value=float(value); self.grad=True
    def detach(self): return self
    def clone(self): return Scalar(self.value)
    def norm(self): return Scalar(abs(self.value))
    def clamp_min(self, value): return Scalar(max(self.value,value))
    def __sub__(self, other): return Scalar(self.value-other.value)
    def __truediv__(self, other): return Scalar(self.value/other.value)
    def __float__(self): return self.value
    def backward(self): pass


class ProjectedUpdateDefinition(unittest.TestCase):
    def test_extra_s_d_change_uses_projected_value_and_max_one_denominator(self):
        s, d = Scalar(0), Scalar(0)
        model=types.SimpleNamespace(mass=types.SimpleNamespace(device='cpu'),
            parameters=lambda: [s,d], named_parameters=lambda: [('s_raw',s),('d_raw',d)],
            project_runtime_=lambda: setattr(d,'value',max(0.,d.value)))
        def fake_step(): s.value=.4; d.value=-.3
        opt=types.SimpleNamespace(zero_grad=Mock(),step=fake_step,state={})
        finite=types.SimpleNamespace(all=lambda:True)
        torch=types.SimpleNamespace(isfinite=lambda v:finite,nn=types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=lambda p,n:Scalar(1))))
        sgm=types.SimpleNamespace(synchronize=lambda d:None,graph=lambda m:nullcontext(),
            objective=lambda *a,**k:(Scalar(1),{}),old=types.SimpleNamespace(_adam_states_finite=lambda s:True))
        with patch.object(study,'deps',return_value=(None,torch,sgm,None)),patch.object(study,'journal'):
            row=study.one_update(model,opt,None,[],5.,base='synthetic',category='formal',run_id='S_30311',step=1)
        self.assertEqual(row['block_updates']['s_raw']['relative_l2'],.4)
        self.assertEqual(row['block_updates']['d_raw']['relative_l2'],0.)
        s.value=2.;d.value=2.
        with patch.object(study,'deps',return_value=(None,torch,sgm,None)),patch.object(study,'journal'):
            row=study.one_update(model,opt,None,[],5.,base='synthetic',category='formal',run_id='P_30311',step=1)
        self.assertEqual(row['block_updates']['s_raw']['relative_l2'],.8)
        self.assertEqual(row['block_updates']['d_raw']['relative_l2'],1.)


if __name__ == '__main__':
    unittest.main()
