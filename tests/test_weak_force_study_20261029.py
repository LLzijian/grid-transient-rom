"""Data-free guards and ledger checks; no numerical fixtures or optimizer runs."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import weak_force_study_20261029 as study


def profile(misses=21,risk=2.,stable=1.,timing=.1):
    return dict(classification_actual15s=dict(missed_resolved_unstable=misses,false_los_resolved_stable=0),
        classification_8s=dict(missed_resolved_unstable=3,false_los_resolved_stable=0),
        unstable_actual15_risk=risk,unstable_actual15_numerical_score_perturbation=.001,
        horizon_risks={'5':dict(value=stable),'8':dict(value=stable)},
        common_frequency_rmse_8s=.3,normalized_p_rmse_8s=.4,
        timing_actual15={'one':dict(numerically_qualified=True,truth=dict(time_s=1.),
            fine=dict(time_s=1.+timing),coarse=dict(time_s=1.+timing+.001))})


class StudyGuards(unittest.TestCase):
    def setUp(self):
        self.dependencies=patch.object(study,'deps',return_value=(None,None,types.SimpleNamespace(valid=lambda p:p.get('valid',True)),None))
        self.dependencies.start();self.addCleanup(self.dependencies.stop)

    def test_second_seed_requires_gain_and_every_protection(self):
        profiles=dict(D=profile(),R_N=profile(risk=3.),W=profile(misses=20))
        self.assertTrue(study.retention(profiles)['second_seed'])
        for field in ('common_frequency_rmse_8s','normalized_p_rmse_8s'):
            altered=copy.deepcopy(profiles);altered['W'][field]*=1.051
            self.assertFalse(study.retention(altered)['second_seed'])
        altered=copy.deepcopy(profiles);altered['W']['classification_8s']['missed_resolved_unstable']=4
        self.assertFalse(study.retention(altered)['second_seed'])
        self.assertFalse(study.retention(profiles,optimization_unresolved=True)['second_seed'])

    def test_risk_branch_does_not_fabricate_fewer_misses(self):
        profiles=dict(D=profile(),R_N=profile(risk=3.),W=profile(risk=1.89))
        result=study.retention(profiles)
        self.assertTrue(result['second_seed']);self.assertTrue(result['risk_gain']);self.assertFalse(result['fewer_misses'])
        profiles['W']['unstable_actual15_numerical_score_perturbation']=.2
        self.assertFalse(study.retention(profiles)['second_seed'])

    def test_empty_common_time_and_unresolved_rows_do_not_trigger(self):
        p=dict(D=profile(),R_N=profile(risk=3.),W=profile(misses=20))
        p['W']['timing_actual15']={}
        self.assertFalse(study.retention(p)['second_seed'])
        p['W']=profile(misses=20);p['W']['valid']=False
        self.assertFalse(study.retention(p)['second_seed'])

    def test_efficiency_target_uses_frozen_source_not_continuous_endpoint(self):
        candidate=profile(misses=20)
        self.assertTrue(study.efficiency_target(candidate,profile(),profile(risk=3.))['passed'])
        candidate['classification_8s']['missed_resolved_unstable']=4
        self.assertFalse(study.efficiency_target(candidate,profile(),profile(risk=3.))['passed'])

    def test_attempt_is_charged_before_completion_and_reserved_not_automatic(self):
        with tempfile.TemporaryDirectory() as folder:
            study.journal(folder,'optimizer_step_attempt','formal','D_29311',1,'first')
            c=study.counts(folder)
            self.assertEqual(c['attempted_total'],1);self.assertEqual(c['completed_total'],0)
            with self.assertRaises(ValueError):study.journal(folder,'optimizer_step_attempt','formal','W_29311',1,'second')
            study.journal(folder,'optimizer_step_completed','formal','D_29311',1,'first')
            self.assertEqual(study.counts(folder)['completed_total'],1)
            for category in ('repair','replay'):
                with self.assertRaises(ValueError):study.journal(folder,'optimizer_step_attempt',category,'D_29311',1,'reserved')

    def test_976_total_is_not_reset_by_run_name_or_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'optimizer_calls.jsonl'
            records=[]
            for category,n in study.LIMITS.items():
                for j in range(n):
                    row=dict(call_id=f'{category}_{j}',category=category,run_id='historical',step=j)
                    records.extend([dict(row,event='optimizer_step_attempt'),dict(row,event='optimizer_step_completed')])
            path.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
            self.assertEqual(study.counts(folder)['attempted_total'],976)
            with self.assertRaises(ValueError):study.journal(folder,'optimizer_step_attempt','formal','W_29312',1,'new_seed')


if __name__=='__main__':unittest.main()
