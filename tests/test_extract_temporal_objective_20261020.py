"""Pure unittest fixtures, with no private assets or numerical-model imports."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import extract_temporal_objective_20261020 as m


def fixture():
    ids = [f'old_{i}' for i in range(346)]+[f'newtrain_OLD_{i}' for i in range(20)]+[f'newtrain_NEW_{i}' for i in range(19)]
    records = [{'id':eid,'group':'fixture_bus','label':'stable' if i<342 else 'unstable'} for i,eid in enumerate(ids)]
    groups,full = m.load_cohorts({'cohorts':{'original366':records[:366],'new19':records[366:]}})
    extras = []
    for i,r in enumerate(records):
        n = 500 if i%2 == 0 else 501
        classification = {'finite_to_endpoint':True,'los_pair_pass':True,'coi_normalized_rmse':.0001,'common_normalized_rmse':.0001,
                          'fine_classification':{'category':'window_no_los_unsettled','predicted_los':False,'trigger_index':None}}
        physical = {'available':True,'frequency':{'finite':True,'n_scalar_points':n,'physical_sse':4.*n},
                    'mechanical_power':{'finite':True,'n_scalar_points':10*n,'physical_sse':.01*10*n}}
        extras.append({'event_id':r['id'],'group':r['group'],'label':r['label'],'scientific_numeric_8s':True,
            'classification_8s':copy.deepcopy(classification),'classification_15s':copy.deepcopy(classification),
            'actual_interval15':{'n_time_points':n,'available':True,'finite':True,'numerically_admissible':True,
                'coi_normalized_rmse':.0001,'common_normalized_rmse':.0001,'mean_point_error':2.,'frequency_rmse_rad_s':2.,'power_rmse_system_pu':.1},
            'physical_1_2_5_8s':{h:copy.deepcopy(physical) for h in ('1','2','5','8','prefix')},
            'temporal':{'actual_future_points':n,'tail_points':(n+4)//5,'endpoint_group':'le5s' if n<=500 else 'gt5s',
                        'finite':True,'mean_point_error':2.,'tail_point_error':3.}})
    points = [{'event_id':r['id'],'group':r['group'],'label':r['label'],'horizon':h,'available':True,
               'finite':True,'numerically_admissible':True,'mean_point_error':2.} for r in records for h in ('1','2','5','8','prefix')]
    return {'event_count':385,'extras':extras,'rows':{'objective_point_error':points},'test_read':False},groups,full


class ExtractionTest(unittest.TestCase):
    def test_fixed_cohorts_ignore_prefix_and_reject_missing_duplicate(self):
        e,g,full=fixture(); m.validate_evidence(e,full)
        self.assertEqual(len(g['original366']),366); self.assertEqual(len(g['new19']),19)
        self.assertEqual(sum(x.startswith('newtrain_') for x in g['original366']),20)
        broken=copy.deepcopy(e); broken['extras'].pop()
        with self.assertRaises(ValueError): m.validate_evidence(broken,full)
        broken=copy.deepcopy(e); broken['extras'][-1]=broken['extras'][0]
        with self.assertRaises(ValueError): m.validate_evidence(broken,full)
        broken=copy.deepcopy(e); broken['rows']['objective_point_error'].pop()
        with self.assertRaises(ValueError): m.validate_evidence(broken,full)

    def test_risk_units_and_actual_sse_counts(self):
        self.assertAlmostEqual(m.risk([1.,2.,3.]),2.2)
        e,_,_=fixture(); a,b=copy.deepcopy(e['extras'][:2])
        a['actual_interval15'].update(n_time_points=1,frequency_rmse_rad_s=2.)
        b['actual_interval15'].update(n_time_points=3,frequency_rmse_rad_s=4.)
        p=m.pooled([a,b],'frequency'); self.assertEqual(p['sse'],52.); self.assertEqual(p['n_scalar_points'],4)
        self.assertAlmostEqual(p['rmse'],math.sqrt(13.))
        p=m.pooled([a,b],'mechanical_power'); self.assertEqual(p['n_scalar_points'],40)
        self.assertAlmostEqual(p['rmse'],.1); self.assertAlmostEqual(p['rmse_MW'],10.)

    def test_qualification_propagation_keeps_every_denominator(self):
        e,g,_=fixture(); summary=m.summarize(e,g['common385'])
        self.assertEqual(summary['stable']['5']['ceil_risk'],2.)
        self.assertEqual(summary['classification_15s']['unstable_denominator'],43)
        self.assertEqual(summary['classification_15s']['missed'],43)
        e['extras'][0]['scientific_numeric_8s']=False
        changed=m.summarize(e,g['common385'])
        self.assertIsNone(changed['stable']['5']['ceil_risk'])
        self.assertEqual(changed['temporal_actual_interval15']['all']['mean_event_tail20_point_error'],3.)
        e['extras'][0]['classification_15s']['los_pair_pass']=False
        changed=m.summarize(e,g['common385'])
        self.assertIsNone(changed['temporal_actual_interval15']['all']['mean_event_tail20_point_error'])
        self.assertEqual(changed['temporal_actual_interval15']['all']['event_count'],385)
        self.assertEqual(changed['classification_15s']['stable_denominator'],342)
        self.assertEqual(changed['classification_15s']['numerical_unresolved'],1)
        self.assertIsNone(m.differences(changed,summary)['stable_5.ceil_risk'])

    def test_finite_is_insufficient_and_missing_tail_is_explicit(self):
        e,_,_=fixture(); subset=e['extras'][:2]
        self.assertEqual(m.temporal_facts(subset)['all']['mean_event_tail20_point_error'],3.)
        subset[0]['actual_interval15']['numerically_admissible']=False
        self.assertIsNone(m.temporal_facts(subset)['all']['mean_event_tail20_point_error'])
        subset[0]['actual_interval15']['numerically_admissible']=True
        subset[0].pop('temporal')
        facts=m.temporal_facts(subset)['all']
        self.assertEqual(facts['missing_temporal_fact_ids'],[subset[0]['event_id']])
        self.assertEqual(facts['event_count'],2); self.assertIsNone(facts['mean_event_tail20_point_error'])

    def test_endpoint_groups_boundary_and_tail_count(self):
        e,_,_=fixture(); facts=m.temporal_facts(e['extras'])
        self.assertEqual(facts['le5s']['event_count'],193)
        self.assertEqual(facts['gt5s']['event_count'],192)
        self.assertEqual(facts['le5s']['event_count']+facts['gt5s']['event_count'],385)
        self.assertEqual(facts['le5s']['classification_15s']['stable_denominator'],171)
        e['extras'][0]['temporal']['tail_points']=99
        with self.assertRaises(ValueError): m.temporal_facts(e['extras'])


if __name__ == '__main__':
    unittest.main()
