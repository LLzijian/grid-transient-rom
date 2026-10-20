"""Pure-stdlib synthetic QA; no private data, model code or study results."""
import copy
from pathlib import Path
import math
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import extract_spatial_extrema_20261020 as m


def manifest():
    rows=[]
    for s in m.SEEDS:
        for v in m.VARIANTS:
            path=m.role_path(s,v)
            hashes=m.IMPORTED_HASHES if (s,v)==(6101,'O') else ('a'*64,'b'*64,'c'*64)
            rows.append({'seed':s,'variant':v,'stage_step':512,'imported_control':(s,v)==(6101,'O'),
                'model_path':str(path),'training_path':str(path.parent/'training.pt'),'checkpoint_path':str(path.parent/'checkpoint.json'),
                **dict(zip(('model_sha256','training_sha256','checkpoint_sha256'),hashes))})
    return {'endpoint_count':6,'endpoints':rows,'schedule_sha256':m.SCHEDULE_SHA,'test_read':False,'val_read':False}


def fixture():
    ids=[f'old_{i}' for i in range(346)]+[f'newtrain_OLD_{i}' for i in range(20)]+[f'newtrain_NEW_{i}' for i in range(19)]
    rows=[{'id':eid,'group':'bus_fixture','label':'stable' if i<342 else 'unstable'} for i,eid in enumerate(ids)]
    groups,full=m.load_cohorts({'cohorts':{'original366':rows[:366],'new19':rows[366:]}})
    extras=[]; points=[]; records=[]; numerical=[]
    for row in rows:
        pair={'finite_to_endpoint':True,'los_pair_pass':True,'coi_normalized_rmse':.0001,'common_normalized_rmse':.0001,
              'fine_classification':{'category':'window_no_los_unsettled','predicted_los':False,'trigger_index':None}}
        component={'available':True,'frequency':{'finite':True,'physical_sse':4.,'n_scalar_points':1},
                   'mechanical_power':{'finite':True,'physical_sse':.1,'n_scalar_points':10}}
        extras.append({'event_id':row['id'],'group':row['group'],'label':row['label'],'scientific_numeric_8s':True,
            'classification_8s':copy.deepcopy(pair),'classification_15s':copy.deepcopy(pair),
            'actual_interval15':{'available':True,'finite':True,'numerically_admissible':True,'coi_normalized_rmse':.0001,
                'common_normalized_rmse':.0001,'n_time_points':1,'mean_point_error':2.,'frequency_rmse_rad_s':2.,'power_rmse_system_pu':.1},
            'physical_1_2_5_8s':{h:copy.deepcopy(component) for h in ('1','2','5','8','prefix')},
            'spatial':{'actual_future_points':1,'finite':True,'numerically_admissible':True,'combined':3.,
                       'angle_normalized_squared_error':2.,'speed_normalized_squared_error':4.}})
        for h in ('1','2','5','8','prefix'):
            common={'event_id':row['id'],'group':row['group'],'label':row['label'],'horizon':h,'available':True,'finite':True,'numerically_admissible':True}
            points.append(dict(common,mean_point_error=2.))
            records.append(dict(common,rmse=2.,normalized_sse=4.,n=1))
            numerical.append(dict(common,**{k:v for k,v in pair.items() if k!='fine_classification'}))
    return {'event_count':385,'test_read':False,'extras':extras,'rows':{'objective_point_error':points,'records':records,'numerical':numerical}},groups,full


class ExtractionQA(unittest.TestCase):
    def test_exact_control_role_not_new_training(self):
        f=manifest(); roles=m.checked_manifest(f)
        self.assertEqual(len(roles),6)
        self.assertIn('temporal_objective01',roles[(6101,'O')]['model_path'])
        self.assertIn('spatial_extrema01',roles[(6101,'S')]['model_path'])
        bad=copy.deepcopy(f); bad['endpoints'][0]['model_path']=str(m.BASE/'models/6101_O/checkpoints/step000512/model.json')
        with self.assertRaises(ValueError):m.checked_manifest(bad)
        bad=copy.deepcopy(f); bad['endpoints'][0]['model_sha256']='0'*64
        with self.assertRaises(ValueError):m.checked_manifest(bad)
        bad=copy.deepcopy(f); bad['endpoints'][0]['imported_control']=False
        with self.assertRaises(ValueError):m.checked_manifest(bad)

    def test_cohort_sets_and_all_event_denominators(self):
        e,g,full=fixture(); m.validate_evidence(e,full)
        self.assertEqual(len(g['original366']),366);self.assertEqual(len(g['new19']),19)
        self.assertEqual(sum(x.startswith('newtrain_') for x in g['original366']),20)
        s=m.summarize(e,g['common385'])
        self.assertEqual(s['stable_count'],342);self.assertEqual(s['unstable_count'],43)
        self.assertEqual(s['classification_15s']['missed'],43)
        bad=copy.deepcopy(e);bad['extras'].pop()
        with self.assertRaises(ValueError):m.validate_evidence(bad,full)

    def test_units_and_sse_pool_risk_are_distinct(self):
        self.assertAlmostEqual(m.risk([1.,2.,3.]),2.2)
        e,_,_=fixture();a,b=copy.deepcopy(e['extras'][:2])
        a['actual_interval15'].update(n_time_points=1,frequency_rmse_rad_s=2.)
        b['actual_interval15'].update(n_time_points=3,frequency_rmse_rad_s=4.)
        f=m.pooled([a,b],'frequency');self.assertEqual(f['n_scalar_points'],4);self.assertAlmostEqual(f['rmse'],math.sqrt(13.))
        p=m.pooled([a,b],'mechanical_power');self.assertEqual(p['n_scalar_points'],40);self.assertAlmostEqual(p['rmse_MW'],10.)

    def test_coi_arithmetic_mean_pooled_and_ceil_tail(self):
        e,_,_=fixture(); ids={e['extras'][0]['event_id'],e['extras'][1]['event_id']}
        row=[r for r in e['rows']['records'] if r['event_id'] in ids and r['horizon']=='1']
        row[0].update(rmse=1.,normalized_sse=1.,n=1)
        row[1].update(rmse=3.,normalized_sse=27.,n=3)
        s=m.coi_summary(e,ids,'stable','1')
        self.assertEqual(s['mean_event_normalized_rmse'],2.)
        self.assertAlmostEqual(s['pooled_normalized_rmse'],math.sqrt(7.))
        self.assertEqual(s['worst10_mean_event_normalized_rmse'],3.)
        row[1]['numerically_admissible']=False
        s=m.coi_summary(e,ids,'stable','1')
        self.assertEqual(s['eligible_events'],2);self.assertIsNone(s['pooled_normalized_rmse'])

    def test_numeric_propagation_and_spatial_finite_insufficient(self):
        e,g,_=fixture();e['extras'][0]['scientific_numeric_8s']=False
        s=m.summarize(e,g['common385'])
        self.assertIsNone(s['stable']['5']['ceil_risk'])
        self.assertEqual(s['spatial_actual15']['all']['combined'],3.)
        e['extras'][0]['classification_15s']['los_pair_pass']=False
        s=m.summarize(e,g['common385'])
        self.assertIsNone(s['spatial_actual15']['all']['combined'])
        self.assertEqual(s['spatial_actual15']['all']['events'],385)
        self.assertEqual(s['classification_15s']['stable_denominator'],342)
        self.assertEqual(s['classification_15s']['numerical_unresolved'],1)
        e['extras'][0]['classification_15s']['los_pair_pass']=True
        e['extras'][0]['spatial']['numerically_admissible']=False
        self.assertIsNone(m.spatial_facts(e['extras'])['combined'])


if __name__=='__main__':unittest.main()
