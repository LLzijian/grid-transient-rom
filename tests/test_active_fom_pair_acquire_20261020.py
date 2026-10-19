"""Synthetic acquisition contract; no FOM, model weights or restricted data."""
import itertools
import math
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import active_fom_pair_20261020 as a

def synthetic_pool():
    result=[]
    times=((.075,.125),(.175,.225,.275),(.35,.5))
    for bus,counts in zip(a.BUSES,a.EXPECTED_COUNTS):
        for s,count in enumerate(counts):
            choices=list(itertools.product(times[s],(.001,math.sqrt(.00001),.01)))[:count]
            for duration,xf in choices:
                result.append({'bus':bus,'bus_id':str(bus),'duration_s':duration,'xf':xf,
                               'line':f'L{bus}','location':0.,'id':f'{bus}_{duration}_{xf}'})
    return result

class AcquisitionContracts(unittest.TestCase):
    def test_exact_dp_against_exhaustive(self):
        allowed=((0,1),(1,2),(0,2))
        expected=[v for v in itertools.product(*allowed) if all(v.count(j)==1 for j in range(3))]
        assignment,details=a.draw_assignment(allowed,(1,1,1),np.random.default_rng(7))
        self.assertEqual(details['total_count'],len(expected))
        self.assertIn(tuple(assignment),expected)
        for i in range(3):
            self.assertEqual(details['marginal_numerators'][i],[sum(v[i]==s for v in expected) for s in range(3)])
        for row in details['path']:
            self.assertLess(row['draw'],row['total'])
            self.assertEqual(row['total'],sum(row['completion_counts']))

    def test_prelocked_complete_design(self):
        d=a.make_design(synthetic_pool()); repeated=a.make_design(synthetic_pool())
        self.assertEqual(d,repeated)
        self.assertEqual(d['val_dp']['total_count'],4252621230)
        self.assertEqual(len(d['candidates']),48)
        self.assertEqual(len(set(d['random_ids'])),12)
        for s in a.STRATA:
            self.assertEqual(sum(r['stratum']==s for r in d['val_conditions']),8)
            self.assertEqual(sum(r['stratum']==s for r in d['candidates']),16)
            self.assertEqual(sum(r['stratum']==s and r['id'] in d['random_ids'] for r in d['candidates']),4)
        for b in a.BUSES:
            self.assertEqual(sum(r['bus']==b for r in d['val_conditions']),1)
            cs=[r for r in d['candidates'] if r['bus']==b]
            self.assertEqual(len(cs),2); self.assertEqual(len({r['stratum'] for r in cs}),2)
        self.assertFalse({a.key(r) for r in d['val_conditions']} & {a.key(r) for r in d['candidates']})

    def test_pool_mismatch_rejects(self):
        with self.assertRaises(ValueError): a.make_design(synthetic_pool()[:-1])
        with self.assertRaises(ValueError): a.draw_assignment(((0,),),(0,1,0),np.random.default_rng(1))

    def test_active_failure_does_not_replace_random(self):
        design=a.make_design(synthetic_pool()); rows=[]
        for c in design['candidates']:
            rows.append({'condition':c,'qualified':c['id'] not in design['random_ids'],
                         'fine':{'total':{'D':2.,'B':.01}},'coarse':{'total':{'D':2.,'B':.01}}})
        chosen=a.choose_active(rows,design['random_ids'])
        self.assertEqual(len(chosen),12)
        self.assertFalse(set(chosen)&set(design['random_ids']))
        for r in rows:
            if r['condition']['stratum']=='short': r['qualified']=False
        self.assertIsNone(a.choose_active(rows,design['random_ids']))

    @patch.object(a.data,'_assert_result_system_compatible')
    @patch.object(a.data,'_load_reference_fingerprint',return_value={})
    def test_real_history_alignment_and_unknown(self,*_):
        arrays={'t':np.arange(11)*.01,'delta':np.zeros((11,10)),'omega':np.ones((11,10))}
        meta={'record_start_absolute_s':1.2001,'label':'unknown','history_eligible':True}
        result=(arrays,meta,{},np.ones(10))
        self.assertTrue(a.history_pair(result,result)['pass'])
        self.assertEqual(meta['label'],'unknown')
        later=dict(meta,event_time_s=10000.)
        self.assertEqual(a.history_pair(result,(arrays,later,{},np.ones(10))),a.history_pair(result,result))
        with self.assertRaises(ValueError): a.history_pair(result,(arrays,dict(meta,record_start_absolute_s=1.2),{},np.ones(10)))
        with self.assertRaises(ValueError): a.history_pair(result,(arrays,dict(meta,label='stable'),{},np.ones(10)))
        with self.assertRaises(ValueError): a.history_pair(result,(arrays,dict(meta,history_eligible=False),{},np.ones(10)))

    def test_canonical_history_never_reads_future(self):
        class Base: pm_star=np.ones(10)
        arrays={'delta':np.zeros((20,10)),'omega':np.ones((20,10)),'Pm':np.ones((20,10))}
        result=(arrays,{'event_time_s':1.}, {'delta':np.zeros(10)},np.ones(10))
        x,meta=a.canonical_history(result,Base())
        arrays['delta'][11:]=np.nan
        xx,mm=a.canonical_history(result,Base())
        np.testing.assert_array_equal(x,xx)
        self.assertIsNone(meta['event_time_s']); self.assertIsNone(mm['event_time_s'])

    def freeze_fixture(self,campaign):
        design=a.make_design(synthetic_pool())
        a.d.write(campaign/'design.json',design)
        a.d.write(campaign/'val_conditions.json',{'val_conditions':design['val_conditions']})
        models=[]
        for seed in (6101,6102,6103):
            for structure in ('full8','delete7'):
                folder=campaign/f'{seed}_{structure}'/'step001280'; folder.mkdir(parents=True)
                a.d.write(folder/'model.json',{'synthetic':True,'seed':seed,'structure':structure})
                (folder/'training.pt').write_bytes(b'synthetic optimizer identity')
                digest=a.d.sha(folder/'model.json')
                a.d.write(folder/'checkpoint.json',{'step':1280,'model_sha256':digest,
                    'training_sha256':a.d.sha(folder/'training.pt'),'identity':{'seed':seed,'structure':structure}})
                models.append({'seed':seed,'structure':structure,'path':str(folder/'model.json'),'sha256':digest})
        path=campaign/'frozen.json'
        a.d.write(path,{'models':models,'conditions':design['val_conditions'],
                       'conditions_sha256':a.d.sha(campaign/'val_conditions.json')})
        return path,models

    def test_val_requires_actual_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): a.verify_val_freeze(Path(tmp),None)
            with self.assertRaises(FileNotFoundError): a.verify_val_freeze(Path(tmp),Path(tmp)/'missing.json')

    def test_val_freeze_model_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign=Path(tmp); path,models=self.freeze_fixture(campaign)
            self.assertEqual(len(a.verify_val_freeze(campaign,path)['models']),6)
            Path(models[0]['path']).write_text('changed',encoding='utf-8')
            with self.assertRaises(ValueError): a.verify_val_freeze(campaign,path)

    def test_val_condition_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign=Path(tmp); path,models=self.freeze_fixture(campaign)
            manifest=a.read(path); manifest['conditions'][0]['duration_s']+=.001
            a.d.write(path,manifest)
            with self.assertRaises(ValueError): a.verify_val_freeze(campaign,path)

def run_qa():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(AcquisitionContracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return {'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors)}

if __name__=='__main__':
    raise SystemExit(0 if run_qa()['status']=='PASS' else 1)
