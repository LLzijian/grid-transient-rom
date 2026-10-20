"""Stdlib-only synthetic extraction QA; never opens active study artifacts."""
import copy
from pathlib import Path
import runpy
import sys
import json
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import extract_quartic_memory as m


def manifest():
    rows=[]
    for seed in m.SEEDS:
        for v in m.VARIANTS:
            p=m.role_path(seed,v)
            rows.append({'seed':seed,'variant':v,'stage_step':1024,'model_path':str(p),'training_path':str(p.parent/'training.pt'),
                'checkpoint_path':str(p.parent/'checkpoint.json'),'model_sha256':'a'*64,'training_sha256':'b'*64,'checkpoint_sha256':'c'*64})
    return {'endpoints':rows,'endpoint_count':6,'schedule_sha256':m.SCHEDULE_SHA,'test_read':False,'val_read':False}


def payload(quartic=False,reorder=False,value=0.):
    old={'channel':'node','expression':{'op':'mul','args':[{'op':'var','name':'p'},{'op':'var','name':'q'}]}}
    terms=[old]+([copy.deepcopy(m.QUARTIC)] if quartic else [])
    coefficients=[.2]+([value] if quartic else [])
    if reorder:terms.reverse();coefficients.reverse()
    return {'terms':terms,'coefficients':coefficients,'state_dict':{'coefficients':copy.deepcopy(coefficients),'local_head.weight':[[.1,.3]]},
            'trainable_parameters':['coefficients','local_head.weight']}


class ExtractionTest(unittest.TestCase):
    def test_root_and_base_constants(self):
        self.assertEqual(m.ROOT,Path(__file__).resolve().parents[5])
        self.assertEqual(m.BASE,Path(__file__).resolve().parents[1])
        self.assertEqual(m.BASE,m.ROOT/'private_data/closure_mechanism_20261019/P2/quartic_memory01')

    def test_source_manifest_exact_roles_paths_stage_schedule(self):
        source=manifest();self.assertEqual(len(m.checked_manifest(source)),6)
        for mutate in ('schedule','path','count','nontrain','duplicate'):
            bad=copy.deepcopy(source)
            if mutate=='schedule':bad['schedule_sha256']='0'*64
            elif mutate=='path':bad['endpoints'][0]['model_path']=str(m.BASE/'models/6101_N/checkpoints/step000512/model.json')
            elif mutate=='count':bad['endpoints'].pop()
            elif mutate=='nontrain':bad['val_read']=True
            else:bad['endpoints'][1]=bad['endpoints'][0]
            with self.assertRaises(ValueError):m.checked_manifest(bad)

    def test_term_key_order_and_shared_parameter_alignment(self):
        pn=payload();cn=copy.deepcopy(pn);cn['coefficients'][0]+=.1;cn['state_dict']['coefficients']=copy.deepcopy(cn['coefficients'])
        pq=payload(True,True,0.);cq=payload(True,True,.4)
        cq['coefficients'][1]+=.2;cq['state_dict']['coefficients']=copy.deepcopy(cq['coefficients'])
        out=m.shared_update(pn,cn,pq,cq)
        self.assertEqual(out['shared_scalar_count'],3)
        self.assertAlmostEqual(out['N_update_l2'],.1);self.assertAlmostEqual(out['Q_common_update_l2'],.2)
        self.assertAlmostEqual(out['QminusN_update_l2'],.1)
        bad=copy.deepcopy(cq);bad['coefficients'][0]=.7
        with self.assertRaises(ValueError):m.parameter_map(bad)

    def test_tested_pure_functions_cohorts_units_null_propagation(self):
        # Execute only the old test module definitions, not its unittest main.
        old=m.ROOT/'tests/test_extract_spatial_extrema_20261020.py'
        fixture=runpy.run_path(str(old))['fixture']
        e,groups,full=fixture();p=m.pure();p['validate_evidence'](e,full)
        self.assertEqual(len(groups['original366']),366);self.assertEqual(len(groups['new19']),19)
        self.assertEqual(sum(x.startswith('newtrain_') for x in groups['original366']),20)
        self.assertAlmostEqual(p['risk']([1.,2.,3.]),2.2)
        original=p['summarize'](e,groups['common385'])
        full_summary=m.summarize(e,groups['common385'])
        for h in ('1','2','5','8'):self.assertEqual(full_summary['stable'][h]['ceil_risk'],2.)
        self.assertIn('coi.stable_1.pooled_normalized_rmse',m.metric_values(full_summary))
        self.assertEqual(original['classification_15s']['unstable_denominator'],43)
        self.assertAlmostEqual(original['stable']['5']['mechanical_power']['rmse_MW'],10.)
        e['extras'][0]['scientific_numeric_8s']=False
        changed=p['summarize'](e,groups['common385'])
        self.assertIsNone(changed['stable']['5']['ceil_risk'])
        self.assertEqual(changed['spatial_actual15']['all']['combined'],3.)
        e['extras'][0]['classification_15s']['los_pair_pass']=False
        self.assertIsNone(p['summarize'](e,groups['common385'])['spatial_actual15']['all']['combined'])

    def test_protocol_fact_booleans_no_tolerance_and_missing_numeric(self):
        fixture=runpy.run_path(str(m.ROOT/'tests/test_extract_spatial_extrema_20261020.py'))['fixture']
        e,groups,_=fixture();summary=m.pure()['summarize'](e,groups['common385'])
        models={f'{s}_{v}':{'common385':copy.deepcopy(summary)} for s in m.SEEDS for v in m.VARIANTS}
        for s in (6101,6102):models[f'{s}_Q']['common385']['classification_15s']['missed']-=1
        facts=m.gate_facts(models)
        self.assertTrue(facts['at_least_two_including6101'])
        models['6103_Q']['common385']['stable']['5']['ceil_risk']+=1e-15
        self.assertFalse(m.gate_facts(models)['per_seed']['6103']['no_increase']['stable_5.ceilrisk'])
        models['6101_Q']['common385']['stable']['5']['ceil_risk']=None
        self.assertFalse(m.gate_facts(models)['per_seed']['6101']['all_relevant_numeric_resolved'])

    def test_complete_synthetic_checkpoint_tree_and_training_statistics(self):
        original_base=m.BASE
        with tempfile.TemporaryDirectory() as temporary:
            try:
                m.BASE=Path(temporary);roles={};sources={s:{'model_sha256':'d'*64,'training_sha256':'e'*64} for s in m.SEEDS}
                for seed in m.SEEDS:
                    for v in m.VARIANTS:
                        folder=m.role_path(seed,v).parent;trace=[]
                        for step in range(1,1025):
                            raw=0. if step%3==0 else 1. if step%3==1 else -1.
                            q={'raw_gradient':raw if v=='Q' else None,'gradient':.5*raw if v=='Q' else None,
                               'before_update':(step-1)*.0001 if v=='Q' else None,'before':step*.0001 if v=='Q' else None,
                               'after':step*.0001 if v=='Q' else None,'adam_delta':.0001 if v=='Q' else None,
                               'projected_update':.0001 if v=='Q' else None,'projected':False}
                            trace.append({'stage_step':step,'horizon_s':(1.,2.,5.)[(step-1)%3],'loss':float(step),
                                'gradient_norm':101. if step%2 else 99.,'components':{k:float(step) for k in ('risk','mean','local','latent_consistency','torque','complexity')},
                                'learning_rates':{k:x*(.3 if step>512 else 1.) for k,x in {'skeleton':.0009,'closure':.00009,'encoder':.000027}.items()},'quartic':q})
                        for step in range(0,1025,32):
                            terms=copy.deepcopy(m.N_TERMS)+([copy.deepcopy(m.QUARTIC)] if v=='Q' else [])
                            coeff=[.2+i*.01+step*.00001 for i in range(7)]+([step*.0001] if v=='Q' else [])
                            if v=='Q':terms.reverse();coeff.reverse()
                            data={'terms':terms,'coefficients':coeff,'state_dict':{'coefficients':copy.deepcopy(coeff),'local_head.weight':[[.1+step*.00001,.3]]},'trainable_parameters':['coefficients','local_head.weight']}
                            p=folder.parent/f'step{step:06d}/model.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(data))
                        identity={'seed':seed,'variant':v,'schedule_sha256':m.SCHEDULE_SHA,'source_model_sha256':'d'*64,'source_training_sha256':'e'*64}
                        (folder/'checkpoint.json').write_text(json.dumps({'step':1024,'identity':identity,'trace':trace}))
                        roles[(seed,v)]={'model_sha256':m.sha(folder/'model.json'),'checkpoint_sha256':m.sha(folder/'checkpoint.json')}
                summary,intervals=m.training_facts(roles,sources)
                q=summary['6101_Q'];self.assertEqual(q['phase_counts'],{1.:342,2.:341,5.:341})
                self.assertEqual(q['clip_count'],512);self.assertEqual(len(intervals['6101']),32)
                stats=q['training_statistics'];self.assertEqual(stats['by_horizon']['1.0']['actual_rows'],342)
                self.assertEqual(stats['by_horizon']['1.0']['first64']['loss_mean'],95.5)
                self.assertEqual(stats['by_horizon']['1.0']['first64']['component_means']['risk'],95.5)
                a=stats['quartic_scalar_statistics']['all1024']['raw_gradient']
                self.assertEqual((a['finite_count'],a['positive_count'],a['negative_count'],a['zero_count']),(1024,342,341,341))
                self.assertEqual(stats['quartic_scalar_statistics']['stage513_1024']['after']['actual_rows'],512)
                self.assertAlmostEqual(q['saved_a_by_step']['1024'],.1024)
                meta_path=m.role_path(6101,'N').parent/'checkpoint.json';original=meta_path.read_text()
                for fault,pattern in (('phase','phase mismatch'),('lr','learning-rate'),('component','components')):
                    meta=json.loads(original)
                    if fault=='phase':meta['trace'][0]['horizon_s']=2.
                    elif fault=='lr':meta['trace'][512]['learning_rates']['closure']=.00009
                    else:meta['trace'][0]['components']['risk']=float('nan')
                    meta_path.write_text(json.dumps(meta));roles[(6101,'N')]['checkpoint_sha256']=m.sha(meta_path)
                    with self.assertRaisesRegex(ValueError,pattern):m.training_facts(roles,sources)
                meta_path.write_text(original);roles[(6101,'N')]['checkpoint_sha256']=m.sha(meta_path)
                bad=json.loads((m.role_path(6101,'Q').parent.parent/'step000032/model.json').read_text())
                bad['terms'][1]=copy.deepcopy(bad['terms'][0])
                with self.assertRaises(ValueError):m.validate_saved_terms(bad,'Q')
                bad_model=json.loads((m.role_path(6101,'Q').parent.parent/'step000032/model.json').read_text())
                index=next(i for i,t in enumerate(bad_model['terms']) if m.term_key(t)==m.term_key(m.QUARTIC))
                bad_model['coefficients'][index]=-.2;bad_model['state_dict']['coefficients'][index]=-.2
                bad_path=m.role_path(6101,'Q').parent.parent/'step000032/model.json';bad_path.write_text(json.dumps(bad_model))
                with self.assertRaises(ValueError):m.training_facts(roles,sources)
            finally:m.BASE=original_base

    def test_local_source_hash_rejected(self):
        local={'source_sha256':'a'*64,'origin':{},'quotient':{},'linear_field':{},'energy_curvature':{}}
        self.assertEqual(m.checked_local(local,'a'*64)['source_sha256'],'a'*64)
        with self.assertRaises(ValueError):m.checked_local(local,'b'*64)


if __name__=='__main__':unittest.main()
