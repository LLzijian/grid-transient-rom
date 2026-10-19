"""Production rejection guards; success identities come from actual probes."""
import copy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import fom_execution_environment_20261020 as env

class EnvironmentContracts(unittest.TestCase):
    fom=None
    gpu=None
    def guarded_refusal(self,expected):
        calls=[]
        with patch.object(env,'_fom_identity',return_value=copy.deepcopy(self.fom)):
            with self.assertRaises((RuntimeError,ValueError)):
                env.guarded_call('short',lambda identity:calls.append(identity),expected=expected)
        self.assertEqual(calls,[])

    def test_category_dispatch_and_unknown(self):
        for k,role in env.ALLOWED.items(): self.assertEqual(env.executable_for(k),env.FOM_PY if role=='fom' else env.GPU_PY)
        with self.assertRaises(ValueError): env.guarded_call('condition',lambda _:self.fail('callback entered'))

    def test_wrong_executable(self):
        with patch.object(sys,'executable',str(env.GPU_PY)):
            with self.assertRaises(RuntimeError): env.guarded_call('short',lambda _:self.fail('callback entered'))

    def test_wrong_prefix(self):
        with patch.object(sys,'executable',str(env.FOM_PY)),patch.object(sys,'prefix',str(ROOT/'wrong-prefix')):
            with self.assertRaises(RuntimeError): env.guarded_call('short',lambda _:self.fail('callback entered'))

    def test_missing_package_before_callback(self):
        calls=[]
        with patch.object(sys,'executable',str(env.FOM_PY)),patch.object(sys,'prefix',str(env.FOM_PY.parent.parent)),patch.object(env,'_load_dependency',side_effect=ImportError('synthetic missing numpy')):
            with self.assertRaises(ImportError): env.guarded_call('short',lambda i:calls.append(i),expected=self.fom)
        self.assertEqual(calls,[])

    def test_child_mismatch(self):
        child=copy.deepcopy(self.fom); child['executable_sha256']='0'*64
        with self.assertRaises(RuntimeError): env.compare_child(self.fom,child)

    def test_source_mutation(self):
        altered=copy.deepcopy(self.fom); altered['source']['production']['sha256']='0'*64
        self.guarded_refusal(altered)

    def test_case_mutation(self):
        altered=copy.deepcopy(self.fom); altered['case']['sha256']='0'*64
        self.guarded_refusal(altered)

    def test_empty_partial_or_wrong_role_expected(self):
        for expected in ({},{'role':'fom'},self.gpu): self.guarded_refusal(expected)

    def test_gpu_unavailable_or_malformed_identity(self):
        altered=copy.deepcopy(self.gpu); altered['torch']['cuda_available']=False
        with patch.object(env,'_gpu_identity',return_value=altered):
            with self.assertRaises(ValueError): env.preflight('gpu')
        altered=copy.deepcopy(self.fom); altered['executable_sha256']=None
        with self.assertRaises(ValueError): env.validate_identity(altered,self.fom,'fom')

def run_qa(real_fom,real_gpu):
    env.validate_identity(real_fom,real_fom,'fom'); env.validate_identity(real_gpu,real_gpu,'gpu')
    EnvironmentContracts.fom=real_fom; EnvironmentContracts.gpu=real_gpu
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EnvironmentContracts))
    return {'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),'log':stream.getvalue(),
            'success_identities_from_real_probes':True,'failure_before_condition_callback':True}
