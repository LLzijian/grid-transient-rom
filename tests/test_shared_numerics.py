"""Pure synthetic checks for extraction; no model, dataset or rollout runner loads."""
import ast
from pathlib import Path
import unittest

import numpy as np

from kiki.numerics import rk4_step


class SharedNumericsTests(unittest.TestCase):
    def test_matches_original_kernel_for_vectors_batches_and_backward_steps(self):
        source=Path(__file__).resolve().parents[1]/'experiments/audit_latent_order_rollout.py'
        tree=ast.parse(source.read_text(encoding='utf-8-sig'))
        original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rk4')
        namespace={'np':np}
        exec(compile(ast.Module(body=[original],type_ignores=[]),str(source),'exec'),namespace)
        def rhs(y):return -.2*y+.03*np.sin(y)
        for shape in [(4,),(3,4)]:
            y=np.arange(np.prod(shape),dtype=np.float64).reshape(shape)/10
            before=y.copy()
            for step in [.01,-.01,.0025]:
                with self.subTest(shape=shape,step=step):
                    actual=rk4_step(rhs,y,step)
                    expected=namespace['rk4'](y,step,rhs)
                    np.testing.assert_array_equal(actual.view(np.uint64),expected.view(np.uint64))
            np.testing.assert_array_equal(y,before)

    def test_fourth_order_accuracy_and_preserved_common_component(self):
        errors=[]
        for count in [4,8]:
            y=np.array([1.,2.])
            for _ in range(count):y=rk4_step(lambda state:-state,y,1/count)
            errors.append(np.max(np.abs(y-np.array([1.,2.])*np.exp(-1))))
        self.assertGreater(errors[0]/errors[1],15)
        np.testing.assert_allclose(rk4_step(lambda state:np.ones_like(state),np.zeros(3),.1),np.full(3,.1),atol=1e-16,rtol=0)

    def test_invalid_inputs_and_wrong_shape_cannot_silently_propagate(self):
        for step in [0.,np.nan,np.inf]:
            with self.subTest(step=step),self.assertRaises(ValueError):rk4_step(lambda y:-y,np.ones(2),step)
        for y in [np.array([np.nan]),np.array([1j])]:
            with self.assertRaises(ValueError):rk4_step(lambda value:-value,y,.01)
        with self.assertRaises(ValueError):rk4_step(lambda y:np.ones(1),np.ones(2),.01)
        with self.assertRaises(FloatingPointError):rk4_step(lambda y:y*np.inf,np.ones(2),.01)
