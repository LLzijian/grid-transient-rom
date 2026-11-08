"""Data-free checks, deliberately NOT executed during implementation preparation.

Run this file separately from the fixed known-ROM scientific qualification CLI.
"""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from weak_force_identification_20261029 import (
    Observations, WeakMatrices, build_matrices, fixture_spec, fixed_windows,
    incidence, kkt_report, raw_coordinates, sensitivity, simpson, solve_once,
    test_functions as functions, weak_window, qualify_fixtures,
)


class WeakForceChecks(unittest.TestCase):
    def test_windows_and_real_interval_coverage(self):
        windows,cov = fixed_windows(834)
        self.assertEqual(windows,[(10+100*j,110+100*j) for j in range(8)])
        self.assertEqual(cov['eligible_intervals'],800)
        windows,cov = fixed_windows(234)
        self.assertEqual(windows,[(10,110),(110,210),(210,230)])
        self.assertEqual(cov['unused_interval_ranges'],[[230,233]])
        self.assertEqual(fixed_windows(30)[0],[])
        self.assertEqual(fixed_windows(31)[0],[(10,30)])

    def test_analytic_functions_and_time_scaling(self):
        for n in (20,100):
            T=n*.01
            s=np.arange(n+1)/n
            p0,d0,dd0=functions(n)[0]
            p1,d1,dd1=functions(n)[1]
            np.testing.assert_allclose(p0,s*s*(1-s)**2,atol=1e-15)
            np.testing.assert_allclose(d0,(2*s-6*s*s+4*s**3)/T,atol=1e-14)
            np.testing.assert_allclose(dd0,(2-12*s+12*s*s)/T**2,atol=1e-12)
            np.testing.assert_allclose(p1,(2*s-1)*p0,atol=1e-15)
            np.testing.assert_allclose(d1,(2*p0+(2*s-1)*d0*T)/T,atol=1e-14)
            np.testing.assert_allclose(dd1,(4*d0*T+(2*s-1)*dd0*T*T)/T**2,atol=1e-12)
            for p,d in ((p0,d0),(p1,d1)):
                np.testing.assert_array_equal(p[[0,-1]],0)
                np.testing.assert_array_equal(d[[0,-1]],0)

    def test_same_endpoint_simpson(self):
        t=np.arange(101)*.01
        for stride in (1,2):
            self.assertAlmostEqual(float(simpson(t**3,stride)),.25,14)
        with self.assertRaises(ValueError):
            simpson(np.ones(100))

    def test_only_observable_width_and_unique_events(self):
        with self.assertRaises(ValueError):
            Observations('r_is_forbidden',False,np.zeros((111,40)))
        s,_,_=fixture_spec()
        e=Observations('same',False,np.zeros((111,30)))
        with self.assertRaises(ValueError):
            build_matrices([e,e],s)

    def test_no_window_is_not_zero_information_row(self):
        s,_,_=fixture_spec()
        m=build_matrices([Observations('short_unstable',True,np.zeros((15,30))),
                          Observations('equilibrium',False,np.zeros((211,30)))],s)
        self.assertEqual(m.A.shape,(40,47))
        self.assertEqual(m.coverage[0]['windows'],[])
        self.assertEqual(m.coverage[1]['unique_used_nodes'],201)
        self.assertEqual(m.coverage[1]['used_intervals'],200)
        report=sensitivity(m)
        self.assertIsNone(report['unstable']['E_X'])
        self.assertIsNone(report['unstable']['passed'])
        self.assertTrue(report['passed'])
        self.assertTrue((m.A==0).all())
        self.assertTrue((m.b==0).all())

    def test_nonuniform_mass_order_and_C_gate_derivative(self):
        s,_,_=fixture_spec()
        t=np.arange(101)*.01
        x=np.zeros((101,30))
        x[:,:10]=t[:,None]*np.arange(10)[None,:]/9
        x[:,10:20]=np.arange(10)[None,:]/9
        X,_=weak_window(x,s,'C')
        B=incidence(); v=x[:,10:20]*s.mass; f=s.force(x)
        # Independent explicit per-edge ten-vector actions; retains g_dot.
        psi,dp,_=functions(100)[0]
        for e in (0,8,44):
            b=B[:,e]; eta=x[:,:10]@b
            g=(1+np.cos(eta))/2
            gd=-.5*np.sin(eta)*(x[:,10:20]@b)
            action=np.outer(f@b,b)*(psi*g)[:,None]
            action+=np.outer(v@b,b)*(dp*g+psi*gd)[:,None]
            np.testing.assert_allclose(X[:10,2+e],-simpson(action),atol=1e-15)

    def test_strict_boundary_no_epsilon_alpha_zero_legal(self):
        z=np.r_[1.,1.,np.zeros(45)]
        self.assertEqual(raw_coordinates(z)['a_bar'],[0.]*45)
        for j in (0,1):
            bad=z.copy(); bad[j]=0
            with self.assertRaises(ValueError): raw_coordinates(bad)
        for val in (-1.,float('nan'),float('inf')):
            bad=z.copy(); bad[2]=val
            with self.assertRaises(ValueError): raw_coordinates(bad)

    def test_kkt_checks_primal_and_dual_independently(self):
        A=np.eye(3); b=np.array([1.,-2.,3.])
        self.assertTrue(kkt_report(A,b,np.array([1.,0.,3.]))['passed'])
        self.assertFalse(kkt_report(A,b,np.array([1.,1.,3.]))['passed'])
        self.assertFalse(kkt_report(A,b,np.array([1.,-1.,3.]))['passed'])

    def test_single_public_call_no_iteration_fabrication(self):
        A=np.eye(47); b=np.ones(47)
        m=WeakMatrices(A,b,A.copy(),b.copy(),np.zeros(47,bool),[])
        events=[]
        with patch('scipy.optimize.nnls',return_value=(b.copy(),0.)) as solver:
            result=solve_once(m,nnls_event=lambda status,detail:events.append(status))
        solver.assert_called_once()
        self.assertEqual(solver.call_args.kwargs,{'maxiter':4700})
        self.assertFalse(result['iterations']['available'])
        self.assertTrue(result['success'])
        self.assertEqual(result['rank'],47)
        self.assertEqual(events,['attempt','completed'])

    def test_failed_solver_preserves_actual_attempt_identity(self):
        A=np.eye(47); b=np.ones(47)
        m=WeakMatrices(A,b,A.copy(),b.copy(),np.zeros(47,bool),[])
        events=[]
        with patch('scipy.optimize.nnls',side_effect=RuntimeError('iteration limit')) as solver:
            result=solve_once(m,nnls_event=lambda status,detail:events.append(status))
        solver.assert_called_once()
        self.assertFalse(result['success'])
        self.assertEqual(events,['attempt','failed'])

    def test_partial_failure_is_durable_and_existing_output_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'new_fixture_suite'
            with patch('scipy.integrate.solve_ivp',side_effect=RuntimeError('injected reference failure')):
                result=qualify_fixtures(output)
            self.assertFalse(result['passed'])
            saved=json.loads((output/'result.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['exception']['message'],'injected reference failure')
            self.assertEqual(saved['stages']['A_reference_0'],'failed')
            self.assertEqual(saved['stages']['C_reference_0'],'not_run')
            self.assertEqual(saved['artificial_nnls_attempts'],0)
            journal=[json.loads(line) for line in (output/'journal.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertIn('stage_failed',[row['event'] for row in journal])
            with self.assertRaises(FileExistsError):
                qualify_fixtures(output)


if __name__=='__main__':
    unittest.main()
