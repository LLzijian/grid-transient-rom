"""Standalone integration and exact coefficient rendering, without real data."""
import importlib.util
import sys
import numpy as np
import pytest
from task4.paths import ROOT
import task4.symbolic_runtime as runtime


@pytest.fixture
def export_script(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('export_test_module',ROOT/'scripts/export_model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_standalone_rk4_uses_once_only_initialization(export_script,monkeypatch):
    monkeypatch.setitem(sys.modules,'runtime',runtime)
    namespace={'__name__':'imported_simulator'}
    exec(compile(export_script.SIMULATOR,'simulate.py','exec'),namespace)
    class Linear:
        calls=0
        def initial(self,h):
            self.calls+=1
            assert h.shape==(1,11,1,3)
            return np.array([[1.,2.]])
        def field(self,x):return -x
    model=Linear()
    t,z=namespace['simulate'](model,np.zeros((11,1,3)),seconds=.05,dt=.005)
    assert model.calls==1 and t[0]==.1 and len(t)==11
    np.testing.assert_allclose(z[0,-1],np.array([1.,2.])*np.exp(-.05),rtol=3e-12,atol=1e-14)


def test_equations_preserve_small_nonzero_coefficients(export_script):
    spec={'coordinates':{'time_unit':'s'},'structure':{'kinematics':'delta_dot=pi/mass'},
          'functions':{'port':{'powers':[[0,0],[1,2]],'coefficients':[[0.,1e-20],[-.125,0.]],'outputs':2}}}
    rendered=export_script.equations(spec)
    assert 'F[0] = -0.125*x0*x1^2' in rendered
    assert 'F[1] = '+format(1e-20,'.17g') in rendered


def test_packaging_checks_authorization_before_source_read(export_script,monkeypatch,tmp_path):
    def denied(synthetic):raise PermissionError('expected gate')
    monkeypatch.setattr(export_script,'check_authorized',denied)
    monkeypatch.setattr(export_script,'load_checkpoint',lambda path:pytest.fail('source read before authorization'))
    with pytest.raises(PermissionError):export_script.package('missing.pt',tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()
