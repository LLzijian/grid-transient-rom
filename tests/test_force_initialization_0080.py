import importlib.util,sys
from pathlib import Path
import numpy as np,pytest
torch=pytest.importorskip('torch');ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
spec=importlib.util.spec_from_file_location('i80',ROOT/'experiments/audit_force_initialization_0080.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Toy:
 def rk4(self,x,dt):return x+dt
 def begin_graph_cache(self):pass
 def end_graph_cache(self):pass
def test_masked_autonomous_rollout_stops_at_lengths():
 x=torch.zeros((2,33),dtype=torch.float64);y=m.rollout(Toy(),x,np.array([1,3]),.01);assert np.allclose(y[0,1],.01) and np.isnan(y[0,2]).all() and np.allclose(y[1,3],.03)
def test_point_and_tail_equal_event_and_nonfinite_numeric():
 p=np.zeros((2,30));t=p.copy();t[0,20]=1;assert m.point(p,t,np.ones(10),np.ones(10))[0]==pytest.approx(.1)
 assert m.risk([0,1,2,3])['tail35']==pytest.approx(1.5)
 with pytest.raises(FloatingPointError):m.numeric(np.full((1,3,33),np.nan),np.zeros((1,2,33)),np.array([1]),np.ones(10),np.ones(10))

def test_numeric_common_uses_raw_speed_not_coi_projected_speed():
 fine=np.zeros((1,3,33)); coarse=np.zeros((1,2,33)); fine[:,::2,10:20]=2.
 row=m.numeric(fine,coarse,np.array([1]),np.ones(10),np.ones(10))[0]
 assert row['coi']==pytest.approx(0.) and row['common']>0

def test_nonfinite_scored_point_and_risk_fail():
 with pytest.raises(FloatingPointError):m.point(np.full((1,30),np.nan),np.zeros((1,30)),np.ones(10),np.ones(10))
 with pytest.raises(FloatingPointError):m.risk([np.nan,1])

def test_global_top35_not_batch_tail():
 v=np.arange(346,dtype=float); r=m.risk(v)
 assert r['tail35']==pytest.approx(np.mean(v[-35:]))
 assert r['risk']==pytest.approx(.8*v.mean()+.2*v[-35:].mean())

def test_only_initial_q_changes_and_future_is_autonomous():
 class Coupled:
  def rk4(self,x,dt):
   out=x.clone();out[:,10:13]+=dt*x[:,30:33];return out
 a=torch.zeros((2,33),dtype=torch.float64); b=a.clone();b[:,30:]=1
 ya=m.rollout(Coupled(),a,np.array([1,3]),.01)
 yb=m.rollout(Coupled(),b,np.array([1,3]),.01)
 np.testing.assert_array_equal(ya[:,0,:30],yb[:,0,:30])
 assert yb[1,3,10]==pytest.approx(.03) and ya[1,3,10]==0

def test_batch32_equals_single_batch_with_variable_lengths():
 x=torch.zeros((35,33),dtype=torch.float64); lengths=np.arange(35)%3+1
 np.testing.assert_allclose(m.rollout32(Toy(),x,lengths,.01),m.rollout(Toy(),x,lengths,.01),equal_nan=True)
