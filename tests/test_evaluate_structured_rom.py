import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
from evaluate_structured_rom import evaluate

class ZeroROM:
    mass=np.ones(10)
    def __init__(self, nonfinite=False, coarse_bad=False, nan_index=4): self.nonfinite=nonfinite; self.coarse_bad=coarse_bad; self.nan_index=nan_index
    def rollout(self, history, count, dt):
        out=torch.zeros((history.shape[0],count+1,33),dtype=torch.float64)
        if self.nonfinite and dt==.005: out[:,self.nan_index]=float('nan')
        if self.coarse_bad and dt==.01: out[:,:,10:20]=.01
        return out

def _fixture():
    events=[]; starts={}
    for i in range(107):
        eid=f'e{i:03d}'; x=np.zeros((811,30)); events.append({'id':eid,'group':str(i%3),'label':'unstable' if i%2 else 'stable','x':x})
        starts[eid]={'history':x[:11],'equilibrium_delta':np.zeros(10),'inertia':np.ones(10),'omega_s':2*np.pi*60,'event_time_s':None}
    return events,starts

def test_fixed_endpoint_metrics_are_on_01_grid_and_complete_denominators():
    events,starts=_fixture(); ev=evaluate(ZeroROM(),events,starts,run_finer=False)
    assert len(ev['records'])==107*5
    assert ev['complete_counts']['5']['finite_scored']==107
    assert not ev['numerical_trigger']
    assert ev['split_guard']['test_enumerated'] is False

def test_nonfinite_rows_are_retained_and_trigger_refinement():
    events,starts=_fixture(); ev=evaluate(ZeroROM(True),events,starts,run_finer=False)
    assert ev['numerical_trigger']
    assert any(r.get('finite') is False for r in ev['records'])
    assert ev['complete_counts']['1']['unavailable_or_nonfinite']>0

def test_one_finer_branch_compares_0025_to_005_without_grid_mismatch():
    events,starts=_fixture(); ev=evaluate(ZeroROM(coarse_bad=True),events,starts,run_finer=True)
    assert ev['finer_executed'] and ev['finer_numeric_gate_pass']
    assert len(ev['numerical_finer'])==107 and len(ev['numerical_scored_finer'])==107*5

class LinearROM:
    mass=np.ones(10)
    def rollout(self, history, count, dt):
        t=torch.arange(count+1,dtype=torch.float64)*dt
        out=torch.zeros((history.shape[0],count+1,33),dtype=torch.float64)
        out[:,:,10]=t
        return out

def test_linear_trajectory_metrics_use_decimated_005_grid():
    events,starts=_fixture()
    for e in events: e['x'][:,10]=(np.arange(811)-10)*.01
    ev=evaluate(LinearROM(),events,starts,run_finer=False)
    assert ev['summary']['1']['pooled'] < 1e-14

def test_short_truth_is_unavailable_not_a_nonfinite_complete_failure():
    events,starts=_fixture()
    for e in events[:11]: e['x']=e['x'][:111]
    ev=evaluate(ZeroROM(),events,starts,run_finer=False)
    assert ev['complete_counts']['2']['eligible_cohort']==96
    assert ev['summary']['2'] is not None

def test_hidden_odd_005_substep_nan_fails_genuine_numerical_qa():
    events,starts=_fixture(); ev=evaluate(ZeroROM(True,nan_index=3),events,starts,run_finer=False)
    # Set the NaN on the odd .005 sample, which cannot be visible on .01 rows.
    assert ev['model_numerical_gate_pass'] is False

class EarlyLOSThenBadROM:
    mass=np.ones(10)
    def rollout(self, history, count, dt):
        out=torch.zeros((history.shape[0],count+1,33),dtype=torch.float64)
        out[:,:,0]=4.; out[:,:,10]=2*np.pi*60*.03
        if dt==.005: out[:,400]=float('nan')
        return out

def test_nonfinite_after_converged_los_and_true_prefix_is_not_a_trigger():
    events,starts=_fixture()
    for e in events: e['x']=e['x'][:111]
    ev=evaluate(EarlyLOSThenBadROM(),events,starts,run_finer=False)
    assert ev['numerical_trigger'] is False

def test_common_frequency_failure_marks_l1_unresolved_without_refinement():
    events,starts=_fixture(); ev=evaluate(ZeroROM(coarse_bad=True),events,starts,run_finer=False)
    assert ev['model_numerical_gate_pass'] is False
    assert ev['l1_numerically_gated_counts']['numerical_unresolved']==107
