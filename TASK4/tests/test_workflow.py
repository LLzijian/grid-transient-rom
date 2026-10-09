import json,sys
from pathlib import Path
import pytest
from task4.paths import ROOT
from task4.workflow import Workflow
from task4.runner import run_managed

def setup(tmp_path):
    root=tmp_path/'TASK4';(root/'state').mkdir(parents=True);(root/'reports').mkdir()
    # Build a clean state rather than reusing a potentially running real project state.
    state={'schema':1,'status':'NOT_STARTED','route_revision':1,'authorization':None,'active_loop':None,'active_jobs':{},'accepted_candidate':None,'accepted_code_ref':None,'active_candidate':None,'complementary_candidates':[],'pending_direction':None,'budget':{'cap_seconds':None,'spent_seconds':0.},'loops':{},'history':[]}
    (root/'state/research.json').write_text(json.dumps(state))
    (root/'reports/spec.md').write_text('test fixture protocol');(root/'reports/result.md').write_text('test fixture report')
    (root/'candidate.pt').write_text('test fixture, not model');return root,Workflow(root)

def test_no_implicit_start(tmp_path):
    root,w=setup(tmp_path)
    with pytest.raises(PermissionError):w.open_loop('T4_0001','reports/spec.md',20)

def test_unused_budget_returned(tmp_path):
    root,w=setup(tmp_path);w.authorize('synthetic test instruction',100)
    w.open_loop('T4_0001','reports/spec.md',80);w.account(5,'synthetic fixture cost')
    w.close_loop('reject','reports/result.md')
    assert w.load()['budget']['spent_seconds']==5
    assert w.load()['loops']['T4_0001']['released_seconds']==75
    w.open_loop('T4_0002','reports/spec.md',95)

def test_accept_then_rollback_pointer(tmp_path):
    root,w=setup(tmp_path);w.authorize('synthetic test',100);w.open_loop('T4_0001','reports/spec.md',20)
    w.close_loop('accept','reports/result.md','candidate.pt','snapshot_001')
    w.open_loop('T4_0002','reports/spec.md',20);w.close_loop('reject','reports/result.md')
    assert w.load()['accepted_candidate']=='candidate.pt' and w.load()['active_candidate']=='candidate.pt'
    assert (root/'candidate.pt').exists()

def test_direction_pauses_until_explicit_approval(tmp_path):
    root,w=setup(tmp_path);w.authorize('synthetic test',100)
    (root/'reports/proposal.md').write_text('new direction fixture')
    w.propose_direction('reports/proposal.md')
    assert w.load()['status']=='WAITING_USER_DIRECTION_APPROVAL'
    with pytest.raises(PermissionError):w.open_loop('T4_0001','reports/spec.md',10)
    with pytest.raises(RuntimeError):w.authorize('generic continue is not approval',100)
    w.approve_direction('explicitly approve proposal fixture')
    assert w.load()['route_revision']==2 and w.load()['status']=='READY'
    w.open_loop('T4_0001','reports/spec.md',10)

def test_runner_failure_is_charged_and_preserved(tmp_path):
    root,w=setup(tmp_path);w.authorize('synthetic runner test',100);w.open_loop('T4_0001','reports/spec.md',20)
    out=run_managed([sys.executable,'-c','print("fixture failure");raise SystemExit(3)'],'T4_0001',10,root=root)
    assert out['exit_code']==3 and w.load()['budget']['spent_seconds']>0
    assert not w.load()['active_jobs'] and Path(out['log']).exists()

def test_reopening_closed_loop_refused(tmp_path):
    root,w=setup(tmp_path);w.authorize('synthetic test',100);w.open_loop('T4_0001','reports/spec.md',20);w.close_loop('inconclusive','reports/result.md')
    with pytest.raises(ValueError):w.open_loop('T4_0001','reports/spec.md',20)


def test_implementation_acceptance_does_not_invent_model(tmp_path):
    root,w=setup(tmp_path);w.authorize('synthetic test',100);w.open_loop('T4_0001','reports/spec.md',20)
    w.close_loop('accept','reports/result.md',code_ref='engine_snapshot')
    assert w.load()['accepted_candidate'] is None and w.load()['accepted_code_ref']=='engine_snapshot'

def test_pending_direction_cannot_run_process(tmp_path):
    root,w=setup(tmp_path);w.authorize('synthetic test',100);(root/'reports/p.md').write_text('proposal')
    w.propose_direction('reports/p.md')
    with pytest.raises(PermissionError):run_managed([sys.executable,'-c','print(1)'],'T4_0001',10,root=root)
