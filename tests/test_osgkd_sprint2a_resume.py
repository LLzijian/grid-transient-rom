import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import torch
import pytest
from experiments.osgkd_v2_sprint2a_20261125 import resume_fit as resume
from experiments.osgkd_v2_sprint2a_20261125 import resume_host as host


def test_scientific_strict_except_resource_cost():
    row = dict(attempt=897, alpha=.4, accepted=True, wall_s=1., cpu_s=2., peak_memory_bytes=3)
    assert resume.scientific(row) == resume.scientific({**row, 'wall_s': 99., 'cpu_s': 4., 'peak_memory_bytes': 8})
    assert resume.scientific(row) != resume.scientific({**row, 'alpha': .40000000000001})
    assert resume.scientific(row) != resume.scientific({**row, 'accepted': False})


def test_work_restores_all_counters():
    snapshot = dict(rhs_forward=123, rhs_recompute=4, encoder_events=9)
    assert resume.restore_work(snapshot).snapshot() == snapshot


def test_adam_model_rng_bitwise_restore():
    import numpy as np
    torch.set_num_threads(1)
    constants = {key: np.ones(2) for key in ('mass', 'damping', 'Sp', 'Se', 'omega_scale', 'S_m')}
    constants.update(F=np.eye(2), theta_ref=np.zeros(2), edge_index=np.array([[0], [1]]), s_coi=2.)
    for kind in ('graph', 'flat'):
        model = resume.fit.new_model(constants, kind, 24111)
        opt = resume.fit.new_optimizer(model)
        sum(p.square().sum() for p in model.parameters()).backward()
        opt.step()
        saved = dict(model=model.checkpoint(), optimizer=opt.state_dict(), torch_rng=torch.get_rng_state().clone())
        expected = torch.rand(7)
        restored, optimizer = resume.restore(saved, constants, kind, 24111)
        assert torch.equal(expected, torch.rand(7))
        assert resume.run.old.prior.p1a.tree_equal(restored.checkpoint(), saved['model'])
        assert resume.run.old.prior.p1a.tree_equal(optimizer.state_dict(), saved['optimizer'])


def test_state_stage_boundary(tmp_path):
    resume.state(tmp_path, 2, 897, 'running')
    payload = json.loads((tmp_path/'state.json').read_text())
    assert payload['slice'] == 2 and payload['next_attempt'] == 897
    resume.state(tmp_path, 2, 898, 'paused')
    assert json.loads((tmp_path/'state.json').read_text())['status'] == 'paused'


def test_repeated_seal_verifies_existing_transaction(monkeypatch):
    from types import SimpleNamespace
    model = SimpleNamespace(checkpoint=lambda: {'x': torch.tensor([1.])})
    opt = SimpleNamespace(state_dict=lambda: {'x': torch.tensor([2.])})
    work = resume.restore_work(dict(rhs_forward=12, rhs_recompute=0, encoder_events=3))
    rows, guards = [{'attempt': 64}], [{'passed': True}]
    saved = dict(attempt=1, rejected=0, model=model.checkpoint(), optimizer=opt.state_dict(), torch_rng=torch.get_rng_state().clone(), work=work.snapshot(), guard_work=work.snapshot())
    tx = dict(checkpoint={'path': 'fake.pt'}, attempts_ref='rows', guards_ref='guards')
    monkeypatch.setattr(resume, 'load', lambda ref: saved)
    monkeypatch.setattr(resume.run, 'read', lambda ref: rows if ref == 'rows' else guards)
    assert resume.verify_seal(tx, model, opt, rows, guards, work, work, 0) == (tx['checkpoint'], 'rows')
    saved['model'] = {'x': torch.tensor([3.])}
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        resume.verify_seal(tx, model, opt, rows, guards, work, work, 0)


def test_progress_has_no_event_ids(capsys):
    resume.emit_progress('graph_24111', 2, 896, 'checkpoint')
    payload = json.loads(capsys.readouterr().out)
    assert payload['next_attempt'] == 897 and 'event_ids' not in payload


def test_safe_weights_checkpoint_load(tmp_path, monkeypatch):
    path = tmp_path/'synthetic.pt'
    torch.save(dict(model={'x': torch.tensor([1.])}, torch_rng=torch.get_rng_state()), path)
    monkeypatch.setattr(resume.run, 'verified', lambda ref: path)
    assert torch.equal(resume.load({})['model']['x'], torch.tensor([1.]))


@pytest.mark.skipif(os.name != 'nt', reason='Windows file-sharing conflict')
def test_live_status_retries_actual_windows_read_lock(tmp_path, monkeypatch):
    old = tmp_path/'state.json'
    new = tmp_path/'state.json.tmp'
    old.write_bytes(b'old')
    new.write_bytes(b'new')
    replace = os.replace
    blocked = Event()

    def observed_replace(source, destination):
        try:
            return replace(source, destination)
        except PermissionError:
            blocked.set()
            raise

    monkeypatch.setattr(host.os, 'replace', observed_replace)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with old.open('rb') as reader:
            replacement = pool.submit(host.replace_with_retry, new, old)
            assert blocked.wait(1.)
            assert reader.read() == b'old'
        assert replacement.result(timeout=2.) >= 1
    assert old.read_bytes() == b'new'


@pytest.mark.skipif(os.name != 'nt', reason='Windows file-sharing conflict')
def test_live_status_read_retries_actual_windows_exclusive_handle(tmp_path, monkeypatch):
    import ctypes
    from ctypes import wintypes
    from pathlib import Path

    status = tmp_path/'state.json'
    status.write_text(json.dumps({'next_attempt': 897}), encoding='utf8')
    open_win = ctypes.windll.kernel32.CreateFileW
    open_win.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                         wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    open_win.restype = wintypes.HANDLE
    close_win = ctypes.windll.kernel32.CloseHandle
    close_win.argtypes = [wintypes.HANDLE]
    blocked = Event()
    read_bytes = Path.read_bytes

    def observed_read(path):
        try:
            return read_bytes(path)
        except PermissionError:
            blocked.set()
            raise

    monkeypatch.setattr(Path, 'read_bytes', observed_read)
    with ThreadPoolExecutor(max_workers=1) as pool:
        handle = open_win(str(status), 0x80000000, 0, None, 3, 0x80, None)
        assert handle not in (None, ctypes.c_void_p(-1).value)
        try:
            result = pool.submit(host.read, status)
            assert blocked.wait(1.)
        finally:
            close_win(handle)
        assert result.result(timeout=2.) == {'next_attempt': 897}
