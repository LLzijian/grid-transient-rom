"""Bounded JSON/hash-only freeze after all three registered W fits complete."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
SOURCE = '399e50d73531dd79a4613cf7e235fc4db061f6ee'
sys.path[:0] = [str(ROOT), str(ROOT/'experiments'), str(ROOT/'src')]
from closure_p3_disagreement_20261020 import own_creation_filetime_utc

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    started = time.perf_counter()
    output = BASE/'freeze01'
    output.mkdir(exist_ok=False)
    (output/'worker.json').write_text(json.dumps({'pid':os.getpid(), 'creation_filetime_utc':own_creation_filetime_utc(), 'executable':sys.executable}), encoding='utf-8')
    timer = threading.Timer(40., lambda:os._exit(124))
    timer.daemon = True
    timer.start()
    code = ROOT/'experiments/frequency_dissipation_20261021.py'
    assert sha(code) == 'f83edeedf24c40078dd92e2bdaaf3af9739b838027b4179167341218d6188c7f'
    receipts = {}
    for seed in (6101,6102,6103):
        path = BASE/f'native_train_{seed}_w/execution.json'
        r = json.loads(path.read_text(encoding='utf-8'))
        assert r['status']=='COMPLETED' and r['source_commit']==SOURCE
        assert r['source_before']==r['source_after']
        assert r['inner']['cleanup_verified'] and r['inner']['fixture_identity_after']=='GONE'
        assert r['outer']['exit_code']==r['inner']['exit_code']==0
        receipts[str(seed)] = sha(path)
    assert not (BASE/'native_active.lock').exists()
    module = runpy.run_path(str(code), run_name='w8_freeze_metadata')
    result = module['freeze_endpoints']()
    assert result['endpoint_count']==3 and result['test_read'] is False and result['val_read'] is False
    assert sha(code) == 'f83edeedf24c40078dd92e2bdaaf3af9739b838027b4179167341218d6188c7f'
    summary = {'status':'COMPLETED', 'source_commit':SOURCE, 'manifest_sha256':sha(BASE/'frozen_evaluation_endpoints.json'), 'endpoint_count':3, 'training_receipts_sha256':receipts, 'seconds':time.perf_counter()-started, 'self_limit_seconds':40, 'predictions_or_updates':0}
    (output/'result.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    timer.cancel()
    print(json.dumps(summary))

if __name__=='__main__':
    main()
