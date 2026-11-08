"""Verify preserved originals, cumulative locations, current readers and mutation scope."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

folder=Path(__file__).resolve().parent
root=folder.parents[2]
plan=json.loads((folder/'preflight.json').read_text(encoding='utf-8'))
prior=json.loads((root/plan['previous_batch']).read_text(encoding='utf-8'))
locations={x['old_path']:dict(x) for x in prior['cumulative_locations']}
for e in plan['moves']:
    assert e['old_path'] not in locations
    locations[e['old_path']]=dict(old_path=e['old_path'],current_path=e['new_path'],bytes=e['bytes'],sha256=e['sha256'],
        git_blob_exact=e['git_blob_exact'],git_text_equivalent_after_line_ending_normalization=True,
        source_record='results/repository_organization/20261108_early_history04/migration.json')
assert len(locations)==206 and len(plan['moves'])==25 and len(plan['protected'])==65
exact=0
for old,e in locations.items():
    data=(root/e['current_path']).read_bytes()
    assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'],old
    assert not (root/old).exists(),old
    blob=subprocess.check_output(['git','show',plan['source_commit']+':'+old],cwd=root)
    assert data.replace(b'\r\n',b'\n')==blob.replace(b'\r\n',b'\n'),old
    exact+=data==blob
for e in plan['protected']:
    data=(root/e['path']).read_bytes()
    assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'],e['path']
    assert e['evidence']
    for edge in e['evidence']:
        assert (root/edge['source']).exists(),edge['source']
baseline=map(json.loads,(root/'results/repository_organization/20261108_inventory01/inventory.jsonl').read_text(encoding='utf-8').splitlines())
metadata_checks=text_checks=0
for row in baseline:
    old=row['current_path']
    if old in {'README.md','reports/README.md','.gitattributes'}:continue
    path=root/locations.get(old,{}).get('current_path',old)
    info=path.stat()
    assert (info.st_size,info.st_mtime_ns)==(row['bytes'],row['mtime_ns']),old
    metadata_checks+=1
    if row['sha256']:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],old
        text_checks+=1
attrs=subprocess.check_output(['git','check-attr','-z','text','--',*[e['current_path'] for e in locations.values()]],cwd=root).decode('utf-8').split('\0')
assert attrs[2::3]==['unset']*206
receipt=dict(plan,state='MOVED_AND_BYTE_VERIFIED',verified_at_utc=datetime.now(timezone.utc).isoformat(),
    cumulative_locations=list(locations.values()),prior_receipts_modified=False,
    released_prior_protection=[],maintained_files=[])
for e in receipt['moves']:
    e.update(size_and_sha256_verified=True,old_path_absent=True,git_text_conversion='disabled',network_link_liveness_checked=False)
for e in plan['maintained_files_before']:
    receipt['maintained_files'].append(dict(path=e['path'],before_sha256=e['sha256'],after_sha256=hashlib.sha256((root/e['path']).read_bytes()).hexdigest()))
(folder/'migration.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
readers=['README.md','reports/README.md','results/repository_organization/README.md',
    'reports/history/early_e_series/README.md','reports/history/early_e_series/catalog.md',
    'reports/history/paper/README.md','reports/history/paper/catalog.md',
    'reports/maintenance/early_gates/README.md','reports/maintenance/early_gates/decision_readers.md']
readers += [f'reports/history/{topic}/{page}.md' for topic in ['private_rom','closure_discovery'] for page in ['README','catalog','protected']]
links=re.compile(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+[\"\'][^)]*)?\)')
local=fixed=0
known_old=set(locations)|{e['path'] for e in plan['protected']}
for name in readers:
    path=root/name
    text=path.read_text(encoding='utf-8-sig')
    assert all(line.rstrip()==line for line in text.splitlines()),name
    for value in links.findall(text):
        if value.startswith(('https://','http://')):
            if plan['source_commit'] in value:
                assert value.split(plan['source_commit']+'/',1)[1] in known_old,(name,value)
                fixed+=1
            continue
        if value.startswith('#'):continue
        target=(path.parent/value.split('#',1)[0]).resolve()
        assert target.is_relative_to(root) and target.exists(),(name,value)
        local+=1
before={line[3:] for line in plan['preexisting_status'].splitlines()}
after=subprocess.check_output(['git','status','--porcelain=v1','-uall'],cwd=root).decode('utf-8')
allowed=plan['allowed_changes']
for line in after.splitlines():
    name=line[3:]
    assert name in before or any(name==a or a.endswith('/') and name.startswith(a) for a in allowed),name
subprocess.run(['git','diff','--check'],cwd=root,check=True,capture_output=True)
result=dict(schema_version=1,candidates=90,moved_this_batch=25,moved_cumulative=206,protected_originals_unchanged=65,
    prior_protected_unchanged=23,newly_considered_protected=42,baseline_files_metadata_checked=metadata_checks,
    baseline_text_hashes_checked=text_checks,original_working_tree_byte_hashes_preserved=206,
    git_blob_exact=exact,git_differs_only_by_line_endings=206-exact,local_reading_links_checked=local,
    fixed_commit_reader_links_checked=fixed,mutation_scope_checked=True,git_diff_check_passed=True,
    raw_trajectories_or_weights_parsed=False,experiments_run=False,network_link_liveness_checked=False)
(folder/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
