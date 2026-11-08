"""Validate this report archive and current reading views without scientific execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

folder=Path(__file__).resolve().parent
root=folder.parents[2]
plan=json.loads((folder/'prepared.json').read_text(encoding='utf-8'))
prior=json.loads((root/plan['previous_batch']).read_text(encoding='utf-8'))
locations={e['old_path']:dict(e) for e in prior['cumulative_locations']}
for e in plan['moves']:
    assert e['old_path'] not in locations
    locations[e['old_path']]=dict(old_path=e['old_path'],current_path=e['new_path'],bytes=e['bytes'],sha256=e['sha256'],
        git_blob_exact=e['git_blob_exact'],git_text_equivalent_after_line_ending_normalization=True,
        source_record='results/repository_organization/20261109_history05/migration.json')
assert len(locations)==416 and len(plan['moves'])==210 and len(plan['protected'])==326
for old,e in locations.items():
    data=(root/e['current_path']).read_bytes()
    assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'],old
    assert not (root/old).exists(),old
for e in plan['protected']:
    data=(root/e['path']).read_bytes()
    assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'],e['path']
    assert e['evidence']
    for edge in e['evidence']:assert (root/edge['source']).exists(),edge['source']
for e in plan['snapshots']:
    data=(root/e['new_path']).read_bytes()
    assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256']
attr_paths=[e['current_path'] for e in locations.values()]+[e['new_path'] for e in plan['snapshots']]
attrs=subprocess.check_output(['git','check-attr','-z','--stdin','text'],input=('\0'.join(attr_paths)+'\0').encode('utf-8'),cwd=root).decode('utf-8').split('\0')
assert attrs[2::3]==['unset']*417
baseline=map(json.loads,(root/'results/repository_organization/20261108_inventory01/inventory.jsonl').read_text(encoding='utf-8').splitlines())
mutable={'.gitattributes','README.md','reports/README.md','results/README.md','tools/research_harness.py',
    'campaigns/gftd_kuramoto_discovery_20261106.md','campaigns/README.md','docs/engineering_loop.md','templates/run_closure.md'}
metadata_checks=hash_checks=0
for row in baseline:
    old=row['current_path']
    if old in mutable:continue
    p=root/locations.get(old,{}).get('current_path',old)
    info=p.stat()
    assert (info.st_size,info.st_mtime_ns)==(row['bytes'],row['mtime_ns']),old
    metadata_checks+=1
    if row['sha256']:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],old
        hash_checks+=1
receipt=dict(plan,state='MOVED_AND_BYTE_VERIFIED',cumulative_locations=list(locations.values()),prior_receipts_modified=False,
    released_prior_protection=[e['path'] for e in prior['protected'] if e['path'] in {x['old_path'] for x in plan['moves']}])
(folder/'migration.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
spec=importlib.util.spec_from_file_location('archive_navigation',root/'tools/research_harness.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
registered=cli.run_navigation(root)
assert registered['valid'] and len(registered['runs'])==21,registered['issues']
assert cli.check_root(root)['valid']
storage=json.loads((folder/'storage_associations.json').read_text(encoding='utf-8'))
storage_summary=json.loads((folder/'storage_association_summary.json').read_text(encoding='utf-8'))
assert len(storage)==339 and storage_summary['public_result_groups']==96
assert len({e['path'] for e in storage})==339
assert sum(e['kind']=='migrated_local_storage' for e in storage)==216
assert sum(e['kind']=='protected_or_restricted_local_root' for e in storage)==27
assert all(e['reference_clues'] or e['registered_materials'] for e in storage if e['kind']=='public_results')
assert len({name for e in storage for name in e['registered_run_contexts']})==21
assert all(e['retention']=='preserve' and e['producer_job'] is None and not e['contents_checked'] for e in storage)
for e in storage:
    for edge in e['reference_clues']:
        assert (root/edge['source_current_path']).is_file(),edge['source_current_path']
        assert not edge['runtime_use_verified']
link_summary=json.loads((folder/'link_resolution.json').read_text(encoding='utf-8'))
assert link_summary['fixed_original_identities_checked']==742
assert link_summary['statuses'].get('missing',0)==0
readers=set(plan['maintained_readers'])
readers|={'reports/history/README.md','reports/history/catalog.md','harness/README.md','reports/run_index.md',
    'results/repository_organization/20261109_history05/README.md'}
local=fixed=opaque=0
known=set(locations)|{e['path'] for e in plan['protected']}
for name in sorted(readers):
    p=root/name
    text=p.read_text(encoding='utf-8-sig')
    for value in cli.LINK_PATTERN.findall(text):
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',value):
            if plan['reading_commit']+'/' in value:
                old=value.split(plan['reading_commit']+'/',1)[1]
                assert old in known,(name,old)
                fixed+=1
            continue
        if value.startswith('#'):continue
        target=(p.parent/value.split('#',1)[0]).resolve()
        assert target.is_relative_to(root),(name,value)
        if cli._private(target.relative_to(root)):
            opaque+=1
            continue
        assert target.exists(),(name,value)
        local+=1
allowed=plan['allowed_changes']
for line in subprocess.check_output(['git','status','--porcelain=v1','-uall'],cwd=root,stderr=subprocess.DEVNULL).decode('utf-8').splitlines():
    name=line[3:]
    assert any(name==a or a.endswith('/') and name.startswith(a) for a in allowed),name
subprocess.run(['git','diff','--check'],cwd=root,check=True,capture_output=True)
result=dict(kind='REPORT_ARCHIVE_VALIDATION',candidates=536,moved_this_batch=210,moved_cumulative=416,
    protected_originals_unchanged=326,prior_protection_released=len(receipt['released_prior_protection']),
    baseline_files_metadata_unchanged=metadata_checks,baseline_text_hashes_unchanged=hash_checks,
    moved_original_byte_hashes_verified=416,maintained_reader_snapshot_verified=1,
    local_reading_links_checked=local,fixed_original_reading_links_checked=fixed,opaque_local_links_not_opened=opaque,
    registered_runs_still_valid=21,public_harness_check_passed=True,mutation_scope_checked=True,
    fixed_original_git_text_identities_checked=742,result_and_local_roots_indexed=339,public_result_groups_linked=96,
    local_storage_payloads_opened=False,registered_run_contexts_in_storage_index=21,
    raw_datasets_weights_predictions_opened=False,scientific_experiments_executed=False,network_link_liveness_checked=False)
(folder/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
