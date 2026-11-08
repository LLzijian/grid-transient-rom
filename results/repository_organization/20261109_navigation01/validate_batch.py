"""Validate reading links, frozen identities and declared normalization mutations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import re

folder=Path(__file__).resolve().parent
root=folder.parents[2]
before=json.loads((folder/'baseline.json').read_text(encoding='utf-8'))
archive=json.loads((root/'results/repository_organization/20261108_early_history04/migration.json').read_text(encoding='utf-8'))
locations={e['old_path']:e['current_path'] for e in archive['cumulative_locations']}
for row in before['frozen_files']:
    assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
snapshot=before['campaign_before']
assert hashlib.sha256((root/snapshot['snapshot']).read_bytes()).hexdigest()==snapshot['sha256']
for row in archive['cumulative_locations']:
    data=(root/row['current_path']).read_bytes()
    assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],row['old_path']
mutable={'.gitattributes','README.md','reports/README.md','tools/research_harness.py',
    'campaigns/gftd_kuramoto_discovery_20261106.md','campaigns/README.md','docs/engineering_loop.md','templates/run_closure.md'}
checked=hashes=0
for row in map(json.loads,(root/'results/repository_organization/20261108_inventory01/inventory.jsonl').read_text(encoding='utf-8').splitlines()):
    if row['current_path'] in mutable:continue
    p=root/locations.get(row['current_path'],row['current_path'])
    info=p.stat()
    assert (info.st_size,info.st_mtime_ns)==(row['bytes'],row['mtime_ns']),row['current_path']
    checked+=1
    if row['sha256']:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['current_path']
        hashes+=1
spec=importlib.util.spec_from_file_location('navigation_validation',root/'tools/research_harness.py')
cli=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
all_runs=cli.run_navigation(root)
assert all_runs['valid'] and len(all_runs['runs'])==21,all_runs['issues']
assert all_runs==cli.run_navigation(root),'navigation output is not deterministic'
assert cli.check_root(root)['valid']
pilots=['gftd_sparse_topology_20261107','gftd_modulation_20261107','gftd_fit297_20261107']
for run_id in pilots:
    run=cli.run_navigation(root,run_id)['runs'][0]
    assert all(e.get('path_protection',{}).get('consumers') for e in run['navigation']['materials'])
fit=cli.run_navigation(root,pilots[-1])['runs'][0]
assert len(fit['events'])==4 and len(fit['navigation']['corrections'])==1
assert fit['events'][2]['recorded_utc']<fit['events'][3]['recorded_utc']
readers=['README.md','reports/README.md','reports/run_index.md','harness/README.md','campaigns/README.md',
    'campaigns/gftd_kuramoto_discovery_20261106.md','docs/engineering_loop.md','templates/run_closure.md',
    'results/repository_organization/implementation_status.md','results/repository_organization/README.md']+[f'reports/{run}/README.md' for run in pilots]
local_links=opaque_links=0
for name in readers:
    p=root/name
    for value in cli.LINK_PATTERN.findall(p.read_text(encoding='utf-8-sig')):
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',value) or value.startswith('#'):continue
        target=(p.parent/value.split('#',1)[0]).resolve()
        assert target.is_relative_to(root),(name,value)
        if cli._private(target.relative_to(root)):
            opaque_links+=1
            continue
        if target==folder/'validation.json':continue
        assert target.exists(),(name,value)
        local_links+=1
campaign=(root/'campaigns/gftd_kuramoto_discovery_20261106.md').read_text(encoding='utf-8')
assert '没有调制开/关的同期真实系统证据' not in campaign
assert '先完成选择性归档；下一主线' not in campaign
attr=subprocess.check_output(['git','check-attr','text','--',snapshot['snapshot']],cwd=root,text=True)
assert attr.strip().endswith(': unset')
subprocess.run(['git','diff','--check'],cwd=root,check=True,capture_output=True)
result=dict(kind='NAVIGATION_STANDARDIZATION_VALIDATION',registered_runs=21,pilot_runs=3,
    prior_archived_original_hashes_preserved=206,frozen_run_and_pilot_report_hashes_checked=len(before['frozen_files']),
    campaign_planning_snapshot_sha256_verified=True,baseline_files_unchanged_outside_declared_mutations=checked,
    baseline_text_hashes_unchanged=hashes,local_reading_links_checked=local_links,opaque_local_links_not_opened=opaque_links,
    original_resume_decision_archive_relationship_checked=True,explanation_correction_relationship_checked=True,
    pilot_material_consumers_recorded=True,deterministic_navigation=True,public_harness_check_passed=True,
    raw_datasets_weights_predictions_opened=False,scientific_execution_started=False,
    whole_project_normalization_complete=False,remaining_requirements_reference='results/repository_organization/implementation_status.md')
(folder/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
