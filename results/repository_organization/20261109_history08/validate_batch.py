"""Verify released paths and maintained navigation without executing old science."""
import ast,csv,hashlib,importlib.util,json,re,subprocess,sys
from pathlib import Path
folder=Path(__file__).resolve().parent;root=folder.parents[2]
p=json.loads((folder/'migration.json').read_text(encoding='utf-8'))
locations={e['old_path']:e['current_path'] for e in p['cumulative_locations']}
assert len(locations)==459 and len(p['moves'])==6
for e in p['cumulative_locations']:
    data=(root/e['current_path']).read_bytes()
    assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'],e['old_path']
for e in p['moves']:assert not (root/e['old_path']).exists()
with (folder/'catalog.csv').open(encoding='utf-8-sig',newline='') as f:catalog=list(csv.DictReader(f))
assert len(catalog)==536 and sum(e['decision']=='protected' for e in catalog)==320
for e in catalog:assert hashlib.sha256((root/e['current_path']).read_bytes()).hexdigest()==e['sha256']
with (folder/'protected.csv').open(encoding='utf-8-sig',newline='') as f:protected=list(csv.DictReader(f))
assert len(protected)==320
archived=set(locations.values())
for row in protected:
    assert (root/row['original_path']).is_file() and row['release_condition']
    refs=json.loads(row['consumers']);assert refs
    for e in refs:assert (root/e['source']).is_file() and e['source'] not in archived,e
with (root/'results/repository_organization/20261109_history07/code_index.csv').open(encoding='utf-8-sig',newline='') as f:code=list(csv.DictReader(f))
for e in code:
    data=(root/e['current_path']).read_bytes();assert hashlib.sha256(data).hexdigest()==e['sha256']
    ast.parse(data.decode('utf-8-sig'))
baseline=json.loads((root/'results/repository_organization/20261109_code06/migration.json').read_text(encoding='utf-8'))
for name,digest in baseline['preserved_scientific_source_hashes'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest

updates=json.loads((folder/'association_path_updates.json').read_text(encoding='utf-8'))
before=subprocess.check_output(['git','show',p['source_commit']+':'+updates['path']],cwd=root)
before_exact=hashlib.sha256(before).hexdigest()==updates['before_sha256']
before_crlf=hashlib.sha256(before.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')).hexdigest()==updates['before_sha256']
assert before_exact or before_crlf  # Maintained navigation follows Git text normalization.
assoc=json.loads(before);changed=0
for row in assoc:
    for e in row['reference_clues']:
        if e['source_current_path'] in updates['source_locations']:
            e['source_current_path']=updates['source_locations'][e['source_current_path']];e['source_role']='historical_frozen_source';changed+=1
actual=(root/updates['path']).read_bytes()
assert hashlib.sha256(actual).hexdigest()==updates['after_sha256'] and json.loads(actual)==assoc and changed==4
for row in assoc:
    for e in row['reference_clues']:assert (root/e['source_current_path']).is_file()

nav=json.loads((folder/'navigation_updates.json').read_text(encoding='utf-8'))
html=(root/nav['html_reader']).read_text(encoding='utf-8')
for e in nav['html_href_updates']:
    assert html.count('href="'+e['new_href']+'"')==e['count']
    html=html.replace('href="'+e['new_href']+'"','href="'+e['old_href']+'"')
before_html=subprocess.check_output(['git','show',p['source_commit']+':'+nav['html_reader']],cwd=root).decode()
assert html.replace('\r\n','\n')==before_html.replace('\r\n','\n'),'reading HTML changes must be limited to released target hrefs'
spec=importlib.util.spec_from_file_location('release08_harness',root/'tools/research_harness.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
assert cli.check_root(root)['valid'] and cli.run_navigation(root)['valid']
readers=nav['rewritten_readers']+nav['protected_pages']+[
 'reports/history/catalog.md','reports/history/private_rom/README.md','reports/history/closure_discovery/README.md','reports/history/force_memory/README.md',
 'reports/gftd_fit297_20261107/README.md','reports/gftd_modulation_20261107/README.md',
 'results/repository_organization/README.md','results/repository_organization/implementation_status.md',
 'results/repository_organization/20261109_history05/README.md','results/repository_organization/20261109_history08/README.md']
links=0
for name in set(readers):
    path=root/name
    for value in cli.LINK_PATTERN.findall(path.read_text(encoding='utf-8-sig')):
        if value.startswith('#') or re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:',value):continue
        target=(path.parent/value.split('#',1)[0]).resolve()
        if target==folder/'validation.json':continue
        assert target.is_relative_to(root) and target.exists(),(name,value)
        links+=1
sys.path.insert(0,str(root/'experiments'));import verify_artifacts
json_path=root/locations['reports/failed/0026_m4_evidence.json']
findings=verify_artifacts.inspect_json_text(json_path.read_text(encoding='utf-8-sig'),source=str(json_path),strict_nonfinite=False)
assert not any(e.level=='error' for e in findings)
allowed=set(readers)|{updates['path'],nav['html_reader']}
allowed|={e['old_path'] for e in p['moves']}|{e['new_path'] for e in p['moves']}
changed_paths=subprocess.check_output(['git','diff','--name-only','--no-renames','HEAD','-z'],cwd=root)
untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=root,stderr=subprocess.DEVNULL)
for name in (changed_paths+untracked).decode().split('\0'):
    assert not name or name in allowed or name.startswith('results/repository_organization/20261109_history08/'),name
subprocess.run(['git','diff','--check'],check=True,capture_output=True,cwd=root)
result=dict(cumulative_byte_hashes_verified=459,released_report_paths=6,remaining_report_exceptions=320,protected_consumers_exist_and_are_not_archived=True,
    original_code_hashes_and_AST_verified=len(code),original_scientific_library_hashes_verified=len(baseline['preserved_scientific_source_hashes']),
    storage_source_updates=changed,local_reading_links_checked=links,public_navigation_valid=True,html_only_target_href_changes=True,
    legacy_json_structural_errors=0,legacy_json_nonfinite_warnings=sum(e.level=='warning' for e in findings),
    scientific_execution_started=False,private_payloads_opened=False)
(folder/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
