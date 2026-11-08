"""Check archive identities and reading links without importing scientific runners."""
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

folder=Path(__file__).resolve().parent
root=folder.parents[2]
p=json.loads((folder/'migration.json').read_text(encoding='utf-8'))
locations={e['old_path']:e['current_path'] for e in p['cumulative_locations']}
assert len(p['moves'])==28 and len(locations)==453
for e in p['cumulative_locations']:
    data=(root/e['current_path']).read_bytes()
    assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'],e['old_path']
for e in p['moves']:assert not (root/e['old_path']).exists(),e['old_path']
for e in p['items']:
    if e['decision']=='protected':
        assert hashlib.sha256((root/e['path']).read_bytes()).hexdigest()==e['sha256']
    if e.get('closure_evidence'):assert (root/e['closure_evidence']).is_file()

with (folder/'code_index.csv').open(encoding='utf-8-sig',newline='') as f:
    scripts=list(csv.DictReader(f))
assert len(scripts)==238
assert sum('/history/' in e['current_path'] for e in scripts)==29
assert not any(e['decision']=='pending_review' for e in scripts)
for e in scripts:
    data=(root/e['current_path']).read_bytes()
    assert len(data)==int(e['bytes']) and hashlib.sha256(data).hexdigest()==e['sha256'],e['current_path']
    ast.parse(data.decode('utf-8-sig'),filename=e['current_path'])
previous=json.loads((root/p['previous_batch']).read_text(encoding='utf-8'))
for name,digest in previous['preserved_scientific_source_hashes'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
with (root/'results/repository_organization/20261109_history05/catalog.csv').open(encoding='utf-8-sig',newline='') as f:
    reports=list(csv.DictReader(f))
for e in reports:
    name=locations.get(e['original_path'],e['current_path'])
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==e['sha256'],name

request=''.join('145847532bacd3454dca8869e819778efc8258f6:'+e['old_path']+'\n' for e in p['moves']).encode()
blob=subprocess.check_output(['git','cat-file','--batch'],input=request,cwd=root)
offset=0;exact_git=0
for e in p['moves']:
    end=blob.index(b'\n',offset);header=blob[offset:end].split();assert header[1]==b'blob'
    size=int(header[2]);original=blob[end+1:end+1+size];offset=end+size+2
    actual=(root/e['new_path']).read_bytes();exact_git+=original==actual
    assert original.replace(b'\r\n',b'\n')==actual.replace(b'\r\n',b'\n')
assert offset==len(blob)
attrs=subprocess.check_output(['git','check-attr','-z','--stdin','text'],input=('\0'.join(e['new_path'] for e in p['moves'])+'\0').encode(),cwd=root).decode().split('\0')
assert attrs[2::3]==['unset']*28

updates=json.loads((folder/'association_path_updates.json').read_text(encoding='utf-8'))
baseline=subprocess.check_output(['git','show',p['source_commit']+':'+updates['path']],cwd=root)
assert hashlib.sha256(baseline).hexdigest()==updates['before_sha256']
assoc=json.loads(baseline);changed=0
for row in assoc:
    for e in row['reference_clues']:
        if e['source_current_path'] in updates['source_locations']:
            e['source_current_path']=updates['source_locations'][e['source_current_path']]
            e['source_role']='historical_frozen_source';changed+=1
actual=(root/updates['path']).read_bytes()
assert hashlib.sha256(actual).hexdigest()==updates['after_sha256']
assert assoc==json.loads(actual) and changed==48
for row in assoc:
    for e in row['reference_clues']:assert (root/e['source_current_path']).is_file()

html_path=root/'reports/history/force_memory/project_overview_20261029_reading.html'
original=(root/locations['reports/kiki_project_overview_20261029.html']).read_text(encoding='utf-8-sig')
reading=html_path.read_text(encoding='utf-8')
links=json.loads((folder/'html_links.json').read_text(encoding='utf-8'))
assert len(links)==76
reversed_reading=re.sub(r'<aside style="padding:16px;.*?</aside>','',reading,count=1)
lookup={e['reading_href']:e['original_href'] for e in links}
reversed_reading=re.sub(r'(href=["\x27])([^"\x27]+)(["\x27])',lambda m:m.group(1)+lookup.get(m.group(2),m.group(2))+m.group(3),reversed_reading)
assert reversed_reading.rstrip()==original.rstrip(),'HTML body may only change href values and the attributed banner'
for e in links:assert (root/e['current_target']).is_file()

spec=importlib.util.spec_from_file_location('archive07_harness',root/'tools/research_harness.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
assert cli.check_root(root)['valid'] and cli.run_navigation(root)['valid']
readers=['experiments/README.md','experiments/history/early_e_series/README.md','reports/history/README.md',
 'reports/history/private_rom/README.md','reports/history/closure_discovery/README.md','reports/history/closure_development/README.md',
 'reports/history/force_memory/README.md','reports/history/force_memory/protected.md','reports/maintenance/execution_runtime/README.md',
 'experiments/history/private_rom/README.md','experiments/history/closure_discovery/README.md','experiments/history/force_memory/README.md',
 'results/repository_organization/README.md','results/repository_organization/implementation_status.md',
 'results/repository_organization/20261109_history05/README.md',
 'results/repository_organization/20261109_history07/README.md','results/repository_organization/20261109_history07/originals.md',
 'results/repository_organization/20261109_history07/local_roots_review.md']
link_count=0
for name in readers:
    path=root/name
    for value in cli.LINK_PATTERN.findall(path.read_text(encoding='utf-8-sig')):
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',value) or value.startswith('#'):continue
        target=(path.parent/value.split('#',1)[0]).resolve()
        if target==folder/'validation.json':continue  # Written only after all checks pass.
        assert target.is_relative_to(root) and target.exists(),(name,value)
        link_count+=1
allowed=set(readers)|{updates['path']}
allowed|={e['old_path'] for e in p['moves']}|{e['new_path'] for e in p['moves']}|{html_path.relative_to(root).as_posix()}
changed_paths=subprocess.check_output(['git','diff','--name-only','--no-renames','HEAD','-z'],cwd=root)
untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=root,stderr=subprocess.DEVNULL)
for name in (changed_paths+untracked).decode().split('\0'):
    if not name:continue
    assert name in allowed or name.startswith('results/repository_organization/20261109_history07/'),name
subprocess.run(['git','diff','--check'],check=True,capture_output=True,cwd=root)
result=dict(cumulative_originals_byte_verified=453,report_catalog_hashes_verified=len(reports),scripts_indexed_and_AST_checked=238,
    moved_this_batch=28,protected_this_batch=3,new_archive_exact_git_bytes=exact_git,new_archive_git_text_equivalent=28,
    old_scientific_library_files_unchanged=len(previous['preserved_scientific_source_hashes']),
    html_local_links_rebased=76,html_body_equivalence_checked=True,maintained_source_paths_updated=48,
    public_navigation_valid=True,local_reading_links_checked=link_count,scientific_workers_started=False,
    raw_data_models_or_predictions_read=False,private_worker_read=False,local_roots_moved=False)
(folder/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
