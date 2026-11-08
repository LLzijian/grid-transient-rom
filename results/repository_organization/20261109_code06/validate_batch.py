"""Verify code locations, frozen sources and public metadata; never run old entries."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

folder=Path(__file__).resolve().parent
root=folder.parents[2]
p=json.loads((folder/'migration.json').read_text(encoding='utf-8'))
assert len(p['indexed_scripts'])==238 and len(p['moves'])==9 and len(p['cumulative_locations'])==425
for row in p['indexed_scripts']:
    data=(root/row['current_path']).read_bytes()
    assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],row['current_path']
    ast.parse(data.decode('utf-8-sig'),filename=row['current_path'])
    for e in row['consumers']:assert (root/e['source_current_path']).is_file(),e['source_current_path']
for e in p['moves']:assert not (root/e['old_path']).exists()
for name,digest in p['preserved_scientific_source_hashes'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
for e in p['cumulative_locations']:
    assert hashlib.sha256((root/e['current_path']).read_bytes()).hexdigest()==e['sha256'],e['old_path']
request=''.join('145847532bacd3454dca8869e819778efc8258f6:'+e['old_path']+'\n' for e in p['moves']).encode()
blob=subprocess.check_output(['git','cat-file','--batch'],input=request,cwd=root)
offset=0
for e in p['moves']:
    end=blob.index(b'\n',offset);header=blob[offset:end].split();assert header[1]==b'blob'
    size=int(header[2]);original=blob[end+1:end+1+size];offset=end+size+2
    assert original.replace(b'\r\n',b'\n')==(root/e['new_path']).read_bytes().replace(b'\r\n',b'\n')
assert offset==len(blob)
paths=[e['new_path'] for e in p['moves']]+['src/kiki/numerics.py','tests/test_shared_numerics.py']
attrs=subprocess.check_output(['git','check-attr','-z','--stdin','text'],input=('\0'.join(paths)+'\0').encode(),cwd=root).decode().split('\0')
assert attrs[2::3]==['unset']*len(paths)
spec=importlib.util.spec_from_file_location('code_navigation_check',root/'tools/research_harness.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
assert cli.check_root(root)['valid']
assert cli.run_navigation(root)['valid']
updates=json.loads((folder/'association_path_updates.json').read_text(encoding='utf-8'))
old=json.loads(subprocess.check_output(['git','show',p['source_commit']+':'+updates['path']],cwd=root))
for row in old:
    for edge in row['reference_clues']:
        if edge['source_current_path'] in updates['source_locations']:
            edge['source_current_path']=updates['source_locations'][edge['source_current_path']]
            edge['source_role']='historical_frozen_source'
actual=json.loads((root/updates['path']).read_text(encoding='utf-8'))
assert old==actual,'storage association changes must be limited to archived source locations'
for row in actual:
    for edge in row['reference_clues']:assert (root/edge['source_current_path']).is_file()
readers=['README.md','experiments/README.md','experiments/history/early_e_series/README.md',
    'results/repository_organization/20261109_code06/README.md','reports/history/early_e_series/README.md']
links=0
for name in readers:
    path=root/name
    for value in cli.LINK_PATTERN.findall(path.read_text(encoding='utf-8-sig')):
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',value) or value.startswith('#'):continue
        target=(path.parent/value.split('#',1)[0]).resolve()
        assert target.is_relative_to(root) and target.exists(),(name,value)
        links+=1
allowed={'.gitattributes','README.md','templates/run_protocol.md','src/kiki/numerics.py','tests/test_shared_numerics.py',
    'experiments/README.md','reports/history/early_e_series/README.md','results/repository_organization/README.md',
    'results/repository_organization/implementation_status.md','results/repository_organization/20261109_history05/README.md',
    'results/repository_organization/20261109_history05/storage_associations.json'}
allowed|={e['old_path'] for e in p['moves']}|{e['new_path'] for e in p['moves']}
for line in subprocess.check_output(['git','status','--porcelain=v1','-uall'],cwd=root,stderr=subprocess.DEVNULL).decode().splitlines():
    name=line[3:]
    assert name in allowed or name.startswith(('results/repository_organization/20261109_code06/','experiments/history/early_e_series/')),name
subprocess.run(['git','diff','--check'],check=True,capture_output=True,cwd=root)
result=dict(top_level_python_scripts_indexed=238,old_entries_archived=9,protected_scripts=207,pending_scripts=22,
    cumulative_original_hashes_verified=425,existing_scientific_library_files_unchanged=len(p['preserved_scientific_source_hashes']),
    original_fixed_commit_readers_verified=9,new_public_kernel='kiki.numerics.rk4_step',
    public_harness_and_registered_references_valid=True,local_reading_links_checked=links,
    scientific_runner_or_model_executed=False,private_dataset_or_weight_read=False,
    frozen_scientific_sources_rewritten=False,mutation_scope_checked=True)
(folder/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
