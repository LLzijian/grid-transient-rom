"""Isolated regressions for denied reads, conflicting states and current links."""
import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from harness_runtime import organization as org
import test_harness_cli as fixtures
import test_repository_inventory as inventory_tests


class OrganizationTests(unittest.TestCase):
    setUp = fixtures.CLITests.setUp
    write_json = fixtures.CLITests.write_json
    invoke = fixtures.CLITests.invoke

    def pointer(self, inputs):
        self.write_json(org.CURRENT, dict(schema_version=1, inputs={key: dict(path=name,
            sha256_lf=hashlib.sha256((self.root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()) for key,name in inputs.items()}))

    def test_document_reconciliation_never_opens_private_target(self):
        self.write_json(org.DOCUMENTS, dict(schema_version=1,source_commit='fixture',entries=[dict(
            old_path='docs/a.md',new_path='private_data/a.md',bytes=1,sha256='a'*64)]))
        private = self.root/'private_data/a.md'
        private.parent.mkdir();private.write_bytes(b'x')
        actual = Path.open;opened=[]
        def recording(path,*args,**kwargs):
            opened.append(path)
            return actual(path,*args,**kwargs)
        with patch.object(Path,'open',recording),self.assertRaises(ValueError):
            inventory_tests.inv.reconcile(self.root)
        self.assertNotIn(private,opened)

    def test_cleanup_summary_reparse_point_is_rejected_and_audited(self):
        name='results/private_data_cleanup_20261107/summary.json'
        self.write_json(name,dict(secret='must not read'))
        original=Path.lstat;access=[]
        targets={self.root/name,self.root.resolve()/name}
        def attributes(path,*args,**kwargs):
            info=original(path,*args,**kwargs)
            return SimpleNamespace(st_mode=info.st_mode,st_file_attributes=0x400) if path in targets else info
        with patch.object(Path,'lstat',attributes),self.assertRaisesRegex(ValueError,'reparse'):
            org.read_public(self.root,name,access=access)
        self.assertEqual(access[0]['outcome'],'refused')
        with patch.object(Path,'lstat',attributes),self.assertRaisesRegex(ValueError,'reparse'):
            self._inventory_with_summary()
        self.assertFalse((self.root/'results/repository_organization/snapshot').exists())

    def _inventory_with_summary(self):
        inv=inventory_tests.inv
        for name in inv.ROOT_FILES:
            (self.root/name).write_text('# fixture\n',encoding='utf-8')
        with patch.object(inv,'git',return_value=('', '')):
            inv.inventory(self.root,'results/repository_organization/snapshot')

    def test_type_size_and_symlink_reads_are_refused(self):
        self.write_json('results/a/model.json',{})
        for name in ('private_data/x.md','reports/private_data/x.md','../outside','results/a/model.json','.env','secret.json','reports/private_data./x.md'):
            with self.subTest(name=name),self.assertRaises(ValueError):org.read_public(self.root,name)
        big=self.root/'reports/big.md';big.parent.mkdir(exist_ok=True);big.write_bytes(b'x'*20)
        with patch.object(org,'MAX_BYTES',8),self.assertRaisesRegex(ValueError,'small'):org.read_public(self.root,'reports/big.md')
        link=self.root/'reports/link.md'
        try:
            link.symlink_to(big)
        except OSError:
            original=Path.lstat
            def link_stat(path,*args,**kwargs):
                return SimpleNamespace(st_mode=stat.S_IFLNK) if path==link else original(path,*args,**kwargs)
            with patch.object(Path,'lstat',link_stat),self.assertRaisesRegex(ValueError,'symlink'):
                org.read_public(self.root,'reports/link.md')
        else:
            with self.assertRaisesRegex(ValueError,'symlink'):org.read_public(self.root,'reports/link.md')

    def test_private_root_and_malformed_maps_are_rejected(self):
        private=self.root/'private_data';private.mkdir()
        with self.assertRaisesRegex(ValueError,'private'):org.Organization(private)
        self.write_json(org.DOCUMENTS,dict(schema_version=1,source_commit='fixture',entries=[None]))
        with self.assertRaisesRegex(ValueError,'object'):org.Organization(self.root)

    def test_existing_reading_page_is_linked_and_view_preserves_unknown_availability(self):
        self.write_json('harness/runs/example.json',dict(run_id='example'))
        self.write_json('harness/runs/second.json',dict(run_id='second'))
        page=self.root/'reports/example/README.md';page.parent.mkdir(parents=True);page.write_text('# reading',encoding='utf-8')
        rendered=fixtures.cli.render_run_navigation(fixtures.cli.run_navigation(self.root))
        self.assertIn('[example](reports/example/README.md)',rendered)
        self.assertFalse((self.root/'reports/second/README.md').exists())
        scope='results/repository_organization/fixture/scope.csv'
        file=self.root/scope;file.parent.mkdir(parents=True)
        file.write_text('original_path,current_path\nreports/example/README.md,reports/example/README.md\n',encoding='utf-8')
        self.pointer(dict(scope=scope))
        row=org.Organization(self.root)
        self.assertEqual(row.view(),row.view())
        entry=row.view('reports/example/README.md')['entries'][0]
        self.assertEqual(entry['local_presence'],'present')
        self.assertEqual(entry['remote_availability'],'not_verified')
        self.assertIsNone(entry['in_git'])

    def test_same_record_conflicts_are_visible_in_state_and_both_renderings(self):
        self.write_json('harness/runs/example.json',dict(run_id='example',recorded_utc='2026-11-01T00:00:00Z',
            scientific_work=dict(budget_status='closed'),root_decisions=dict(scientific_budget_status='open',code='retain_branch',code_decision='reject')))
        result=fixtures.cli.run_navigation(self.root,'example');run=result['runs'][0]
        self.assertEqual(run['states']['budget_status'],'conflict')
        self.assertEqual(run['states']['code_decision'],'conflict')
        for c in run['conflicts']:
            self.assertEqual(c['source_reference'],'harness/runs/example.json')
            self.assertEqual(c['recorded_utc'],'2026-11-01T00:00:00Z')
        rendered=fixtures.cli.render_run_navigation(result)
        self.assertIn('| 工作量／预算 | conflict |',rendered)
        self.assertIn('| 代码决定 | conflict |',rendered)
        self.assertIn('retain_branch',rendered)
        result['runs'].append(dict(run,run_id='second'))
        self.assertIn('| conflict |',fixtures.cli.render_run_navigation(result))

    def test_archive_aliases_drive_all_navigation_links_and_check(self):
        fixtures.CLITests.public_fixture(self)
        target=self.root/'reports/history/source.md';target.parent.mkdir(parents=True);target.write_bytes(b'# source\n')
        entry=dict(old_path='reports/old.md',current_path='reports/history/source.md',bytes=9,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        archive='results/repository_organization/fixture/migration.json'
        self.write_json(archive,dict(cumulative_locations=[entry]));self.pointer(dict(archive_locations=archive))
        self.write_json('harness/runs/example.json',dict(run_id='example',evidence=dict(references=['reports/old.md'])))
        self.write_json('harness/navigation.json',dict(kind='RETROSPECTIVE_NAVIGATION',runs=dict(example=dict(
            campaign_reference='reports/old.md',materials=[dict(role='source',reference='reports/old.md')],
            corrections=[dict(original_reference='reports/old.md',corrected_by_reference='reports/old.md',adopted_reference='reports/old.md',scope='fixture')]))))
        result=fixtures.cli.run_navigation(self.root,'example');self.assertTrue(result['valid'])
        rendered=fixtures.cli.render_run_navigation(result)
        self.assertNotIn('](reports/old.md)',rendered)
        self.assertIn('](reports/history/source.md)',rendered)
        self.assertTrue(fixtures.cli.check_root(self.root)['valid'])
        code,_=self.invoke('show-run','--root',str(self.root),'--run','example','--output',str(self.root/'reports/example/README.md'))
        self.assertEqual(code,0)
        text=(self.root/'reports/example/README.md').read_text(encoding='utf-8')
        for link in fixtures.cli.LINK_PATTERN.findall(text):self.assertTrue((self.root/'reports/example'/link).exists(),link)
        self.assertEqual(org.Organization(self.root).view('reports/old.md')['entries'][0]['current_path'],'reports/history/source.md')
        manifest='results/repository_organization/fixture/reading.json'
        url='https://github.com/LLzijian/grid-transient-rom/blob/'+'a'*40+'/reports/old.md'
        self.write_json(manifest,dict(reading_commit='a'*40,materials=[dict(old_path='reports/old.md',raw_sha256=entry['sha256'],reading_url=url)]))
        self.pointer(dict(archive_locations=archive,reading_manifest=manifest))
        rendered=fixtures.cli.render_run_navigation(fixtures.cli.run_navigation(self.root,'example'))
        self.assertIn(']('+url+')',rendered)
        self.assertIn('](reports/history/source.md)',rendered)
        (self.root/'reports/old.md').write_text('# active source',encoding='utf-8')
        self.assertNotIn(']('+url+')',fixtures.cli.render_run_navigation(fixtures.cli.run_navigation(self.root,'example')))
        row=org.Organization(self.root).view('reports/old.md')['entries'][0]
        self.assertEqual(row['current_path'],'reports/old.md')
        self.assertEqual(row['frozen_original_path'],'reports/history/source.md')
        target.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'bytes differ'):org.Organization(self.root)

    def test_pointer_versions_and_new_types_do_not_bypass_material_scope(self):
        name='results/repository_organization/fixture/migration.json'
        self.write_json(name,dict(cumulative_locations=[]));self.pointer(dict(archive_locations=name))
        (self.root/name).write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'version differs'):org.Organization(self.root)
        self.write_json(name,dict(cumulative_locations=[dict(old_path='reports/old.json',current_path='private_data/payload.json',bytes=1,sha256='a'*64)]))
        self.pointer(dict(archive_locations=name))
        with self.assertRaisesRegex(ValueError,'private'):org.Organization(self.root)


if __name__=='__main__':unittest.main()
