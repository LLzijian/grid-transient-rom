"""Navigation checks use public synthetic metadata, never execute an experiment."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_harness_cli as fixtures

cli=fixtures.cli

class RunNavigationTests(unittest.TestCase):
    setUp=fixtures.CLITests.setUp
    write_json=fixtures.CLITests.write_json
    invoke=fixtures.CLITests.invoke
    def registry(self, **changes):
        record=dict(run_id='example',evidence=dict(references=['reports/conclusion.md','artifacts/example/model.npz']))
        record.update(changes)
        self.write_json('harness/runs/example.json',record)
        p=self.root/'reports/conclusion.md'
        p.parent.mkdir(exist_ok=True)
        p.write_text('# Example\n',encoding='utf-8')
        return record

    def index(self, **changes):
        self.write_json('harness/navigation.json',dict(kind='RETROSPECTIVE_NAVIGATION',runs=dict(example=changes)))

    def test_old_record_unknown_fields_and_opaque_private_input_are_deterministic(self):
        self.registry()
        self.index(title='中文研究问题')
        before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()}
        first=cli.run_navigation(self.root,'example')
        self.assertEqual(first,cli.run_navigation(self.root,'example'))
        self.assertTrue(first['valid'])
        self.assertIsNone(first['runs'][0]['states']['execution_status'])
        self.assertEqual(first['runs'][0]['materials'][1]['status'],'local_not_checked')
        self.assertEqual(before,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()})

    def test_missing_unsafe_and_encoded_escape_paths_are_reported(self):
        self.registry(evidence=dict(references=['reports/missing.md','../outside.json','C:/secret','reports/%2e%2e/secret.md']))
        result=cli.run_navigation(self.root,'example')
        self.assertFalse(result['valid'])
        self.assertEqual(result['runs'][0]['materials'][0]['status'],'missing')
        self.assertEqual([x['status'] for x in result['runs'][0]['materials'][1:3]],['unsafe','unsafe'])
        code,_=self.invoke('show-run','--root',str(self.root),'--run','../example')
        self.assertEqual(code,2)

    def test_existing_public_migration_requires_preserved_bytes(self):
        self.registry(evidence=dict(references=['docs/old.md']))
        payload=b'# Historical design\n'
        target=self.root/'designs/old.md'
        target.parent.mkdir()
        target.write_bytes(payload)
        self.write_json('harness/documentation_migration_20261108.json',dict(schema_version=1,source_commit='synthetic',entries=[dict(old_path='docs/old.md',new_path='designs/old.md',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())]))
        result=cli.run_navigation(self.root,'example')
        self.assertTrue(result['valid'])
        self.assertEqual(result['runs'][0]['materials'][0]['resolved_reference'],'designs/old.md')
        target.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'bytes differ'):cli.run_navigation(self.root,'example')

    def test_frozen_report_archive_supersedes_an_identical_design_copy(self):
        old = 'reports/design.md'
        self.registry(evidence=dict(references=[old]))
        self.index(materials=[dict(reference=old, role='冻结设计',
            path_protection=dict(status='preserve_current_path', reason='执行时绑定旧路径'))])
        payload = b'# Frozen design\n'
        copy = 'designs/design.md'
        archive = 'reports/history/example/originals/design.md'
        for name in (copy, archive):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        identity = dict(old_path=old, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        self.write_json('harness/documentation_migration_20261108.json', dict(schema_version=1,
            source_commit='synthetic', entries=[dict(identity, new_path=copy, operation='copy_keep_frozen_source')]))
        mapping = 'results/repository_organization/example/migration.json'

        def pin(entry):
            manifest = self.write_json(mapping, dict(cumulative_locations=[dict(entry, current_path=archive)]))
            digest = hashlib.sha256(manifest.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            self.write_json('results/repository_organization/current.json', dict(schema_version=1,
                inputs=dict(archive_locations=dict(path=mapping, sha256_lf=digest))))

        pin(identity)
        result = cli.run_navigation(self.root, 'example')
        self.assertTrue(result['valid'])
        self.assertEqual(result['runs'][0]['materials'][0]['resolved_reference'], archive)
        rendered = cli.render_run_navigation(result)
        self.assertIn('已核验迁移', rendered)
        self.assertNotIn('原位保留', rendered)
        self.assertEqual((self.root / copy).read_bytes(), payload)
        pin(dict(identity, sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'duplicate migration old_path'):
            cli.run_navigation(self.root, 'example')

    def test_same_record_conflict_is_not_resolved_by_picking_a_value(self):
        self.registry(scientific_work=dict(budget_status='closed'),root_decisions=dict(scientific_budget_status='open'))
        result=cli.run_navigation(self.root,'example')
        self.assertFalse(result['valid'])
        self.assertEqual(result['runs'][0]['conflicts'][0]['resolution'],'not_selected')

    def test_later_archive_event_is_distinct_from_earlier_decision(self):
        self.registry()
        self.index(events=[dict(label='当时决定',reference='results/example/root_adoption.json',relation='科学决定'),
                           dict(label='后续归档',reference='results/example/archive_verified.json',relation='后续事件')])
        self.write_json('results/example/root_adoption.json',dict(root_decisions=dict(recorded_utc='2026-11-07T18:22:07Z',next_action='archive pending')))
        self.write_json('results/example/archive_verified.json',dict(recorded_utc='2026-11-07T18:32:38Z',remote_head='abc'))
        result=cli.run_navigation(self.root,'example')
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['runs'][0]['events']),2)
        self.assertLess(result['runs'][0]['events'][0]['recorded_utc'],result['runs'][0]['events'][1]['recorded_utc'])
        self.assertEqual(result['runs'][0]['conflicts'],[])

    def test_only_named_receipts_are_opened_and_output_does_not_overwrite(self):
        self.registry()
        self.index(events=[dict(reference='results/example/predictions.json')])
        self.write_json('results/example/predictions.json',dict(payload='not execution metadata'))
        with self.assertRaisesRegex(ValueError,'receipt'):cli.run_navigation(self.root,'example')
        self.index(title='Question')
        target=self.root/'reports/example/README.md'
        code,_=self.invoke('show-run','--root',str(self.root),'--run','example','--output',str(target))
        self.assertEqual(code,0)
        self.assertIn('[reports/conclusion.md](../conclusion.md)',target.read_text(encoding='utf-8'))
        first=target.read_bytes()
        code,_=self.invoke('show-run','--root',str(self.root),'--run','example','--output',str(target))
        self.assertEqual(code,2)
        self.assertEqual(first,target.read_bytes())

    def test_symlinked_public_reference_is_rejected_before_reading(self):
        self.registry()
        p=self.root/'reports/linked.md'
        with patch.object(cli,'_link_or_reparse',side_effect=lambda path:path.name=='linked.md'):
            self.registry(evidence=dict(references=['reports/linked.md']))
            result=cli.run_navigation(self.root,'example')
        self.assertFalse(result['valid'])
        self.assertEqual(result['runs'][0]['materials'][0]['status'],'unsafe')

    def test_all_runs_render_as_a_compact_table_and_invalid_sidecar_is_rejected(self):
        self.registry()
        self.write_json('harness/runs/second.json',dict(run_id='second'))
        result=cli.run_navigation(self.root)
        rendered=cli.render_run_navigation(result)
        self.assertIn('| 中文问题 |',rendered)
        self.assertNotIn('### 材料与字段来源',rendered)
        self.index(events=[{}])
        with self.assertRaisesRegex(ValueError,'invalid navigation events'):cli.run_navigation(self.root,'example')
