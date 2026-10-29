"""Pure execution admission tests; no native launch or real payload access."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import weak_force_execution_20261029 as execution


class ExecutionAdmission(unittest.TestCase):
    def test_unknown_release_refuses_before_files_jobs_or_phases(self):
        with patch.object(execution,'RELEASE_SHA','UNKNOWN'),patch.object(execution,'sha') as digest:
            for action in (lambda:execution.require_release(),
                           lambda:execution.freeze('a'*40),
                           lambda:execution.launch(execution.BASE,'a'*40,'weak'),
                           lambda:execution.run(execution.BASE,'a'*40)):
                with self.assertRaisesRegex(ValueError,'UNKNOWN'):action()
            digest.assert_not_called()

    def test_release_requires_actual_parent_archive_and_review(self):
        required=dict(parent_remote_archival_verified=True,independent_implementation_review_passed=True,
            root_execution_authorized=True,parent_complete_sha256=execution.PARENT_COMPLETE_SHA,optimizer_upper_bound=976)
        with patch.object(execution,'RELEASE_SHA','a'*64),patch.object(execution,'sha',return_value='a'*64):
            for key in ('parent_remote_archival_verified','independent_implementation_review_passed','root_execution_authorized'):
                value=dict(required);value[key]=False
                with patch.object(execution,'read',return_value=value):
                    with self.assertRaises(ValueError):execution.require_release()
            with patch.object(execution,'read',return_value=required):
                self.assertEqual(execution.require_release(),required)

    def test_only_registered_directory_and_weak_once_in_loop(self):
        self.assertEqual(execution.checked_base(execution.BASE),execution.BASE.resolve())
        with self.assertRaises(ValueError):execution.checked_base(execution.BASE/'replacement')
        self.assertEqual(execution.PHASES,('prepare','weak','qualify','qa','study','summary'))
        self.assertEqual(execution.PHASES.count('weak'),1)


if __name__=='__main__':unittest.main()
