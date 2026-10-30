"""CLI contract tests; call main directly, never spawn subprocesses."""

from contextlib import redirect_stdout
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


CLI_PATH = Path(__file__).resolve().parents[1] / "tools/research_harness.py"
SPEC = importlib.util.spec_from_file_location("research_harness_cli_for_test", CLI_PATH)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def task(task_id, **changes):
    value = {
        "task_id": task_id, "role": "reviewer", "question": "Review synthetic evidence",
        "inputs": [], "outputs": ["review"], "write_paths": [], "depends_on": [],
        "resources": {}, "status": "pending", "acceptance": "Bounded conclusion",
        "stop_condition": "Return evidence",
    }
    value.update(changes)
    return value


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_json(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def invoke(self, *arguments):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(list(arguments))
        return code, out.getvalue()

    def public_fixture(self, base=None):
        base = base or self.root
        for name in cli.REQUIRED_FILES:
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                path.write_text(json.dumps({"schema_version": 2, "permissions": {}}), encoding="utf-8")
            else:
                path.write_text("# synthetic public fixture\n", encoding="utf-8")
        return base

    def test_paused_snapshot_has_no_ready_candidates_or_allocations(self):
        path = self.write_json("tasks.json", {"task_state": "PAUSED_BY_USER", "tasks": [task("live", status="running", resources={"gpu:0": 1}), task("review"), task("writer", write_paths=["src"])]})
        before = set(self.root.rglob("*"))
        code, output = self.invoke("schedule", "--tasks", str(path), "--resources", '{"gpu:0":1}', "--slots", "3")
        value = json.loads(output)
        self.assertEqual(code, 0)
        self.assertTrue(value["paused"])
        self.assertEqual(value["ready"], [])
        self.assertEqual(value["running"], ["live"])
        self.assertEqual(value["slots"]["remaining_after_selection"], 2)
        self.assertEqual(value["resources"]["remaining_after_selection"], {"gpu:0": 0})
        self.assertFalse(value["dispatch_performed"])
        self.assertEqual(before, set(self.root.rglob("*")))

    def test_bare_list_is_advisory_and_missing_dependencies_fail(self):
        path = self.write_json("tasks.json", [task("reader")])
        code, output = self.invoke("schedule", "--tasks", str(path), "--resources", "{}", "--slots", "1")
        value = json.loads(output)
        self.assertEqual(code, 0)
        self.assertEqual(value["ready"], ["reader"])
        self.assertFalse(value["live_state_verified"])
        self.assertIn("not_granted", value["authorization"])
        path.write_text(json.dumps([task("bad", depends_on=["unknown"])]), encoding="utf-8")
        code, output = self.invoke("schedule", "--tasks", str(path), "--resources", "{}", "--slots", "1")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output)["ready"], [])

    def test_render_stdout_only_unless_output_is_explicit(self):
        path = self.write_json("snapshot.json", {"schema_version": 2, "task_state": "PAUSED_BY_USER", "snapshot": {"as_of_utc": "2026-10-30T01:46:06Z", "source_sha": "example", "live_verified": False}, "current_run": {"run_id": "historical", "status": "COMPLETED"}, "active_jobs": [], "pointers": {}, "next_action": "Await authorization"})
        before = set(self.root.rglob("*"))
        code, output = self.invoke("render-state", "--registry", str(path))
        self.assertEqual(code, 0)
        self.assertIn("PAUSED_BY_USER", output)
        self.assertIn("0 active job", output)
        self.assertIn("Live process state was not inspected", output)
        self.assertEqual(before, set(self.root.rglob("*")))
        target = self.root / "explicit/state.md"
        code, output = self.invoke("render-state", "--registry", str(path), "--output", str(target))
        self.assertEqual(code, 0)
        self.assertTrue(target.is_file())
        self.assertFalse(json.loads(output)["dispatch_performed"])

    def test_check_does_not_require_registry_or_read_private_payloads(self):
        self.public_fixture()
        secret = self.root / "private_data/never_read.json"
        secret.parent.mkdir()
        secret.write_text("not JSON; never inspect", encoding="utf-8")
        (self.root / "README.md").write_text("[opaque](private_data/never_read.json)\n", encoding="utf-8")
        private_config = self.root / "harness/private_data/never_read.json"
        private_config.parent.mkdir()
        private_config.write_text("not JSON either", encoding="utf-8")
        actual_read = Path.read_text
        reads = []

        def guarded_read(path, *args, **kwargs):
            self.assertNotIn("private_data", path.parts)
            reads.append(path)
            return actual_read(path, *args, **kwargs)

        with patch.object(Path, "read_text", guarded_read):
            result = cli.check_root(self.root)
        self.assertTrue(result["valid"], result["issues"])
        self.assertFalse(result["active_registry_required"])
        self.assertFalse(result["private_payloads_read"])
        self.assertNotIn(secret, reads)

    def test_check_json_and_safe_relative_references(self):
        self.public_fixture()
        (self.root / "README.md").write_text("[rules](docs/research_policy.md)\n[bad](../../outside.txt)\n", encoding="utf-8")
        self.write_json("harness/examples/record.json", {"evidence": {"references": ["templates/missing.md"]}})
        (self.root / "harness/broken.json").write_text("{bad", encoding="utf-8")
        result = cli.check_root(self.root)
        self.assertFalse(result["valid"])
        codes = {row["code"] for row in result["issues"]}
        self.assertTrue({"reference_outside_root", "missing_reference", "invalid_public_metadata"} <= codes)

    def test_overlay_missing_existing_repository_reference_is_explicit_warning(self):
        base = self.public_fixture(self.root / "overlay")
        (base / "README.md").write_text("[history](reports/existing.md)\n", encoding="utf-8")
        result = cli.check_root(self.root)
        self.assertTrue(result["valid"], result["issues"])
        self.assertTrue(any(x["code"] == "missing_reference" and x["severity"] == "warning" for x in result["issues"]))

    def test_manifest_escape_and_symlinked_private_parent_are_not_read(self):
        self.public_fixture()
        self.write_json("manifest.json", {"files": ["../outside.json"]})
        result = cli.check_root(self.root)
        self.assertFalse(result["valid"])
        self.assertIn("unsafe_manifest_path", {x["code"] for x in result["issues"]})
        hidden = self.root / "private_data"
        hidden.mkdir()
        (hidden / "policy.json").write_text("private invalid JSON", encoding="utf-8")
        (self.root / "harness").rename(self.root / "original_harness")
        try:
            (self.root / "harness").symlink_to(hidden, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable on this host")
        result = cli.check_root(self.root)
        self.assertFalse(result["valid"])
        self.assertIn("missing_required_file", {x["code"] for x in result["issues"]})
        self.assertNotIn("harness/policy.json", result["checked_files"])

    def test_installed_check_covers_all_bundled_files_and_role_links(self):
        self.public_fixture()
        result = cli.check_root(self.root)
        self.assertTrue(result["valid"], result["issues"])
        self.assertEqual(len(cli.REQUIRED_FILES), 39)
        self.assertEqual(set(result["checked_files"]), set(cli.REQUIRED_FILES))
        (self.root / "agents/reviewer.md").write_text("[absent](missing-review.md)\n", encoding="utf-8")
        result = cli.check_root(self.root)
        self.assertFalse(result["valid"])
        self.assertIn("missing_reference", {i["code"] for i in result["issues"]})
        (self.root / "docs/architecture_design.md").unlink()
        result = cli.check_root(self.root)
        self.assertIn("missing_required_file", {i["code"] for i in result["issues"]})

    def test_windows_junction_is_not_traversed(self):
        try:
            import _winapi
        except ImportError:
            self.skipTest("Windows junction API is unavailable")
        self.public_fixture()
        outside = self.root / "private_data"
        outside.mkdir()
        (outside / "secret.json").write_text("private malformed JSON", encoding="utf-8")
        redirect = self.root / "harness/redirect"
        _winapi.CreateJunction(str(outside), str(redirect))
        self.assertFalse(redirect.is_symlink())
        self.assertTrue(cli._link_or_reparse(redirect))
        real_iterdir = Path.iterdir

        def guarded_iterdir(path):
            self.assertNotEqual(path, redirect, "must not enumerate a junction target")
            return real_iterdir(path)

        with patch.object(Path, "iterdir", guarded_iterdir):
            result = cli.check_root(self.root)
        self.assertTrue(result["valid"], result["issues"])
        self.assertFalse(any("redirect" in name for name in result["checked_files"]))
        # Required files reached through a junction are rejected before resolve.
        (self.root / "harness").rename(self.root / "original_harness")
        _winapi.CreateJunction(str(outside), str(self.root / "harness"))
        result = cli.check_root(self.root)
        self.assertFalse(result["valid"])
        self.assertNotIn("harness/policy.json", result["checked_files"])

    def test_windows_overlay_junction_is_rejected(self):
        try:
            import _winapi
        except ImportError:
            self.skipTest("Windows junction API is unavailable")
        outside = self.root / "external_overlay"
        outside.mkdir()
        _winapi.CreateJunction(str(outside), str(self.root / "overlay"))
        with patch.object(Path, "read_text", side_effect=AssertionError("must not read an overlay junction")):
            result = cli.check_root(self.root)
        self.assertFalse(result["valid"])
        self.assertEqual(result["checked_files"], [])

    def test_manifest_overlay_relative_paths_and_new_hashes(self):
        base = self.public_fixture(self.root / "overlay")
        payload = (base / "AGENTS.md").read_bytes()
        entry = {"path": "AGENTS.md", "base_sha256": "old-base-not-current-bytes", "base_git_blob": "historical-identity", "new_sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        self.write_json("manifest.json", {"files": [entry]})
        result = cli.check_root(self.root)
        self.assertTrue(result["valid"], result["issues"])
        self.assertFalse(result["base_hashes_checked"])
        (base / "AGENTS.md").write_text("# changed bytes\n", encoding="utf-8")
        result = cli.check_root(self.root)
        self.assertFalse(result["valid"])
        self.assertTrue({"manifest_hash_mismatch", "manifest_size_mismatch"} <= {x["code"] for x in result["issues"]})

    def test_invalid_input_returns_error_without_writes(self):
        path = self.root / "bad.json"
        path.write_text("{broken", encoding="utf-8")
        before = set(self.root.rglob("*"))
        code, output = self.invoke("schedule", "--tasks", str(path), "--resources", "{}", "--slots", "1")
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(output)["dispatch_performed"])
        self.assertEqual(before, set(self.root.rglob("*")))

    def test_private_root_is_rejected_without_reading(self):
        private = self.root / "private_data"
        private.mkdir()
        with patch.object(Path, "read_text", side_effect=AssertionError("must not read private root")):
            result = cli.check_root(private)
        self.assertFalse(result["valid"])
        self.assertEqual(result["checked_files"], [])

    def test_assess_reports_schema_without_granting_acceptance(self):
        path = self.write_json("record.json", {"schema_version": 2, "run_id": "synthetic_cli", "purpose": "probe", "scope": "Synthetic CLI fixture only", "evidence": {"validity": "unknown"}, "proposal": {"scientific_claim": "unresolved", "model_decision": "retain_evidence", "science_decision": "unresolved", "next_action": "discriminate"}})
        code, output = self.invoke("assess", "--record", str(path))
        value = json.loads(output)
        self.assertEqual(code, 0, value)
        self.assertTrue(value["valid"])
        self.assertFalse(value["promotion_eligible"])
        self.assertIn("allowed_claims", value)


if __name__ == "__main__":
    unittest.main()
