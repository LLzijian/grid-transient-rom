"""Synthetic checks for path safety, incomplete access, and static dependency labels."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("repository_inventory", Path(__file__).resolve().parents[1] / "tools/repository_inventory.py")
inv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inv)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_unsafe_output_and_existing_snapshot_are_rejected(self):
        for name in ("../elsewhere", "results/repository_organization/../escape", "artifacts/test", "C:/outside"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                inv.inventory(self.root, name)
        path = self.root / "results/repository_organization/existing"
        path.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            inv.inventory(self.root, "results/repository_organization/existing")

    def test_links_are_not_followed_and_permission_failure_is_not_absence(self):
        with patch.object(inv.os, "scandir", side_effect=PermissionError):
            self.root.joinpath("reports").mkdir()
            rows = list(inv.walk(self.root, "reports"))
            self.assertEqual(rows[0][2:], ("unreadable", "PermissionError"))
        with patch.object(inv.Path, "lstat", side_effect=PermissionError):
            self.assertEqual(inv.observe(self.root, "private_data/blocked")["status"], "inaccessible")
        with patch.object(inv, "linked", return_value=True):
            rows = list(inv.walk(self.root, "reports"))
            self.assertEqual(rows[0][2], "link")
            with self.assertRaises(ValueError):
                inv.safe_path(self.root, "reports/a.md")

    def test_imports_hash_bindings_and_dynamic_scans_remain_distinct(self):
        rows = inv.references("experiments/new.py", "import old\np.rglob('*.json')\nx='results/old/model.json'\n",
                              {"old": ["experiments/old.py"]})
        self.assertTrue(any(r["target"] == "experiments/old.py" and r["kind"] == "python_import_clue" for r in rows))
        self.assertTrue(any(r["kind"] == "dynamic_scan_requires_review" and r["target"] is None for r in rows))
        self.assertEqual(inv.hash_bindings("release.json", {"source_sha256": {"experiments/old.py": "a" * 64}})[0]["kind"], "path_hash_binding")
        links = inv.references("reports/x.md", "[safe](../results/a.json) [escape](../../outside)", {})
        self.assertEqual([r["target"] for r in links], ["results/a.json"])

    def test_payloads_and_ignored_files_are_not_read_as_metadata(self):
        row = dict(kind="file", storage="versioned", bytes=100)
        for name in ("results/a/observations.json", "results/a/model.npz", "results/a/model.json"):
            self.assertFalse(inv.text_allowed(name, row))
        self.assertTrue(inv.text_allowed("results/a/execution_release.json", row))
        self.assertFalse(inv.text_allowed("results/a/execution_release.json", dict(row, storage="local_ignored")))

    def test_inventory_covers_files_without_changing_originals(self):
        for name in inv.ROOT_FILES:
            self.write(name, "# fixture\n")
        original = self.write("reports/old.md", "# unchanged\n")
        self.write("experiments/old.py", "x=1\n")
        self.write("tests/caller.py", "import old\n")
        self.write("results/run/summary.json", '{"value": 1}')
        before = original.read_bytes()
        def git(*args):
            return ("test-head\n", "") if args[1:3] == ("rev-parse", "HEAD") else ("", "")
        with patch.object(inv, "git", side_effect=git):
            summary = inv.inventory(self.root, "results/repository_organization/test")
        self.assertFalse(summary["original_file_changes_observed"])
        self.assertEqual(original.read_bytes(), before)
        folder = self.root / "results/repository_organization/test"
        rows = [json.loads(line) for line in (folder / "inventory.jsonl").read_text(encoding="utf-8").splitlines()]
        by_path = {row["current_path"]: row for row in rows}
        self.assertEqual(by_path["experiments/old.py"]["handling"], "protected_in_place")
        self.assertIsNone(by_path["results/run/summary.json"]["sha256"])
        self.assertTrue(all(row["move_authorized"] is False for row in rows))


if __name__ == "__main__":
    unittest.main()
