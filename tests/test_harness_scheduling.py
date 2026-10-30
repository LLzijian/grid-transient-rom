"""Scheduling contract tests: no jobs, data access, or model calls."""

import copy
import unittest
from unittest.mock import patch

from harness_runtime.scheduling import normalize_write_path, ready_tasks


def task(task_id, **changes):
    value = {
        "task_id": task_id, "role": "reviewer", "question": "Check a frozen, synthetic claim",
        "inputs": [{"path": "fixture.json", "sha256": "0" * 64}],
        "outputs": ["review conclusion"], "write_paths": [], "depends_on": [],
        "resources": {}, "status": "pending", "acceptance": "State the evidence and limits",
        "stop_condition": "Return the bounded review",
    }
    value.update(changes)
    return value


class SchedulingTests(unittest.TestCase):
    def test_read_only_tasks_parallel_and_input_order(self):
        tasks = [task("first", priority=-1), task("second", priority=100), task("third")]
        result = ready_tasks(tasks, {}, 2)
        self.assertTrue(result["valid"])
        self.assertEqual(result["ready"], ["first", "second"])
        self.assertFalse(result["dispatch_performed"])
        self.assertIn("not_granted", result["authorization"])
        self.assertEqual(result["blocked"]["third"], ["host_slots_unavailable"])

    def test_dependencies_require_completion_not_selection(self):
        tasks = [task("done", status="completed"), task("parent", depends_on=["done"]), task("child", depends_on=["parent"])]
        result = ready_tasks(tasks, {}, 5)
        self.assertEqual(result["ready"], ["parent"])
        self.assertIn("dependency_not_completed:parent", result["blocked"]["child"])

    def test_missing_dependency_and_cycle_are_errors(self):
        for tasks, code in [
            ([task("x", depends_on=["missing"])], "missing_dependency"),
            ([task("a", depends_on=["b"]), task("b", depends_on=["a"])], "dependency_cycle"),
            ([task("self", depends_on=["self"])], "dependency_cycle"),
            ([task("same"), task("same")], "duplicate_task_id"),
        ]:
            with self.subTest(code=code):
                result = ready_tasks(tasks, {}, 4)
                self.assertFalse(result["valid"])
                self.assertEqual(result["ready"], [])
                self.assertIn(code, {i["code"] for i in result["issues"]})

    def test_running_tasks_occupy_slots_resources_and_writes(self):
        tasks = [
            task("live", status="running", resources={"gpu:0": 1}, write_paths=["src/model"]),
            task("needs_gpu", resources={"gpu:0": 1}),
            task("writes_child", write_paths=["src/model/head.py"]),
            task("review"),
            task("extra"),
        ]
        result = ready_tasks(tasks, {"gpu:0": 1}, 2)
        self.assertTrue(result["valid"])
        self.assertEqual(result["ready"], ["review"])
        self.assertEqual(result["running"], ["live"])
        self.assertIn("resource_unavailable:gpu:0", result["blocked"]["needs_gpu"])
        self.assertTrue(any(x.startswith("write_conflict:live:") for x in result["blocked"]["writes_child"]))
        self.assertEqual(result["slots"]["remaining_after_selection"], 0)

    def test_normalized_prefix_conflict_but_not_similar_sibling(self):
        tasks = [task("writer", write_paths=["src//./model"]), task("child", write_paths=["src/model/a.py"]), task("ancestor", write_paths=["src"]), task("sibling", write_paths=["src2/a.py"])]
        result = ready_tasks(tasks, {}, 9)
        self.assertEqual(result["ready"], ["writer", "sibling"])
        self.assertIn("child", result["blocked"])
        self.assertIn("ancestor", result["blocked"])
        self.assertEqual(normalize_write_path("src//./model/"), "src/model")

    def test_unsafe_write_paths_are_rejected(self):
        paths = ["../escape", "src/../escape", "/absolute", "C:/absolute", "src/a:stream", "src\\file", "src/*", "src/[ab]", "src/{a,b}", ".", "a\x00b"]
        for path in paths:
            with self.subTest(path=path):
                result = ready_tasks([task("bad", write_paths=[path])], {}, 1)
                self.assertFalse(result["valid"])
                self.assertEqual(result["ready"], [])
                self.assertIn("invalid_write_path", {i["code"] for i in result["issues"]})

    def test_windows_case_aliases_share_one_writer(self):
        tasks = [task("writer", write_paths=["src/Model"]),
                 task("alias", write_paths=["SRC/model/HEAD.py"]),
                 task("sibling", write_paths=["src/models"])]
        with patch("harness_runtime.scheduling.os.name", "nt"):
            result = ready_tasks(tasks, {}, 4)
            self.assertEqual(result["ready"], ["writer", "sibling"])
            self.assertIn("write_conflict:writer:src/Model", result["blocked"]["alias"])
            tasks[0]["status"] = tasks[1]["status"] = "running"
            result = ready_tasks(tasks, {}, 4)
            self.assertFalse(result["valid"])
            self.assertIn("active_write_conflict", {i["code"] for i in result["issues"]})

    def test_windows_alias_and_device_components_are_rejected(self):
        for path in ("src/file.", "src /file.py", "src./file.py", "src/AUX",
                     "src/CON.txt", "src/AUX .txt", "src/nul/data", "src/com1.json", "src/LPT²",
                     "src/CONOUT$", "src/a|b", 'src/a"b'):
            with self.subTest(path=path):
                self.assertFalse(ready_tasks([task("bad", write_paths=[path])], {}, 1)["valid"])
        self.assertEqual(normalize_write_path("src/console.py"), "src/console.py")
        self.assertEqual(normalize_write_path("src/COM10.md"), "src/COM10.md")

    def test_conflicting_active_allocations_fail_without_cancelling_work(self):
        cases = [
            ([task("a", status="running"), task("b", status="running")], {}, 1, "active_slots_over_capacity"),
            ([task("a", status="running", resources={"gpu:0": 1})], {}, 1, "active_resource_over_capacity"),
            ([task("a", status="running", write_paths=["src"]), task("b", status="running", write_paths=["src/a.py"])], {}, 2, "active_write_conflict"),
            ([task("a"), task("b", status="running", depends_on=["a"])], {}, 2, "active_dependency_incomplete"),
        ]
        for tasks, resources, slots, code in cases:
            with self.subTest(code=code):
                original = copy.deepcopy(tasks)
                result = ready_tasks(tasks, resources, slots)
                self.assertFalse(result["valid"])
                self.assertEqual(result["ready"], [])
                self.assertIn(code, {i["code"] for i in result["issues"]})
                self.assertEqual(tasks, original)

    def test_zero_slots_and_unavailable_resources_are_not_fabricated_capacity(self):
        result = ready_tasks([task("x")], {}, 0)
        self.assertTrue(result["valid"])
        self.assertEqual(result["ready"], [])
        result = ready_tasks([task("gpu", resources={"gpu:0": 1}), task("cpu")], {}, 5)
        self.assertTrue(result["valid"])
        self.assertEqual(result["ready"], ["cpu"])

    def test_quantities_and_task_fields_checked(self):
        for tasks, resources, slots in [
            ([task("x")], {}, True), ([task("x")], {"gpu:0": True}, 1),
            ([task("x", resources={"gpu:0": -1})], {}, 1),
            ([task("x", resources={"gpu:0": 0.5})], {}, 1),
            ([task("x", priority=True)], {}, 1),
            ([task("x", read_only=True, write_paths=["src"])], {}, 1),
            ([task("x", read_only=False)], {}, 1),
        ]:
            with self.subTest(tasks=tasks, resources=resources, slots=slots):
                self.assertFalse(ready_tasks(tasks, resources, slots)["valid"])

    def test_failed_and_paused_tasks_are_not_ready(self):
        result = ready_tasks([task("failed", status="failed"), task("paused", status="paused"), task("child", depends_on=["failed"])], {}, 5)
        self.assertEqual(result["ready"], [])
        self.assertIn("dependency_not_completed:failed", result["blocked"]["child"])

    def test_inputs_unchanged_and_explicit_implementation_contract(self):
        tasks = [task("implement", read_only=False, write_paths=["src/model.py"], outputs=["src/model.py", "test evidence"])]
        original = copy.deepcopy(tasks)
        capacities = {"gpu:0": 1}
        result = ready_tasks(tasks, capacities, 1)
        self.assertTrue(result["valid"])
        self.assertEqual(tasks, original)
        self.assertEqual(capacities, {"gpu:0": 1})
        self.assertFalse(result["input_hashes_verified"])


if __name__ == "__main__":
    unittest.main()
