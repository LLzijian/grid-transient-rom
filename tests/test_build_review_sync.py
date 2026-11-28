import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_review_sync.py"


def test_report_lead_preserves_explicit_and_legacy_conclusions():
    from scripts.build_review_sync import report_lead

    report = "# 结果\n\n**全量验证已完成。** 其余细节单列。[摘要](run01/summary.json)\n\n方法。"
    legacy = "# 结果\n\n2026-11-23。冻结父点完成验证。[摘要](run01/summary.json)\n\n方法。"
    assert report_lead(report.encode(), "run01") == "**全量验证已完成。**"
    assert report_lead(legacy.encode(), "run01") == "2026-11-23。冻结父点完成验证。"


def test_ledger_uses_registered_conclusion_without_replacing_application():
    from scripts.build_review_sync import ledger_table

    application = "reports/run01/result.md"
    conclusion = "reports/run01/conclusion.md"
    registration = "harness/runs/run01.json"
    files = {
        application: "# 冻结申请\n\n**等待授权。**\n".encode(),
        conclusion: "# 结案\n\n**固定评分完成。**\n".encode(),
    }
    record = {"execution": {"status": "BLOCKED"}, "evidence": {"report": {"path": conclusion}}}
    files[registration] = json.dumps(record).encode()
    assert "等待授权" in ledger_table(files, list(files))
    record["execution"]["status"] = "COMPLETED"
    files[registration] = json.dumps(record).encode()
    ledger = ledger_table(files, list(files))
    assert "固定评分完成" in ledger and f"[报告]({conclusion})" in ledger
    assert "等待授权" not in ledger


def test_review_bundle_is_a_snapshot_of_one_push(tmp_path):
    def git(*args, data=None):
        return subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
             "-c", "commit.gpgsign=false", *args],
            cwd=tmp_path, input=data, check=True, capture_output=True,
        ).stdout.decode().strip()

    def put(path, content):
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)

    git("init", "-q")
    put("docs/current_state.md", b"Project state\n")
    git("add", "--", "docs")
    git("commit", "-qm", "Starting point")
    for revision in range(9):
        git("commit", "--allow-empty", "-qm", f"Prior push {revision}")
    before = git("rev-parse", "HEAD")

    report_path = "reports/run01/result.md"
    report = "# 正文\n\n| RMSE |\n| --- |\n| 0.25 |\n".encode()
    analysis_path = "results/run01/analysis.json"
    saved_scores = {
        "scores": {"control": {
            "splits": {"train_all": {"stable_8s": {"official": {"motion_risk_v1": 1.115}}}},
            "classification": {"train_all": {
                "8s": {"false_los_ids": [], "stable_denominator": 267},
                "actual": {"detected_unstable_ids": 3, "unstable_denominator": 30},
            }},
        }},
        "diagnostic": "Per-sample diagnostic data\n" * 1600,
    }
    analysis = (json.dumps(saved_scores, indent=2) + "\n").encode()
    archived_path = "reports/history/originals/run00_conclusion.md"
    put(report_path, report)
    put(analysis_path, analysis)
    put("harness/runs/run01.json", b'{"run_id":"run01"}\n')
    put("harness/runs/run00.json", b'{"run_id":"run00"}\n')
    put("results/run00/analysis.json", b'{invalid json\n')
    put(archived_path, "# 旧报告\n\n历史对照通过 12/12。\n".encode())
    put("results/run01/metrics.csv", b"rmse\n0.25\n")
    put("results/run01/job/summary.json", b'{"stable": 267}\n')
    put("reports/run01/checkpoint.pt", b"binary payload")
    put("reports/run01/binary.md", b"not a text file\0")
    put("reports/run01/oversized.md", b"a" * (512 * 1024 + 1))
    put("private_data/result.md", b"Unpublished inputs\n")
    put("src/kiki/model.py", b"def predict():\n    return 0.25\n")
    git("add", "--all")
    link = git("hash-object", "-w", "--stdin", data=b"../../private_data/result.md")
    git("update-index", "--add", "--cacheinfo", f"120000,{link},reports/run01/link.md")
    git("commit", "-qm", "Archive run01")
    report_date = git("log", "-1", "--format=%cI", "--", report_path)

    put("experiments/run02.py", b"print('second commit')\n")
    git("add", "--", "experiments")
    git("commit", "-qm", "Prepare run02")
    put(report_path, b"Working tree only\n")
    put("reports/local/result.md", b"Uncommitted report\n")

    out = tmp_path / "bundle"
    args = [sys.executable, str(SCRIPT), "--out", str(out), "--base", before]
    subprocess.run(args, cwd=tmp_path, check=True, capture_output=True)
    assert {file.relative_to(out).as_posix() for file in out.rglob("*") if file.is_file()} == {
        "00_INDEX.md", "00_LEDGER.md", "LATEST_DIGEST.md", "LATEST_GIT_CHANGES.md", report_path,
        "docs/current_state.md", "results/run01/metrics.csv", "results/run01/job/summary.json",
        analysis_path, archived_path, "harness/runs/run01.json", "harness/runs/run00.json",
        "results/run00/analysis.json",
    }
    assert (out / report_path).read_bytes() == report
    assert (out / analysis_path).read_bytes() == analysis
    assert f"| `{report_path}` | {report_date} |" in (out / "00_INDEX.md").read_text(encoding="utf-8")
    digest = (out / "LATEST_DIGEST.md").read_text(encoding="utf-8")
    assert "Prepare run02" in digest and "# 正文" in digest and "diagnostic" not in digest
    ledger = (out / "00_LEDGER.md").read_text(encoding="utf-8")
    rows = [line for line in ledger.splitlines() if line.startswith("| `")]
    assert len(rows) == 2
    run01 = next(row for row in rows if row.startswith("| `run01` |"))
    run00 = next(row for row in rows if row.startswith("| `run00` |"))
    assert "对照组 1.115（0/267）" in run01 and "对照组 3/30" in run01
    assert f"[原始 analysis.json]({analysis_path})" in run01
    assert "历史对照通过 12/12。" in run00 and f"[报告]({archived_path})" in run00
    assert "\n".join(rows) in digest.split("## 提交说明", 1)[0]
    changes = (out / "LATEST_GIT_CHANGES.md").read_text(encoding="utf-8")
    head = git("rev-parse", "HEAD")
    assert f"`{before}` → `{head}`（本次推送前的分支提交）" in changes
    assert changes.count("\ncommit ") == 10 and "Prior push 1" in changes and "Prior push 0" not in changes
    assert "+    return 0.25" in changes and "+print('second commit')" in changes
    assert subprocess.run(args, cwd=tmp_path, capture_output=True).returncode != 0
    assert (out / report_path).read_bytes() == report
