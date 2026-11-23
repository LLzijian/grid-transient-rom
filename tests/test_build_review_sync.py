import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_review_sync.py"


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
    before = git("rev-parse", "HEAD")

    report_path = "reports/run01/result.md"
    report = "# 正文\n\n| RMSE |\n| --- |\n| 0.25 |\n".encode()
    put(report_path, report)
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
        "00_INDEX.md", "LATEST_DIGEST.md", "LATEST_GIT_CHANGES.md", report_path,
        "docs/current_state.md", "results/run01/metrics.csv", "results/run01/job/summary.json",
    }
    assert (out / report_path).read_bytes() == report
    assert f"| `{report_path}` | {report_date} |" in (out / "00_INDEX.md").read_text(encoding="utf-8")
    digest = (out / "LATEST_DIGEST.md").read_text(encoding="utf-8")
    assert "Prepare run02" in digest and "# 正文" in digest and "rmse\n0.25" in digest
    changes = (out / "LATEST_GIT_CHANGES.md").read_text(encoding="utf-8")
    assert f"`{before}`" in changes
    assert "+    return 0.25" in changes and "+print('second commit')" in changes
    assert subprocess.run(args, cwd=tmp_path, capture_output=True).returncode != 0
    assert (out / report_path).read_bytes() == report
