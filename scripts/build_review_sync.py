"""Build the plain-text review bundle from committed files at HEAD."""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


REPORTS = ("reports/", "TASK4/reports/")
ROOTS = ("reports", "docs", "results", "campaigns", "harness/runs", "TASK4/reports")
MAX_FILE_BYTES = 512 * 1024
CORE_PYTHON = tuple(f":(glob){root}/**/*.py" for root in ("src", "experiments", "scripts", "tools"))


def git(*args, data=None):
    return subprocess.run(["git", "-c", "core.quotepath=false", *args],
                          input=data, stdout=subprocess.PIPE, check=True).stdout


def git_text(*args):
    return git(*args).decode("utf-8", errors="replace")


def is_metric(path):
    if not path.startswith((*REPORTS, "results/")):
        return False
    name = PurePosixPath(path).name.lower()
    if name.endswith(".csv"):
        return any(term in name for term in ("metric", "summary", "comparison", "paired_events", "scores"))
    if name.endswith(".json"):
        return name in ("analysis.json", "release.json") or "summary" in name or "metrics" in name
    return False


def is_review_file(path):
    if not path.isprintable() or "\\" in path:
        return False
    if path.endswith(".md"):
        return path.startswith((*REPORTS, "campaigns/")) or (
            path.startswith("docs/") and path.count("/") == 1
        )
    return is_metric(path) or (path.startswith("harness/runs/") and path.endswith(".json"))


def read_review_files(head):
    entries = {}
    for record in git("ls-tree", "-r", "-z", "--long", head, "--", *ROOTS).split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, _, blob, size = metadata.split()
        path = raw_path.decode("utf-8", errors="surrogateescape")
        if mode in (b"100644", b"100755") and int(size) <= MAX_FILE_BYTES and is_review_file(path):
            entries[path] = blob

    # Read Git blobs in one process so local edits never enter the bundle.
    data = git("cat-file", "--batch", data=b"".join(blob + b"\n" for blob in entries.values()))
    files = {}
    offset = 0
    for path in entries:
        end = data.index(b"\n", offset)
        size = int(data[offset:end].split()[2])
        start = end + 1
        content = data[start : start + size]
        offset = start + size + 1
        if b"\0" in content or content.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            continue
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        files[path] = content
    return files


def modification_dates(head, files):
    history = git_text("log", "--first-parent", "--diff-merges=first-parent", "--no-renames",
                       "--format=%x00%cI", "--name-only", "-z", head, "--", *ROOTS)
    dates = {}
    date = "unknown"
    for field in history.split("\0"):
        field = field.removeprefix("\n")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}", field):
            date = field
        elif field in files:
            dates.setdefault(field, date)
    return dates


def comparison_base(head, before):
    if before:
        exists = subprocess.run(["git", "cat-file", "-e", f"{before}^{{commit}}"],
                                stderr=subprocess.DEVNULL).returncode == 0
        if exists:
            return before, "推送前的提交"
    parent = git_text("show", "-s", "--format=%P", head).split()
    fallback = "推送前的提交不可用，使用" if before else "使用"
    if parent:
        return parent[0], f"{fallback}上一个提交"
    return git("hash-object", "-t", "tree", "--stdin", data=b"").decode().strip(), f"{fallback}空树"


def code_block(value, language="", limit=20_000):
    trimmed = value[:limit].rstrip()
    fence = "```"
    while fence in trimmed:
        fence += "`"
    ending = "\n\n内容已截取。" if len(value) > limit else ""
    return f"{fence}{language}\n{trimmed or '无'}\n{fence}{ending}\n"


def build_bundle(out, before):
    head = git_text("rev-parse", "HEAD").strip()
    base, comparison = comparison_base(head, before)
    changed = set(git_text("diff", "--name-only", "-z", "--no-renames", base, head).split("\0"))
    files = read_review_files(head)
    dates = modification_dates(head, files)
    recent = [*dates, *(path for path in files if path not in dates)]

    reports = [path for path in recent if path.startswith(REPORTS) and path.endswith(".md")]
    reports.sort(key=lambda path: (path not in changed, not path.endswith("/result.md")))
    metrics = [path for path in recent if is_metric(path)]
    metrics.sort(key=lambda path: (path not in changed, not path.endswith((".csv", "/analysis.json"))))

    commit = git_text("log", "-1", "--format=%B", head).strip()
    digest = ["# 最新研究简报", f"提交 `{head}`", f"差异从 `{base}` 开始比较（{comparison}）。",
              "## 提交说明", code_block(commit, limit=3000),
              "完整路径见 `00_INDEX.md`，提交和源码差异见 `LATEST_GIT_CHANGES.md`。",
              "## 最近的报告"]
    for path in reports[:3]:
        body = files[path].decode("utf-8-sig").strip()
        digest.extend([f"### `{path}`", "\n".join(f"> {line}" for line in body[:7000].splitlines())])
        if len(body) > 7000:
            digest.append("内容已截取。全文保存在上面的原路径。")
    digest.append("## 核心指标")
    for path in metrics[:3]:
        digest.extend([f"### `{path}`", code_block(files[path].decode("utf-8-sig"),
                                                   PurePosixPath(path).suffix[1:], limit=3500)])

    index = ["# 同步文件索引", "", f"提交 `{head}` · {len(files)} 个文件", "",
             "时间取自当前分支中最后一次修改该路径的 Git 提交。", "",
             "| 仓库路径 | 最后修改时间 |", "| --- | --- |"]
    for path in sorted(files):
        label = path.replace("|", r"\|")
        index.append(f"| `{label}` | {dates.get(path, 'unknown')} |")

    log = git_text("log", "-n", "10", "--stat", "--stat-count=40", head)
    stats = git_text("diff", "--stat", "--stat-count=80", base, head)
    patch = git_text("diff", "--no-ext-diff", "--no-textconv", "--no-color", "-U2",
                     base, head, "--", *CORE_PYTHON)
    changes = ["# 最新 Git 变更", f"比较 `{base}` → `{head}`（{comparison}）",
               "## 最近 10 条提交", code_block(log), "## 变更统计", code_block(stats),
               "## 核心 Python 差异", code_block(patch, "diff")]
    generated = {"00_INDEX.md": "\n".join(index),
                 "LATEST_DIGEST.md": "\n\n".join(digest),
                 "LATEST_GIT_CHANGES.md": "\n\n".join(changes)}

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".review-sync-") as staging:
        bundle = Path(staging) / "bundle"
        for path, content in files.items():
            target = bundle / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        bundle.mkdir(exist_ok=True)
        for name, text in generated.items():
            (bundle / name).write_text(text + "\n", encoding="utf-8", newline="\n")
        bundle.rename(out)
    print(f"Collected {len(files)} committed text files in {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("_review_sync"), help="New output directory")
    parser.add_argument("--base", default="", help="Commit before the push")
    args = parser.parse_args()
    if args.base and not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", args.base):
        parser.error("--base must be a full Git commit SHA")
    if args.out.exists() or args.out.is_symlink():
        parser.error("--out must be a new directory")
    build_bundle(args.out, args.base)


if __name__ == "__main__":
    main()
