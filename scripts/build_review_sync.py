"""Build the plain-text review bundle from committed files at HEAD."""

import argparse
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote


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
            return before, "本次推送前的分支提交"
    parent = git_text("show", "-s", "--format=%P", head).split()
    fallback = "推送前的提交不可用，使用" if before else "未提供推送前的提交，使用"
    if parent:
        return parent[0], f"{fallback}当前提交的第一父提交"
    return git("hash-object", "-t", "tree", "--stdin", data=b"").decode().strip(), f"{fallback}空树"


def code_block(value, language="", limit=20_000):
    trimmed = value[:limit].rstrip()
    fence = "```"
    while fence in trimmed:
        fence += "`"
    ending = "\n\n内容已截取。" if limit is not None and len(value) > limit else ""
    return f"{fence}{language}\n{trimmed or '无'}\n{fence}{ending}\n"


def field(data, *keys):
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def read_json(files, path):
    try:
        return json.loads(files.get(path, b"null"))
    except (ValueError, RecursionError):
        return None


def number(value):
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value):
        return f"{value:.5g}"
    return "unknown"


def event_count(scores, window, ids_key, denominator_key):
    value = field(scores, window, ids_key)
    count = len(value) if isinstance(value, list) else value
    denominator = field(scores, window, denominator_key)
    if type(count) is int and type(denominator) is int and denominator > 0 and 0 <= count <= denominator:
        return f"{count}/{denominator}"
    return "unknown"


def analysis_summary(analysis):
    # shortcut: Extract the archived OSGKD score formats; use the report lead for other formats.
    scores = field(analysis, "official_scores") or field(analysis, "scores")
    flat = isinstance(field(scores, "stable_8s", "refined"), dict)
    variants = field(scores, "stable_8s") if flat else scores
    if not isinstance(variants, dict):
        return ""

    labels = {"C1_dynamic": "C1 动态候选", "parent": "原 GRU" if flat else "父模型",
              "2s": "2 s 组", "8s": "8 s 组", "refined": "一步历史精修",
              "control": "对照组", "candidate": "候选组"}
    stable = []
    detected = []
    for name, row in variants.items():
        score = row if flat else field(row, "splits", "train_all", "stable_8s")
        classes = (field(analysis, "classifications", name) if flat
                   else field(row, "classification", "train_all"))
        risk = number(field(score, "official", "motion_risk_v1"))
        if risk == "unknown":
            continue
        label = labels.get(name, name)
        false_alarms = event_count(classes, "8s", "false_los_ids", "stable_denominator")
        hits = event_count(classes, "actual", "detected_unstable_ids", "unstable_denominator")
        stable.append(f"{label} {risk}（{false_alarms}）")
        detected.append(f"{label} {hits}")
    if not stable:
        return ""

    parts = [f"train 稳定样本 8 s 运动风险（括号内为误报数/样本数） {'，'.join(stable)}。",
             f"失稳样本真实区间检出 {'，'.join(detected)}。"]
    history = field(analysis, "history", "history_objective", "fine")
    if isinstance(history, dict):
        before = number(history.get("before_sum"))
        after = number(history.get("effective_history_after_sum"))
        stable_n = field(analysis, "fine_history_by_label", "stable", "denominator")
        unstable_n = field(analysis, "fine_history_by_label", "unstable", "denominator")
        n = stable_n + unstable_n if type(stable_n) is int and type(unstable_n) is int else "unknown"
        parts.append(f"train 归一化历史平方残差合计（{n} 条）为原 GRU {before}、一步历史精修 {after}。")
    early = field(analysis, "early", "parent_misses", "capped_0.25")
    if isinstance(early, dict):
        control = field(early, "control")
        candidate = field(early, "candidate")
        n = number(field(control, "denominator"))
        metrics = []
        for name, label in (("tau_e_normalized_mse", "总电气归一化均方误差"),
                            ("motion_mse", "运动均方误差")):
            a = number(field(control, "metrics", name, "mean"))
            b = number(field(candidate, "metrics", name, "mean"))
            metrics.append(f"{label} {a} → {b}")
        parts.append(f"train 原 {n} 条漏检样本在各自真实区间内（最长 0.25 s），对照组到候选组的"
                     f"{'，'.join(metrics)}。")
    return " ".join(parts)


def report_lead(content, run):
    for paragraph in re.split(r"\n\s*\n", content.decode("utf-8-sig")):
        lines = (line.strip() for line in paragraph.splitlines())
        prose = " ".join(line for line in lines if line and not line.startswith("#"))
        if not prose or prose.startswith(("|", "```", "<!--", "---")):
            continue
        prose = re.sub(r"(?:[ ·]*\[[^\]]+\]\([^)]+\))+$", "", prose)
        if run in prose and re.match(r"(?:Run\b|日期|\d{4}-\d{2}-\d{2})", prose):
            continue
        if conclusion := re.match(r"^\*\*([^*]+)\*\*", prose):
            prose = conclusion.group(0)
        return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", prose)
    return ""


def ledger_table(files, recent):
    runs = {PurePosixPath(path).stem for path in files if path.startswith("harness/runs/")}
    runs.update(path.split("/")[1] for path in files if path.count("/") == 2 and (
        path.startswith("reports/") and path.endswith("/result.md") or
        path.startswith("results/") and path.endswith("/analysis.json")
    ))
    order = {path: i for i, path in enumerate(recent)}
    archived = {PurePosixPath(path).name: path for path in reversed(recent)
                if path.startswith("reports/") and path.endswith(("_conclusion.md", "_scientific_assessment.md"))}
    rows = []
    for run in runs:
        analysis = f"results/{run}/analysis.json"
        registration = f"harness/runs/{run}.json"
        run_record = read_json(files, registration)
        conclusion = f"reports/{run}/conclusion.md"
        final_report = (
            field(run_record, "execution", "status") == "COMPLETED"
            and field(run_record, "evidence", "report", "path") == conclusion
            and conclusion in files
        )
        report = conclusion if final_report else next((path for path in (
            f"reports/{run}/result.md", conclusion,
            archived.get(f"{run}_conclusion.md"), f"reports/{run}/final_assessment.md",
            f"reports/{run}/scientific_assessment.md", archived.get(f"{run}_scientific_assessment.md"),
        ) if path in files), None)
        summary = "" if final_report else analysis_summary(read_json(files, analysis))
        if not summary and report:
            summary = report_lead(files[report], run)
        sources = [(label, path) for label, path in (
            ("原始 analysis.json", analysis), ("报告", report), ("Run 登记", registration),
        ) if path in files]
        links = " · ".join(f"[{label}]({quote(path)})" for label, path in sources)
        rank = min(order.get(path, len(order)) for path in (analysis, report, registration))
        label = run.replace("|", r"\|")
        summary = " ".join((summary or "unknown").splitlines()).replace("|", r"\|")
        rows.append((rank, run, f"| `{label}` | {summary} | {links} |"))
    return "\n".join(["| Run | 核心指标和结论 | 来源 |", "| --- | --- | --- |",
                      *(row for _, _, row in sorted(rows))])


def build_bundle(out, before):
    head = git_text("rev-parse", "HEAD").strip()
    base, comparison = comparison_base(head, before)
    changed = set(git_text("diff", "--name-only", "-z", "--no-renames", base, head).split("\0"))
    files = read_review_files(head)
    dates = modification_dates(head, files)
    recent = [*dates, *(path for path in files if path not in dates)]

    reports = [path for path in recent if path.startswith(REPORTS) and path.endswith(".md")]
    reports.sort(key=lambda path: (path not in changed, not path.endswith("/result.md")))

    commit = git_text("log", "-1", "--format=%B", head).strip()
    table = ledger_table(files, recent)
    digest = ["# 最新研究简报", "## Run 总账", table, f"提交 `{head}`",
              f"本次差异比较 `{base}` → `{head}`（{comparison}）。",
              "## 提交说明", code_block(commit, limit=3000),
              "总账也可单独查看 `00_LEDGER.md`。完整路径见 `00_INDEX.md`，"
              "提交和源码差异见 `LATEST_GIT_CHANGES.md`。",
              "## 最近的报告"]
    for path in reports[:3]:
        body = files[path].decode("utf-8-sig").strip()
        digest.extend([f"### `{path}`", "\n".join(f"> {line}" for line in body[:7000].splitlines())])
        if len(body) > 7000:
            digest.append("内容已截取。全文保存在上面的原路径。")

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
               "## 最近 10 条提交", "从本次同步的提交沿分支历史回看，附各提交的文件统计。",
               code_block(log, limit=None), "## 本次比较的变更统计", code_block(stats),
               "## 本次比较的核心 Python 差异", code_block(patch, "diff")]
    generated = {"00_INDEX.md": "\n".join(index),
                 "00_LEDGER.md": f"# Run 总账\n\n提交 `{head}`\n\n{table}",
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
