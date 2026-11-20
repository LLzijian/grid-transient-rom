"""Summarize completed frozen jobs without opening arrays or running models."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[2]
RUN = "osgkd_fit297_20261113"
RESULTS, ARTIFACTS = ROOT/"results"/RUN, ROOT/"artifacts"/RUN
SOURCES = {}
DIAGNOSTIC_IDS = ["expansion_0138", "newtrain_0586bd3a72003148", "newtrain_e6041b33464f3be1",
                  "pilot_0246", "pilot_0324"]


def read(path, lines=False):
    raw = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return [json.loads(line) for line in raw.splitlines()] if lines else json.loads(raw)


def verify(ref):
    path = ROOT/ref["path"]
    assert path.resolve().is_relative_to(ROOT), "reference outside repository"
    value = read(path)
    assert SOURCES[ref["path"]] == {k:ref[k] for k in ("sha256","bytes")}, "reference identity differs"
    return value


def class_ids(rows, scope):
    keys = ("detected_unstable", "missed_unstable", "false_los", "numerical_unresolved", "latency_uncovered",
            "unsettled", "settled", "uncovered", "autonomous_los", "autonomous_settled", "autonomous_unsettled")
    counts = {key+"_ids":[] for key in keys}
    for row in rows:
        eid, item = row["event_id"], row["classification_"+scope]
        if item is None:
            counts["uncovered_ids"].append(eid)
            continue
        c = item["classification"]
        if c["category"] in ("numerical_unresolved", "latency_uncovered"):
            counts[c["category"]+"_ids"].append(eid)
            continue
        state = "los" if c["predicted_los"] else "settled" if c["window_settled"] else "unsettled"
        counts["autonomous_"+state+"_ids"].append(eid)
        if not item["truth_supported"]:
            counts["uncovered_ids"].append(eid)
        elif c["predicted_los"]:
            counts["detected_unstable_ids" if row["label"] == "unstable" else "false_los_ids"].append(eid)
        else:
            if row["label"] == "unstable":
                counts["missed_unstable_ids"].append(eid)
            counts["settled_ids" if c["window_settled"] else "unsettled_ids"].append(eid)
    return counts


def validate_score(score, rows, sides):
    assert len(rows) == len({r["event_id"] for r in rows}) == 297
    assert Counter(r["label"] for r in rows) == {"stable":267,"unstable":30}
    assert [r["event_id"] for r in rows] == sides["ordered_ids"]
    for split, ids in (("train_all",sides["ordered_ids"]),("train_fit",sides["fit_ids"]),("train_check",sides["check_ids"])):
        cohort = [r for r in rows if r["event_id"] in set(ids)]
        for name, metrics in score["splits"][split].items():
            kind, scope = name.split("_")
            population = [r for r in cohort if kind == "all" or r["label"] == kind]
            scope = scope.removesuffix("s")
            unresolved = [r["event_id"] for r in population if r["scopes"][scope]["metrics"] is None]
            short = [r["event_id"] for r in population if not r["scopes"][scope]["full_horizon_truth"]]
            assert metrics["denominator"] == len(population) and metrics["covered"] == len(population)-len(unresolved)
            assert metrics["unresolved_ids"] == unresolved and metrics["short_record_ids"] == short
            assert (metrics["official"] is None) == (bool(unresolved) or not population)
        for scope, counts in score["classification"][split].items():
            population = cohort if scope == "8s" else [r for r in cohort if r["label"] == "unstable"]
            assert counts["denominator"] == len(population)
            assert counts["unstable_denominator"] == sum(r["label"] == "unstable" for r in population)
            assert counts["stable_denominator"] == sum(r["label"] == "stable" for r in population)
            assert all(counts[k] == v for k,v in class_ids(population,scope).items())


def timing(left, right, scope):
    ids = [eid for eid in left if (scope != "actual" or left[eid]["label"] == "unstable") and all(rows[eid]["classification_"+scope] is not None
           and rows[eid]["classification_"+scope]["truth_supported"]
           and rows[eid]["classification_"+scope]["timing_comparable"] for rows in (left,right))]
    errors = {"C0":[],"C1_dynamic":[]}
    for eid in ids:
        a,b = (rows[eid]["classification_"+scope] for rows in (left,right))
        if a["true_time_record_s"] != b["true_time_record_s"]:
            return dict(ids=ids,statistics=None,reason=f"true record time differs: {eid}")
        for name,item in (("C0",a),("C1_dynamic",b)):
            error = item["trigger_time_error_s"]
            if (error is None or not math.isfinite(error) or item["predicted_time_record_s"] is None
                    or item["true_time_record_s"] is None
                    or not math.isclose(error,item["predicted_time_record_s"]-item["true_time_record_s"],rel_tol=0,abs_tol=1e-12)):
                return dict(ids=ids,statistics=None,reason=f"inconsistent record-time error: {name}/{eid}")
            errors[name].append(error)
    return dict(ids=ids,denominator=len(ids),coordinate="predicted minus true, original record seconds",
                statistics={name:dict(mean_error_s=sum(v)/len(v),mean_absolute_error_s=sum(map(abs,v))/len(v))
                            for name,v in errors.items()} if ids else None,
                reason=None if ids else "no common truth-supported, timing-comparable events")


def main():
    output = RESULTS/"analysis.json"
    assert not output.exists(), "analysis already exists"
    receipts = {name:read(RESULTS/job/"summary.json") for name,job in
                (("C0","train_C0"),("C1","train_C1"),("evaluate","eval01"))}
    for name,receipt in receipts.items():
        assert receipt["run_id"] == RUN and receipt["status"] == "completed"
        assert receipt["phase"] == ("evaluate" if name == "evaluate" else "train")
        for key in ("audit_ref","source_hashes","torch_version","device","threads","qualification_ref"):
            assert receipt["release"][key] == receipts["evaluate"]["release"][key], key
    evaluated = receipts["evaluate"]
    audit = verify(evaluated["release"]["audit_ref"])
    assert audit["status"] == "adopted_for_prediction"
    sides = verify(audit["materialization"])
    assert sides["status"] == "completed" and sides["run_id"] == RUN
    trials, attempts, training = {}, {}, {}
    for name in ("C0","C1"):
        assert verify(evaluated["release"]["train_refs"][name]) == receipts[name]
        trial = receipts[name]["result"]["trials"][name]
        trials[name] = trial
        attempts[name] = read(ARTIFACTS/("train_"+name)/name/"attempts.jsonl",lines=True)
        logs = attempts[name]
        assert [r["attempt"] for r in logs] == list(range(1,len(logs)+1))
        assert trial["attempt_count"] == len(logs) and trial["variant"] == name
        assert all(set(r["event_ids"]) <= set(sides["fit_ids"]) for r in logs)
        assert trial["accepted"] == sum(r["accepted"] for r in logs)
        assert trial["rejected"] == sum(not r["accepted"] for r in logs)
        assert trial["clipped"] == sum(r["clipped"] for r in logs)
        target = "C0" if name == "C0" else "C1_dynamic"
        assert evaluated["result"]["selection"][target] == trial["selected"]
        assert evaluated["release"]["selected_model_refs"][target] == (trial["selected"]["artifact"] if trial["selected"] else None)
        training[name] = {k:trial[k] for k in ("parameter_count","attempt_count","accepted","rejected","clipped","stop","adaptation","function_costs","cost","work")}
        training[name]["selected"] = trial["selected"]
        training[name]["selected_attempt"] = trial["selected"]["attempt"] if trial["selected"] else None
        training[name]["selected_check_J"] = trial["selected"]["check"]["J"] if trial["selected"] else None
        training[name]["checkpoint_curve"] = [{k:r[k] for k in ("attempt","fit","check","numeric")} for r in trial["checkpoints"]]
        training[name]["job_cost"] = receipts[name]["cost"]
    common = min(map(len,attempts.values()))
    assert all((a["event_ids"],a["horizon_s"]) == (b["event_ids"],b["horizon_s"])
               for a,b in zip(attempts["C0"][:common],attempts["C1"][:common],strict=True)), "paired sequence differs"
    rows, scores = {}, {}
    for name,score in evaluated["result"]["scores"].items():
        if score is None:
            scores[name], rows[name] = None, {}
            continue
        raw = read(ARTIFACTS/"eval01"/name/"rows.json")
        validate_score(score,raw,sides)
        assert set(DIAGNOSTIC_IDS) <= {r["event_id"] for r in raw}
        rows[name] = {r["event_id"]:r for r in raw}
        scores[name] = {**score,"splits":{split:{key:{k:v for k,v in item.items() if k != "covered_only_diagnostic"}
                               for key,item in values.items()} for split,values in score["splits"].items()}}
    source = evaluated["result"]["C1_score_source"]
    assert source == ("C0" if evaluated["result"]["selection"]["C1_used_static_fallback"] else "C1_dynamic")
    assert evaluated["result"]["selection"]["C1"] == evaluated["result"]["selection"][source]
    classification_counts = {name: {split: {scope: {key:len(value) for key,value in counts.items() if key.endswith("_ids")}
        for scope,counts in scopes.items()} for split,scopes in score["classification"].items()}
        for name,score in scores.items() if score is not None}
    comparisons = {}
    if all(scores[n] is not None for n in ("C0","C1_dynamic")):
        for split in scores["C0"]["classification"]:
            comparisons[split] = {scope:{key:dict(C0_only=sorted(set(a[key])-set(b[key])),C1_dynamic_only=sorted(set(b[key])-set(a[key])))
                for key in a if key.endswith("_ids")}
                for scope in ("8s","actual")
                for a,b in [(scores["C0"]["classification"][split][scope],scores["C1_dynamic"]["classification"][split][scope])]}
    result = dict(run_id=RUN,status="completed",training=training,paired_sequence=dict(passed=True,common_attempts=common,
                  extra_attempts={n:len(v)-common for n,v in attempts.items()}),scores=scores,C1_score_source=source,
                  classification_counts=classification_counts,classification_differences=comparisons,timing={scope:timing(rows["C0"],rows["C1_dynamic"],scope)
                  if all(rows.values()) else dict(statistics=None,reason="candidate absent") for scope in ("8s","actual")},
                  source_diagnostic_ids_retained=DIAGNOSTIC_IDS,evaluation_cost=evaluated["cost"],evaluation_function_costs=evaluated["function_costs"])
    queue = RESULTS/"resource_queue.json"
    if queue.exists():
        result["resource_wait"] = read(queue)
    raw = Path(__file__).read_bytes()
    result["script"] = dict(path=Path(__file__).relative_to(ROOT).as_posix(),sha256=hashlib.sha256(raw).hexdigest())
    result["sources"] = SOURCES
    with output.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+"\n")
    print(output)


if __name__ == "__main__":
    main()
