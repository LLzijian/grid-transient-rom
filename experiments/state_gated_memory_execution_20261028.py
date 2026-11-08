"""Finite native ownership for the preregistered matched exchange study.

Import and pure checks never load torch/events or start a native Job. No retries,
reserved repairs, external optimizer replay, or process quotas are enabled here.
"""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import force_graph_execution_20261028 as parent
from experiments.r_memory_execution_20261025 import commit_id, read, sha, write_new, typed_identity, source_closure, python_command
BASE = ROOT / "private_data/state_gated_memory_20261028/study01"
PARENT = ROOT / "private_data/force_graph_20261028/study01"
STUDY_MODULE = "experiments.state_gated_memory_study_20261028"
PHASES = ("prepare", "qualify", "qa", "study", "summary")
PHASE_OUTPUTS = {"prepare":"prepare.json", "qualify":"qualification/summary.json", "qa":"qa/summary.json", "study":"study/summary.json", "summary":"summary.json"}
SOURCE_ROOTS=("experiments/state_gated_memory_execution_20261028.py","experiments/state_gated_memory_study_20261028.py","experiments/state_gated_memory_model_20261028.py","tests/test_state_gated_memory_execution_20261028.py","tests/test_state_gated_memory_study_20261028.py","tests/test_state_gated_memory_model_20261028.py")
DOCUMENTS=("AGENTS.md","docs/engineering_loop.md","docs/research_execution_20261028.md","reports/state_gated_memory_20261028_protocol.md","reports/astra6_structure_identification_next_20261028_architecture.md","reports/state_gated_memory_20261028_design_review.md","results/state_gated_memory_20261028/model_pre_freeze_qa.json","results/state_gated_memory_20261028/study_pre_freeze_qa.json","results/state_gated_memory_20261028/execution_pre_freeze_qa.json")
R_N_SOURCE = "private_data/force_graph_20261028/study01/study/N6101_M_28102/checkpoints/step000192"
SOURCE_ASSETS = {R_N_SOURCE+"/model.json":"937f00b740875913923487b8d772f40d2a8145a88eba2cdaf61f2d889715bfc9", R_N_SOURCE+"/checkpoint.json":"2083fcbd3f25679350b7b63f2d7285412995a9ae141be1fb2a4a5cb8f60d0ff9", R_N_SOURCE+"/training.pt":"197ab52b8061de7319f08345ffe3829036caa38c9fcbd7594a0c6936f0ea210b"}
FIXED_RECORDS = {"experiments/force_exchange_model_20261028.py":"010ff2210fb84767fa6a5dee4cfe38826ea8526787a277e09db08e5c7b02d1b4", "tests/test_force_exchange_model_20261028.py":"671c94b5c54478e863b9597f85fc758c3ffa49b529adb7f10fe609c08a39f6ae", "results/force_exchange_20261028/model_pre_freeze_qa.json":"f947ac00d1dfc7cfdc46f8fde15069a52da84abcd7470c8fa25b909c5c0137b4", "results/force_exchange_20261028/matched_control_cpu_qualification.json":"cc566fa31b0fe9b77777f563c14aebcfe90ed24662a8d53adf6b7aadf8493da7"}
NORMAL_CATEGORY_LIMITS = {"qa":108, "formal":1152, "repair":0, "replay":0}
RESERVED_CATEGORY_LIMITS = {"repair":108, "replay":96}
TOTAL_LIMIT = 1464
PARENT_COMPLETE_SHA = "807b99a7814e4c16d03cec99a6100bc30b3a865eff959c3c128ec7ba9c36e71d"
PARENT_SUMMARY_SHA = "cadd7d330e1ea3fc775596e30d3555d694a24501806ced68b87f0ec9cfff44d8"
PER_VARIANT_GATES = ("cpu_qualified", "cpu_cuda_pair_pass", "finite_12x8_backward", "fullfit297_qualified")
CROSS_MODEL_SCOPE = "R_N_A_same_source_fit297_bidirectional_RK4_fine005_base01_8s_actual15"


def cross_model_evidence(base, result):
    """Validate the exact scientific prefix and bind it to following native phases."""
    path = (base / "qualification/R_N_A_cross_model.json").resolve()
    link = result.get("R_N_A_cross_model")
    if (result.get("R_N_A_cross_model_qualified") is not True or not isinstance(link, dict)
            or not isinstance(link.get("path"), str) or Path(link["path"]).resolve() != path
            or not path.is_file() or link.get("sha256") != sha(path)):
        raise ValueError("cross-model explicit qualification and exact prefix path/hash required")
    value, prepared = read(path), read(base / "prepare.json")
    expected_ids = prepared.get("fit_ids", [])
    alignment = {"fine_dt": .005, "output_dt": .01, "stride": 2, "coarse_stride": 2,
        "history_rows": 11, "history_dt": .01, "initial_observation_index": 10,
        "eight_second_shapes": [1601, 801]}
    if (value.get("status") != "PASS" or value.get("qualified") is not True
            or type(value.get("record_count")) is not int or value["record_count"] != 297
            or len(expected_ids) != 297 or len(set(expected_ids)) != 297
            or value.get("event_ids") != expected_ids or value.get("source") != prepared.get("source")
            or value.get("scope") != CROSS_MODEL_SCOPE or value.get("bidirectional") is not True
            or value.get("predictions_recomputed") is not False
            or any(value.get("alignment", {}).get(k) != v for k, v in alignment.items())):
        raise ValueError("cross-model fit297 source, alignment or physical comparison scope differs")
    rows = value.get("rows", [])
    if len(rows) != 297 or [r.get("event_id") for r in rows] != expected_ids:
        raise ValueError("cross-model prefix must contain every registered fit event exactly once")
    for scope in ("8s", "actual15"):
        aggregate = value.get("aggregates", {}).get(scope, {})
        if aggregate.get("event_count") != 297 or aggregate.get("qualified_count") != 297:
            raise ValueError("cross-model prefix has unresolved physical comparison rows")
        for row in rows:
            interval = row.get("intervals", {}).get(scope, {})
            n = row.get("actual_steps")
            if (row.get("qualified") is not True or interval.get("qualified") is not True
                    or type(n) is not int or not 1 <= n <= 1500
                    or interval.get("actual_output_steps") != (800 if scope == "8s" else n)
                    or interval.get("full_R_N_fine_finite") is not True
                    or interval.get("full_A_fine_finite") is not True):
                raise ValueError("cross-model genuine interval, bidirectional qualification or fine finiteness differs")
    return {str(path): sha(path)}


def checked_base(base=BASE):
    base=Path(base).resolve()
    if base != ROOT / "private_data/state_gated_memory_20261028/study01":
        raise ValueError("only the registered exchange study01 output is allowed")
    return base


def check_locks(base=BASE, *, state=None, owned_identity=None):
    base=checked_base(base)
    # Parent checks exact immutable r-memory and training_eff locks and actual
    # GONE identities. A completed parent must have no active force_graph lock.
    proofs=parent.check_locks(PARENT, state=state)
    own=base/"active.lock"
    if own.exists():
        actual=typed_identity(read(own).get("worker"))
        if owned_identity is None or any(actual[k]!=owned_identity[k] for k in ("pid","creation_filetime_utc")):
            raise ValueError("new lock is another or unowned native worker")
    elif owned_identity is not None:
        raise ValueError("owned worker needs its actual new lock")
    return proofs


def clean_receipt(value, source_commit, phase):
    value=parent.clean_receipt(value, source_commit, phase)
    if any(value.get(k) is not None for k in ("runtime_limits","cpu_limits","active_process_limit")):
        raise ValueError("registered native execution must have no process quotas")
    return value


def preserved_cpu_records(files):
    for name,expected in FIXED_RECORDS.items():
        actual=sha(ROOT/name)
        if actual!=expected: raise ValueError("frozen CPU/model evidence changed: "+name)
        files[str(ROOT/name)]=actual
    failed=read(ROOT/"results/force_exchange_20261028/model_pre_freeze_qa.json")
    matched=read(ROOT/"results/force_exchange_20261028/matched_control_cpu_qualification.json")
    if failed.get("training_admission") is not False or matched.get("training_admission") is not False or matched.get("status")!="PASS_FOR_MATCHED_CONTROLS":
        raise ValueError("matched controls must never relabel the failed old admission")
    files[str(ROOT/"results/state_gated_memory_20261028/model_pre_freeze_qa.json")]=sha(ROOT/"results/state_gated_memory_20261028/model_pre_freeze_qa.json")
    return {"legacy_dropin":"FAILED_NOT_ADOPTED", "matched_cpu_only":"PASS_FOR_MATCHED_CONTROLS", "real_training_admission":False}


def parent_runtime_proof(files, *, state=None):
    from experiments import uncapped_windows_job_20261021 as job
    state=job.identity_state if state is None else state
    if (sha(PARENT/"complete.json")!=PARENT_COMPLETE_SHA or sha(PARENT/"summary.json")!=PARENT_SUMMARY_SHA
            or (PARENT/"active.lock").exists()):
        raise ValueError("the exact closed parent and absent parent lock are required")
    binding=read(PARENT/"binding.json");runtime=binding["runtime"]
    expected=python_command([])
    if any(expected.get(k)!=v for k,v in runtime["command"].items()):
        raise ValueError("current direct-base command differs from owned parent runtime")
    # Reuse exact previously frozen runtime identities; no new CUDA probe or
    # duplicate binary hashing here. Native supervise verifies bound files.
    for name in runtime["files"]:files[name]=binding["files"][name]
    old=read(ROOT/"results/r_memory_20261025/recovery_20261028.json")["binding_comparison"]
    hashes={r["path"]:r["actual_sha256"] for r in old["files"]}
    if old["total"]!=87 or old["matched"]!=87 or old["changed"] or old["missing"]:
        raise ValueError("original full runtime audit evidence is required")
    if any(files[n]!=hashes.get(n) for n in runtime["files"]):raise ValueError("parent runtime differs from historical proven runtime")
    receipts={}
    for phase in parent.PHASES:
        paths,_=parent._completed_phase(PARENT,binding["source_commit"],phase)
        files.update(paths)
        rp=PARENT/"native"/phase/"receipt.json";r=clean_receipt(read(rp),binding["source_commit"],phase)
        if state(r["worker"])!="GONE":raise ValueError("a previous native owner is live, unknown or reused")
        for name in ("experiments/uncapped_windows_job_20261021.py","experiments/rational_python_runtime_20261022.py"):
            if r["source_before"].get(str(ROOT/name))!=sha(ROOT/name):raise ValueError("native ownership helper changed")
        receipts[phase]={"path":str(rp),"sha256":sha(rp),"worker_pid_decimal":str(r["worker"]["pid"]),"worker_filetime_decimal":str(r["worker"]["creation_filetime_utc"])}
    files.update({str(PARENT/n):sha(PARENT/n) for n in ("complete.json","summary.json","binding.json","prepare.json","schedule.json")})
    return runtime,{"six_actual_parent_receipts":receipts,"runtime_probe_rerun":False,"large_runtime_binaries_rehashed_during_preparation":False}


def freeze(source_commit, base=BASE):
    commit_id(source_commit);base=checked_base(base)
    if base.exists():raise FileExistsError("preserve existing output; no re-freeze")
    locks=check_locks(base)
    paths=source_closure(ROOT,SOURCE_ROOTS)+[ROOT/n for n in DOCUMENTS]+[ROOT/n for n in SOURCE_ASSETS]
    files={str(p.resolve()):sha(p) for p in paths}
    for name,expected in SOURCE_ASSETS.items():
        if files[str(ROOT/name)]!=expected:raise ValueError("fixed F asset changed: "+name)
    cpu=preserved_cpu_records(files)
    runtime,proof=parent_runtime_proof(files)
    prepared=read(ROOT/parent.OLD/"prepare.json")
    for name,key in (("fit_events.npz","fit_cache_sha256"),("no_update_events.npz","no_update_cache_sha256")):
        p=ROOT/parent.OLD/name
        if sha(p)!=prepared[key]:raise ValueError("original prepared cache bytes changed")
        files[str(p)]=prepared[key]
    files[str(ROOT/parent.OLD/"prepare.json")]=sha(ROOT/parent.OLD/"prepare.json")
    binding={"source_commit":source_commit,"scope":"state_gated_memory_20261028 finite matched A/B/C","base":str(base),"files":files,"runtime":runtime,"runtime_proof":proof,"cpu_evidence":cpu,"historical_locks":locks,"source_assets":SOURCE_ASSETS,"no_process_quotas":True,"project_val_test_accessed":False,"optimizer_call_upper_bound":TOTAL_LIMIT,"normal_category_limits":NORMAL_CATEGORY_LIMITS,"reserved_category_limits":RESERVED_CATEGORY_LIMITS,"automatic_resume":False,"repair_or_external_resume_requires_separate_explicit_root_execution":True}
    base.mkdir(parents=True,exist_ok=False);write_new(base/"binding.json",binding);return binding


def call_counts(base=BASE):
    path=Path(base)/"optimizer_calls.jsonl";calls={};counts={k:{"attempted":0,"completed":0} for k in NORMAL_CATEGORY_LIMITS}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r=json.loads(line);category=r.get("category");cid=r.get("call_id")
            if category not in counts or not isinstance(cid,str) or not cid:raise ValueError("registered category and actual call_id required")
            if r.get("event")=="optimizer_step_attempt" and cid not in calls:
                calls[cid]=dict(r,completed=False);counts[category]["attempted"]+=1
            elif r.get("event")=="optimizer_step_completed" and cid in calls and not calls[cid]["completed"] and all(r.get(k)==calls[cid].get(k) for k in ("category","run_id","step")):
                calls[cid]["completed"]=True;counts[category]["completed"]+=1
            else:raise ValueError("optimizer call identity/order mismatch")
    if any(v["attempted"]>NORMAL_CATEGORY_LIMITS[k] for k,v in counts.items()):raise ValueError("registered category bound or disabled repair/replay exceeded")
    total=sum(v["attempted"] for v in counts.values())
    if total>TOTAL_LIMIT:raise ValueError("1464-call upper bound exceeded")
    return {"categories":counts,"attempted_total":total,"completed_total":sum(v["completed"] for v in counts.values()),"pending_attempts":[{"call_id":cid,"category":r["category"],"run_id":r.get("run_id"),"step":r.get("step")} for cid,r in calls.items() if not r["completed"]],"reserved_repair_calls_used":0}


def phase_inputs(base, source_commit, phase):
    if phase not in PHASES:raise ValueError("unregistered phase")
    files={str(base/"binding.json"):sha(base/"binding.json")}
    dependencies=PHASES[:PHASES.index(phase)]
    for previous in dependencies:
        paths,_=_completed_phase(base,source_commit,previous);files.update(paths)
    counts=call_counts(base)
    if counts["pending_attempts"] and phase!="summary":raise ValueError("pending actual optimizer call requires root reconciliation")
    if "qa" in dependencies:
        expected=read(base/"qa/summary.json")["actual_optimizer_updates"]
        if counts["categories"]["qa"]!={"attempted":expected,"completed":expected}:raise ValueError("actual per-variant QA journal differs")
    return files


def check_binding(binding, source_commit, base=BASE):
    from experiments import uncapped_windows_job_20261021 as job
    base = checked_base(base)
    if (binding.get("source_commit") != commit_id(source_commit) or binding.get("base") != str(base)
            or binding.get("no_process_quotas") is not True or not binding.get("files")
            or binding.get("optimizer_call_upper_bound") != TOTAL_LIMIT
            or binding.get("normal_category_limits") != NORMAL_CATEGORY_LIMITS):
        raise ValueError("a current same-source exchange binding is required")
    job.verify_files(binding["files"])
    command = python_command([])
    expected = binding["runtime"]["command"]
    if any(command.get(k) != v for k, v in expected.items()):
        raise ValueError("current direct-base runtime differs from frozen command")
    return command

def _completed_phase(base, source_commit, phase):
    receipt_path = base / "native" / phase / "receipt.json"
    receipt = clean_receipt(read(receipt_path), source_commit, phase)
    bound_files = read(base / "binding.json")["files"]
    if any(receipt["source_before"].get(name) != digest for name, digest in bound_files.items()):
        raise ValueError("the completed phase was not executed against this exact source binding")
    folder = base / "ownership" / phase
    admission_path = folder / "admission.json"
    admission = read(admission_path)
    if (admission.get("source_commit") != source_commit or admission.get("phase") != phase
            or receipt["source_before"].get(str(admission_path)) != sha(admission_path)
            or any(receipt["source_before"].get(name) != digest or sha(name) != digest
                   for name, digest in admission["inputs"].items())):
        raise ValueError("the completed phase prerequisite evidence or admission changed")
    result_path = base / PHASE_OUTPUTS[phase]
    result = read(result_path)
    completion = read(folder / "complete.json")
    if (completion.get("status") != "COMPLETED" or completion.get("source_commit") != source_commit
            or completion.get("phase") != phase or completion.get("scientific_output_sha256") != sha(result_path)):
        raise ValueError("the native worker's completed scientific output changed")
    prefix_files = {}
    if phase == "prepare":
        if result.get("val_read") is not False or result.get("test_read") is not False:
            raise ValueError("prepared cohort must explicitly preserve closed val/test")
    elif phase == "qualify":
        eligible=result.get("eligible_variants",[]);variants=result.get("per_variant",{})
        if result.get("status")!="COMPLETED" or result.get("training_method")!="rk4" or set(variants)!={"A","B","C"} or len(set(eligible))!=len(eligible) or not set(eligible)<={"A","B","C"}:
            raise ValueError("explicit per-variant scientific qualification required")
        if result.get("history_dt_main")!=.0025 or result.get("history_dt_reference")!=.00125:raise ValueError("registered paired history refinement differs")
        for name,row in variants.items():
            if row.get("training_admission") is not (name in eligible):raise ValueError("eligible branches must match actual individual admission")
            if name in eligible and any(row.get(k) is not True for k in PER_VARIANT_GATES):
                raise ValueError("eligible variant has an unresolved scientific gate")
        if "A" in eligible:prefix_files=cross_model_evidence(base,variants["A"])
    elif phase == "qa":
        eligible=result.get("eligible_variants",[]);models=result.get("models",[])
        if result.get("status")!="COMPLETED" or not 0<=result.get("actual_optimizer_updates",-1)<=108 or len(set(eligible))!=len(eligible):raise ValueError("bounded per-variant real QA required")
        if set(eligible)!={r["variant"] for r in models if r.get("status")=="PASS"}:raise ValueError("eligible variants must match passed QA")
        for row in models:
            if row.get("status")=="PASS" and row.get("actual_optimizer_calls")!=36:raise ValueError("each eligible variant requires actual36 Adam/RNG calls")
    # A completed study/summary can honestly contain UNRESOLVED science. This
    # controller does not turn a native exit into a model-qualification decision.
    return {str(receipt_path): sha(receipt_path), str(result_path): sha(result_path), **prefix_files}, result

def _study_call(base, phase):
    module = importlib.import_module(STUDY_MODULE)
    return getattr(module, phase)(base) if phase in ("prepare", "summary") else getattr(module, phase)(base, device="cuda")

def worker(base, source_commit, phase):
    from experiments import uncapped_windows_job_20261021 as job
    base = checked_base(base)
    binding = read(base / "binding.json")
    command = binding["runtime"]["command"]
    folder = base / "ownership" / phase
    identity = {**job.identity(), "physical_image": job.physical_image(job.api(), job.api().GetCurrentProcess()),
        "logical_executable": command["logical_executable"], "runtime_scheme": command["runtime_scheme"],
        "source_commit": source_commit, "phase": phase}
    identity["pid_decimal"] = str(identity["pid"])
    identity["creation_filetime_decimal"] = str(identity["creation_filetime_utc"])
    write_new(folder / "worker.json", identity)
    while not (folder / "owned").is_file():
        time.sleep(.02)
    try:
        check_binding(binding, source_commit, base)
        job.verify_files(read(folder / "admission.json")["inputs"])
        phase_inputs(base, source_commit, phase)
        check_locks(base, owned_identity=identity)
        result = _study_call(base, phase)
        counts = call_counts(base)
        write_new(folder / "complete.json", {"status": "COMPLETED", "source_commit": source_commit,
            "phase": phase, "scientific_result_type": type(result).__name__, "optimizer_calls": counts,
            "scientific_output_sha256": sha(base / PHASE_OUTPUTS[phase])})
    except BaseException as error:
        write_new(folder / "failure.json", {"source_commit": source_commit, "phase": phase,
            "error": repr(error), "traceback": traceback.format_exc()})
        raise

def launch(base, source_commit, phase):
    from experiments import uncapped_windows_job_20261021 as job
    base = checked_base(base)
    binding = read(base / "binding.json")
    check_binding(binding, source_commit, base)
    check_locks(base)
    inputs = phase_inputs(base, source_commit, phase)
    folder, native = base / "ownership" / phase, base / "native" / phase
    if folder.exists() or native.exists() or (base / PHASE_OUTPUTS[phase]).exists():
        raise FileExistsError("preserve the sole phase attempt; no automatic resume/repair")
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / "admission.json", {"source_commit": source_commit, "phase": phase,
        "binding_sha256": sha(base / "binding.json"), "inputs": inputs,
        "historical_locks": check_locks(base), "device": "cuda" if phase not in ("prepare", "summary") else "cpu",
        "dtype": "float64", "automatic_resume": False})
    files = dict(binding["files"])
    files.update(inputs)
    files[str(folder / "admission.json")] = sha(folder / "admission.json")
    command = python_command(["-B", str(Path(__file__).resolve()), phase,
        "--source-commit", source_commit, "--base", str(base), "--worker"])

    def owned(_identity):
        with (folder / "owned").open("x", encoding="utf-8") as stream:
            stream.write("owned\n")

    receipt = job.supervise(command, native, base / "active.lock", files, ROOT,
        worker_receipt=folder / "worker.json", on_worker_owned=owned,
        provenance={"source_commit": source_commit, "phase": phase})
    clean_receipt(receipt, source_commit, phase)
    _completed_phase(base, source_commit, phase)
    return receipt

def run(base, source_commit):
    base = checked_base(base)
    if not (base / "binding.json").exists():
        freeze(source_commit, base)
    for phase in PHASES:
        # Completed phases can be inspected but never silently rerun. This also
        # permits run after an explicit standalone freeze, before any Job exists.
        launch(base, source_commit, phase)
    write_new(base / "complete.json", {"status": "COMPLETED", "source_commit": source_commit,
        "optimizer_calls": call_counts(base), "model_adopted": False,
        "scope": "finite registered execution ended; scientific decisions remain root-owned"})

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", *PHASES, "run"))
    parser.add_argument("--source-commit", required=True, type=commit_id)
    parser.add_argument("--base", default=BASE, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    base = checked_base(args.base)
    if args.worker:
        if args.phase not in PHASES:
            parser.error("only an actual registered phase may be a worker")
        worker(base, args.source_commit, args.phase)
    elif args.phase == "freeze":
        freeze(args.source_commit, base)
    elif args.phase == "run":
        run(base, args.source_commit)
    else:
        launch(base, args.source_commit, args.phase)

if __name__ == "__main__":
    main()
