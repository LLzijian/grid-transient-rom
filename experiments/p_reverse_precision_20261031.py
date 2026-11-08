"""Fixed reverse-p numerical probe: no rollout, optimizer or parameter fitting.

Wrapper logs must be outside private_data/p_reverse_precision_20261031, since
an existing output directory is deliberately treated as an unreconciled job.
"""
from __future__ import annotations
import copy
from decimal import Decimal
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
PRIVATE = ROOT / "private_data/p_reverse_precision_20261031"
PUBLIC = ROOT / "results/p_reverse_precision_20261031/public_probe.json"
PARENT = ROOT / "private_data/p_port_diagnostic_20261031"
MODELS = ("A96", "step000016")
HISTORY_DT = (.0025, .00125)
SCHEMA = "p_reverse_precision_20261031_v1"
PARENT_COMMIT = "f48987abcc10f5361194ca96419a197f62fa0269"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ident(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(ROOT.resolve()).as_posix(), "sha256": sha(path)}


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray,list,tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.bool_,bool)):
        return bool(value)
    if isinstance(value, (np.integer,int)):
        return int(value)
    if isinstance(value, (float,np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(safe(value),f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write("\n")


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def tensor_hash(value):
    return array_hash(value.detach().cpu().numpy())


def state_identity(model):
    return {name:tensor_hash(value) for name,value in model.state_dict().items()}


def generator_q(g, rpp, storage):
    """Form a and then Q in binary64, exactly as the frozen primary input."""
    g,rpp,storage=(np.asarray(v,dtype=np.float64) for v in (g,rpp,storage))
    n=len(g)
    p=np.zeros((3*n,3*n),dtype=np.float64)
    p[:n,:n],p[:n,n:2*n],p[n:2*n,2*n:]=-np.diag(rpp*storage),-np.diag(g),np.eye(n)
    return -.01*p


def high_precision_reference(q, dps):
    """Independent per-machine 3x3 expm and analytic row on exact binary64 Q."""
    import mpmath as mp
    q=np.asarray(q,dtype=np.float64)
    n=q.shape[0]//3
    exp_rows, analytic_rows = [],[]
    with mp.workdps(dps):
        for i in range(n):
            indices=(i,n+i,2*n+i)
            block=mp.matrix([[mp.mpf(float(q[j,k])) for k in indices] for j in indices])
            if any(block[j,k]!=0 for j,k in ((0,2),(1,0),(1,1),(2,0),(2,1),(2,2))):
                raise ValueError("reference requires the registered upper triangular generator")
            value=mp.expm(block)
            d,b,c=block[0,0],block[0,1],block[1,2]
            f1=mp.expm1(d)/d if d else mp.mpf(1)
            f2=(mp.expm1(d)-d)/(d*d) if d else mp.mpf(1)/2
            exp_rows.append([value[0,j] for j in range(3)])
            analytic_rows.append([mp.exp(d),b*f1,b*c*f2])
    return exp_rows,analytic_rows


def reference_set(q):
    import mpmath as mp
    rows80,_=high_precision_reference(q,80)
    rows120,analytic120=high_precision_reference(q,120)
    with mp.workdps(120):
        scale=max(mp.mpf(1),*(abs(v) for row in rows120 for v in row))
        diff80=max(abs(a-b) for ra,rb in zip(rows80,rows120) for a,b in zip(ra,rb))
        diffanalytic=max(abs(a-b) for ra,rb in zip(analytic120,rows120) for a,b in zip(ra,rb))
        qualified=diff80<=mp.mpf("1e-60")*scale and diffanalytic<=mp.mpf("1e-60")*scale
        n=len(rows120); rounded=np.zeros((n,3*n))
        for i,row in enumerate(rows120):
            rounded[i,[i,n+i,2*n+i]]=[float(v) for v in row]
        record={"qualified":qualified,"dps":[80,120],"reference80_vs120_maxabs":mp.nstr(diff80,110),
            "analytic120_vs_expm120_maxabs":mp.nstr(diffanalytic,110),"relative_reference_threshold":"1e-60*max(1,maxabs(reference))",
            "reference120_rows":[[mp.nstr(v,125) for v in row] for row in rows120],"binary64_rounded_sha256":array_hash(rounded)}
    return rounded,record


def ordered_bits(value):
    bits=np.asarray(value,dtype=np.float64).view(np.uint64)
    sign=np.uint64(1)<<np.uint64(63)
    return np.where((bits&sign)!=0,~bits,bits|sign)


def comparison(value, reference):
    value,reference=np.asarray(value,dtype=np.float64),np.asarray(reference,dtype=np.float64)
    if value.shape!=reference.shape or not np.isfinite(value).all() or not np.isfinite(reference).all():
        raise ValueError("finite aligned comparison arrays required")
    delta=value-reference
    bits_a,bits_b=ordered_bits(value),ordered_bits(reference)
    ulp=np.maximum(bits_a,bits_b)-np.minimum(bits_a,bits_b)
    denominator=np.linalg.norm(reference)
    rel=np.divide(np.abs(delta),np.abs(reference),out=np.full(value.shape,np.nan),where=reference!=0)
    n=reference.shape[0]
    return {"max_abs":float(np.max(np.abs(delta))),"frobenius_relative":float(np.linalg.norm(delta)/denominator) if denominator else None,
        "max_ULP":int(np.max(ulp)),"element_absolute_error":np.abs(delta),"element_relative_error_nonzero_reference":rel,"element_ULP":ulp,
        "block_max_abs":[float(np.max(np.abs(delta[:,start:start+n]))) for start in (0,n,2*n)],
        "old_1e12_max_standardized_violation":float(np.max(np.abs(delta)/(1e-12+1e-12*np.abs(reference)))),
        "old_1e12_gate_passed":bool(np.allclose(value,reference,atol=1e-12,rtol=1e-12)),"array_sha256":array_hash(value)}


def precise_assembly_effect(g,rpp,storage,reference):
    """One secondary input-rounding diagnosis, never replaces the primary Q."""
    import mpmath as mp
    n=len(g);result=np.zeros((n,3*n))
    with mp.workdps(80):
        h=mp.mpf(float(.01))
        for i in range(n):
            d=h*mp.mpf(float(rpp[i]))*mp.mpf(float(storage[i]))
            b=h*mp.mpf(float(g[i]));c=-h
            block=mp.matrix([[d,b,0],[0,0,c],[0,0,0]])
            value=mp.expm(block)
            result[i,[i,n+i,2*n+i]]=[float(value[0,j]) for j in range(3)]
    return comparison(result,reference)


def cached_precise_assembly(cache,g,rpp,storage,reference):
    # Distinct raw factors can produce the same rounded a/Q. Deduplicate only
    # when all three original binary64 arrays agree, never by Q alone.
    key=hashlib.sha256("|".join(array_hash(np.asarray(value,dtype=np.float64)) for value in (g,rpp,storage)).encode()).hexdigest()
    if key not in cache:
        cache[key]={**precise_assembly_effect(g,rpp,storage,reference),"raw_constant_identity_sha256":key}
    return cache[key]


def error_against_unrounded_reference(value, reference_record):
    """True high-precision max error, separate from rounded-reference ULP tables."""
    import mpmath as mp
    value=np.asarray(value,dtype=np.float64)
    n=value.shape[0]
    with mp.workdps(120):
        rows=[[mp.mpf(text) for text in row] for row in reference_record["reference120_rows"]]
        error=mp.mpf(0)
        for i in range(n):
            for j in range(3*n):
                ref=rows[i][j//n] if j in (i,n+i,2*n+i) else mp.mpf(0)
                error=max(error,abs(mp.mpf(float(value[i,j]))-ref))
        return mp.nstr(error,125)


def json_and_provenance(path,model):
    import mpmath as mp
    raw=json.loads(Path(path).read_text(encoding="utf-8"),parse_float=Decimal)
    normal=read(path)
    results={}
    for name in ("jpi_p","rpp","storage_base","p_backward"):
        tokens=np.asarray(raw["state_dict"][name],dtype=object)
        binary=np.asarray(tokens,dtype=np.float64)
        loaded=getattr(model,name).detach().cpu().numpy()
        with mp.workdps(80):
            rounding=max(abs(mp.mpf(str(token))-mp.mpf(float(value))) for token,value in zip(tokens.flat,binary.flat))
            results[name]={"JSON_binary64_vs_loaded_bitwise_equal":array_hash(binary)==array_hash(loaded),"maximum_decimal_token_to_binary64_rounding":mp.nstr(rounding,85),
                "loaded_dtype":str(loaded.dtype),"loaded_device":str(getattr(model,name).device),"config_vs_effective_exact":np.array_equal(np.asarray(normal["config"][name]),loaded) if name in normal["config"] else None}
    ancestors=[]
    def walk(value):
        if not isinstance(value,dict):return
        if "source_path" in value and "source_sha256" in value:
            path=Path(value["source_path"])
            try:
                path.resolve().relative_to((ROOT/"private_data").resolve())
                actual=sha(path)
                ancestors.append({"path":str(path),"expected_sha256":value["source_sha256"],"actual_sha256":actual,"matches":actual==value["source_sha256"]})
            except Exception as exc:
                ancestors.append({"path":str(path),"status":"unknown","reason":repr(exc)})
        for nested in value.values():
            if isinstance(nested,dict):walk(nested)
    walk(normal.get("source_identity",{}))
    return {"fields":results,"ancestors":ancestors,"historical_generation_runtime_recreated":False,
        "historical_dtype_cause":"unknown unless explicitly bound by source/metadata; a float32 fingerprint alone is not proof",
        "constructor_evidence":"ForceGraphROM.__init__ allocates P and eye explicitly torch.float64; export uses tensor.cpu().tolist; from_export builds config then loads supplied state tensors",
        "frozen_model_identity":ident(path)}


def qualify_artificial():
    """Registered zero/constant/linear scalar check; no data or ROM call."""
    from scipy.linalg import expm
    result=[]
    for a in (0.,1e-10,1.,100.):
        q=generator_q([.7],[a],[1.])
        reference,record=reference_set(q)
        if not record["qualified"]:raise ArithmeticError("artificial high-precision reference unresolved")
        np.testing.assert_allclose(reference,expm(q)[:1],atol=1e-12,rtol=1e-12)
        # A reverse interval with physical forward slope, starting at its right end.
        for slope in (0.,.4):
            h=.01;p0=.2;w0=-.3
            augmented=np.array([[-a,-.7,0],[0,0,1],[0,0,0]],dtype=float)
            forward=expm(h*augmented)@np.array([p0,w0,slope])
            backward=(reference@np.array([forward[0],forward[1],slope])).item()
            np.testing.assert_allclose(backward,p0,atol=1e-12,rtol=1e-12)
        result.append({"a":a,"reference_qualified":True})
    return {"status":"PASS","cases":result,"atol":1e-12,"rtol":1e-12}


def validate_archival_receipt(receipt):
    if receipt.get("local_commit")!=PARENT_COMMIT or receipt.get("remote_commit")!=PARENT_COMMIT or receipt.get("ordinary_push_verified") is not True:
        raise ValueError("parent archival receipt does not verify the frozen local/remote parent commit")


def load_sources(protocol,design):
    """Read only model constants and NPZ history members, never future members."""
    import closure_probe_20261030 as probe
    import checkpoint_selection_20261030 as old
    parent_complete=read(PARENT/"complete.json")
    if parent_complete.get("status")!="COMPLETED":raise ValueError("parent p diagnostic not complete")
    archive_marker=PARENT/"archival_verified.json"
    validate_archival_receipt(read(archive_marker))
    parent=read(PARENT/"manifest.json")
    if parent_complete.get("manifest_sha256")!=sha(PARENT/"manifest.json"):
        raise ValueError("parent manifest binding differs")
    for name in ("force_graph_model_20261028.py","state_gated_memory_model_20261028.py","memory_transport_model_20261029.py"):
        source=parent["source"][name]
        if sha(ROOT/source["path"])!=source["sha256"]:raise ValueError("parent model implementation changed")
    previous=read(ROOT/parent["previous_manifest"]["path"])
    if sha(ROOT/parent["previous_manifest"]["path"])!=parent["previous_manifest"]["sha256"]:
        raise ValueError("frozen previous manifest differs")
    models,histories,bindings={}, {},{}
    for label in MODELS:
        item=parent["models"][label]
        for name in ("summary","complete","per_event","model"):
            # per_event contains future-derived scalar metrics: hash bytes only,
            # never parse it in this history-only probe.
            asset=item[name]
            if sha(ROOT/asset["path"])!=asset["sha256"]:raise ValueError("parent source asset changed")
        receipt=read(ROOT/item["complete"]["path"])
        if receipt.get("status")!="COMPLETED" or receipt["summary_sha256"]!=item["summary"]["sha256"] or receipt["manifest_sha256"]!=parent["previous_manifest"]["sha256"]:
            raise ValueError("model completion receipt differs")
        spec=old.model_spec(label)
        if Path(spec["export"]).resolve()!=(ROOT/item["model"]["path"]).resolve():raise ValueError("model loader registry differs from parent source")
        models[label]=probe.load_model(spec)
        rows={};members=[]
        for prediction in item["predictions"]:
            path=Path(prediction["path"]).resolve()
            path.relative_to((ROOT/"private_data").resolve())
            if sha(path)!=prediction["sha256"]:raise ValueError("history archive byte hash changed")
            bound={"path":path.relative_to(ROOT.resolve()).as_posix(),"sha256":prediction["sha256"]}
            if bound not in receipt["artifacts"]:raise ValueError("history archive not bound by original completion")
            with np.load(path,allow_pickle=False) as data:
                ids=[str(value) for value in data["event_ids"]]
                history=data["histories"]
                if history.shape!=(len(ids),11,30) or history.dtype!=np.float64:
                    raise ValueError("causal history member shape/dtype changed")
                for eid,h in zip(ids,history):
                    if eid in rows:raise ValueError("duplicate history event")
                    legal=np.zeros_like(h)
                    legal[:,:20]=h[:,:20]
                    legal[-1,20:30]=h[-1,20:30]
                    if not np.isfinite(legal).all():raise ValueError("nonfinite legal initialization information")
                    rows[eid]=legal
                members.append({"archive":bound,"event_ids_member_sha256":array_hash(np.asarray(ids)),"legal_histories_member_sha256":array_hash(np.stack([rows[eid] for eid in ids])),"loaded_members":["event_ids","histories"],"future_members_loaded":False})
        ids=sorted(rows)
        if len(ids)!=297 or set(ids)!=set(previous["data"]["fit_ids"]):raise ValueError("fit297 history ID set differs")
        histories[label]=np.stack([rows[eid] for eid in ids])
        bindings[label]={"model":item["model"],"summary":item["summary"],"complete":item["complete"],"history_members":members,"event_ids":ids,"legal_history_sha256":array_hash(histories[label])}
    if not np.array_equal(histories[MODELS[0]],histories[MODELS[1]]):raise ValueError("two frozen legal histories differ")
    import torch,scipy,mpmath
    source_names=("p_reverse_precision_20261031.py","force_graph_model_20261028.py","state_gated_memory_model_20261028.py","memory_transport_model_20261029.py","closure_probe_20261030.py","checkpoint_selection_20261030.py")
    manifest={"schema":SCHEMA,"protocol":ident(protocol),"design":ident(design),"parent_manifest":ident(PARENT/"manifest.json"),"parent_complete":ident(PARENT/"complete.json"),"parent_archival_verified":ident(archive_marker),"parent_verified_commit":PARENT_COMMIT,"models":bindings,
        "source":{name:ident(ROOT/"experiments"/name) for name in source_names},
        "runtime":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"mpmath":mpmath.__version__,"device":"cpu","dtype":"float64","torch_default_dtype":str(torch.get_default_dtype()),"torch_threads":torch.get_num_threads()},
        "histories":{"event_count":297,"past_p_slots":"zero placeholders; only terminal true p0 retained","future_members_loaded":False},"optimizer_calls":0,"shared_parameter_estimates":0,"new_ROM_rollouts":0,"new_fom":0,"val_read":False,"test_read":False}
    write_once(PRIVATE/"manifest.json",manifest)
    return models,histories,bindings,manifest


def summarize(records):
    result={}
    for key in records[0]:
        values=[row[key] for row in records if row[key] is not None]
        result[key]={"max":float(np.max(values)) if values else None,"median":float(np.median(values)) if values else None,"available":len(values),"denominator":len(records)}
    return result


def run(protocol,design):
    if PRIVATE.exists() or PUBLIC.exists():raise FileExistsError("output exists: reconcile prior job; put wrapper logs outside tool output directory")
    PRIVATE.mkdir(parents=True)
    started=time.perf_counter()
    write_once(PRIVATE/"started.json",{"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"optimizer_calls":0,"new_ROM_rollouts":0})
    try:
        import torch
        from scipy.linalg import expm
        torch.set_num_threads(1)
        stage=time.perf_counter()
        models,histories,bindings,manifest=load_sources(protocol,design)
        read_seconds=time.perf_counter()-stage
        artificial=qualify_artificial()
        original_states={label:state_identity(model) for label,model in models.items()}
        stage=time.perf_counter()
        math_cache={};assembly_cache={};p1={};references={};p0={}
        for label,model in models.items():
            g=model.jpi_p.detach().cpu().numpy();rpp=model.rpp.detach().cpu().numpy();storage=model.storage_base.detach().cpu().numpy()
            q=generator_q(g,rpp,storage)
            key=array_hash(q)
            if key not in math_cache:
                reference,qualification=reference_set(q)
                if qualification["qualified"]:
                    low={"torch64":torch.matrix_exp(torch.as_tensor(q,dtype=torch.float64))[:10].numpy(),"scipy64":expm(q)[:10],"torch32_fingerprint":torch.matrix_exp(torch.as_tensor(q,dtype=torch.float32))[:10].double().numpy(),"reference_rounded64":reference}
                    comparisons={name:{**comparison(value,reference),"max_abs_against_unrounded_mp120":error_against_unrounded_reference(value,qualification)} for name,value in low.items()}
                else:comparisons={}
                math_cache[key]={"Q_sha256":key,"Q_float64":q,"reference":qualification,"comparisons":comparisons}
                references[key]=reference
            math_record=math_cache[key]
            p0[label]=call_source(model)
            p1[label]={"math_input_sha256":key,"frozen_buffer":comparison(model.p_backward.detach().cpu().numpy(),references[key]) if math_record["reference"]["qualified"] else None,
                "JSON_dtype_ancestry":json_and_provenance(ROOT/bindings[label]["model"]["path"],model),
                "precise_raw_constant_assembly_vs_primary_input":cached_precise_assembly(assembly_cache,g,rpp,storage,references[key]) if math_record["reference"]["qualified"] else None,
                "buffers":{name:{"sha256":tensor_hash(getattr(model,name)),"dtype":str(getattr(model,name).dtype),"device":str(getattr(model,name).device)} for name in ("jpi_p","rpp","storage_base","p_backward","mass","p_scale","rstar")}}
            if math_record["reference"]["qualified"]:
                p1[label]["frozen_buffer"]["max_abs_against_unrounded_mp120"]=error_against_unrounded_reference(model.p_backward.detach().cpu().numpy(),math_record["reference"])
                old_hash=tensor_hash(model.p_backward)
                reproduced=old_hash==math_record["comparisons"]["torch64"]["array_sha256"]
                p1[label]["numerical_source_evidence"]={"current_torch64_bitwise_reproduces_frozen_buffer":reproduced,
                    "current_torch64_fails_original_1e12_vs_reference":not math_record["comparisons"]["torch64"]["old_1e12_gate_passed"],
                    "supports_current_same_algorithm_expm_discrepancy":reproduced and not math_record["comparisons"]["torch64"]["old_1e12_gate_passed"],
                    "JSON_precision_loss_supported":not all(cell["JSON_binary64_vs_loaded_bitwise_equal"] for cell in p1[label]["JSON_dtype_ancestry"]["fields"].values()),
                    "unique_historical_cause":"unknown; current reproducibility and dtype fingerprints do not reconstruct an unbound historical runtime"}
        reference_seconds=time.perf_counter()-stage
        stage=time.perf_counter();p2={};per_event={};initialize_calls=0;field_calls=0
        for label,model in models.items():
            key=p1[label]["math_input_sha256"]
            if not math_cache[key]["reference"]["qualified"]:
                p2[label]={"status":"reference_unresolved"}
                continue
            reference=references[key]
            copied=replace_copy(model,reference)
            p2[label]={};per_event[label]={}
            for dt in HISTORY_DT:
                old,trace=capture_initialize(model,histories[label],dt,trace=True)
                initialize_calls+=1;field_calls+=1
                p0[label][f"actual_trace_{dt}"]=trace
                if trace["status"]!="active_buffer":
                    p2[label][str(dt)]={"status":"use_unknown","replacement_skipped":True}
                    continue
                new,_=capture_initialize(copied,histories[label],dt)
                initialize_calls+=1;field_calls+=1
                rows,checks=sensitivity_rows(old,new,histories[label],model,reference)
                per_event[label][str(dt)]={"event_ids":bindings[label]["event_ids"],"rows":rows}
                p2[label][str(dt)]={"status":"explained_on_registered_identities" if all(checks.values()) else "propagation_unresolved","checks":checks,"events":len(rows),"summary":summarize(rows)}
            if state_identity(model)!=original_states[label]:raise ValueError("original model state changed")
        initialization_seconds=time.perf_counter()-stage
        result={"schema":SCHEMA,"status":"COMPLETED","manifest_sha256":sha(PRIVATE/"manifest.json"),"artificial_qualification":artificial,"P0":p0,"P1":p1,"unique_mathematical_inputs":math_cache,"P2":p2,
            "runtime":{"read_seconds":read_seconds,"reference_seconds":reference_seconds,"initialization_seconds":initialization_seconds,"wall_seconds_before_publication":time.perf_counter()-started,"initialize_calls":initialize_calls,"initial_field_calls":field_calls,"optimizer_calls":0,"shared_parameter_estimates":0,"new_ROM_rollouts":0,"new_fom":0},
            "model_adopted":False,"old_p_diagnostic_results_modified":False,"old_1e12_gate_relaxed":False,"long_horizon_prediction_impact":"unknown; no forward integration"}
        write_once(PRIVATE/"per_event.json",per_event)
        write_once(PRIVATE/"probe.json",result)
        public={key:value for key,value in result.items() if key not in ("P0","P1","unique_mathematical_inputs")}
        public["P0"]={label:{"actual_class":p0[label]["actual_class"],"MRO":p0[label]["MRO"],"traces":{key:value for key,value in p0[label].items() if key.startswith("actual_trace_")}} for label in MODELS}
        compact=lambda row:{key:value for key,value in row.items() if key not in ("element_absolute_error","element_relative_error_nonzero_reference","element_ULP")}
        public["P1"]={label:{"math_input_sha256":p1[label]["math_input_sha256"],"frozen_buffer":compact(p1[label]["frozen_buffer"]) if p1[label]["frozen_buffer"] else None,
            "JSON_field_identity":p1[label]["JSON_dtype_ancestry"]["fields"],"historical_dtype_cause":p1[label]["JSON_dtype_ancestry"]["historical_dtype_cause"],"numerical_source_evidence":p1[label].get("numerical_source_evidence"),"raw_constant_assembly":compact(p1[label]["precise_raw_constant_assembly_vs_primary_input"]) if p1[label]["precise_raw_constant_assembly_vs_primary_input"] else None} for label in MODELS}
        public["mathematical_inputs"]={key:{"reference":{k:v for k,v in record["reference"].items() if k!="reference120_rows"},"comparisons":{name:compact(c) for name,c in record["comparisons"].items()}} for key,record in math_cache.items()}
        write_once(PUBLIC,public)
        write_once(PRIVATE/"complete.json",{"status":"COMPLETED","probe":ident(PRIVATE/"probe.json"),"per_event":ident(PRIVATE/"per_event.json"),"public":ident(PUBLIC),"manifest_sha256":result["manifest_sha256"],"actual_total_wall_seconds":time.perf_counter()-started})
        return result
    except BaseException as exc:
        write_once(PRIVATE/"failure.json",{"status":"FAILED","error":repr(exc),"traceback":traceback.format_exc(),"wall_seconds":time.perf_counter()-started,"optimizer_calls":0,"new_ROM_rollouts":0})
        raise


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["run"])
    parser.add_argument("--protocol",type=Path,default=ROOT/"reports/p_reverse_precision_20261031_protocol.md")
    parser.add_argument("--design",type=Path,default=ROOT/"reports/p_reverse_precision_20261031_scientist.md")
    args=parser.parse_args(argv)
    run(args.protocol,args.design)


def call_source(model):
    methods={}
    for name in ("initialize","encode","history_system","from_export"):
        method=getattr(model,name)
        fn=method.__func__ if hasattr(method,"__func__") else method
        source=inspect.getsource(fn)
        lines,start=inspect.getsourcelines(fn)
        hits=[start+i for i,line in enumerate(lines) if "self.p_backward" in line and "@" in line]
        methods[name]={"qualname":fn.__qualname__,"file":str(Path(inspect.getsourcefile(fn)).relative_to(ROOT)),"source_sha256":hashlib.sha256(source.encode()).hexdigest(),"buffer_matmul_lines":hits}
    return {"actual_class":type(model).__module__+"."+type(model).__qualname__,"MRO":[x.__module__+"."+x.__qualname__ for x in type(model).__mro__],"methods":methods}


def capture_initialize(model, histories, dt, *, trace=False):
    """Exactly one initialize; capture its real history_system result, no rerun."""
    import torch
    original=model.history_system
    function=original.__func__ if hasattr(original,"__func__") else original
    source_lines,start=inspect.getsourcelines(function)
    hit_lines={start+i for i,line in enumerate(source_lines) if "self.p_backward" in line and "@" in line}
    captured={};calls=[];line_hits=[]
    code_names={}
    for name in ("initialize","encode","history_system"):
        fn=getattr(model,name)
        fn=fn.__func__ if hasattr(fn,"__func__") else fn
        code_names[fn.__code__]=name
    def tracer(frame,event,arg):
        if frame.f_code in code_names:
            if event=="call":calls.append(code_names[frame.f_code])
            if event=="line" and frame.f_code is function.__code__ and frame.f_lineno in hit_lines:
                line_hits.append(frame.f_lineno)
            return tracer
        return None
    def capture(*args,**kwargs):
        out=original(*args,**kwargs)
        for key in ("p_history","r0","design","bscaled","F","q"):
            captured[key]=out[key].detach().cpu().double().numpy().copy()
        return out
    previous=sys.gettrace()
    owned="history_system" in model.__dict__
    previous_override=model.__dict__.get("history_system")
    model.history_system=capture
    model.clear_cache()
    try:
        if trace:sys.settrace(tracer)
        with torch.no_grad():
            initial=model.initialize(torch.as_tensor(histories,dtype=torch.float64),history_dt=dt)
            field=model.field(initial)
        captured["initial"]=initial.detach().cpu().numpy().copy()
        captured["field"]=field.detach().cpu().numpy().copy()
    finally:
        if trace:sys.settrace(previous)
        if owned:model.history_system=previous_override
        else:delattr(model,"history_system")
        model.clear_cache()
    for value in captured.values():
        if not np.isfinite(value).all():raise FloatingPointError("nonfinite causal initialization")
    trace_record={"calls":calls,"p_backward_line_hits":line_hits,"expected_history_matmul_count":10,
        "status":"active_buffer" if trace and len(line_hits)==10 and all(name in calls for name in ("initialize","encode","history_system")) else "use_unknown"}
    return captured,trace_record


def replace_copy(model, reference):
    import torch
    old_identity=state_identity(model)
    clone=copy.deepcopy(model)
    before_grad={k:v.requires_grad for k,v in clone.named_parameters()}
    before_training=clone.training
    with torch.no_grad():clone.p_backward.copy_(torch.as_tensor(reference,dtype=clone.p_backward.dtype))
    clone.clear_cache()
    after=state_identity(clone)
    if any(old_identity[name]!=digest for name,digest in after.items() if name!="p_backward"):
        raise ValueError("replacement altered another state tensor")
    if before_grad!={k:v.requires_grad for k,v in clone.named_parameters()} or clone.training!=before_training:
        raise ValueError("replacement altered gradient/training flags")
    return clone


def delta_p_recurrence(old_block,reference,history,old_path):
    n=old_block.shape[0]
    delta=np.zeros(old_path.shape,dtype=np.float64)
    for i in range(9,-1,-1):
        slope=(history[:,i+1,n:2*n]-history[:,i,n:2*n])/.01
        vector=np.concatenate((old_path[:,i+1],history[:,i+1,n:2*n],slope),axis=1)
        delta[:,i]=vector@(reference-old_block).T+delta[:,i+1]@reference[:,:n].T
    return delta


def sensitivity_rows(old,new,history,model,reference):
    import torch
    old_block=model.p_backward.detach().cpu().numpy()
    pscale=model.p_scale.detach().cpu().numpy();rstar=model.rstar.detach().cpu().numpy();mass=model.mass.detach().cpu().numpy()
    dp=new["p_history"]-old["p_history"];dr=new["r0"]-old["r0"]
    predicted=delta_p_recurrence(old_block,reference,history,old["p_history"])
    recurrence_ok=np.allclose(dp,predicted,atol=1e-12,rtol=1e-12)
    design_same=np.array_equal(old["design"],new["design"])
    q_same=np.array_equal(old["q"],new["q"])
    target_delta=np.concatenate((new["bscaled"]-old["bscaled"],np.zeros((len(history),10))),axis=1)
    dr_prediction=None
    if design_same and q_same:
        with torch.no_grad():
            design=torch.as_tensor(old["design"],dtype=torch.float64)
            qr,tri=torch.linalg.qr(design,mode="reduced")
            sol=torch.linalg.solve_triangular(tri,qr.transpose(-1,-2)@torch.as_tensor(target_delta[...,None],dtype=torch.float64),upper=True).squeeze(-1)
            dr_prediction=sol.numpy()*rstar
    qr_ok=dr_prediction is not None and np.allclose(dr,dr_prediction,atol=1e-12,rtol=1e-12)
    dfield=new["field"]-old["field"]
    field_ok=np.allclose(dfield[:,10:20],-dr/mass,atol=1e-12,rtol=1e-12) and np.array_equal(dfield[:,20:30],np.zeros_like(dr))
    observed_ok=np.array_equal(old["initial"][:,:30],new["initial"][:,:30]) and np.array_equal(new["initial"][:,:30],history[:,-1,:30])
    records=[]
    for i in range(len(history)):
        target_old=np.r_[old["bscaled"][i],-np.sqrt(1e-4)*old["q"][i]]
        target_new=np.r_[new["bscaled"][i],-np.sqrt(1e-4)*new["q"][i]]
        res_old=old["design"][i]@(old["r0"][i]/rstar)-target_old
        res_new=new["design"][i]@(new["r0"][i]/rstar)-target_new
        denom=np.linalg.norm(old["r0"][i])
        records.append({"delta_p_maxabs":float(np.max(np.abs(dp[i]))),"delta_p_rms":float(np.sqrt(np.mean(dp[i]**2))),
            "delta_p_normalized_rms":float(np.sqrt(np.mean((dp[i]/pscale)**2))),"delta_r0_maxabs":float(np.max(np.abs(dr[i]))),
            "delta_r0_L2":float(np.linalg.norm(dr[i])),"delta_r0_normalized_L2":float(np.linalg.norm(dr[i]/rstar)),"delta_r0_relative_L2":float(np.linalg.norm(dr[i])/denom) if denom else None,
            "delta_design_maxabs":float(np.max(np.abs(new["design"][i]-old["design"][i]))),"delta_bscaled_L2":float(np.linalg.norm(new["bscaled"][i]-old["bscaled"][i])),
            "QR_residual_old_L2":float(np.linalg.norm(res_old)),"QR_residual_new_L2":float(np.linalg.norm(res_new)),"delta_QR_residual_L2":float(np.linalg.norm(res_new-res_old)),
            "delta_omega_dot_maxabs":float(np.max(np.abs(dfield[i,10:20]))),"delta_omega_dot_normalized_rms":float(np.sqrt(np.mean((dfield[i,10:20]/model.input_scale_omega.detach().cpu().numpy())**2))),
            "delta_p_dot_maxabs":float(np.max(np.abs(dfield[i,20:30]))),"p_recurrence_residual_maxabs":float(np.max(np.abs(dp[i]-predicted[i]))),"r0_QR_propagation_residual_maxabs":float(np.max(np.abs(dr[i]-dr_prediction[i]))) if dr_prediction is not None else None})
    checks={"p_recurrence_at_1e12":bool(recurrence_ok),"design_bitwise_unchanged":bool(design_same),"QR_propagation_at_1e12":bool(qr_ok),"initial_observed_state_unchanged":bool(observed_ok),"initial_field_identity_at_1e12":bool(field_ok)}
    return records,checks


if __name__=="__main__":
    main()
