"""One-way new-train discrimination and frozen new-val confirmation."""
import hashlib
import os
from pathlib import Path
import json
import time
import numpy as np
import torch


class RecoveryUnresolved(RuntimeError):
    """An existing artifact is incomplete or its scientific identity has changed."""


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value):
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_sha256(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once_json(path, value):
    """Create an immutable JSON artifact or require the existing value to match."""
    path = Path(path)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            raise RecoveryUnresolved(f"existing artifact is unreadable: {path}") from error
        if _canonical_json(existing) != _canonical_json(value):
            raise RecoveryUnresolved(f"existing immutable artifact differs: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(_json_value(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return _write_once_json(path, value)
    return True


def _normalized_generation_conditions(conditions, split, data_module):
    normalized = data_module._conditions_for_split(conditions, split)
    return [data_module._attempt_record(row, split) for row in normalized]


def _verify_and_record_raw_hashes(output, manifest, expected_conditions):
    output = Path(output).resolve()
    attempts = manifest.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != len(expected_conditions):
        raise RecoveryUnresolved("completed generation has a partial attempt list")
    if any(_canonical_json(row.get("condition")) != _canonical_json(expected)
           for row, expected in zip(attempts, expected_conditions)):
        raise RecoveryUnresolved("generation attempts do not match normalized conditions")
    raw_hashes = {}
    for attempt in attempts:
        steps = attempt.get("steps")
        if not isinstance(steps, dict):
            raise RecoveryUnresolved("completed generation has a missing step manifest")
        for step, record in steps.items():
            raw_file = record.get("raw_file")
            if not raw_file:
                if record.get("status") in {"completed", "raw_validation_failed"}:
                    raise RecoveryUnresolved(f"completed raw step {step} has no file path")
                continue
            raw_path = (output / raw_file).resolve()
            try:
                raw_path.relative_to(output)
            except ValueError as error:
                raise RecoveryUnresolved("raw artifact path escapes generation directory") from error
            if not raw_path.is_file():
                raise RecoveryUnresolved(f"generation raw file is missing: {raw_file}")
            relative_raw = raw_path.relative_to(output).as_posix()
            raw_hashes[relative_raw] = _file_sha256(raw_path)
            detail_path = raw_path.with_suffix(".json")
            if not detail_path.is_file():
                raise RecoveryUnresolved(f"generation raw metadata is missing: {detail_path.name}")
            relative_detail = detail_path.relative_to(output).as_posix()
            raw_hashes[relative_detail] = _file_sha256(detail_path)
    identity = {
        "split": manifest.get("split"),
        "conditions_sha256": _json_sha256(expected_conditions),
        "freeze_sha256": _json_sha256(manifest.get("freeze")),
        "generation_sha256": _file_sha256(output / "generation.json"),
    }
    integrity = {"schema_version": 1, "identity": identity, "raw_sha256": raw_hashes}
    _write_once_json(output / "generation.integrity.json", integrity)
    return raw_hashes


def generate_or_reuse(conditions, out, split, cap_seconds, *, freeze_path=None, simulator=None):
    """Generate once, then reuse only a completed identity-matched artifact.

    Any existing RUNNING, incomplete, or identity-mismatched generation is preserved
    and raises :class:`RecoveryUnresolved`; it never resamples or overwrites raw data.
    """
    import closure_discovery_data as data

    if split not in {"train", "val"}:
        raise ValueError("new data generation is restricted to train or val")
    output = Path(out).expanduser().resolve()
    normalized = _normalized_generation_conditions(conditions, split, data)
    freeze = data._freeze_snapshot(freeze_path) if split == "val" and freeze_path is not None else None
    if split == "val" and freeze is None:
        raise ValueError("VAL generation requires a verified immutable freeze")
    manifest_path = output / "generation.json"
    reused = manifest_path.exists()
    if reused:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise RecoveryUnresolved("existing generation manifest is unreadable") from error
    else:
        raw_dir = output / "raw"
        if raw_dir.exists() and any(raw_dir.iterdir()):
            raise RecoveryUnresolved("raw attempts exist without a generation manifest")
        manifest = data.generate_conditions(conditions, output, split, cap_seconds,
                                            freeze_path=freeze_path, simulator=simulator)
        if not manifest_path.is_file():
            raise RecoveryUnresolved("generator returned without a persistent generation manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != split:
        raise RecoveryUnresolved("generation split changed")
    if _canonical_json(manifest.get("conditions")) != _canonical_json(normalized):
        raise RecoveryUnresolved("normalized generation conditions changed")
    if _canonical_json(manifest.get("freeze")) != _canonical_json(freeze):
        raise RecoveryUnresolved("generation freeze summary changed")
    if manifest.get("status") != "complete":
        raise RecoveryUnresolved(f"generation remains unresolved with status {manifest.get('status')!r}")
    hashes = _verify_and_record_raw_hashes(output, manifest, normalized)
    manifest["_reused"] = reused
    manifest["_raw_sha256"] = hashes
    return manifest


def _cohort_identity(events):
    rows = []
    for event in events:
        row = {key: event.get(key) for key in ("id", "group", "label", "new_split", "trajectory_end_s")}
        for key in ("x", "torque", "t"):
            value = event.get(key)
            if value is not None:
                array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
                row[f"{key}_shape"] = list(array.shape)
                row[f"{key}_sha256"] = hashlib.sha256(array.tobytes()).hexdigest()
        rows.append(row)
    rows.sort(key=lambda row: str(row.get("id")))
    return {"event_count": len(rows), "sha256": _json_sha256(rows)}


def evaluate_or_reuse(model, events, output, model_sha256,*,deadline=None):
    """Reuse a completed evaluation only for the same model hash and event cohort."""
    path = Path(output)
    identity = {"model_sha256": str(model_sha256), "cohort": _cohort_identity(events)}
    if path.exists():
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            raise RecoveryUnresolved(f"existing evaluation is unreadable: {path}") from error
        if result.get("_identity") != identity:
            raise RecoveryUnresolved(f"evaluation identity changed: {path}")
    else:
        result = evaluate_events(model, events, path, identity=identity,**({'deadline':deadline} if deadline is not None else {}))
    records = result.get("records")
    expected_ids = sorted(str(event["id"]) for event in events)
    row_fields = {"id", "group", "label", "numerical_pass", "complete_5s", "mse",
                  "metrics", "pair", "selected_pair", "scored_numerics"}
    summary_fields = {"primary_risk", "primary_complete_stable_count", "all_numerical_pass",
                      "classification_counts", "scope", "test_read"}
    if (result.get("_identity") != identity or not isinstance(records, list)
            or result.get("event_count") != len(events)
            or sorted(str(row.get("id")) for row in records) != expected_ids
            or any(not isinstance(row, dict) or not row_fields <= set(row) for row in records)
            or not summary_fields <= set(result)):
        raise RecoveryUnresolved(f"evaluation result is incomplete for its declared cohort: {path}")
    return result


def evaluate_events(model,events,output,*,identity=None,force_finer=False,deadline=None):
    """Generic cohort adapter around the unchanged historical metric/grid helpers."""
    from evaluate_structured_rom import _rollouts,_pair_row,_scored_numerics,_metric_safe,_needs_finer
    from train_structured_rom import _point_error
    from train_local_memory_0098 import risk
    output=Path(output);rows=[];mass=model.mass.detach().numpy()
    for e in events:
        if deadline is not None and time.perf_counter()+30>=deadline:
            raise TimeoutError('incomplete numerical evaluation at stage cap; no partial aggregate')
        hist=e['x'][:11];batch=hist[None]
        # Preserve substeps for numerical checks. No true trajectory is extended.
        fine=_rollouts(model,batch,1600,.005)[0]
        coarse=_rollouts(model,batch,800,.01)[0]
        pair=_pair_row(fine,coarse,hist,e,e['classification_meta'],mass)
        scored=_scored_numerics(fine,coarse,e,mass)
        ref=fine[::2];selected_pair=pair
        if force_finer or _needs_finer([pair]+[r for r in scored if r['available']]):
            finer=_rollouts(model,batch,3200,.0025)[0]
            selected_pair=_pair_row(finer,fine,hist,e,e['classification_meta'],mass,stride=2,output_dt=.005)
            scored=_scored_numerics(finer,fine,e,mass,stride=4,coarse_stride=2)
            # Primary prediction remains .005; finer is qualification only.
        numerical=not _needs_finer([selected_pair]+[r for r in scored if r['available']])
        metrics=[_metric_safe(ref,e,h,mass) for h in (1,2,5,8,'prefix')]
        actual=min(500,len(e['x'])-11)
        truth=torch.as_tensor(e['x'][11:11+actual],dtype=torch.float64)
        pred=torch.as_tensor(ref[1:1+actual],dtype=torch.float64)
        mse=float(_point_error(pred,truth,model).mean()) if actual>0 and np.isfinite(pred.numpy()).all() else None
        rows.append({'id':e['id'],'group':e['group'],'label':e['label'],'numerical_pass':numerical,
                     'actual_seconds':actual*.01,'complete_5s':actual==500,'mse':mse,
                     'metrics':metrics,'pair':pair,'selected_pair':selected_pair,'scored_numerics':scored})
    primary=[r for r in rows if r['label']=='stable' and r['complete_5s']]
    qualified=bool(primary) and all(r['numerical_pass'] and r['mse'] is not None for r in primary)
    summary={'records':rows,'event_count':len(events),'primary_complete_stable_count':len(primary),
             'primary_risk':float(risk(torch.as_tensor([r['mse'] for r in primary]))) if qualified else None,
             'all_numerical_pass':all(r['numerical_pass'] for r in rows),'test_read':False,
             'scope':'new frozen val under same simulator/system, not frozen original test or safety guarantee'}
    classified=[r for r in rows if r['numerical_pass']]
    summary['classification_counts']={
        'true_unstable':sum(r['label']=='unstable' for r in rows),
        'missed_unstable':sum(r['label']=='unstable' and not r['selected_pair']['fine_classification']['predicted_los'] for r in classified),
        'false_los':sum(r['label']=='stable' and r['selected_pair']['fine_classification']['predicted_los'] for r in classified),
        'numerically_unresolved':len(rows)-len(classified)}
    if identity is not None:summary['_identity']=identity
    _write_once_json(output,summary);return summary


def _model_file(row):
    path = Path(row["path"]).expanduser().resolve()
    if path.is_dir():
        path = path / "model.json"
    if not path.is_file():
        raise RecoveryUnresolved(f"frozen model file is missing: {path}")
    return path


def _verify_model_rows(models, source_models=None, output_dir=None):
    if not isinstance(models, list) or not models:
        raise RecoveryUnresolved("confirmation freeze must bind its complete endpoint model set")
    try:
        identities = [(str(row["name"]), int(row["seed"])) for row in models]
    except (KeyError, TypeError, ValueError) as error:
        raise RecoveryUnresolved("confirmation model identity is incomplete") from error
    if len(set(identities)) != len(identities) or not {("main",2701), ("fixed",2701)} <= set(identities):
        raise RecoveryUnresolved("confirmation endpoint identities changed")
    if source_models is not None:
        if len(models) != len(source_models):
            raise RecoveryUnresolved("confirmation endpoint count differs from the frozen C cohort")
        for endpoint, source in zip(models, source_models):
            source_path = _model_file(source)
            source_sha = source.get("sha256")
            endpoint_source = endpoint.get("source_path")
            endpoint_sha = endpoint.get("source_sha256")
            endpoint_metadata = {key:value for key,value in endpoint.items()
                                 if key not in {"path","sha256","source_path","source_sha256"}}
            source_metadata = {key:value for key,value in source.items() if key not in {"path","sha256"}}
            if (endpoint_metadata != source_metadata or endpoint_source != str(source_path)
                    or endpoint_sha != source_sha or _file_sha256(source_path) != source_sha):
                raise RecoveryUnresolved("confirmation models no longer match the frozen C endpoint cohort")
    for row in models:
        path = _model_file(row)
        if output_dir is not None:
            expected_path=(Path(output_dir)/'D/models'/str(row['name'])/str(row['seed'])/'model.json').resolve()
            if path!=expected_path:
                raise RecoveryUnresolved(f"confirmation model path changed: {path}")
        expected = row.get("sha256")
        if not isinstance(expected, str) or _file_sha256(path) != expected:
            raise RecoveryUnresolved(f"confirmation model hash changed: {path}")


def _confirmation_freeze_payload(models, val_conditions, condition_manifest_path):
    return {
        "models": models,
        "primary": "stable complete5s normalized event risk",
        "main_name": "main",
        "main_seed": 2701,
        "control_name": "fixed",
        "control_seed": 2701,
        "bootstrap_seed": 5701,
        "bootstrap_resamples": 2000,
        "test_read": False,
        "conditions": val_conditions,
        "conditions_sha256": _file_sha256(condition_manifest_path),
    }


def _freeze_or_reuse(path, models, val_conditions, condition_manifest_path, *, source_models=None):
    path = Path(path)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            raise RecoveryUnresolved("existing confirmation freeze is unreadable") from error
        if models is None:
            models = existing.get("models")
    if models is None:
        raise RecoveryUnresolved("confirmation endpoint models are unavailable")
    _verify_model_rows(models, source_models)
    expected = _confirmation_freeze_payload(models, val_conditions, condition_manifest_path)
    if path.exists():
        if _canonical_json(existing) != _canonical_json(expected):
            raise RecoveryUnresolved("confirmation freeze differs from its bound models or cohort")
        return existing, True
    _write_once_json(path, expected)
    return expected, False


def _physical_condition_key(row):
    bus = str(row.get("bus_id", row.get("bus", row.get("train_bus"))))
    line = str(row.get("line", row.get("fault_line", "")))
    location = float(row.get("location", row.get("fault_location", 0.0)))
    duration = float(row.get("duration_s", row.get("fault_duration_s")))
    xf = float(row.get("xf", row.get("fault_xf")))
    return bus, line, location, duration, xf


def _mandatory_a_train_conditions(out, data):
    """Read the completed A TRAIN condition manifest and bind its two conditions.

    This reads condition metadata only. It never opens generated raw artifacts or
    any validation trajectory.
    """
    generation_path = Path(out).resolve() / "A/fom_cost_retry2/generation.json"
    if not generation_path.is_file():
        raise RecoveryUnresolved(f"mandatory A TRAIN generation manifest is missing: {generation_path}")
    try:
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RecoveryUnresolved("mandatory A TRAIN generation manifest is unreadable") from error
    if generation.get("split") != "train" or generation.get("status") != "complete":
        raise RecoveryUnresolved("mandatory A generation is not a completed TRAIN-only artifact")
    declared = generation.get("conditions")
    if not isinstance(declared, list) or len(declared) != 2:
        raise RecoveryUnresolved("mandatory A TRAIN generation must declare exactly two conditions")
    if any(not isinstance(row, dict) or row.get("split") != "train" for row in declared):
        raise RecoveryUnresolved("mandatory A generation contains a non-TRAIN condition")
    try:
        conditions = data._conditions_for_split(declared, "train")
    except Exception as error:
        raise RecoveryUnresolved("mandatory A TRAIN conditions do not match registered physical TRAIN metadata") from error
    keys = [_physical_condition_key(row) for row in conditions]
    if len(set(keys)) != 2:
        raise RecoveryUnresolved("mandatory A TRAIN conditions contain a physical duplicate")
    strata = {name: sum(("short" if row["duration_s"] < 0.15 else
                         "medium" if row["duration_s"] <= 0.3 else "long") == name
                        for row in conditions)
              for name in ("short", "medium", "long")}
    if strata != {"short": 1, "medium": 1, "long": 0}:
        raise RecoveryUnresolved(f"mandatory A TRAIN conditions have unexpected duration strata: {strata}")
    return conditions, _file_sha256(generation_path)


def _selection_identity(manifest, frozen_models, old_train_events,
                        mandatory_conditions, mandatory_generation_sha256):

    source_hashes = {}
    for row in frozen_models:
        path = _model_file(row)
        actual = _file_sha256(path)
        if actual != row.get("sha256"):
            raise RecoveryUnresolved(f"C-stage model hash changed: {path}")
        identity_key = f"{row['name']}:{int(row['seed'])}"
        if identity_key in source_hashes:
            raise RecoveryUnresolved(f"duplicate frozen model identity: {identity_key}")
        source_hashes[identity_key] = actual
    return {
        "candidate_pool_sha256": _json_sha256(manifest["train_pool"]),
        "mandatory_a_train_conditions_sha256": _json_sha256(mandatory_conditions),
        "mandatory_a_train_generation_sha256": str(mandatory_generation_sha256),
        "source_model_sha256": source_hashes,
        "train_cohort": _cohort_identity(old_train_events),
        "metric": "train_structured_rom._point_error(a,b,main).mean()",
    }


def train_prediction_disagreement(main_prediction, alternative_prediction, model):
    """Compare TRAIN predictions using the model's fixed structured error scales."""
    from train_structured_rom import _point_error
    return float(_point_error(main_prediction, alternative_prediction, model).mean())


def train_disagreement_rows(main,alternative,events,metadata,*,deadline=None):
    """Identical per-event, normalized predictions evaluated in finite batches."""
    from train_local_memory_0098 import graph,histories_for
    from train_structured_rom import _point_error
    rows=[]
    for start in range(0,len(events),32):
        if deadline is not None and time.perf_counter()+10>=deadline:raise TimeoutError('discrimination predictions incomplete')
        block=events[start:start+32];history=histories_for([(e,10) for e in block])
        with torch.no_grad(),graph(main):a=main.rollout(history,100,.01)[...,:30]
        with torch.no_grad(),graph(alternative):b=alternative.rollout(history,100,.01)[...,:30]
        values=_point_error(a,b,main).reshape(len(block),-1).mean(-1).numpy()
        if not np.isfinite(values).all():raise FloatingPointError('nonfinite train disagreement; no events removed')
        rows.extend({**metadata[e['id']],'score':float(v)} for e,v in zip(block,values))
    return rows


def _select_or_reuse_train_conditions(out, manifest, frozen_models, old_train_events, main, alternative):
    import closure_discovery_data as data

    selection_path = Path(out) / "D/selected_new_train.json"
    mandatory, mandatory_generation_sha256 = _mandatory_a_train_conditions(out, data)
    mandatory_keys = {_physical_condition_key(row) for row in mandatory}
    identity = _selection_identity(manifest, frozen_models, old_train_events,
                                   mandatory, mandatory_generation_sha256)
    if selection_path.exists():
        try:
            saved = json.loads(selection_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise RecoveryUnresolved("existing TRAIN selection is unreadable") from error
        if not isinstance(saved, dict) or set(saved) != {"identity", "conditions"}:
            raise RecoveryUnresolved("existing TRAIN selection has no verifiable metric/cohort identity")
        conditions = saved["conditions"]
        if saved["identity"].get("base") != identity or saved["identity"].get("conditions_sha256") != _json_sha256(conditions):
            raise RecoveryUnresolved("existing TRAIN selection identity changed")
        pool_rows = [{**row, "split": "train"} for row in manifest["train_pool"]]
        pool = data._conditions_for_split(pool_rows, "train")
        chosen = data._conditions_for_split(conditions, "train")
        pool_keys = {_physical_condition_key(row) for row in pool}
        chosen_keys = [_physical_condition_key(row) for row in chosen]
        if len(chosen_keys) != 24 or len(set(chosen_keys)) != 24 or not set(chosen_keys) <= pool_keys:
            raise RecoveryUnresolved("existing TRAIN selection is not the original 24-condition subset")
        if not mandatory_keys <= set(chosen_keys):
            raise RecoveryUnresolved("existing TRAIN selection omits a mandatory A TRAIN condition; no resampling is permitted")
        chosen_strata = {name: sum(("short" if row["duration_s"] < 0.15 else
                                    "medium" if row["duration_s"] <= 0.3 else "long") == name
                                   for row in chosen)
                         for name in ("short", "medium", "long")}
        if chosen_strata != {"short": 8, "medium": 8, "long": 8}:
            raise RecoveryUnresolved("existing TRAIN selection does not preserve the registered 8/8/8 strata")
        return conditions

    metadata = {str(row["event_id"]): row for row in data.allowed_train_conditions()}
    discrepancy = train_disagreement_rows(main,alternative,old_train_events,metadata)
    conditions = data.select_new_train_conditions(manifest["train_pool"], discrepancy, count=24,
                                                  mandatory_conditions=mandatory)
    if len(conditions) != 24:
        raise RecoveryUnresolved("TRAIN selector did not return the fixed 24 conditions")
    selected_keys = {_physical_condition_key(row) for row in conditions}
    if not mandatory_keys <= selected_keys:
        raise RecoveryUnresolved("TRAIN selector omitted a mandatory A TRAIN condition")
    payload = {"identity": {"base": identity, "conditions_sha256": _json_sha256(conditions)},
               "conditions": conditions}
    _write_once_json(selection_path, payload)
    return conditions


def _fit_with_recovery(fit_fn, model_class, model, events, force_scale, target, path, seed,
                       deadline, source_model_sha256):
    """Bind a D refit to its source/cohort and resume only a verified checkpoint."""
    path=Path(path)
    checkpoint_path=path/'checkpoint.json'
    model_path=path/'model.json'
    training_path=path/'training.pt'
    fit_identity_path=path/'fit_identity.json'
    identity={
        'source_model_sha256':str(source_model_sha256),
        'train_cohort':_cohort_identity(events),
        'force_scale':float(force_scale),
        'target_updates':int(target),
        'seed':int(seed),
        'terms_sha256':_json_sha256(model.terms),
        'direction_mask':model.template_mask.detach().cpu().numpy().tolist(),
    }
    artifacts=[checkpoint_path,model_path,training_path]
    existing=path.exists() and any(path.iterdir())
    if existing and not fit_identity_path.exists():
        raise RecoveryUnresolved(f"D fit artifacts lack a bound identity: {path}")
    if existing and any(item.exists() for item in artifacts) and not all(item.is_file() for item in artifacts):
        raise RecoveryUnresolved(f"D fit checkpoint is partial: {path}")
    _write_once_json(fit_identity_path,identity)
    if checkpoint_path.exists():
        try:
            checkpoint=json.loads(checkpoint_path.read_text(encoding='utf-8'))
        except Exception as error:
            raise RecoveryUnresolved(f"D fit checkpoint is unreadable: {checkpoint_path}") from error
        if (_file_sha256(model_path)!=checkpoint.get('model_sha256')
                or _file_sha256(training_path)!=checkpoint.get('training_sha256')):
            raise RecoveryUnresolved(f"D fit checkpoint hash changed: {path}")
        try:
            step=int(checkpoint['step'])
        except (KeyError,TypeError,ValueError) as error:
            raise RecoveryUnresolved(f"D fit checkpoint has no valid update count: {path}") from error
        if step<0 or step>target:
            raise RecoveryUnresolved(f"D fit checkpoint update count is outside its fixed budget: {step}")
        saved_model=model_class.from_export(model_path)
        if (_json_sha256(saved_model.terms)!=identity['terms_sha256']
                or not torch.equal(saved_model.template_mask.detach().cpu(),model.template_mask.detach().cpu())):
            raise RecoveryUnresolved(f"D fit checkpoint structure differs from the registered source: {path}")
        if step==target:
            return saved_model,{'seconds':0.0,'updates':step,'path':str(path),'reused_checkpoint':True}
    return fit_fn(model,events,force_scale,target,path,seed,deadline=deadline)


def paired_primary(main,control):
    """Cluster bootstrap of the registered nonlinear risk, not point/time samples."""
    from train_local_memory_0098 import risk
    left={r['id']:r for r in main['records'] if r['label']=='stable' and r['complete_5s']}
    right={r['id']:r for r in control['records'] if r['label']=='stable' and r['complete_5s']}
    if set(left)!=set(right) or not left:return {'status':'unresolved','reason':'primary cohort mismatch/empty'}
    ids=sorted(left)
    if any(not left[k]['numerical_pass'] or not right[k]['numerical_pass'] or left[k]['mse'] is None or right[k]['mse'] is None for k in ids):
        return {'status':'unresolved','reason':'numerically unresolved primary event retained'}
    groups=sorted({left[k]['group'] for k in ids});rng=np.random.default_rng(5701);diff=[]
    def delta(chosen):
        return float(risk(torch.as_tensor([left[k]['mse'] for k in chosen]))-risk(torch.as_tensor([right[k]['mse'] for k in chosen])))
    for _ in range(2000):
        sampled=rng.choice(groups,len(groups),replace=True)
        chosen=[k for g in sampled for k in ids if left[k]['group']==g]
        diff.append(delta(chosen))
    low,high=np.quantile(diff,[.025,.975])
    return {'status':'complete','difference':delta(ids),'ci95':[float(low),float(high)],
            'supports_primary_improvement':bool(high<0),'events':len(ids),'groups':len(groups),'resamples':2000}


def run_confirmation(out,deadline):
    from discover_closure import fit,train_inputs
    from symbolic_closure_rom import SymbolicClosureROM
    import closure_discovery_data as data

    out=Path(out).expanduser().resolve()
    manifest_path=out/'new_fault_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    freeze_path=out/'D/confirmation_freeze.json'
    if freeze_path.exists():
        frozen=json.loads((out/'C/frozen.json').read_text(encoding='utf-8'))
        freeze,freeze_reused=_freeze_or_reuse(
            freeze_path,None,manifest['val24'],manifest_path,source_models=frozen['models'])
        endpoints=freeze['models']
        new_train_count=None
        excluded=None
    else:
        frozen=json.loads((out/'C/frozen.json').read_text(encoding='utf-8'))
        main_path=Path(out/'C/main/2701/model.json')
        alternative_path=Path(out/'C/alternative/2701/model.json')
        main=SymbolicClosureROM.from_export(main_path)
        alternative=SymbolicClosureROM.from_export(alternative_path)
        old,_,scale=train_inputs()
        conditions=_select_or_reuse_train_conditions(out,manifest,frozen['models'],old,main,alternative)
        train_generation=generate_or_reuse(
            conditions,out/'D/new_data/train','train',max(1.,deadline-time.perf_counter()-60))
        new,excluded=data.load_new_events(out/'D/new_data/train','train')
        if not new:
            raise RecoveryUnresolved('new TRAIN generation has no qualified events; no replacement sampled')
        combined=old+new
        endpoints=[]
        for row in frozen['models']:
            source_path=_model_file(row)
            if _file_sha256(source_path)!=row['sha256']:
                raise RecoveryUnresolved('C source model changed')
            model=SymbolicClosureROM.from_export(source_path)
            dest=out/'D/models'/row['name']/str(row['seed'])
            model,_=_fit_with_recovery(fit,SymbolicClosureROM,model,combined,scale,256,dest,
                                       row['seed'],deadline,row['sha256'])
            model_path=dest/'model.json'
            endpoints.append({**row,'source_path':str(source_path.resolve()),
                              'source_sha256':row['sha256'],'path':str(model_path),
                              'sha256':_file_sha256(model_path)})
        freeze,freeze_reused=_freeze_or_reuse(
            freeze_path,endpoints,manifest['val24'],manifest_path,source_models=frozen['models'])
        endpoints=freeze['models']
        new_train_count=len(new)

    # Existing freezes are immutable and their endpoint files are checked on every resume.
    frozen=json.loads((out/'C/frozen.json').read_text(encoding='utf-8'))
    _verify_model_rows(endpoints,frozen['models'],out)
    generated_val=generate_or_reuse(
        manifest['val24'],out/'D/new_data/val','val',
        max(1.,deadline-time.perf_counter()-60),freeze_path=freeze_path)
    val,excluded_val=data.load_new_events(out/'D/new_data/val','val',freeze_path=freeze_path)
    evaluations={}
    for row in endpoints:
        if deadline-time.perf_counter()<60:raise TimeoutError('confirmation incomplete; val must not be reused for selection')
        model_path=_model_file(row)
        if _file_sha256(model_path)!=row['sha256']:
            raise RecoveryUnresolved(f"confirmation model changed before evaluation: {model_path}")
        m=SymbolicClosureROM.from_export(model_path)
        key=f"{row['name']}_{row['seed']}"
        evaluations[key]=evaluate_or_reuse(
            m,val,out/f'D/evaluation/{key}.json',row['sha256'],deadline=deadline)
    _verify_model_rows(endpoints,frozen['models'],out)
    primary=paired_primary(evaluations['main_2701'],evaluations['fixed_2701'])
    if excluded_val:
        primary={'status':'unresolved','reason':'new val has unavailable/numerically unqualified events; no aggregate improvement claim',
                 'excluded':excluded_val,'qualified_cohort_diagnostic':primary}
    _write_once_json(out/'D/primary_comparison.json',primary)
    return {'stage':'D','new_train':new_train_count,'new_train_excluded':excluded,'new_val':len(val),
            'new_val_excluded':excluded_val,'frozen_sha256':_file_sha256(freeze_path),'primary':primary,
            'generation_reused':bool(generated_val.get('_reused')),
            'freeze_reused':bool(freeze_reused),
            'status':'frozen_confirmation_complete','test_read':False}
