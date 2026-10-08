"""Loop0089 cached physical-output modes of G3000 and final0088 systems."""
from __future__ import annotations
import argparse, json, time, os
from pathlib import Path
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from compare_physical_modes import (assignment, ordinary_modes, physical_output,
                                    relmax, sha, FREQ_GATE, DAMP_GATE, MAC_GATE,
                                    OMEGA_S)

ROOT = Path(__file__).resolve().parents[1]
P83 = ROOT / "private_data/task2_coefficient_poles_0083"
P88 = ROOT / "private_data/task2_guarded_rollout_0088"
P78 = ROOT / "private_data/task2_control_modes_0078"
BASE = ROOT / "private_data/task2_structure_0078_extend_G-fixed/step3000"
TARGETS = [62, 64, 66, 68, 71, 73, 77, 79, 84, 91, 98, 100, 104]
BASE_MODEL_SHA = "2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b"
CANDIDATE_MODEL_SHA = "5bab5fba84fb6261821c8654ffa6e33b4e015d12f472114028d8fe0e119577ba"
ANCHORS = {
    "0083_matrices": "46bcaa3480fdb422f3f98ca069243ca8f911bd44ae7c061f4eba75b77199d212",
    "0083_evidence": "acb8adf7a42a1d6857ae8c24e6fb736dec80e9daa20c7105c524a1a6ef1cde6a",
    "0088_fixed": "9725ee435f9a75600917dac3a90c5621d5d4237f8e157c7b3a1d450cd77407d6",
    "0088_evidence": "647d287fcaef90edc009de5f4225ec3fd623d91d726fedc0bb21cb18864cc92f",
    "0078_modal": "96ee08704a5dde68e8e65206109c993e0768dc378e4d3c344e0bdd43bb5025da",
    "0078_evidence": "cb69655ad6ce4bcb79b90d56e479ed5b5164f699d90472ce4a5452fb79d801ac",
    "0078_independent": "7ebd8ce0c1718e6d56d78eedff0306f43aad7deda27d06860a20901a39b37525",
}


def _paths():
    from check_control_modes_0078 import _source_paths, DEFAULT_FULL, DEFAULT_VERIFIED, DEFAULT_COUPLED, DEFAULT_M3, CHECKPOINT
    historical = {"old78_"+key: ROOT/path if not path.is_absolute() else path
                  for key, path in _source_paths(DEFAULT_FULL, DEFAULT_VERIFIED, DEFAULT_COUPLED, DEFAULT_M3, CHECKPOINT).items()}
    return {**historical, "p83": P83 / "matrices.npz", "p83e": P83 / "evidence.json",
            "p88": P88 / "fixed_arrays.npz", "p88e": P88 / "evidence.json",
            "m78": P78 / "modal_arrays.npz", "e78": P78 / "evidence.json",
            "i78": P78 / "independent_check.json", "base_model": BASE / "model.pt",
            "base_norm": BASE / "normalization.npz", "base_export": BASE / "structured_numeric_export.json",
            "base_meta": BASE / "checkpoint.json", "candidate_meta": P88 / "candidate/checkpoint.json",
            "training88": ROOT / "private_data/task2_guarded_rollout_0088_independent/training_independent.json",
            "training88_checker": ROOT / "private_data/task2_guarded_rollout_0088_independent/check_training.py",
            "source_model": ROOT / "private_data/task2_long_0060_5s/model.pt",
            "source_norm": ROOT / "private_data/task2_long_0060_5s/normalization.npz",
            "grammar": ROOT / "experiments/structure_grammar.py",
            "joint_latent": ROOT / "experiments/train_joint_latent.py",
            "joint_campaign": ROOT / "experiments/train_joint_campaign.py",
            "audit83": ROOT / "experiments/audit_coefficient_poles_0083.py",
            "audit88": ROOT / "experiments/train_guarded_rollout_0088.py",
            "candidate_model": P88 / "candidate/model.pt", "candidate_norm": P88 / "candidate/normalization.npz", "candidate_export": P88 / "candidate/structured_numeric_export.json",
            "network": ROOT / "private_data/task2_electrical_ports_0048/physical_network.npz",
            "m3": ROOT / "private_data/task2_m3_0020/model.npz", "m3_linearization": ROOT / "private_data/task2_verified_modes_0035b/M3_linearization.npz",
            "descriptor": ROOT / "private_data/task2_fullorder_descriptor_0035/descriptor.npz", "verified": ROOT / "private_data/task2_verified_modes_0035b/evidence.json",
            "coupled": ROOT / "private_data/task2_coupled_equilibrium_0065/coupled_equilibrium.npz", "coupled_evidence": ROOT / "private_data/task2_coupled_equilibrium_0065/evidence.json",
            "compare": ROOT / "experiments/compare_physical_modes.py",
            "code": Path(__file__)}


def _pair_rows(rows):
    fields = ("fom_index", "rom_index", "frequency_hz", "damping_ratio", "frequency_relative_difference",
              "damping_absolute_difference", "d_f", "d_z", "d_m", "mac", "eligible", "cost")
    return np.asarray([[float(r[k]) for k in fields] for r in rows], float)


def bounded(value, limit):
    return isinstance(value, (int, float)) and np.isfinite(value) and 0 <= value <= limit


def validate_inherited(independent, e83, e88, before):
    if independent.get("pair_difference_count") != 0 or not bounded(independent.get("max_pair_numeric_difference"), 1e-11) or not isinstance(independent.get("matched_count"), int) or independent.get("matched_count") != independent.get("reported_matched_count"):
        raise AssertionError("0078 independent QA")
    q = e83["models"]["baseline"]["qa"].get("flags", {})
    if not all(q.get(k) is True for k in ("finite", "origin", "gauge", "common_angle", "Q", "Q_common_angle", "eigen_residual", "fd_ad", "fd_spectral", "sensitivity")):
        raise AssertionError("0083 inherited QA")
    if before["base_model"] != BASE_MODEL_SHA or before["candidate_model"] != CANDIDATE_MODEL_SHA:
        raise AssertionError("model SHA")
    if e88.get("candidate_hashes", {}).get("model.pt") != CANDIDATE_MODEL_SHA:
        raise AssertionError("0088 candidate hash link")
    qa = e88.get("candidate_qa", {})
    if not all(bounded(qa.get('affinity', {}).get(k), 1e-10) for k in ('direct_column_relative_max', 'candidate_affinity_relative')) or not bounded(qa.get('actual_relative'), 1e-10):
        raise AssertionError("0088 cached QA")
    structural = qa.get('structural', {})
    if not all(bounded(structural.get(k), 1e-10) for k in ('origin_max', 'kinematic_max', 'gauge_field_relative', 'full_reload_relative', 'numeric_export_relative', 'power_identity_relative', 'J_antisymmetry', 'R_symmetry')) or not np.isfinite(structural.get('R_min_eigenvalue', np.nan)) or structural['R_min_eigenvalue'] < -1e-10:
        raise AssertionError('0088 inherited structure')


def validate_links(e83, e88, e78, before):
    for prior in (e83, e88, e78):
        if not prior.get('hashes_before') or prior['hashes_before'] != prior.get('hashes_after'):
            raise AssertionError('historical before/after hashes')
    for key, expected in e78['hashes_before'].items():
        if before.get('old78_'+key) != expected:
            raise AssertionError('0078 source link '+key)
    links83 = {'baseline_model':'base_model', 'baseline_meta':'base_meta', 'baseline_norm':'base_norm', 'baseline_export':'base_export',
               'base':'m3', 'network':'network', 'audit':'audit83', 'trainer':'old78_trainer', 'core':'old78_structured_core',
               'grammar':'grammar', 'joint_latent':'joint_latent', 'joint_campaign':'joint_campaign',
               'baseline_source_model':'source_model', 'baseline_source_norm':'source_norm'}
    for old, current in links83.items():
        if e83['hashes_before'].get(old) != before[current]: raise AssertionError('0083 source link '+old)
    for filename, current in {'model.pt':'candidate_model', 'normalization.npz':'candidate_norm', 'structured_numeric_export.json':'candidate_export', 'checkpoint.json':'candidate_meta'}.items():
        if e88['candidate_hashes'].get(filename) != before[current]: raise AssertionError('0088 candidate link '+filename)
    if e88['hashes_before'].get('baseline_model') != before['base_model'] or e88['hashes_before'].get('audit0088') != before['audit88']:
        raise AssertionError('0088 input/source link')


def validate_cached_objects(A, Q, A32, mass, baseline_A=None):
    A, Q, A32, mass = map(np.asarray, (A, Q, A32, mass))
    if A.shape != (33, 33) or Q.shape != (33, 32) or A32.shape != (32, 32) or mass.shape != (10,): raise ValueError("cached shape")
    if not all(np.isfinite(x).all() for x in (A, Q, A32, mass)): raise ValueError("cached finite")
    if not np.all(mass > 0): raise ValueError('positive inertia')
    if relmax(Q.T @ Q, np.eye(32)) > 1e-10: raise AssertionError("Q orthonormality")
    projector = np.eye(33); projector[:10, :10] -= np.ones((10, 10))/10
    if relmax(Q @ Q.T, projector) > 1e-10: raise AssertionError('Q common-frequency projector')
    if relmax(Q.T @ A @ Q, A32) > 1e-10: raise AssertionError("cached reduced matrix")
    if np.max(np.abs(A @ np.r_[np.ones(10), np.zeros(23)])) > 1e-10: raise AssertionError("common angle")
    if baseline_A is not None and relmax(A, baseline_A) > 1e-10: raise AssertionError("cached baseline mismatch")


def run(output: Path) -> dict:
    if output.exists(): raise FileExistsError(output)
    start = time.perf_counter(); deadline = start + 600.; paths = _paths()
    before = {}; current = {"stage": "load"}; arrays = {}; results = {}
    output.mkdir(parents=True, exist_ok=False)
    try:
        before = {k: sha(v) for k, v in paths.items()}
        for key, expected in ANCHORS.items():
            path = {"0083_matrices": paths["p83"], "0083_evidence": paths["p83e"], "0088_fixed": paths["p88"], "0088_evidence": paths["p88e"], "0078_modal": paths["m78"], "0078_evidence": paths["e78"], "0078_independent": paths["i78"]}[key]
            if before[{"0083_matrices":"p83","0083_evidence":"p83e","0088_fixed":"p88","0088_evidence":"p88e","0078_modal":"m78","0078_evidence":"e78","0078_independent":"i78"}[key]] != expected: raise AssertionError("anchor hash " + key)
        with np.load(paths["p83"], allow_pickle=False) as z: p83 = {k: z[k] for k in z.files}
        with np.load(paths["p88"], allow_pickle=False) as z: p88 = {k: z[k] for k in z.files}
        with np.load(paths["m78"], allow_pickle=False) as z: m78 = {k: z[k] for k in z.files}
        e83, e88, e78, i78 = (json.loads(paths[k].read_text(encoding="utf-8")) for k in ("p83e", "p88e", "e78", "i78"))
        validate_links(e83, e88, e78, before)
        desc = e78.get("descriptor", {})
        if e78.get("integrity_pass") is not True or desc.get("finite_modes") != 150 or desc.get("infinite_modes") != 549 or desc.get("indeterminate_pairs") != 0 or desc.get("source_integrity") is not True or desc.get("equilibrium_identity") is not True or desc.get("max_descriptor_residual", float("inf")) > 1e-8 or desc.get("pole_match_max_relative", float("inf")) > 1e-6: raise AssertionError("0078 inherited descriptor QA")
        if e78.get("mapping", {}).get("omega_s") != OMEGA_S or not all(bounded(e78.get('mapping', {}).get(k), 1e-10) for k in ('tf_delta_rel', 'tf_omega_rel', 'delta_dot_omega_rel')): raise AssertionError("0078 inherited units")
        validate_inherited(i78, e83, e88, before); training = json.loads(paths['training88'].read_text(encoding="utf-8"))
        if training.get("qa_pass") is not True or training.get("errors") != [] or training.get("hashes_before") != training.get("hashes_after"): raise AssertionError("0088 independent training QA")
        for key, expected in (("evidence", ANCHORS["0088_evidence"]), ("fixed", ANCHORS["0088_fixed"])):
            if training.get("hashes_before", {}).get(key) != expected: raise AssertionError("0088 training anchor " + key)
        if training['hashes_before'].get('checker') != before['training88_checker'] or training['hashes_before'].get('source88') != before['audit88']: raise AssertionError('0088 independent checker/source link')
        mass = np.asarray(m78["mass"], float); Q = np.asarray(p83["baseline_Q"], float)
        if relmax(Q, p88["Q"]) > 1e-10: raise AssertionError("Q cached mismatch")
        if Q.shape != (33, 32) or mass.shape != (10,): raise ValueError("Q/mass shape")
        if relmax(Q.T @ Q, np.eye(32)) > 1e-10 or np.max(np.abs(Q @ Q.T - np.block([[np.eye(10)-np.ones((10,10))/10, np.zeros((10,23))],[np.zeros((23,10)),np.eye(23)]]))) > 1e-10: raise AssertionError("Q projector")
        validate_cached_objects(p83["baseline_A33"], Q, p83["baseline_A32"], mass, p88["A33"])
        with np.load(paths["m3"], allow_pickle=False) as mz: m3_mass = np.asarray(mz["mass"], float)
        if relmax(mass, m3_mass) > 1e-10: raise AssertionError("inherited inertia")
        targets = e78.get("targets") or e78.get("target_modes")
        if targets is None or [int(t["index"]) for t in targets] != TARGETS: raise AssertionError("target IDs")
        verified = json.loads(paths['verified'].read_text())
        if targets != verified['fullorder_electromechanical_modes'] or not np.array_equal(m78['target_ids'], TARGETS): raise AssertionError('verified target metadata')
        if any(t['imag'] <= 1e-6 or t['participation_identifiable'] is not True or t['electromechanical_participation'] < .1 for t in targets): raise AssertionError('target eligibility')
        target_idx = np.asarray(m78["target_recomputed_indices"], int); fom_vec = np.asarray(m78["descriptor_right_vectors"])
        if target_idx.shape != (13,) or fom_vec.shape != (699,150) or not np.isfinite(fom_vec).all() or np.any(target_idx < 0) or np.any(target_idx >= 150): raise AssertionError('FOM cache dimensions')
        target_values = np.asarray(m78["descriptor_eigenvalues"])[target_idx]
        if relmax(target_values, np.array([complex(t['real'], t['imag']) for t in targets])) > 1e-6: raise AssertionError('target pole identity')
        target_vectors = [physical_output(fom_vec[:, i], mass, omega_scale=OMEGA_S) for i in target_idx]
        if not all(np.isfinite(v).all() and np.linalg.norm(v) > 1e-12 for v in target_vectors): raise AssertionError('target finite/output norm')
        if np.max(np.abs(np.column_stack(target_vectors) - m78["target_outputs"])) > 1e-10: raise AssertionError("cached target outputs")
        for arm, A, A32_saved in (("baseline", p83["baseline_A33"], p83["baseline_A32"]), ("0088", p88["actual_A33"], p88["actual_A32"])):
            current = {"stage": "eigensystem", "arm": arm}
            A = np.asarray(A, float); A32 = Q.T @ A @ Q
            validate_cached_objects(A, Q, A32_saved, mass)
            if time.perf_counter() > deadline: raise TimeoutError('0089 pre-eigensolve cap')
            values, vectors, residual = ordinary_modes(A32)
            arrays[arm+'_eigenvalues'], arrays[arm+'_eigenvectors'] = values, vectors
            cond = float(np.linalg.cond(vectors)); sep = np.inf
            for x in range(len(values)):
                for y in range(x): sep = min(sep, abs(values[x]-values[y])/max(1.,abs(values[x]),abs(values[y])))
            s = assignment(targets, target_vectors, values, vectors, mass, Q)
            availability = {str(t): {name: any(predicate(row) for row in s['all_pair_metrics'] if row['fom_index']==t)
                            for name, predicate in {'frequency': lambda r:r['frequency_relative_difference']<=.15,
                            'damping':lambda r:r['damping_absolute_difference']<=.05, 'shape':lambda r:r['mac']>=.8,
                            'all':lambda r:r['eligible']}.items()} for t in TARGETS}
            results[arm] = {"spectral_abscissa": float(np.max(values.real)), "eigen_residual_max": residual,
                            "condition": cond, "min_relative_separation": float(sep), "sensitivity_flag": bool(cond > 1e8 or sep < 1e-6),
                            "matched_count": len(s["matched"]), "all_pair_metrics": s["all_pair_metrics"], "matched": s["matched"], "gate_availability": availability,
                            "unmatched_fom": s["unmatched_fom"], "unmatched_rom": s["unmatched_rom"],
                            "excluded_null_output_rom": s["excluded_null_output_rom"], "oscillatory_rom_count": s["oscillatory_rom_count"]}
            arrays[arm+"_eigenvalues"], arrays[arm+"_eigenvectors"], arrays[arm+"_outputs"] = values, vectors, np.column_stack([physical_output(vectors[:,i],mass,Q) for i in range(32)])
            if time.perf_counter() > deadline: raise TimeoutError("0089 wall cap")
        after = {k: sha(v) for k,v in paths.items()}
        if before != after: raise RuntimeError("source/cache changed")
        result = {"loop":"0089", "test_read":False, "models_loaded":False, "inherited_qa":True, "source_sha256_before":before, "source_sha256_after":after,
                  "thresholds":{"frequency_relative":FREQ_GATE,"damping_absolute":DAMP_GATE,"mac":MAC_GATE}, "targets":TARGETS, "results":results,
                  "seconds":time.perf_counter()-start}
        if time.perf_counter() > deadline: raise TimeoutError('0089 save cap')
        np.savez_compressed(output/"modal_arrays.npz", Q=Q, mass=mass, target_ids=np.asarray(TARGETS), target_outputs=np.column_stack(target_vectors), **arrays)
        input_arrays = {"Q": Q, "mass": mass, "target_ids": np.asarray(TARGETS), "target_eigenvalues": target_values, "target_vectors": fom_vec[:, target_idx], "target_outputs": np.column_stack(target_vectors)}
        output_arrays = {}
        for arm in ("baseline", "0088"):
            input_arrays[arm+"_A33"] = np.asarray(p83["baseline_A33"] if arm == "baseline" else p88["actual_A33"])
            input_arrays[arm+"_A32"] = np.asarray(p83["baseline_A32"] if arm == "baseline" else p88["actual_A32"])
            input_arrays[arm+"_eigenvalues"] = arrays[arm+"_eigenvalues"]; input_arrays[arm+"_eigenvectors"] = arrays[arm+"_eigenvectors"]
            output_arrays[arm+"_matched_target_ids"] = np.asarray([x["fom_index"] for x in results[arm]["matched"]], int)
        np.savez_compressed(output/"input_arrays.npz", **input_arrays); np.savez_compressed(output/"output_arrays.npz", **output_arrays)
        if time.perf_counter() > deadline: raise TimeoutError('0089 final cap')
        result['seconds'] = time.perf_counter()-start
        result['source_paths'] = {k: str(v.resolve()) for k,v in paths.items()}
        (output/"evidence.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"); return result
    except Exception as exc:
        if arrays: np.savez_compressed(output/'partial_arrays.npz', **arrays)
        (output/"failure_evidence.json").write_text(json.dumps({"type":type(exc).__name__,"message":str(exc),"current":current,"source_sha256_before":before,"test_read":False,'seconds':time.perf_counter()-start,'results':results}, indent=2), encoding="utf-8"); raise


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True); run(p.parse_args().output)


if __name__ == "__main__": main()
