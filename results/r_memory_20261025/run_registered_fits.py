"""Continue the preregistered finite queue after actual calibration completion.

Uses the unchanged frozen controller. No retries, source edits, promotion,
new models, or new optimization settings are introduced by this entry point.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "src")]
from experiments import r_memory_execution_20261025 as control
from experiments.uncapped_windows_job_20261021 import identity

SOURCE = "6409d55f48aff1fd7c0091a26f44252a170502a3"


def main():
    base = control.BASE
    control.check_binding(control.read(base / "binding.json"), SOURCE, base=base)
    selected = control.read(base / "calibration/selection.json")["selected"]
    folder = base / "ownership/registered_fit_queue"
    owner = identity()
    control.write_new(folder / "started.json", {
        "owner": owner, "creation_filetime_decimal": str(owner["creation_filetime_utc"]),
        "source_commit": SOURCE, "driver_sha256": control.sha(Path(__file__)),
        "modes": list(control.MODES), "seeds": list(control.SEEDS),
        "updates_each": 2048, "resume": False,
        "scope": "existing registered fit/noise/summary only; final science decision belongs to root"})
    skipped, failed = [], []
    for mode in control.MODES:
        for seed in control.SEEDS:
            if selected.get(mode) not in (1, 3):
                skipped.append({"mode": mode, "seed": seed, "reason": "registered finite calibration"})
                continue
            try:
                control.launch(base, SOURCE, "fit", mode, seed, "cuda")
                result = {"status": "NATIVE_COMPLETED", "mode": mode, "seed": seed}
            except Exception as error:
                control.check_binding(control.read(base / "binding.json"), SOURCE, base=base)
                key = control.phase_key("fit", mode, seed)
                control.clean_receipt(control.read(base / "native" / key / "receipt.json"),
                    SOURCE, key, allow_exception=True)
                if not (base / "models" / f"{mode}_{seed}" / "failure.json").is_file():
                    raise
                result = {"status": "ISOLATED_FAILURE", "mode": mode, "seed": seed,
                          "error": repr(error)}
                failed.append(result)
                control.check_locks(base=base)
            control.write_new(folder / f"{mode}_{seed}.json", result)
    if not skipped and not failed:
        control.launch(base, SOURCE, "noise", device="cuda")
    control.launch(base, SOURCE, "summary", device="cuda")
    control.write_new(folder / "complete.json", {
        "status": "ENDPOINTS_READY_FOR_ROOT_REVIEW" if not failed and not skipped else "PARTIAL_REQUIRES_ROOT_REVIEW",
        "source_commit": SOURCE, "skipped": skipped, "failed": failed, "owner_end": identity(),
        "model_adoption": "none", "test_access": False})


if __name__ == "__main__":
    main()
