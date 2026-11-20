"""Post-hoc scalar decomposition of the frozen L1 risk, with one shared worst set."""
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = "osgkd_fit297_20261113"
RESULTS = ROOT/"results"/RUN
SOURCES = {}


def read(path):
    raw = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return json.loads(raw)


def main():
    started = time.perf_counter()
    output = RESULTS/"risk_components.json"
    assert not output.exists(), "risk decomposition already exists"
    analysis = read(RESULTS/"analysis.json")
    assert analysis["status"] == "completed" and analysis["run_id"] == RUN
    s_coi = read(RESULTS/"port_audit.json")["constants"]["s_coi"]
    assert s_coi == 1.7119010596239577
    results = {}
    for variant in ("C0","C1_dynamic"):
        path = ROOT/"artifacts"/RUN/"eval01"/variant/"rows.json"
        rows = read(path)
        assert SOURCES[path.relative_to(ROOT).as_posix()] == analysis["sources"][path.relative_to(ROOT).as_posix()]
        assert len(rows) == len({r["event_id"] for r in rows}) == 297
        results[variant] = {}
        for name in ("stable_1s","stable_2s","stable_5s","stable_8s","unstable_actual"):
            label, scope = name.split("_")
            scope = scope.removesuffix("s")
            population = [r for r in rows if r["label"] == label]
            frozen = analysis["scores"][variant]["splits"]["train_all"][name]
            assert len(population) == (267 if label == "stable" else 30) == frozen["denominator"] == frozen["covered"]
            assert frozen["official"] is not None and not frozen["unresolved_ids"]
            metrics = [r["scopes"][scope]["metrics"] for r in population]
            assert all(m is not None and m["points"] > 0 for m in metrics)
            total = [m["mean_motion_error"] for m in metrics]
            common = [m["common_speed_sse"]/(m["points"]*s_coi**2) for m in metrics]
            remainder = [a-b for a,b in zip(total,common,strict=True)]
            assert all(math.isfinite(v) and v >= -1e-12 for v in total+common+remainder)
            order = sorted(range(len(total)),key=lambda i:total[i],reverse=True)
            k = math.ceil(.1*len(total))
            worst = order[:k]
            ambiguous = k < len(order) and total[order[k-1]] == total[order[k]]
            weighted = lambda values: .8*math.fsum(values)/len(values)+.2*math.fsum(values[i] for i in worst)/k
            a,b,risk = weighted(common),weighted(remainder),weighted(total)
            official = frozen["official"]["motion_risk_v1"]
            error = a+b-official
            assert abs(error) <= 1e-10 and abs(risk-official) <= 1e-10
            component = lambda value: dict(risk_contribution=value,percent=100*value/official if official else None)
            results[variant][name] = dict(denominator=len(population),covered=frozen["covered"],worst_count=k,
                official_motion_risk_v1=official,reconstruction_error=error,
                common_speed=None if ambiguous else component(a),relative_angle_and_speed=None if ambiguous else component(b),
                worst_event_ids=None if ambiguous else [population[i]["event_id"] for i in worst],
                status="unknown_boundary_tie" if ambiguous else "decomposed",
                boundary_tie_ids=[population[i]["event_id"] for i in order if total[i] == total[order[k-1]]] if ambiguous else [])
    script = Path(__file__).resolve()
    result = dict(run_id=RUN,status="completed",interpretation="Descriptive decomposition of existing risk; remainder combines COI-relative angle and speed errors; both terms use the total-risk worst set.",
                  s_coi_rad_s=s_coi,results=results,sources=SOURCES,
                  script=dict(path=script.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(script.read_bytes()).hexdigest()),
                  wall_seconds=time.perf_counter()-started)
    with output.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+"\n")
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
