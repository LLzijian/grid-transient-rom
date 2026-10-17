"""Synthetic-only tests for the D JSON summary calculations."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("summary", HERE / "summarize_confirmation.py")
summary = importlib.util.module_from_spec(spec); spec.loader.exec_module(summary)

def toy(numerical=True, finite=True, sse=2.0):
    return {"id": "toy", "label": "stable", "complete_5s": True, "actual_seconds": 5, "mse": 4.0,
            "numerical_pass": numerical,
            "metrics": [{"horizon": str(h), "available": True, "finite": True, "normalized_sse": sse, "n": 2} for h in (1,2,5,8)] + [{"horizon":"prefix","available":True,"normalized_sse":sse,"n":2}],
            "components": {str(h): {"available":True,"frequency": {"finite":finite,"normalized_mse":1.,"physical_sse":4.,"n_scalar_points":4},
                                       "mechanical_power":{"finite":finite,"normalized_mse":2.,"physical_sse":9.,"n_scalar_points":9}} for h in (1,2,5,8)} | {"prefix":{"available":True,"frequency":{"finite":finite,"normalized_mse":1.,"physical_sse":4.,"n_scalar_points":4},"mechanical_power":{"finite":finite,"normalized_mse":2.,"physical_sse":9.,"n_scalar_points":9}}}}

def run():
    e=toy(); a=summary._aggregate([e],"1s"); assert a["status"]=="complete" and a["normalized_risk"]==4.0
    short=toy(); short["actual_seconds"]=1; short["metrics"][2]["available"]=False; short["components"]["5"]["available"]=False
    assert summary._aggregate([short],"1s")["available_event_count"]==1
    assert summary._aggregate([e],"8s")["available_event_count"]==1
    bad=summary._aggregate([toy(False)],"1s"); assert bad["status"]=="unresolved" and bad["affected"]
    # Saved primary risk may differ by float32 aggregate rounding; per-event equality governs.
    assert summary.summarize_model({"records":[e],"primary_risk":4.000000000001,"classification_counts":{}})["horizons"]["5s"]["per_event_mse_match"]
    mismatch=toy(); mismatch["mse"]=99.
    bad5=summary.summarize_model({"records":[mismatch],"primary_risk":4.0,"classification_counts":{}})["horizons"]["5s"]
    assert bad5["status"]=="unresolved" and bad5["per_event_mse_match"] is False
    overflow=toy(sse=1e308); overflow["metrics"][0]["normalized_sse"]=1e308
    overflow["components"]["1"]["frequency"]["normalized_mse"]=1e308
    overflow["components"]["1"]["mechanical_power"]["normalized_mse"]=1e308
    assert summary._aggregate([overflow],"1s")["status"]=="unresolved"

if __name__ == "__main__": run()
