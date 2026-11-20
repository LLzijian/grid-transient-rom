"""Small, public fixtures for the motion-only fit297 adapter."""
import numpy as np
import pytest

from kiki.data import osgkd


def fixture(event_id="one", group="fault|bus:1", amplitude=1., points=31):
    t = np.arange(points)*.01
    theta = np.tile(np.arange(10,dtype=float),(points,1)) + t[:,None]
    omega = np.full((points,10),amplitude)
    torque = np.full_like(omega,2.)
    torque[-1,0] = np.nan
    return osgkd.event_record(event_id,group,"stable",t,theta,omega,torque,torque,torque,
                              {"record_start_delay_s":.001},np.arange(1,11,dtype=float))


def test_motion_packet_uses_one_gauge_and_keeps_common_speed():
    event = fixture()
    packet = osgkd.motion_packet(event,"fixture")
    assert set(packet) == {"history","constants_id"}
    assert set(packet["history"]) == {"t","theta","omega"}
    assert packet["history"]["theta"].shape == (11,10)
    assert np.all(packet["history"]["omega"] == 1)
    assert abs(packet["history"]["theta"][0]@np.arange(1,11)) < 1e-12
    np.testing.assert_allclose(np.diff(packet["history"]["theta"],axis=0), .01, atol=1e-14)
    np.testing.assert_array_equal(packet["history"]["theta"][-1],event["targets"]["theta"][0])
    assert event["t"][0] == 0 and not event["masks"]["tau_m"][-1,0]
    packet["history"]["theta"][0,0] = 123.
    assert event["history"]["theta"][0,0] != 123.


def test_calibration_equal_weights_events_and_masks_missing_ports():
    left,right = fixture("a",amplitude=1.,points=21),fixture("b",amplitude=3.,points=31)
    left["calibration"]["mechanical_input"][:,0] = np.nan
    scales,counts = osgkd.calibrate([left,right])
    np.testing.assert_allclose(scales["omega_scale"],np.sqrt(5))
    np.testing.assert_allclose(scales["Sp"],2.)
    assert counts["Sp"] == [1]+[2]*9
    assert scales["theta_scale"] == 1.


def test_fault_group_split_is_deterministic_and_keeps_repeats_together():
    items = [fixture(f"event_{i}",f"fault|bus:{i//2}") for i in range(12)]
    fit,check = osgkd.split_events(items)
    assert {e["group"] for e in check} == set(osgkd.group_order(items)[::5])
    assert not ({e["group"] for e in fit}&{e["group"] for e in check})
    assert {e["id"] for e in fit+check} == {e["id"] for e in items}
    fit_reversed,check_reversed = osgkd.split_events(items[::-1])
    assert {e["id"] for e in fit_reversed} == {e["id"] for e in fit}
    assert {e["id"] for e in check_reversed} == {e["id"] for e in check}
    with pytest.raises(ValueError,match="two or more"):
        osgkd.split_events(items[:2])


def test_balance_uses_recorded_endpoint_for_odd_coarse_interval():
    t = np.arange(14)*.01
    mass = np.arange(1,11,dtype=float)
    omega = 2*t[:,None]+np.zeros((t.size,10))
    tm = np.broadcast_to(2*mass,(t.size,10)).copy()
    te = np.zeros_like(tm)
    result = osgkd.balance_window(t,omega,tm,te,mass,np.zeros(10),0.,.13)
    assert result["status"] == "passed" and result["points"] == 14
    assert result["fine_rms"] < 1e-14 and result["mesh_difference_rms"] < 1e-14
    assert osgkd.balance_window(t,omega,tm,te,mass,np.zeros(10),.12,.13)["status"] == "not_applicable"
    tm[0,0] = np.nan
    assert osgkd.balance_window(t,omega,tm,te,mass,np.zeros(10),0.,.1)["status"] == "missing"


def test_audit_loader_rejects_nonfrozen_reference():
    with pytest.raises(ValueError,match="unexpected audit path"):
        osgkd.load_fit297({"path":"private_data/not_allowed.npz"})


def test_observation_release_chain_and_tampering(tmp_path, monkeypatch):
    from pathlib import Path
    script_dir = Path(__file__).resolve().parents[1]/"experiments"/osgkd.RUN
    monkeypatch.syspath_prepend(str(script_dir))
    import audit_ports as audit
    import materialize
    materialize.gauge_check()
    monkeypatch.setattr(osgkd,"ROOT",tmp_path)
    monkeypatch.setattr(audit,"ROOT",tmp_path)
    results = tmp_path/"results"/osgkd.RUN
    results.mkdir(parents=True)
    bundle = tmp_path/"artifacts"/osgkd.RUN/"port_audit/data.npz"
    monkeypatch.setattr(audit,"BUNDLE",bundle)
    def write(path, value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(osgkd.canonical_json(value))
        return dict(path=path.relative_to(tmp_path).as_posix(),bytes=path.stat().st_size,
                    sha256=osgkd.sha_bytes(path.read_bytes()))
    events = [fixture(f"e{i}",f"fault|bus:{i%10}") for i in range(297)]
    fit, check = osgkd.split_events(events)
    constants = {"constants_id":"synthetic","mass":list(range(1,11))}
    identities = dict(ordered_ids=[e["id"] for e in events],fit_ids=[e["id"] for e in fit],
                      check_ids=[e["id"] for e in check],constants_id="synthetic")
    source = {"path":"private_data/unopened.npz","bytes":0,"sha256":"synthetic"}
    original = dict(run=osgkd.RUN,status="blocked",**identities,constants=constants,
                    source_notes={"cache_offset_difference":0},sources={"fit_events.npz":source},
                    rows=[{**{k:e[k] for k in ("id","group","label")},
                           "metadata":{k:v for k,v in e["metadata"].items() if k != "angle_gauge_offset"}}
                          for e in events])
    refs = {"original_audit":write(results/"port_audit.json",original)}
    refs["recheck"] = write(results/"port_audit_recheck.json",
                            dict(run=osgkd.RUN,status="blocked",**identities,original_audit=refs["original_audit"]))
    refs["attribution"] = write(results/"source_time_attribution.json",dict(run_id=osgkd.RUN,status="unresolved"))
    source_hashes = {}
    for key,name in (("adapter","src/kiki/data/osgkd.py"),("materializer",f"experiments/{osgkd.RUN}/materialize.py")):
        source_hashes[key] = write(tmp_path/name,{"synthetic":key})["sha256"]
    adoption = dict(run_id=osgkd.RUN,status="adopted_for_prediction",materialization_released=True,
                    **refs,source_hashes=source_hashes,design=write(tmp_path/"design.md",{}),review=write(tmp_path/"review.md",{}))
    refs["observation_adoption"] = write(results/"observation_adoption.json",adoption)
    refs["bundle"] = audit.export(events,constants)
    material = dict(run_id=osgkd.RUN,status="completed",**refs,source=source,**identities)
    refs["materialization"] = write(results/"materialization.json",material)
    release = dict(run=osgkd.RUN,status="adopted_for_prediction",**refs,constants_id="synthetic")
    release_ref = write(results/"port_audit_release.json",release)
    loaded, frozen = osgkd.load_fit297(release_ref)
    assert len(loaded) == 297 and frozen["fit_ids"] == identities["fit_ids"]
    # A fresh outer hash cannot conceal a changed adopted decision.
    adoption["materialization_released"] = False
    release["observation_adoption"] = write(results/"observation_adoption.json",adoption)
    material["observation_adoption"] = release["observation_adoption"]
    release["materialization"] = write(results/"materialization.json",material)
    release_ref = write(results/"port_audit_release.json",release)
    with pytest.raises(ValueError,match="frozen evidence"):
        osgkd.load_fit297(release_ref)
    # The raw angle restoration differs from the old restoration by one scalar only.
    x = np.zeros((31,30),dtype=np.float64)
    te = np.ones((31,10),dtype=np.float64)
    row = dict(id="sample",label="stable",group=f"{audit.FAULT}|bus:1",points=31,
               metadata=dict(source_root="fixture",trajectory_end_s=.3),
               coverage={key:[31]*10 for key in osgkd.PORTS})
    desc = dict(id="sample",label="stable",group=1,root="fixture")
    record = materialize.reconstruct(0,desc,row,x,te,{"theta_ref":list(range(10)),"mass":list(range(1,11))})
    np.testing.assert_array_equal(record["targets"]["tau_e"],te[10:])
    np.testing.assert_allclose(record["history"]["theta"][0]@np.arange(1,11),0,atol=1e-12)
