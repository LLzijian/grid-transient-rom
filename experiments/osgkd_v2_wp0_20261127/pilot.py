"""Single-event, zero-update diagnostics on the frozen Sprint 2A endpoints."""

from unittest.mock import patch

import h5py
import numpy as np
import torch

from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io
from experiments.osgkd_v2_p2a_post_audit_20261124 import contributions
from experiments.osgkd_v2_fast_zero_sum_20261124 import scoring
from kiki.data import osgkd_train_val as data
from kiki.evaluation.osgkd import motion_metrics, numerical_gate
from kiki.evaluation.osgkd_autonomous import autonomous_outcome
from kiki.models.osgkd_fitting import IntegrationWork

PILOT = "pilot_0232"
FAILED = ("graph_24111", "graph_24131")
DONOR = "graph_24121"
STEPS = 200


def packet(constants, trace_ref):
    with np.load(io.verified(trace_ref), allow_pickle=False) as previous:
        truth = previous["observed"][:STEPS+1].copy()
        times = previous["t"][:STEPS+1].copy()
    io.require(truth.shape == (STEPS+1, 20)
               and np.allclose(times, np.arange(STEPS+1)*.01, atol=1e-10, rtol=0),
               "old pilot prediction coordinates differ")
    rows, splits = data._metadata("validation")
    event = rows[PILOT]
    io.require(PILOT in splits["validation"] and PILOT not in splits["test"]
               and event["label"] == "stable", "pilot outside the designated cohort")
    location = io.ROOT / "private_data" / event["source_root"] / "dataset.h5"
    with h5py.File(location, "r") as raw:
        for label,key in (("omega_s","omega_s"),("case_sha256","source_case_sha256"),
                          ("postfault_signature","postfault_signature")):
            io.require(raw.attrs[label] == constants[key], "pilot source differs: " + label)
        position = raw["events"][PILOT]
        t = position["t"][:11]
        theta = position["delta"][:11]
        omega = (position["omega"][:11] - 1.) * constants["omega_s"]
    io.require(np.allclose(t, np.arange(11)*.01, atol=1e-10, rtol=0), "history grid differs")
    mass = np.asarray(constants["mass"])
    theta -= float(theta[0] @ (mass / mass.sum()))
    io.require(np.array_equal(np.r_[theta[-1],omega[-1]],truth[0]),
               "history endpoint differs from archived truth")
    return [torch.as_tensor(x,dtype=torch.float64) for x in (theta,omega)],truth


def envelope(path):
    theta = path[:,:10]
    span = np.ptp(theta,axis=1)
    peaks = np.flatnonzero((span[1:-1]>span[:-2]) & (span[1:-1]>=span[2:]))+1
    first = int(peaks[0]) if len(peaks) else None
    index = int(span.argmax())
    return dict(first_peak_s=first*.01 if first is not None else None,
                first_peak_rad=float(span[first]) if first is not None else None,
                prefix_max_s=index*.01,prefix_max_rad=float(span[index]),
                max_pair_zero_based=[int(theta[index].argmax()),int(theta[index].argmin())],
                max_1s_rad=float(span[:101].max()),end_rad=float(span[-1]))


def edge_stats(model, theta, omega):
    n = model.n
    phase = theta.unsqueeze(-1)-theta.unsqueeze(-2)
    reference = model.theta_ref.unsqueeze(-1)-model.theta_ref.unsqueeze(-2)
    relative = omega.unsqueeze(-1)-omega.unsqueeze(-2)
    scale = model.omega_scale
    divisor = ((scale[:,None].square()+scale[None,:].square())/2).sqrt()
    features = torch.stack((phase.sin()-reference.sin(),phase.cos()-reference.cos(),relative),-1)
    layer = model.encoder.edge_mlp[0]
    mask = (~torch.eye(n,dtype=torch.bool)).expand(len(theta),n,n)
    result = {}
    names = ("sin","cos","relative_velocity")
    for key,value in (("native",features),("pair_scale",torch.cat(
            (features[...,:2],(relative/divisor)[...,None]),-1))):
        pre = layer(value)[mask]
        derivative = 1.-pre.tanh().square()
        weighted = value[...,None,:]*layer.weight
        result[key] = dict(input_rms=dict(zip(names,value[mask].square().mean(0).sqrt().tolist(),strict=True)),
                           weighted_input_rms=dict(zip(names,weighted[mask].square().mean((0,1)).sqrt().tolist(),strict=True)),
                           evaluated_units=derivative.numel(),
                           saturated_fraction=float((derivative<=.01).double().mean()),
                           p05_derivative=float(torch.quantile(derivative,.05)),
                           max_abs_preactivation=float(pre.abs().max()))
    return result,divisor


class ExchangeScale:
    def __init__(self,model,factor):
        self.model,self.factor = model,factor

    def rhs(self,t,state):
        model = self.model
        n = model.n
        rhs = model.rhs(t,state)
        delta = (1.-self.factor)*model.ports(state).Fc @ model.inverse_mass.T
        return torch.cat((rhs[...,:n],rhs[...,n:2*n]+delta,rhs[...,2*n:]),-1)


def rollout(model,initial,work,truth,constants):
    coarse = scoring.integrate(model,initial,STEPS,internal_step=.005,work=work).numpy()
    fine = scoring.integrate(model,initial,STEPS,internal_step=.0025,work=work).numpy()
    gate = numerical_gate(coarse,fine,np.asarray(constants["omega_scale"]))
    if not gate["passed"]:
        return {"numerical":gate},fine
    mass = np.asarray(constants["mass"])
    motion = {}
    for h in (.1,.25,.5,1.,2.):
        stop = round(h*100)+1
        motion[str(h)] = motion_metrics(fine[1:stop,:20],truth[1:stop],mass/mass.sum(),
                                        constants["omega_scale"],constants["s_coi"])
    gap = np.max(np.abs(np.ptp(coarse[:,:10],axis=1)-np.ptp(fine[:,:10],axis=1)))
    return dict(numerical=gate,motion=motion,angle=envelope(fine),mesh_angle_gap_rad=float(gap)),fine


def force(model,path):
    value = torch.from_numpy(path)
    rhs = model.rhs(None,value)
    parts = contributions.fields(model,value)
    io.require(torch.allclose(sum(parts.values()),rhs,atol=1e-11,rtol=1e-10),"field decomposition differs")
    _,omega,_,_,e = value.split(model.n,-1)
    port = model.ports(value)
    torques = dict(mechanical=port.tau_m,internal_reciprocal=-port.Fc,environment_memory=-model.Se*e,
                   environment_other=-port.tau_env+model.Se*e,damping=-omega@model.damping.T)
    out = {}
    for name,torque in torques.items():
        normalized = torque@model.inverse_mass.T/model.omega_scale
        out[name] = dict(node_t0=torque[0].tolist(),node_impulse=torch.trapezoid(torque,dx=.01,dim=0).tolist(),
                         acceleration_mean_square=float(torch.trapezoid(normalized.square().mean(-1),dx=.01)/2),
                         signed_power_integral=float(torch.trapezoid((omega*torque).sum(-1),dx=.01)))
    out["memory_derivative"] = {name:float(torch.trapezoid((x[...,-model.n:]/model.init_bounds[-model.n:]).square().mean(-1),dx=.01)/2)
                                for name,x in parts.items() if name.startswith("memory_")}
    return out


def linear_region(model,path,spectrum):
    detail = io.read(io.verified(spectrum["detail_ref"]))
    scales = np.asarray(detail["scales"])
    basis = np.asarray(detail["basis"])
    J = np.asarray(detail["quotient_jacobian"])
    x = (path-np.asarray(detail["reference"]))/scales
    dx = (model.rhs(None,torch.from_numpy(path)).numpy()/scales)@basis
    error = np.linalg.norm(dx-(x@basis)@J.T,axis=1)/np.maximum(np.linalg.norm(dx,axis=1),1e-15)
    angle = np.ptp(x[:,:10],axis=1)
    other = np.max(np.abs(x[:,10:]),axis=1)
    valid = (angle<=.2)&(other<=.1)&(error<=.1)
    poles = np.linalg.eigvals(J)
    return dict(max_real_per_s=spectrum["max_real"],native_positive_modes=spectrum["positive_modes"],
                linear_points_1s=int(valid[:101].sum()),linear_points_2s=int(valid.sum()),
                t0_angle_offset_rad=float(angle[0]),t0_nonangular_max=float(other[0]),
                t0_residual=float(error[0]),max_prefix_residual=float(error.max()),
                mesh_max_rk4_amplification={str(h):float(abs(sum((h*poles)**k/f for k,f in enumerate((1,1,2,6,24)))).max())
                                            for h in (.005,.0025)})


@torch.no_grad()
def audit(models,scores,constants,trace_ref,local):
    histories,truth = packet(constants,trace_ref)
    work = IntegrationWork()
    initial = {name:model.initialize(*histories) for name,model in models.items()}
    work.encoder_events += len(models)
    report = {}
    traces = {"observed":truth,"theta_history":histories[0].numpy(),"omega_history":histories[1].numpy()}
    for name,model in models.items():
        before = model.checkpoint()["state_dict"]
        result = scores[name]
        rows = io.read(io.verified(result["score"]["rows_ref"]))
        row, = [entry for entry in rows if entry["event_id"]==PILOT]
        outcome = autonomous_outcome(row)
        io.require(row["label"]=="stable" and (outcome=="autonomous_trigger")== (name in FAILED),"frozen pilot outcome differs")
        point,traces[name] = rollout(model,initial[name],work,truth,constants)
        io.require(point["numerical"]["passed"],"frozen endpoint fails paired prefix")
        for h in (1,2):
            actual = point["motion"][str(float(h))]["mean_motion_error"]
            expected = row["scopes"][str(h)]["metrics"]["mean_motion_error"]
            io.require(np.isclose(actual,expected,rtol=1e-9,atol=1e-10),"prefix differs from official row")
        raw = initial[name][20:]
        ratio = raw/model.init_bounds
        entry = dict(native=point,latent_t0=raw.reshape(3,10).tolist(),latent_t0_normalized=ratio.reshape(3,10).tolist(),
                     at_least_95pct_bound=(ratio.abs().reshape(3,10)>=.95).sum(-1).tolist(),
                     archived_outcome=outcome,archived_trigger_record_s=row["classification_8s"]["predicted_time_record_s"],
                     force=force(model,traces[name]),reference_spectrum=linear_region(model,traces[name],result["spectrum"]),
                     edge_message=None,scale_only=None,swaps={})
        if name.startswith("graph_"):
            entry["edge_message"],divisor = edge_stats(model,*histories)
            mlp = model.encoder.edge_mlp
            original = mlp.forward

            def scaled(x):
                return original(torch.cat((x[...,:2],(x[...,2]/divisor)[...,None]),-1))

            with patch.object(mlp,"forward",scaled):
                z = model.initialize(*histories)
            work.encoder_events += 1
            point,traces[name+"_scale"] = rollout(model,z,work,truth,constants)
            entry["scale_only"] = {**point,"latent_t0":z[20:].reshape(3,10).tolist()}
        io.require(all(torch.equal(x,model.state_dict()[key]) for key,x in before.items()),"frozen model tensor changed")
        report[name] = entry

    for variant in FAILED:
        for recipient, donor in ((variant,DONOR),(DONOR,variant)):
            z = torch.cat((initial[recipient][:20],initial[donor][20:]))
            point,traces[recipient+"_init_"+donor] = rollout(models[recipient],z,work,truth,constants)
            report[recipient]["swaps"][donor] = point
    exchange = {}
    for multiplier in (.9,1.1):
        label = str(multiplier)
        system = ExchangeScale(models[FAILED[0]],multiplier)
        exchange[label],traces["exchange_"+label] = rollout(system,initial[FAILED[0]],work,truth,constants)

    observed = envelope(truth)
    gates = {}
    for name in FAILED:
        entry = report[name]
        native,altered,swap = entry["native"],entry["scale_only"],entry["swaps"][DONOR]
        loss = native["motion"]["1.0"]["mean_motion_error"]
        if loss>0 and altered["numerical"]["passed"] and swap["numerical"]["passed"]:
            gain = 1.-altered["motion"]["1.0"]["mean_motion_error"]/loss
            target = observed["max_1s_rad"]
            angle_error = abs(native["angle"]["max_1s_rad"]-target)
            angle_gain = angle_error-abs(altered["angle"]["max_1s_rad"]-target)
            swap_angle_gain = angle_error-abs(swap["angle"]["max_1s_rad"]-target)
            floor = 10*max(x["mesh_angle_gap_rad"] for x in (native,altered,swap))
            passed = (gain>=.25 and angle_gain>floor and swap_angle_gain>floor
                      and swap["motion"]["1.0"]["mean_motion_error"]<loss
                      and entry["reference_spectrum"]["linear_points_1s"]==0)
        else:
            gain,angle_gain,swap_angle_gain,passed = None,None,None,False
        gates[name] = dict(passed=bool(passed),motion_relative_gain=gain,angle_error_reduction_rad=angle_gain,
                           swap_angle_error_reduction_rad=swap_angle_gain)
    trace = local/"pilot_prefix.npz"
    with trace.open("xb") as stream:
        np.savez_compressed(stream,**traces)
    return dict(event_id=PILOT,observation=observed,models=report,exchange_sensitivity=exchange,
                scale_gate_by_seed=gates,release_wp1b=all(item["passed"] for item in gates.values()),
                trace_ref=io.identity(trace),work=work.snapshot())
