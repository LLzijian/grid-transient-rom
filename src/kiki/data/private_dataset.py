"""One read/coordinate adapter for the private raw-ANDES dataset."""
from __future__ import annotations

import h5py
import numpy as np

from kiki.physics import coi_project, coi_project_power, complete_incidence


def canonical_arrays(raw, equilibrium, inertia, omega_s):
    """Raw speed is absolute pu; raw mechanical/electrical power is system pu."""
    inertia = np.asarray(inertia, dtype=float)
    result = {"t": np.asarray(raw["t"]), "mass": inertia / float(omega_s)}
    for key in ("delta", "omega"):
        scale = float(omega_s) if key == "omega" else 1.0
        result[key] = coi_project(np.asarray(raw[key]) * scale, inertia)
        result[f"equilibrium_{key}"] = coi_project(np.asarray(equilibrium[key]) * scale, inertia)
    for key in ("Pm", "Pe"):
        result[key] = coi_project_power(np.asarray(raw[key]), inertia)
        result[f"equilibrium_{key}"] = coi_project_power(np.asarray(equilibrium[key]), inertia)
    result["incidence"] = complete_incidence(len(inertia))[0]
    # Preserve common frequency for explicit governor identification. These
    # transforms live here, not in separate training/generation implementations.
    result["delta_sync"] = np.asarray(raw["delta"]) - np.asarray(equilibrium["delta"])
    result["omega_sync"] = (np.asarray(raw["omega"]) - 1.0) * float(omega_s)
    result["Pm_raw"] = np.asarray(raw["Pm"])
    result["Pm_star"] = np.asarray(equilibrium["Pm"])
    return result


def load_private_event(path, event_id):
    with h5py.File(path, "r") as file:
        event = file["events"][event_id]
        raw = {key: event[key][...] for key in ("t", "delta", "omega", "Pm", "Pe")}
        equilibrium = {key: file["equilibrium"][key][...] for key in ("delta", "omega", "Pm", "Pe")}
        data = canonical_arrays(raw, equilibrium, file["inertia"][...], file.attrs["omega_s"])
    return data


def los_index(delta, omega_pu, inertia, config):
    """First endpoint of sustained angle AND relative-speed divergence."""
    angle = np.ptp(np.asarray(delta), axis=1)
    speed = np.max(np.abs(coi_project(np.asarray(omega_pu), np.asarray(inertia))), axis=1)
    exceeds = (angle > config["los_angle_rad"]) & (speed > config["los_relative_speed_pu"])
    hold = int(config["los_hold_samples"])
    if len(exceeds) < hold:
        return None
    hits = np.flatnonzero(np.convolve(exceeds.astype(int), np.ones(hold, dtype=int), mode="valid") == hold)
    return int(hits[0] + hold - 1) if len(hits) else None


def check_arrays(data, full_state, config):
    time = np.asarray(data["t"])
    if len(time) < 2 or abs(time[0]) > 1e-10 or np.any(np.diff(time) <= 0):
        raise ValueError("invalid_time_axis")
    if not np.allclose(np.diff(time), config["sample_interval_s"], atol=1e-9, rtol=0):
        raise ValueError("invalid_sampling_interval")
    for key in ("delta", "omega", "Pm", "Pe"):
        if np.shape(data[key]) != (len(time), 10) or not np.isfinite(data[key]).all():
            raise ValueError(f"invalid_{key}_shape_or_finiteness")
    if len(full_state) != len(time) or not np.isfinite(full_state).all():
        raise ValueError("invalid_full_state")
    repeated = np.max(np.abs(np.diff(full_state, axis=0)), axis=1) < 1e-13
    if len(repeated) >= 5 and np.any(np.convolve(repeated.astype(int), np.ones(5, int), "valid") == 5):
        raise ValueError("frozen_or_padded_states")
