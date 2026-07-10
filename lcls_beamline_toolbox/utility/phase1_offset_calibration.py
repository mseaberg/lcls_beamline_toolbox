from __future__ import annotations

import importlib
import math

import bluesky.plan_stubs as bps

from lcls_beamline_toolbox.utility.bluesky_alignment import scan_fit_center


def learn_phase1_offset(
    *,
    RE_sim,
    sim_backend,
    RE_hw,
    hw_backend,
    motor_attr,
    detector_attr,
    offset_pv,
    start,
    stop,
    steps,
    shots_per_step=1,
    move=True,
    write=False,
    caput_fn=None,
):
    """
    Run one paired Phase 1 scan on simulation and hardware, fit both centers,
    and learn the additive offset between them.

    The returned ``offset`` is in the native motor units. For the Phase 1 theta
    motors, ``offset_deg`` is the value typically written to the offset PV.
    """
    sim_motor = getattr(sim_backend, motor_attr)
    sim_detector = getattr(sim_backend, detector_attr)
    hw_motor = getattr(hw_backend, motor_attr)
    hw_detector = getattr(hw_backend, detector_attr)

    sim_result = scan_fit_center(
        RE_sim,
        sim_motor,
        sim_detector,
        start=start,
        stop=stop,
        steps=steps,
        shots_per_step=shots_per_step,
        move=False,
    )
    hw_result = scan_fit_center(
        RE_hw,
        hw_motor,
        hw_detector,
        start=start,
        stop=stop,
        steps=steps,
        shots_per_step=shots_per_step,
        move=False,
    )

    sim_center = float(sim_result["center"])
    hw_center = float(hw_result["center"])

    if move:
        RE_sim(bps.mv(sim_motor, sim_center))
        RE_hw(bps.mv(hw_motor, hw_center))

    offset = hw_center - sim_center
    offset_deg = math.degrees(offset)

    if write:
        if caput_fn is None:
            caput_fn = importlib.import_module("epics").caput
        caput_fn(offset_pv, offset_deg)

    return {
        "motor": motor_attr,
        "detector": detector_attr,
        "offset_pv": offset_pv,
        "sim_center": sim_center,
        "hw_center": hw_center,
        "offset": offset,
        "offset_deg": offset_deg,
        "sim_result": sim_result,
        "hw_result": hw_result,
        "moved": move,
        "wrote_pv": write,
    }


def write_phase1_offset(result, caput_fn=None):
    """
    Write a learned Phase 1 offset to the corresponding EPICS PV in degrees.
    """
    if caput_fn is None:
        caput_fn = importlib.import_module("epics").caput

    caput_fn(result["offset_pv"], result["offset_deg"])
    return result["offset_pv"], result["offset_deg"]

