from __future__ import annotations

import importlib

import bluesky.plan_stubs as bps

from lcls_beamline_toolbox.utility.bluesky_alignment import scan_fit_center


def learn_phase1_offset(
    *,
    RE,
    sim_backend,
    hw_backend,
    motor_attr,
    detector_attr,
    norm_detector_attr,
    offset_pv,
    start,
    stop,
    steps,
    shots_per_step=1,
    duration=2,
    move=True,
    write=False,
):
    """
    Run one paired Phase 1 scan on simulation and hardware, fit both centers,
    and learn the additive offset between them.

    Parameters
    ----------
    RE : bluesky.RunEngine
        RunEngine used for both the model and hardware scans.
    sim_backend, hw_backend : object
        Backend namespaces with matching motor and diagnostic attribute names.
        In the hardware workflow, these can be the LUME/model-PV backend and
        raw-diode hardware backend from ``scripts/SnD_Phase1.py``.
    motor_attr : str
        Name of the motor attribute to scan.
    detector_attr : str
        Name of the diagnostic signal to fit.
    norm_detector_attr : str
        Name of the reference signal used to fit ``detector / norm_detector``.
    offset_pv : str
        EPICS PV that receives the learned offset when ``write=True``.
    start, stop : float
        Relative scan limits in the backend motor units.
    steps : int
        Number of scan points.
    shots_per_step : int, optional
        Number of samples to average at each scan point.
    duration : float, optional
        AvgSignal averaging duration passed to the scan helper.
    move : bool, optional
        If True, move each backend motor to its fitted center after scanning.
    write : bool, optional
        If True, write the learned offset to ``offset_pv``.

    Returns
    -------
    dict
        Result dictionary containing fitted centers, the learned ``offset``,
        scan results, and write/move metadata. ``offset`` is in the backend
        motor units.
    """
    sim_motor = getattr(sim_backend, motor_attr)
    sim_detector = getattr(sim_backend, detector_attr)
    sim_norm_detector = getattr(sim_backend, norm_detector_attr)
    hw_motor = getattr(hw_backend, motor_attr)
    hw_detector = getattr(hw_backend, detector_attr)
    hw_norm_detector = getattr(hw_backend, norm_detector_attr)

    sim_result = scan_fit_center(
        RE,
        sim_motor,
        sim_detector,
        sim_norm_detector,
        start=start,
        stop=stop,
        steps=steps,
        shots_per_step=shots_per_step,
        averaging_duration=duration,
        move=False,
    )
    hw_result = scan_fit_center(
        RE,
        hw_motor,
        hw_detector,
        hw_norm_detector,
        start=start,
        stop=stop,
        steps=steps,
        shots_per_step=shots_per_step,
        averaging_duration=duration,
        move=False,
    )

    sim_center = float(sim_result["center"])
    hw_center = float(hw_result["center"])

    if move:
        RE(bps.mv(sim_motor, sim_center))
        RE(bps.mv(hw_motor, hw_center))

    offset = hw_center - sim_center

    if write:
        importlib.import_module("epics").caput(offset_pv, offset)

    return {
        "motor": motor_attr,
        "detector": detector_attr,
        "norm_detector": norm_detector_attr,
        "offset_pv": offset_pv,
        "sim_center": sim_center,
        "hw_center": hw_center,
        "offset": offset,
        "sim_result": sim_result,
        "hw_result": hw_result,
        "moved": move,
        "wrote_pv": write,
    }
