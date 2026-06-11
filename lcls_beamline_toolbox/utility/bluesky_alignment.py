from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from pcdsdevices.signal import AvgSignal


DEFAULT_PHASE1_STEPS = (
    ("X1", "t1_th1", "t1_dh_sum"),
    ("X2", "t1_th2", "dd_sum"),
    ("X3", "t4_th2", "t4_dh_sum"),
    ("X4", "t4_th1", "do_sum"),
    ("CC1", "t2_th", "dcc_sum"),
    ("CC2", "t3_th", "dco_sum"),
)

DEFAULT_PHASE2_STEPS = (
    ("X1", "t1_chi1", "IP_cy"),
    ("X2", "t1_chi2", "IP_cy"),
    ("X3", "t4_chi2", "IP_cy"),
    ("X4", "t4_chi1", "IP_cy"),
)


def gaussian(x, center, sigma, amplitude, yoffset):
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) + yoffset


def scan_fit_center(
    RE,
    motor,
    detector,
    *,
    start,
    stop,
    steps,
    shots_per_step=1,
    averaging_duration=2,
    move=True,
):
    """
    Shared scan-fit-center primitive for a motor/diagnostic pair.

    This mirrors the logic already used in the hardware GUI:
    - perform a relative list scan
    - average shots per position
    - fit a Gaussian
    - optionally move to the fitted center
    """
    positions = np.linspace(start, stop, steps)
    if shots_per_step > 1 and hasattr(detector, "get"):
        detector = AvgSignal(
            detector,
            shots_per_step,
            averaging_duration,
            name=f"avg_{detector.name}",
        )
    else:
        positions = np.repeat(positions, shots_per_step)
    data_x = []
    data_y = []

    def collect(name, doc):
        if name == "event":
            data_x.append(doc["data"][motor.name])
            data_y.append(doc["data"][detector.name])

    token = RE.subscribe(collect)
    RE(bp.rel_list_scan([detector], motor, positions))
    RE.unsubscribe(token)

    xy = {}
    for x, y in zip(data_x, data_y):
        xy.setdefault(x, []).append(y)

    x_unique = np.array(sorted(xy.keys()))
    y_avg = np.array([np.mean(xy[x]) for x in x_unique])

    initial_guess = [
        np.mean(x_unique),
        np.std(x_unique),
        np.max(y_avg),
        np.min(y_avg),
    ]
    popt, _ = curve_fit(gaussian, x_unique, y_avg, p0=initial_guess)

    center, sigma, amplitude, yoffset = popt

    if move:
        RE(bps.mv(motor, float(center)))

    return {
        "x": x_unique,
        "y": y_avg,
        "center": center,
        "sigma": sigma,
        "amplitude": amplitude,
        "yoffset": yoffset,
    }


def scan_minimize_vertical_error(
    RE,
    motor,
    detector,
    *,
    start,
    stop,
    steps,
    shots_per_step=1,
    averaging_duration=2,
    move=True,
    objective="abs",
):
    """
    Shared scan-and-minimize primitive for image-plane alignment.

    This mirrors the notebook's Phase 2 workflow:
    - perform a relative list scan
    - average shots per position
    - evaluate an objective on the averaged detector values
    - optionally move to the position that minimizes the objective

    Parameters
    ----------
    objective :
        Currently supports ``"abs"`` to minimize the absolute value of the
        averaged detector signal, matching the notebook's ``argmin(abs(cy))``
        logic.
    """
    if objective != "abs":
        raise ValueError(f"Unsupported objective {objective!r}")

    positions = np.linspace(start, stop, steps)
    if shots_per_step > 1 and hasattr(detector, "get"):
        detector = AvgSignal(
            detector,
            shots_per_step,
            averaging_duration,
            name=f"avg_{detector.name}",
        )
    else:
        positions = np.repeat(positions, shots_per_step)
    data_x = []
    data_y = []

    def collect(name, doc):
        if name == "event":
            data_x.append(doc["data"][motor.name])
            data_y.append(doc["data"][detector.name])

    token = RE.subscribe(collect)
    RE(bp.rel_list_scan([detector], motor, positions))
    RE.unsubscribe(token)

    xy = {}
    for x, y in zip(data_x, data_y):
        xy.setdefault(x, []).append(y)

    x_unique = np.array(sorted(xy.keys()))
    y_avg = np.array([np.mean(xy[x]) for x in x_unique])
    objective_y = np.abs(y_avg)

    best_index = int(np.argmin(objective_y))
    best_position = float(x_unique[best_index])
    best_signal = float(y_avg[best_index])
    best_objective = float(objective_y[best_index])

    if move:
        RE(bps.mv(motor, best_position))

    return {
        "x": x_unique,
        "y": y_avg,
        "objective_y": objective_y,
        "best_index": best_index,
        "best_position": best_position,
        "best_signal": best_signal,
        "best_objective": best_objective,
        "objective": objective,
    }


def align_phase1(
    RE,
    backend,
    *,
    start,
    stop,
    steps,
    shots_per_step=1,
    averaging_duration=2,
    move=True,
    sequence=DEFAULT_PHASE1_STEPS,
):
    """
    Run the notebook's Phase 1 rocking-curve alignment sequence.

    Parameters
    ----------
    RE :
        Bluesky RunEngine.
    backend :
        Object exposing the canonical motor/diagnostic names used by both
        hardware and simulator backends.
    start, stop, steps :
        Relative scan parameters passed to each rocking-curve alignment.
    shots_per_step :
        Number of repeated shots to average at each position.
    averaging_duration :
        AvgSignal averaging duration at each scan point.
    move :
        If True, move each motor to the fitted center.
    sequence :
        Iterable of ``(label, motor_attr, detector_attr)`` triples.
    """
    results = {}

    for label, motor_attr, detector_attr in sequence:
        motor = getattr(backend, motor_attr)
        detector = getattr(backend, detector_attr)
        result = scan_fit_center(
            RE,
            motor,
            detector,
            start=start,
            stop=stop,
            steps=steps,
            shots_per_step=shots_per_step,
            averaging_duration=averaging_duration,
            move=move,
        )
        results[label] = {
            "motor": motor_attr,
            "detector": detector_attr,
            **result,
        }

    return results


def align_phase2(
    RE,
    backend,
    *,
    start,
    stop,
    steps,
    shots_per_step=1,
    averaging_duration=2,
    move=True,
    sequence=DEFAULT_PHASE2_STEPS,
    objective="abs",
):
    """
    Run the Phase 2 spatial-overlap alignment sequence.

    Parameters
    ----------
    RE :
        Bluesky RunEngine.
    backend :
        Object exposing the canonical motor/diagnostic names used by both
        hardware and simulator backends.
    start, stop, steps :
        Relative scan parameters passed to each centroid alignment.
    shots_per_step :
        Number of repeated shots to average at each position.
    averaging_duration :
        AvgSignal averaging duration at each scan point.
    move :
        If True, move each motor to the position that minimizes the objective.
    sequence :
        Iterable of ``(label, motor_attr, detector_attr)`` triples.
    objective :
        Objective passed to :func:`scan_minimize_vertical_error`.
    """
    results = {}

    for label, motor_attr, detector_attr in sequence:
        motor = getattr(backend, motor_attr)
        detector = getattr(backend, detector_attr)
        result = scan_minimize_vertical_error(
            RE,
            motor,
            detector,
            start=start,
            stop=stop,
            steps=steps,
            shots_per_step=shots_per_step,
            averaging_duration=averaging_duration,
            move=move,
            objective=objective,
        )
        results[label] = {
            "motor": motor_attr,
            "detector": detector_attr,
            **result,
        }

    return results
