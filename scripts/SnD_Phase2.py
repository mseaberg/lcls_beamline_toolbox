from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from IPython.display import display

from lcls_beamline_toolbox.models.split_and_delay_motion import SND
from lcls_beamline_toolbox.models.split_and_delay_ophyd import SndOphyd
from lcls_beamline_toolbox.utility.bluesky_alignment import (
    DEFAULT_PHASE1_STEPS,
    DEFAULT_PHASE2_STEPS,
    align_phase1,
    align_phase2,
    gaussian,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHASE1_OFFSETS = {
    "t1_th1": 60e-6,
    "t1_th2": -45e-6,
    "t4_th2": 35e-6,
    "t4_th1": -55e-6,
}

PHASE2_OFFSETS = {
    "t1_chi2": 80e-6,
    "t4_chi2": -60e-6,
    "t4_chi1": 50e-6,
}

ALL_OFFSETS = {**PHASE1_OFFSETS, **PHASE2_OFFSETS}

def apply_offsets(snd, offsets):
    for attr, delta in offsets.items():
        getattr(snd, attr).mvr(delta)
    snd.propagate_delay()


def capture_positions(snd, motor_attrs):
    return {motor_attr: getattr(snd, motor_attr).wm() for motor_attr in motor_attrs}


def make_run_engine(with_bec=True):
    RE = RunEngine({})
    if with_bec:
        RE.subscribe(BestEffortCallback())
    return RE


def phase1_summary_df(sequence, initial_positions, final_snd, truth_snd, results):
    rows = []
    for label, motor_attr, detector_attr in sequence:
        truth_position = getattr(truth_snd, motor_attr).wm()
        final_position = getattr(final_snd, motor_attr).wm()
        initial_offset = initial_positions[motor_attr] - truth_position
        final_offset = final_position - truth_position
        rows.append(
            {
                "crystal": label,
                "motor": motor_attr,
                "detector": detector_attr,
                "initial_offset_urad": initial_offset * 1e6,
                "final_offset_urad": final_offset * 1e6,
                "fit_center_urad": results[label]["center"] * 1e6,
                "improved": abs(final_offset) < abs(initial_offset),
            }
        )
    return pd.DataFrame(rows)


def phase2_summary_df(sequence, initial_positions, final_snd, results):
    rows = []
    for label, motor_attr, detector_attr in sequence:
        final_position = getattr(final_snd, motor_attr).wm()
        rows.append(
            {
                "crystal": label,
                "motor": motor_attr,
                "detector": detector_attr,
                "initial_position_urad": initial_positions[motor_attr] * 1e6,
                "final_position_urad": final_position * 1e6,
                "best_position_urad": results[label]["best_position"] * 1e6,
                "best_ip_cy": results[label]["best_signal"],
                "best_vertical_error": results[label]["best_objective"],
            }
        )
    return pd.DataFrame(rows)


def phase2_outcome_df(initial_ip_cy, final_ip_cy, initial_ip_cx, final_ip_cx):
    return pd.DataFrame(
        [
            {
                "initial_ip_cy": initial_ip_cy,
                "final_ip_cy": final_ip_cy,
                "initial_abs_ip_cy": abs(initial_ip_cy),
                "final_abs_ip_cy": abs(final_ip_cy),
                "initial_ip_cx": initial_ip_cx,
                "final_ip_cx": final_ip_cx,
                "reduced_vertical_error": abs(final_ip_cy) < abs(initial_ip_cy),
            }
        ]
    )

phase2_initial_positions = capture_positions(
    snd,
    [motor_attr for _, motor_attr, _ in DEFAULT_PHASE2_STEPS],
)
phase2_initial_ip_cy = snd.get_IP_cy()
phase2_initial_ip_cx = snd.get_IP_cx()

print("Before Phase 2 alignment")
print(f"  IP_cy: {phase2_initial_ip_cy:.6g}")
print(f"  IP_cx: {phase2_initial_ip_cx:.6g}")

phase2_results = align_phase2(
    RE,
    snd_ophyd,
    start=-100e-6,
    stop=100e-6,
    steps=81,
    shots_per_step=1,
)

snd.propagate_delay()
phase2_final_ip_cy = snd.get_IP_cy()
phase2_final_ip_cx = snd.get_IP_cx()

phase2_summary = phase2_summary_df(
    DEFAULT_PHASE2_STEPS,
    phase2_initial_positions,
    snd,
    phase2_results,
)
phase2_outcome = phase2_outcome_df(
    phase2_initial_ip_cy,
    phase2_final_ip_cy,
    phase2_initial_ip_cx,
    phase2_final_ip_cx,
)

display(phase2_summary)
display(phase2_outcome)

print("After Phase 2 alignment")
print(f"  IP_cy: {phase2_final_ip_cy:.6g}")
print(f"  IP_cx: {phase2_final_ip_cx:.6g}")

assert phase2_outcome.loc[0, "reduced_vertical_error"], (
    "Phase 2 did not reduce the absolute vertical image-plane error in simulation."
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (label, _, _) in zip(axes.flat, DEFAULT_PHASE2_STEPS):
    x = phase2_results[label]["x"] * 1e6
    y = phase2_results[label]["y"]
    objective = phase2_results[label]["objective_y"]

    ax.plot(x, y, ".-", label="IP_cy")
    ax.plot(x, objective, "--", label="|IP_cy|")
    ax.axvline(phase2_results[label]["best_position"] * 1e6, color="k", linestyle=":", label="chosen")
    ax.set_title(label)
    ax.set_xlabel("motor position (urad)")
    ax.set_ylabel("image-plane vertical error")
    ax.grid(True)

axes[0].legend(loc="best")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.bar(["before", "after"], [abs(phase2_initial_ip_cy), abs(phase2_final_ip_cy)], alpha=0.7)
plt.ylabel("absolute IP_cy")
plt.title("Phase 2 vertical error")
plt.grid(True, axis="y")
plt.show()