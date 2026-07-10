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

snd_truth = SND(9500)
snd = SND(9500)
apply_offsets(snd, ALL_OFFSETS)

snd_ophyd = SndOphyd(snd)
RE = make_run_engine()

phase1_initial_positions = capture_positions(
    snd,
    [motor_attr for _, motor_attr, _ in DEFAULT_PHASE1_STEPS],
)

print("Before Phase 1 alignment")
for _, detector_attr, getter_name in [
    ("X1", "t1_dh_sum", "get_t1_dh_sum"),
    ("X2", "dd_sum", "get_dd_sum"),
    ("X3", "t4_dh_sum", "get_t4_dh_sum"),
    ("X4", "do_sum", "get_do_sum"),
]:
    print(f"  {detector_attr}: {getattr(snd, getter_name)():.3f}")

phase1_results = align_phase1(
    RE,
    snd_ophyd,
    start=-100e-6,
    stop=100e-6,
    steps=81,
    shots_per_step=1,
)

snd.propagate_delay()
phase1_summary = phase1_summary_df(
    DEFAULT_PHASE1_STEPS,
    phase1_initial_positions,
    snd,
    snd_truth,
    phase1_results,
)

display(phase1_summary)

assert phase1_summary["improved"].all(), "Phase 1 did not improve every rocking motor in simulation."

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (label, _, _) in zip(axes.flat, DEFAULT_PHASE1_STEPS):
    x = phase1_results[label]["x"] * 1e6
    y = phase1_results[label]["y"]
    ax.plot(x, y, ".", label="scan")
    ax.plot(
        x,
        gaussian(
            phase1_results[label]["x"],
            phase1_results[label]["center"],
            phase1_results[label]["sigma"],
            phase1_results[label]["amplitude"],
            phase1_results[label]["yoffset"],
        ),
        "--",
        label="fit",
    )
    ax.axvline(phase1_results[label]["center"] * 1e6, color="k", linestyle=":", label="center")
    ax.set_title(label)
    ax.set_xlabel("motor position (urad)")
    ax.set_ylabel("diagnostic signal")
    ax.grid(True)

axes[0, 0].legend(loc="best")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.bar(phase1_summary["crystal"], np.abs(phase1_summary["initial_offset_urad"]), alpha=0.6, label="before")
plt.bar(phase1_summary["crystal"], np.abs(phase1_summary["final_offset_urad"]), alpha=0.6, label="after")
plt.ylabel("absolute motor error vs nominal (urad)")
plt.title("Phase 1 motor errors")
plt.yscale("log")
plt.grid(True, axis="y")
plt.legend()
plt.show()