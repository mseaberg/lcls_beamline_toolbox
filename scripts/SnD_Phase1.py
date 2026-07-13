from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from IPython.display import display
import sys
from ophyd.signal import EpicsSignalRO, EpicsSignal
from bluesky import plan_stubs as bps
from pcdsdevices.signal import AvgSignal
import time

from types import SimpleNamespace
from hxrsnd.sndsystem import SplitAndDelay
sys.path.append("/cds/home/s/seaberg/dev/lcls_beamline_toolbox")

#from lcls_beamline_toolbox.models.split_and_delay_motion import SND
#from lcls_beamline_toolbox.models.split_and_delay_ophyd import SndOphyd
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
    "t1_th1": 40e-6,
    "t1_th2": -45e-6,
    "t4_th2": 35e-6,
    "t4_th1": -55e-6,
}

PHASE2_OFFSETS = {
    "t1_chi2": 80e-6*0,
    "t4_chi2": -60e-6*0,
    "t4_chi1": 50e-6*0,
}

ALL_OFFSETS = {**PHASE1_OFFSETS, **PHASE2_OFFSETS}

d12 = EpicsSignalRO("XCS:SND:DIO:AMPL_12", name="diode 12")
d12_lume = EpicsSignalRO("XCS:USER:SND:DD_SUM_LUME",name='dd')
d8 = EpicsSignalRO("XCS:SND:DIO:AMPL_8", name="diode 8")
d8_lume = EpicsSignalRO("XCS:USER:SND:DCC_SUM_LUME", name='dcc')
d9 = EpicsSignalRO("XCS:SND:DIO:AMPL_9", name="diode 9")
d9_lume = EpicsSignalRO("XCS:USER:SND:DCO_SUM_LUME",name='dco')
d11 = EpicsSignalRO("XCS:SND:DIO:AMPL_11", name="diode 11")
d11_lume = EpicsSignalRO("XCS:USER:SND:T1_DH_SUM_LUME", name='t1_dh')
d10 = EpicsSignalRO("XCS:SND:DIO:AMPL_10", name="diode 10")
d10_lume = EpicsSignalRO("XCS:USER:SND:DI_SUM_LUME", name='di')
d15 = EpicsSignalRO("XCS:SND:DIO:AMPL_15", name="diode 15")
d15_lume = EpicsSignalRO("XCS:USER:SND:T4_DH_SUM_LUME", name='t4_dh')
d14 = EpicsSignalRO("XCS:SND:DIO:AMPL_14", name="diode 14")
d14_lume = EpicsSignalRO("XCS:USER:SND:DO_SUM_LUME", name='do')
d13 = EpicsSignalRO("XCS:SND:DIO:AMPL_13", name="diode 13")
d13_lume = EpicsSignalRO("XCS:USER:SND:DCI_SUM_LUME",name='dci')
ipm4 = EpicsSignalRO("XCS:SB1:BMMON:SUM",name='ipm4')

snd_system = SplitAndDelay('XCS:SND', name='snd')
time.sleep(1)
t1th1 = snd_system.t1.th1
t1th2 = snd_system.t1.th2
t2th = snd_system.t2.th
t3th = snd_system.t3.th
t4th1 = snd_system.t4.th1
t4th2 = snd_system.t4.th2
t1chi1 = snd_system.t1.chi1
t1chi2 = snd_system.t1.chi2
t4chi1 = snd_system.t4.chi1
t4chi2 = snd_system.t4.chi2
cc_shutter = EpicsSignal("XCS:USR:ao1:4", name='cc_shutter')
delay_shutter = EpicsSignal("XCS:USR:ao1:5", name='delay_shutter')
#t1th1 = EpicsSignal("XCS:SND:T1:TH1", name="x1 motor")
#t1th2 = EpicsSignal("XCS:SND:T1:TH2", name="x2 motor")
#t2th = EpicsSignal("XCS:SND:T2:TH", name="cc1 motor")
#t3th = EpicsSignal("XCS:SND:T3:TH", name="cc2 motor")
#t4th1 = EpicsSignal("XCS:SND:T4:TH1", name="x4 motor")
#t4th2 = EpicsSignal("XCS:SND:T4:TH2", name="x3 motor")
#t1chi1 = EpicsSignal"XCS:SND:T1:CHI1:POSITION", name="x1 chi")

MODEL_BACKEND = SimpleNamespace(
    t1_th1=t1th1,
    t1_dh_sum=AvgSignal(d11_lume,1, 2, name='t1_dh'),
    t1_th2=t1th2,
    dd_sum=AvgSignal(d12_lume,1, 2, name='dd'),
    t4_th2=t4th2,
    t4_dh_sum=AvgSignal(d15_lume,1,2,name='t4_dh'),
    t4_th1=t4th1,
    do_sum=AvgSignal(d14_lume,1,2,name='do'),
    t2_th=t2th,
    dcc_sum=AvgSignal(d8_lume,1,2,name='dcc'),
    t3_th=t3th,
    dci_sum=AvgSignal(d13_lume,1,2,name='dci'),
    dco_sum=AvgSignal(d9_lume,1,2,name='dco'),
    di_sum=AvgSignal(d10_lume,1,2,name='di'),
    t1_chi2=t1chi2,
    t1_chi1=t1chi1,
    t4_chi2=t4chi2,
    t4_chi1=t4chi1,
    cc_shutter=cc_shutter,
    delay_shutter=delay_shutter
)

shots = 60
duration = .5

HARDWARE_BACKEND = SimpleNamespace(
    t1_th1=t1th1,
    t1_dh_sum=AvgSignal(d11,shots,duration,name='t1_dh'),
    t1_th2=t1th2,
    dd_sum=AvgSignal(d12,shots,duration,name='dd'),
    t4_th2=t4th2,
    t4_dh_sum=AvgSignal(d15,shots,duration,name='t4_dh'),
    t4_th1=t4th1,
    do_sum=AvgSignal(d14,shots,duration,name='do'),
    t2_th=t2th,
    dcc_sum=AvgSignal(d8,shots,duration,name='dcc'),
    t3_th=t3th,
    dci_sum=AvgSignal(d13,shots,duration,name='dci'),
    dco_sum=AvgSignal(d9,shots,duration,name='dco'),
    di_sum=AvgSignal(d10,shots,duration,name='di'),
    t1_chi2=t1chi2,
    t1_chi1=t1chi1,
    t4_chi2=t4chi2,
    t4_chi1=t4chi1,
    cc_shutter=cc_shutter,
    delay_shutter=delay_shutter,
    ipm4_sum=ipm4
)


def apply_offsets(RE, snd, offsets):
    for attr, delta in offsets.items():
        time.sleep(0.1)
        motor = getattr(snd, attr)
        motor.mvr(delta*180/np.pi)
    #    new_pos = motor.get() + delta*180/np.pi
    #    RE(bps.mv(motor, new_pos))
    #snd.propagate_delay()


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

#snd_truth = SND(9500)
#snd = SND(9500)
snd = HARDWARE_BACKEND 
#apply_offsets(snd, ALL_OFFSETS)
#
#snd_ophyd = SndOphyd(snd)
RE = make_run_engine()
apply_offsets(RE, snd, ALL_OFFSETS)
#
phase1_initial_positions = capture_positions(
    snd,
    [motor_attr for _, motor_attr, _, _ in DEFAULT_PHASE1_STEPS],
)

print("Before Phase 1 alignment")
for _, detector_attr, getter_name in [
    ("X1", "t1_dh_sum", "get_t1_dh_sum"),
    ("X2", "dd_sum", "get_dd_sum"),
    ("X3", "t4_dh_sum", "get_t4_dh_sum"),
    ("X4", "do_sum", "get_do_sum"),
]:
    print(f"  {detector_attr}: {getattr(snd, detector_attr).get():.3f}")

phase1_results = align_phase1(
    RE,
    snd,
    start=-100e-6*180/np.pi,
    stop=100e-6*180/np.pi,
    steps=81,
    shots_per_step=1,
)
#
#snd.propagate_delay()
#phase1_summary = phase1_summary_df(
#    DEFAULT_PHASE1_STEPS,
#    phase1_initial_positions,
#    snd,
#    snd_truth,
#    phase1_results,
#)
#
#display(phase1_summary)
#
#assert phase1_summary["improved"].all(), "Phase 1 did not improve every rocking motor in simulation."
#
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (label, _, _, _) in zip(axes.flat, DEFAULT_PHASE1_STEPS):
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
#
#plt.figure(figsize=(8, 4))
#plt.bar(phase1_summary["crystal"], np.abs(phase1_summary["initial_offset_urad"]), alpha=0.6, label="before")
#plt.bar(phase1_summary["crystal"], np.abs(phase1_summary["final_offset_urad"]), alpha=0.6, label="after")
#plt.ylabel("absolute motor error vs nominal (urad)")
#plt.title("Phase 1 motor errors")
#plt.yscale("log")
#plt.grid(True, axis="y")
#plt.legend()
#plt.show()
