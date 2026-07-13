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

shots = 60
duration = 0.5

PHASE1_OFFSETS = {
    "t1_th1": 60e-6*0,
    "t1_th2": -45e-6*0,
    "t4_th2": 35e-6*0,
    "t4_th1": -55e-6*0,
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

IP_cx_lume = EpicsSignalRO("XCS:USER:SND:IP_X_CENTROID_LUME",name='cx')
IP_cy_lume = EpicsSignalRO("XCS:USER:SND:IP_Y_CENTROID_LUME",name='cy')
dd_cy_lume = EpicsSignalRO("XCS:USER:SND:DD_Y_CENTROID_LUME",name='dd_cy')
do_cy_lume = EpicsSignalRO("XCS:USER:SND:DO_Y_CENTROID_LUME",name='do_cy')
cx = EpicsSignalRO("XCS:USER:SND:IP_X_CENTROID_SHMEM",name='cx')
cy = EpicsSignalRO("XCS:USER:SND:IP_Y_CENTROID_SHMEM",name='cy')

snd_system = SplitAndDelay('XCS:SND', name='snd')
time.sleep(1)
t1th1 = snd_system.t1.th1
t1th2 = snd_system.t1.th2
t2th = snd_system.t2.th
t3th = snd_system.t3.th
t4th1 = snd_system.t4.th1
t4th2 = snd_system.t4.th2
t1chi1 = EpicsSignal("XCS:SND:T1:CHI1:CMD:TARGET",name='t1_chi1')
t1chi2 = EpicsSignal("XCS:SND:T1:CHI2:CMD:TARGET",name='t1_chi2')
t4chi1 = EpicsSignal("XCS:SND:T4:CHI1:CMD:TARGET",name='t4_chi1')
t4chi2 = EpicsSignal("XCS:SND:T4:CHI2:CMD:TARGET",name='t4_chi2')
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
    IP_cx = AvgSignal(IP_cx_lume,1,2,name='IP_cx'),
    IP_cy = AvgSignal(IP_cy_lume,1,2,name='IP_cy'),
    do_cy = AvgSignal(do_cy_lume,1,2,name='do_cy'),
    dd_cy = AvgSignal(dd_cy_lume,1,2,name='dd_cy'),
    t1_chi2=t1chi2,
    t1_chi1=t1chi1,
    t4_chi2=t4chi2,
    t4_chi1=t4chi1,
    cc_shutter=cc_shutter,
    delay_shutter=delay_shutter
)



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
    IP_cx=AvgSignal(cx,int(shots/2),duration,name='cx'),
    IP_cy=AvgSignal(cy,int(shots/2),duration,name='cy')
)


def apply_offsets(snd, offsets):
    for attr, delta in offsets.items():
        time.sleep(0.1)
        motor = getattr(snd, attr)
        pos = motor.get() + delta*180/np.pi
        #getattr(snd, attr).mvr(delta*180/np.pi)
        motor.set(pos)
    #snd.propagate_delay()


def capture_positions(snd, motor_attrs):
    return {motor_attr: getattr(snd, motor_attr).get() for motor_attr in motor_attrs}


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
        final_position = getattr(final_snd, motor_attr).get()
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

snd = MODEL_BACKEND
RE = make_run_engine()
apply_offsets(snd, PHASE2_OFFSETS)
phase2_initial_positions = capture_positions(
    snd,
    [motor_attr for _, motor_attr, _ in DEFAULT_PHASE2_STEPS],
)



phase2_initial_ip_cx = snd.IP_cx.get()
phase2_initial_ip_cy = snd.IP_cy.get()

print("Before Phase 2 alignment")
print(f"  IP_cy: {phase2_initial_ip_cy:.6g}")
print(f"  IP_cx: {phase2_initial_ip_cx:.6g}")
time.sleep(1)

phase2_results = align_phase2(
    RE,
    snd,
    start=-20e-6*180/np.pi,
    stop=20e-6*180/np.pi,
    steps=21,
    shots_per_step=1,
)

#snd.propagate_delay()
phase2_final_ip_cy = snd.IP_cy.get()
phase2_final_ip_cx = snd.IP_cx.get()

#phase2_summary = phase2_summary_df(
#    DEFAULT_PHASE2_STEPS,
#    phase2_initial_positions,
#    snd,
#    phase2_results,
#)
#phase2_outcome = phase2_outcome_df(
#    phase2_initial_ip_cy,
#    phase2_final_ip_cy,
#    phase2_initial_ip_cx,
#    phase2_final_ip_cx,
#)

#display(phase2_summary)
#display(phase2_outcome)

print("After Phase 2 alignment")
print(f"  IP_cy: {phase2_final_ip_cy:.6g}")
print(f"  IP_cx: {phase2_final_ip_cx:.6g}")

#assert phase2_outcome.loc[0, "reduced_vertical_error"], (
#    "Phase 2 did not reduce the absolute vertical image-plane error in simulation."
#)

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
