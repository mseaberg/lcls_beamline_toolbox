from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from IPython.display import display
import sys
from ophyd.signal import EpicsSignalRO, EpicsSignal
from bluesky import plan_stubs as bps
from pcdsdevices.signal import AvgSignal
from pcdsdevices import analog_signals
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
from lcls_beamline_toolbox.utility.phase1_offset_calibration import learn_phase1_offset

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

aio = analog_signals.Acromag(name='xcs_aio',prefix='XCS:USR')

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
cc_shutter = aio.ao1_4
delay_shutter = aio.ao1_5
#cc_shutter = EpicsSignal("XCS:USR:ao1:4", name='cc_shutter')
#delay_shutter = EpicsSignal("XCS:USR:ao1:5", name='delay_shutter')
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
    ipm4_sum=AvgSignal(d10_lume,1,2,name='di'),
    t1_chi2=t1chi2,
    t1_chi1=t1chi1,
    t4_chi2=t4chi2,
    t4_chi1=t4chi1,
    cc_shutter=cc_shutter,
    delay_shutter=delay_shutter
)

shots = 120
duration = 1

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
def capture_positions(snd, motor_attrs):
    return {motor_attr: getattr(snd, motor_attr).wm() for motor_attr in motor_attrs}


def make_run_engine(with_bec=True):
    RE = RunEngine({})
    if with_bec:
        RE.subscribe(BestEffortCallback())
    return RE


snd = HARDWARE_BACKEND 
#apply_offsets(snd, ALL_OFFSETS)
#
#snd_ophyd = SndOphyd(snd)
RE = make_run_engine()
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

result = learn_phase1_offset(
        RE=RE,
        sim_backend=MODEL_BACKEND,
        hw_backend=HARDWARE_BACKEND,
        motor_attr='t4_th1',
        detector_attr='do_sum',
        norm_detector_attr='t4_dh_sum',
        offset_pv='XCS:USER:SND:T4_TH1_OFFSET',
        start=-.004,
        stop=.004,
        steps=41,
        shots_per_step=1,
        duration=1,
        move=True,
        write=True
        )


print('learned offset: {}'.format(result['offset']))


phase1_results = result['sim_result']
fig, axes = plt.subplots(1,1, figsize=(6,4))
x = phase1_results["x"] * 1e6
y = phase1_results["y"]
axes.plot(x, y, ".", label="scan")
axes.plot(
        x,
        gaussian(
            phase1_results["x"],
            phase1_results["center"],
            phase1_results["sigma"],
            phase1_results["amplitude"],
            phase1_results["yoffset"],
        ),
        "--",
        label="fit",
    )
axes.axvline(phase1_results["center"] * 1e6, color="k", linestyle=":", label="center")
axes.set_xlabel("motor position (urad)")
axes.set_ylabel("diagnostic signal")
axes.grid(True)
axes.set_title('sim results')

axes.legend(loc="best")
plt.tight_layout()

phase1_results = result['hw_result']
fig, axes = plt.subplots(1,1, figsize=(6,4))
x = phase1_results["x"] * 1e6
y = phase1_results["y"]
axes.plot(x, y, ".", label="scan")
axes.plot(
        x,
        gaussian(
            phase1_results["x"],
            phase1_results["center"],
            phase1_results["sigma"],
            phase1_results["amplitude"],
            phase1_results["yoffset"],
        ),
        "--",
        label="fit",
    )
axes.axvline(phase1_results["center"] * 1e6, color="k", linestyle=":", label="center")
axes.set_xlabel("motor position (urad)")
axes.set_ylabel("diagnostic signal")
axes.grid(True)
axes.set_title('hw results')

axes.legend(loc="best")
plt.tight_layout()



plt.show()

#RE(bps.mv(snd.cc_shutter,5,snd.delay_shutter,5))

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
