from __future__ import annotations

import time
from typing import Callable

from ophyd.device import Device
from ophyd.positioner import PositionerBase
from ophyd.status import DeviceStatus

DEFAULT_DIAGNOSTICS = (
    "t1_dh_sum",
    "dd_sum",
    "t4_dh_sum",
    "do_sum",
    "dci_sum",
    "dcc_sum",
    "dco_sum",
    "IP_sum",
    "IP_cx",
    "IP_cy",
)


class SndMotor(PositionerBase):
    """Ophyd wrapper around a split-and-delay motion axis."""

    def __init__(self, axis, *, name: str, **kwargs):
        self._axis = axis
        super().__init__(name=name, **kwargs)

    @property
    def position(self):
        return self._axis.wm()

    @property
    def hints(self):
        return {"fields": [self.name]}

    def set(self, position):
        status = DeviceStatus(self)
        if self._axis.mv(position):
            status.set_finished()
        else:
            status.set_exception(ValueError(f"Move failed for {self.name}"))
        return status

    def read(self):
        return {
            self.name: {
                "value": self.position,
                "timestamp": time.time(),
            }
        }

    def describe(self):
        return {
            self.name: {
                "source": self.name,
                "dtype": "number",
                "shape": [],
            }
        }


class SndDiagnostic(Device):
    """
    Ophyd diagnostic wrapper.

    `trigger_fn` is optional. For the split-and-delay simulation it will
    usually be `snd.propagate_delay`.
    """

    def __init__(self, getter: Callable[[], float], name: str, trigger_fn=None):
        self._getter = getter
        self._trigger_fn = trigger_fn
        super().__init__(name=name)

    def trigger(self):
        if self._trigger_fn is not None:
            self._trigger_fn()
        status = DeviceStatus(self)
        status.set_finished()
        return status

    @property
    def hints(self):
        return {"fields": [self.name]}

    def read(self):
        return {
            self.name: {
                "value": self._getter(),
                "timestamp": time.time(),
            }
        }

    def describe(self):
        return {
            self.name: {
                "source": self.name,
                "dtype": "number",
                "shape": [],
            }
        }


class SndOphyd:
    """Bluesky/Ophyd adapter built from an existing split-and-delay `snd`."""

    def __init__(self, snd, diagnostics=DEFAULT_DIAGNOSTICS):
        self.snd = snd

        for motor in snd.motor_list:
            setattr(self, motor.name, SndMotor(motor, name=motor.name))

        for diagnostic_name in diagnostics:
            getter = getattr(snd, f"get_{diagnostic_name}")
            setattr(
                self,
                diagnostic_name,
                SndDiagnostic(
                    getter,
                    name=diagnostic_name,
                    trigger_fn=snd.propagate_delay,
                ),
            )
