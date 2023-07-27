#  ********************************************************************************
#
#    _________ __________ _
#   / ___/ __ `/ ___/ __ `/    Python toolkit
#  / /__/ /_/ (__  ) /_/ /     for control and analysis
#  \___/\__,_/____/\__, /      of superconducting qubits
#                    /_/
#
#  Copyright (c) 2023 Sinan Inel <sinan.inel@aalto.fi>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ********************************************************************************
"""Pulse gate tests."""
from __future__ import annotations

from typing import Any, Optional

import numpy.typing as npt
from qiskit import pulse

from casq.gates.pulse_gate import PulseGate


class DummyPulseGate(PulseGate):
    """DummyPulseGate class."""

    def __init__(
        self,
        duration: int,
        amplitude: float,
        angle: float = 0,
        limit_amplitude: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """Initialize DummyPulseGate."""
        super().__init__(1, duration, amplitude, angle, limit_amplitude, name)

    def pulse(self, parameters: dict[str, float]) -> pulse.library.Pulse:
        """DummyPulseGate.pulse method."""
        return pulse.Gaussian(duration=self.duration, amp=self.amplitude, **parameters)

    def to_parameters_dict(self, parameters: npt.NDArray) -> dict[str, Any]:
        """DummyPulseGate.to_parameters_dict method."""
        return {"sigma": parameters[0]}


def test_schedule() -> None:
    """Unit test for PulseGate.schedule."""
    dummy = DummyPulseGate(1, 1)
    schedule = dummy.schedule({"sigma": 1}, 0)
    assert isinstance(schedule, pulse.Schedule)
    assert schedule.name.endswith("DummyPulseGateSchedule")
