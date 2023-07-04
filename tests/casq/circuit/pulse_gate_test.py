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

from loguru import logger
from matplotlib.figure import Figure
from qiskit import pulse

from casq.circuit.pulse_gate import PulseGate


class DummyPulseGate(PulseGate):
    """DummyPulseGate class."""

    def pulse(self) -> pulse.library.Pulse:
        """GaussianPulseGate.pulse method.

        Returns:
            :py:class:`qiskit.pulse.library.Pulse`
        """
        return pulse.Gaussian(duration=256, amp=1, sigma=128)


def test_schedule() -> None:
    """Unit test for PulseGate.schedule."""
    dummy = DummyPulseGate(1, 1)
    schedule = dummy.schedule(0)
    assert isinstance(schedule, pulse.Schedule)
    assert schedule.name.endswith("DummyPulseGateSchedule")


def test_draw_signal() -> None:
    """Unit test for draw_schedule method."""
    dummy = DummyPulseGate(1, 1)
    figure = dummy.draw_signal(
        qubit=0, dt=1, carrier_frequency=1, hidden=True
    )
    assert isinstance(figure, Figure)
