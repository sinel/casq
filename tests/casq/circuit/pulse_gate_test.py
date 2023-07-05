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
import pytest
from qiskit import pulse
from qiskit_dynamics import Signal

from casq.circuit import PulseGate
from casq.common import CasqError


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


def test_measured_schedule(backend) -> None:
    """Unit test for PulseGate.schedule with measurement."""
    dummy = DummyPulseGate(1, 1)
    schedule = dummy.schedule(0, backend=backend, measured=True)
    assert isinstance(schedule.instructions[0][1], pulse.Play)
    assert isinstance(schedule.instructions[1][1], pulse.Acquire)
    assert isinstance(schedule, pulse.Schedule)
    assert schedule.name.endswith("DummyPulseGateSchedule")


def test_measured_schedule_without_backend() -> None:
    """Unit test for PulseGate.schedule with measurement and no backend argument."""
    dummy = DummyPulseGate(1, 1)
    with pytest.raises(CasqError) as e:
        dummy.schedule(0, measured=True)
    assert isinstance(e.value, CasqError)
    assert e.value.message == "Backend is required for building schedules with measurements."


def test_discretized_schedule_with_backend(backend) -> None:
    """Unit test for discretized PulseGate.schedule."""
    dummy = DummyPulseGate(1, 1)
    signals = dummy.schedule(0, backend=backend, discretized=True)
    assert isinstance(signals[0], Signal)


def test_discretized_schedule_with_properties(backend_properties) -> None:
    """Unit test for discretized PulseGate.schedule."""
    dummy = DummyPulseGate(1, 1)
    schedule = dummy.schedule(0)
    dt = backend_properties.dt
    frequencies = backend_properties.get_channel_frequencies(list(schedule.channels))
    signals = dummy.schedule(0, dt=dt, channel_frequencies=frequencies, discretized=True)
    assert isinstance(signals[0], Signal)


def test_discretized_schedule_with_missing_arguments() -> None:
    """Unit test for discretized PulseGate.schedule with missing arguments."""
    dummy = DummyPulseGate(1, 1)
    with pytest.raises(CasqError) as e:
        dummy.schedule(0, discretized=True)
    assert isinstance(e.value, CasqError)
    assert e.value.message.startswith("Cannot discretize pulse schedule")


def test_draw_signal() -> None:
    """Unit test for draw_schedule method."""
    dummy = DummyPulseGate(1, 1)
    figure = dummy.draw_signal(
        qubit=0, dt=1, carrier_frequency=1, hidden=True
    )
    assert isinstance(figure, Figure)
