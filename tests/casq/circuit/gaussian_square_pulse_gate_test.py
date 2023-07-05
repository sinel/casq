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
"""Drag pulse gate tests."""
from __future__ import annotations

from qiskit import pulse

from casq.circuit import GaussianSquarePulseGate


def test_pulse_instruction() -> None:
    """Unit test for pulse instruction."""
    qubit = 0
    duration = 256
    amplitude = 1
    sigma = 128
    width = 128
    dummy = GaussianSquarePulseGate(duration, amplitude, sigma, width)
    schedule = dummy.schedule(qubit)
    # noinspection PyTypeChecker
    instruction: pulse.Play = schedule.instructions[0][1]
    assert instruction.pulse.pulse_type == "GaussianSquare"
    # noinspection PyTypeChecker
    waveform: pulse.GaussianSquare = instruction.pulse
    assert schedule.duration == duration
    assert instruction.channel.index == qubit
    assert waveform.name == dummy.ufid
    assert waveform.amp == amplitude
    assert waveform.sigma == sigma
    assert waveform.width == width


def test_symbolic_pulse_instruction() -> None:
    """Unit test for symbolic pulse instruction."""
    qubit = 0
    duration = 256
    amplitude = 1
    sigma = 128
    width = 128
    dummy = GaussianSquarePulseGate(duration, amplitude, sigma, width, jax=True)
    schedule = dummy.schedule(qubit)
    # noinspection PyTypeChecker
    instruction: pulse.Play = schedule.instructions[0][1]
    assert instruction.pulse.pulse_type == "GaussianSquare"
    # noinspection PyTypeChecker
    waveform: pulse.GaussianSquare = instruction.pulse
    assert schedule.duration == duration
    assert instruction.channel.index == qubit
    assert waveform.name == dummy.ufid
    assert waveform.amp == amplitude
    assert waveform.sigma == sigma
    assert waveform.width == width
