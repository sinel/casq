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

from qiskit.providers import BackendV1

from casq.gates.pulse_circuit import PulseCircuit
from casq.gates.gaussian_pulse_gate import GaussianPulseGate


def test_pulse_instruction(backend: BackendV1) -> None:
    """Unit test for PulseCircuit.pulse."""
    gate = GaussianPulseGate(1, 1, 1)
    circuit = PulseCircuit(1)
    instruction = circuit.pulse(gate, backend, 0).instructions[0]
    assert instruction.name == gate.name
    assert instruction.num_qubits == 1
    assert instruction.num_clbits == 0


def test_from_pulse(backend: BackendV1) -> None:
    """Unit test for PulseCircuit.from_pulse."""
    gate = GaussianPulseGate(1, 1, 1)
    circuit = PulseCircuit.from_pulse(gate, backend, 0)
    assert circuit.data[0].operation.name == gate.name
    assert len(circuit.data[0].qubits) == 1
    assert len(circuit.data[0].clbits) == 0
    assert circuit.data[1].operation.name == "measure"
    assert len(circuit.data[1].qubits) == 1
    assert len(circuit.data[1].clbits) == 1
