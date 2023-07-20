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
"""Pulse circuit."""
from __future__ import annotations

from typing import Optional, Union

from qiskit import QuantumCircuit
from qiskit.circuit import Bit, Register
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumcircuit import InstructionSet
from qiskit.providers import BackendV1

from casq.common import dbid, trace, ufid
from casq.gates import PulseGate


class PulseCircuit(QuantumCircuit):
    """PulseCircuit class.

    Extends Qiskit QuantumCircuit class
    with helper methods for adding pulse gates and plotting.

    Args:
        name: Optional user-friendly name for pulse gate.
    """

    @staticmethod
    def from_pulse(gate: PulseGate, backend: BackendV1, qubit: int = 0) -> PulseCircuit:
        """PulseCircuit.from_pulse method.

        Builds simple circuit for solitary usage or testing of pulse gate.

        Args:
            gate: Pulse gate.
            backend: Qiskit backend.
            qubit: Qubit to attach gate to.

        Returns:
            :py:class:`matplotlib.figure.Figure`
        """
        circuit: PulseCircuit = PulseCircuit(1, 1)
        circuit.pulse(gate, backend, qubit)
        circuit.measure(qubit, qubit)
        return circuit

    @trace()
    def __init__(
        self,
        *regs: Union[Register, int, list[Bit]],
        name: Optional[str] = None,
        global_phase: ParameterValueType = 0,
        metadata: Optional[dict] = None,
    ):
        """Initialize PulseCircuit."""
        self.dbid = dbid()
        self.ufid = ufid(self)
        if name is None:
            name = self.ufid
        super().__init__(*regs, name=name, global_phase=global_phase, metadata=metadata)

    def pulse(
        self, gate: PulseGate, backend: BackendV1, qubit: int = 0
    ) -> InstructionSet:  # pragma: no cover
        """PulseGate.gate method.

        Append pulse gate to circuit.

        Args:
            gate: Pulse gate.
            qubit: Qubit to attach pulse gate to.
            backend: Qiskit backend.

        Returns:
            :py:class:`qiskit.pulse.Instruction`
        """
        instructions = self.append(gate, [qubit])
        self.add_calibration(gate.name, [qubit], gate.schedule(qubit, backend), [])
        return instructions
