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
from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit.quantumcircuit import InstructionSet
from qiskit.circuit import Bit, ParameterValueType, Register

from casq.common.decorators import trace
from casq.gates.pulse_gate import PulseGate
from casq.common.helpers import dbid, ufid


class PulseCircuit(QuantumCircuit):
    """PulseCircuit class.

    Extends Qiskit QuantumCircuit class
    with helper methods for adding pulse gates and plotting.

    Args:
        name: Optional user-friendly name for pulse gate.
    """

    @trace()
    def __init__(
        self,
        *regs: Register | int | Sequence[Bit],
        name: str | None = None,
        global_phase: ParameterValueType = 0,
        metadata: dict | None = None,
    ):
        """Initialize PulseCircuit."""
        self.dbid = dbid()
        self.ufid = ufid(self)
        if name is None:
            name = self.ufid
        super().__init__(regs, name, global_phase, metadata)

    def pulse(self, gate: PulseGate, qubit: int) -> InstructionSet:  # pragma: no cover
        """PulseGate.gate method.

        Append pulse gate to circuit.

        Args:
            gate: Pulse gate.
            qubit: Qubit to attach pulse gate to.

        Returns:
            :py:class:`qiskit.pulse.Instruction`
        """
        instructions = self.append(gate, [qubit])
        self.add_calibration(gate.ufid, [qubit], gate.schedule(qubit), [])
        return instructions
