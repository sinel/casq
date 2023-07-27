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

from typing import Any, Optional, Union

from qiskit import QuantumCircuit
from qiskit import schedule as build_schedule
from qiskit.circuit import Bit, Register
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumcircuit import InstructionSet
from qiskit.providers import Backend
from qiskit.pulse import Schedule

from casq.backends.qiskit.backend_characteristics import BackendCharacteristics
from casq.common.decorators import trace
from casq.common.helpers import dbid, ufid
from casq.gates.pulse_gate import PulseGate


class PulseCircuit(QuantumCircuit):
    """PulseCircuit class.

    Extends Qiskit QuantumCircuit class
    with helper methods for adding pulse gates and plotting.

    Args:
        name: Optional user-friendly name for pulse gate.
    """

    @staticmethod
    def from_pulse_gate(gate: PulseGate, parameters: dict[str, Any]) -> PulseCircuit:
        """PulseCircuit.from_pulse method.

        Builds simple circuit for solitary usage or testing of pulse gate.

        Args:
            gate: Pulse gate.
            parameters: Dictionary of pulse parameters that defines the pulse envelope.

        Returns:
            PulseCircuit
        """
        circuit: PulseCircuit = PulseCircuit(1, 1)
        circuit.pulse_gate(gate, parameters, 0)
        circuit.measure(0, 0)
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

    def pulse_gate(
        self, gate: PulseGate, parameters: dict[str, Any], qubit: int = 0
    ) -> InstructionSet:  # pragma: no cover
        """PulseGate.gate method.

        Append pulse gate to circuit.

        Args:
            gate: Pulse gate.
            parameters: Dictionary of pulse parameters that defines the pulse envelope.
            qubit: Qubit to attach pulse gate to.

        Returns:
            :py:class:`qiskit.pulse.Instruction`
        """
        instructions = self.append(gate, [qubit])
        self.add_calibration(gate.name, [qubit], gate.schedule(parameters, qubit), [])
        return instructions

    def to_schedule(self, backend: Backend) -> Schedule:
        """PulseGate.to_schedule method.

        Convert pulse circuit to schedule.

        Args:
            backend: Qiskit backend.

        Returns:
            Schedule
        """
        backend_characteristics = BackendCharacteristics(backend)
        # See: https://github.com/Qiskit/qiskit-terra/issues/9488
        # Need to specify extra info to avoid this error
        return build_schedule(
            self,
            backend,
            dt=backend_characteristics.dt,
            inst_map=backend_characteristics.defaults.instruction_schedule_map,
            meas_map=backend_characteristics.config.meas_map,
        )
