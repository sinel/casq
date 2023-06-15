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

from typing import Optional, Union

from qiskit import QuantumCircuit
from qiskit.providers import BackendV1, BackendV2
from qiskit_dynamics import DynamicsBackend
from qiskit_dynamics.backend.dynamics_job import DynamicsJob

from casq.casq_object import CasqObject
from casq.common.decorators import trace
from casq.gates.pulse_gate import PulseGate


class CasqSimulator(CasqObject):
    """CasqSimulator class.

    Wraps and extends :py:class:`qiskit_dynamics.DynamicsBackend`.

    Args:
        backend: Qiskit backend  to use for extracting simulator model.
        (:py:class:`qiskit.providers.BackendV1`
        or :py:class:`qiskit.providers.BackendV2`)
    """

    @trace()
    def __init__(
        self,
        backend: Union[BackendV1, BackendV2],
        qubits: list[int],
        name: Optional[str] = None,
    ) -> None:
        """Initialize CasqSimulator."""
        super().__init__(name)
        self.source = backend
        self.backend = DynamicsBackend.from_backend(
            backend=backend,
            subsystem_list=qubits,
        )

    def run(
        self,
        circuit: QuantumCircuit,
        pulse_map: Optional[list[tuple[str, PulseGate]]] = None,
    ) -> DynamicsJob:
        """PulseGate.to_circuit method.

        Builds simple circuit for solitary usage or testing of pulse gate.

        Args:
            circuit: Quantum circuit to execute.
            pulse_map: Gate to custom pulse map.

        Returns:
            :py:class:`qiskit_dynamics.backend.dynamics_job.DynamicsJob`
        """
        if pulse_map:
            for gate, pulse in pulse_map:
                for instruction in circuit.data:
                    if instruction.operation.name == gate:
                        circuit.add_calibration(
                            gate, instruction.qubits, pulse.schedule
                        )
        return self.backend.run([circuit]).result()
