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
"""Pulse simulator."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from qiskit import QuantumCircuit
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.quantum_info import DensityMatrix, Operator, Statevector
from qiskit_dynamics import RotatingFrame
from qiskit_dynamics.array import Array
from qiskit_dynamics.solvers import Solver

from casq.backends.pulse_backend import PulseBackend
from casq.backends.pulse_solution import PulseSolution
from casq.backends.qiskit.dynamics_backend_patch import DynamicsBackendPatch
from casq.common import timer, trace
from casq.gates.pulse_circuit import PulseCircuit


class QiskitPulseBackend(PulseBackend):
    """QiskitPulseBackend class."""

    @classmethod
    @trace()
    def from_backend(cls, backend: DynamicsBackendPatch) -> QiskitPulseBackend:
        """Construct a QiskitPulseBackend instance from an existing DynamicsBackendPatch instance."""
        solver = backend.options.solver
        qubit_dict = dict(
            zip(backend.options.subsystem_labels, backend.options.subsystem_dims)
        )
        # noinspection PyProtectedMember
        return cls(
            backend_type=PulseBackend.BackendType.QISKIT,
            static_hamiltonian=solver.model.static_operator,
            hamiltonian_operators=solver.model.operators,
            hamiltonian_channels=solver._hamiltonian_channels,
            qubit_dict=qubit_dict,
            channel_carrier_freqs=solver._channel_carrier_freqs,
            dt=solver._dt,
            rotating_frame=solver.model.rotating_frame,
            evaluation_mode=solver.model.evaluation_mode,
            rwa_cutoff_freq=backend.options.rwa_cutoff_freq,
            steps=backend.options.steps,
        )

    @trace()
    def __init__(
        self,
        static_hamiltonian: Optional[npt.NDArray] = None,
        hamiltonian_operators: Optional[list[Operator]] = None,
        hamiltonian_channels: Optional[list[str]] = None,
        qubit_dict: Optional[dict[int, int]] = None,
        channel_carrier_freqs: Optional[dict] = None,
        dt: Optional[float] = None,
        rotating_frame: Optional[Union[npt.NDArray, RotatingFrame, str]] = "auto",
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        steps: Optional[int] = None,
    ):
        """Instantiate :class:`~casq.PulseBackend`.

        Args:
            static_hamiltonian: Constant Hamiltonian term.
                If a ``rotating_frame`` is specified,
                the ``frame_operator`` will be subtracted
                from the static_hamiltonian.
            hamiltonian_operators: Hamiltonian operators.
            hamiltonian_channels: List of channel names in pulse schedules
                corresponding to Hamiltonian operators.
            qubit_dict: Dictionary of qubits (key=index, value=dimension)
                in the backend to include in the model.
            channel_carrier_freqs: Dictionary mapping channel names to floats
                which represent the carrier frequency of the pulse channel
                with the corresponding name.
            dt: Sample rate for simulating pulse schedules.
            rotating_frame: Rotating frame to transform the model into.
                Rotating frames which are diagonal can be supplied as
                a 1d array of the diagonal elements
                to explicitly indicate that they are diagonal.
            evaluation_mode: Method for model evaluation.
            rwa_cutoff_freq: Rotating wave approximation cutoff frequency.
                If ``None``, no approximation is made.
            steps: Number of steps at which to solve the system.
                Used to automatically calculate an evenly-spaced t_eval range.
        """
        super().__init__(
            PulseBackend.BackendType.QISKIT,
            static_hamiltonian,
            hamiltonian_operators,
            hamiltonian_channels,
            qubit_dict,
            channel_carrier_freqs,
            dt,
            rotating_frame,
            evaluation_mode,
            rwa_cutoff_freq,
            steps,
        )
        if rotating_frame == "auto":
            if "dense" in evaluation_mode:
                rotating_frame = static_hamiltonian
            else:
                rotating_frame = np.diag(static_hamiltonian)
        solver = Solver(
            static_hamiltonian=Array(static_hamiltonian),
            hamiltonian_operators=Array(hamiltonian_operators),
            hamiltonian_channels=hamiltonian_channels,
            channel_carrier_freqs=channel_carrier_freqs,
            dt=dt,
            rotating_frame=rotating_frame,
            evaluation_mode=evaluation_mode,
            rwa_cutoff_freq=rwa_cutoff_freq,
        )
        self.backend = DynamicsBackendPatch(
            solver,
            rwa_cutoff_freq=rwa_cutoff_freq,
            steps=steps,
            subsystem_labels=self.qubits,
            subsystem_dims=self.qubit_dims,
        )

    @trace()
    @timer()
    def run(
        self,
        run_input: list[Union[PulseCircuit, QuantumCircuit, Schedule, ScheduleBlock]],
        qubits: Optional[list[int]] = None,
        initial_state: Optional[Union[DensityMatrix, Statevector]] = None,
        method: Optional[PulseBackend.ODESolverMethod] = None,
        shots: int = 1024,
        seed: Optional[int] = None,
    ) -> dict[str, PulseSolution]:
        """PulseBackend.run."""
        solver_options = self.backend.options.solver_options
        if solver_options:
            if method:
                solver_options.update(method=method.value)
        else:
            if method:
                solver_options = {"method": method.value}
        result = self.backend.run(
            run_input=run_input,
            subsystem_labels=qubits,
            initial_state=initial_state,
            shots=shots,
            seed=seed,
            solver_options=solver_options,
        )
        return PulseSolution.from_qiskit(result.result())
