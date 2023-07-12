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

from abc import abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Optional, Union

import numpy.typing as npt
from qiskit import QuantumCircuit
from qiskit.providers import BackendV1, BackendV2
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.quantum_info import DensityMatrix, Operator, Statevector
from qiskit_dynamics.models import RotatingFrame

from casq.backends.pulse_solution import PulseSolution
from casq.gates.pulse_circuit import PulseCircuit


class PulseBackend:
    """PulseBackend class."""

    class BackendType(Enum):
        """Backend type."""

        C3 = 0
        QCTRL = 1
        QISKIT = 2
        QUTIP = 3

    class ODESolverMethod(str, Enum):
        """Solver methods."""

        QISKIT_DYNAMICS_RK4 = "RK4"
        QISKIT_DYNAMICS_JAX_RK4 = "jax_RK4"
        QISKIT_DYNAMICS_JAX_ODEINT = "jax_odeint"
        SCIPY_BDF = "BDF"
        SCIPY_DOP853 = "DOP853"
        SCIPY_LSODA = "LSODA"
        SCIPY_RADAU = "Radau"
        SCIPY_RK23 = "RK23"
        SCIPY_RK45 = "RK45"

    def __init__(
        self,
        backend_type: PulseBackend.BackendType,
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
        """Instantiate :class:`~casq.backends.PulseBackend`.

        Args:
            backend_type: Backend type.
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
        self.backend_type = backend_type
        self.static_hamiltonian = static_hamiltonian
        self.hamiltonian_operators = hamiltonian_operators
        self.hamiltonian_channels = hamiltonian_channels
        self.qubit_dict = (
            OrderedDict(sorted(qubit_dict.items()))
            if qubit_dict is not None
            else {0: 2}
        )
        self.qubits = list(self.qubit_dict.keys())
        self.qubit_dims = list(self.qubit_dict.values())
        self.channel_carrier_freqs = channel_carrier_freqs
        self.dt = dt
        self.rotating_frame = rotating_frame
        self.evaluation_mode = evaluation_mode
        self.rwa_cutoff_freq = rwa_cutoff_freq
        self.steps = steps
        self.backend: Union[BackendV1, BackendV2] = ...

    @abstractmethod
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
