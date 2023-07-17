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
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional, Union

import numpy.typing as npt
from qiskit import QuantumCircuit
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.quantum_info import DensityMatrix, Statevector

from casq.backends.pulse_solution import PulseSolution
from casq.gates.pulse_circuit import PulseCircuit


class PulseBackend:
    """PulseBackend class."""

    class NativeBackendType(Enum):
        """Native backend type."""

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

    @dataclass
    class Hamiltonian:
        """Hamiltonian properties."""

        static: npt.NDArray
        operators: npt.NDArray
        channels: list[str]

        def to_dict(self) -> dict[str, Any]:
            """Converts to dict."""
            return asdict(self)

    def __init__(
        self,
        native_backend_type: PulseBackend.NativeBackendType,
        hamiltonian_dict: dict,
        qubits: list[int],
        seed: Optional[int] = None,
    ):
        """Instantiate :class:`~casq.backends.PulseBackend`.

        Args:
            native_backend_type: Native backend type.
            hamiltonian_dict: Pulse backend Hamiltonian dictionary.
            qubits: List of qubits to include from the backend.
            seed: Seed to use in random sampling. Defaults to None.
        """
        self._native_backend_type = native_backend_type
        self._hamiltonian_dict = hamiltonian_dict
        self.qubits = qubits
        self._seed = seed
        self._hamiltonian, self.qubit_dims = self._parse_hamiltonian_dict()
        self._native_backend = self._get_native_backend()

    @abstractmethod
    def run(
        self,
        run_input: list[Union[PulseCircuit, QuantumCircuit, Schedule, ScheduleBlock]],
        method: PulseBackend.ODESolverMethod,
        initial_state: Optional[Union[DensityMatrix, Statevector]] = None,
        shots: int = 1024,
        steps: Optional[int] = None,
    ) -> dict[str, PulseSolution]:
        """PulseBackend.run.

        Args:
            run_input: List of pulse schedules or circuits.
            method: ODE solving method to use.
            initial_state: Initial state for simulation,
                either None,
                indicating that the ground state for the system Hamiltonian should be used,
                or an arbitrary Statevector or DensityMatrix.
            shots: Number of shots per experiment. Defaults to 1024.
            steps: Number of steps at which to solve the system.
                Used to automatically calculate an evenly-spaced t_eval range.
        """

    @abstractmethod
    def _get_native_backend(self) -> Any:
        """PulseBackend._get_native_backend."""

    @abstractmethod
    def _parse_hamiltonian_dict(self) -> tuple[PulseBackend.Hamiltonian, list[int]]:
        """PulseBackend._parse_hamiltonian_dict."""
