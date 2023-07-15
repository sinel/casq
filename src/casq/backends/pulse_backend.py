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

import numpy as np
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
    class Options:
        seed: Optional[int] = None

        def to_dict(self) -> dict[str, Any]:
            return asdict(self)

    @dataclass
    class RunOptions:
        initial_state: Optional[Union[DensityMatrix, Statevector]] = None
        method: Optional[PulseBackend.ODESolverMethod] = None
        shots: int = 1024
        steps: Optional[int] = None

        def to_dict(self) -> dict[str, Any]:
            return asdict(self)

    @dataclass
    class Hamiltonian:
        static: npt.NDArray
        operators: npt.NDArray
        channels: list[str]

        def to_dict(self) -> dict[str, Any]:
            return asdict(self)

    def __init__(
        self,
        native_backend_type: PulseBackend.NativeBackendType,
        hamiltonian_dict: dict,
        qubits: Optional[list[int]] = None,
        options: Optional[Options] = None
    ):
        """Instantiate :class:`~casq.backends.PulseBackend`.

        Args:
            native_backend_type: Native backend type.
            hamiltonian_dict: Pulse backend Hamiltonian dictionary.
            qubits: List of qubits to include from the backend.
            options: Pulse backend options.
        """
        self._native_backend_type = native_backend_type
        self._hamiltonian_dict = hamiltonian_dict
        self.qubits = [0] if qubits is None else qubits
        self.options = PulseBackend.Options() if options is None else options
        self._hamiltonian, self.qubit_dims = self._parse_hamiltonian_dict()
        self._native_backend = self._get_native_backend()

    @abstractmethod
    def run(
        self,
        run_input: list[Union[PulseCircuit, QuantumCircuit, Schedule, ScheduleBlock]],
        run_options: Optional[RunOptions] = None
    ) -> dict[str, PulseSolution]:
        """PulseBackend.run."""

    @abstractmethod
    def _get_native_backend(self) -> Any:
        """PulseBackend._get_native_backend."""

    @abstractmethod
    def _parse_hamiltonian_dict(self) -> tuple[PulseBackend.Hamiltonian, list[int]]:
        """PulseBackend._parse_hamiltonian_dict."""
