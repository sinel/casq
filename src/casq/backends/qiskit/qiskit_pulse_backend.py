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

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import BackendV1, BackendV2
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit_dynamics import RotatingFrame
from qiskit_dynamics.backend import parse_backend_hamiltonian_dict
from qiskit_dynamics.array import Array
from qiskit_dynamics.solvers import Solver

from casq.backends.pulse_backend import PulseBackend
from casq.backends.pulse_solution import PulseSolution
from casq.backends.qiskit.backend_characteristics import BackendCharacteristics
from casq.backends.qiskit.dynamics_backend_patch import DynamicsBackendPatch
from casq.common import timer, trace
from casq.gates.pulse_circuit import PulseCircuit


class QiskitPulseBackend(PulseBackend):
    """QiskitPulseBackend class."""

    class EvaluationMode(Enum):
        DENSE = 0
        SPARSE = 1

    @dataclass
    class QiskitOptions(PulseBackend.Options):
        dt: Optional[float] = None
        channel_carrier_freqs: Optional[dict] = None
        control_channel_map: Optional[dict] = None
        rotating_frame: Optional[Union[Array, RotatingFrame]] = None
        evaluation_mode: Optional[QiskitPulseBackend.EvaluationMode] = None
        rwa_cutoff_freq: Optional[float] = None

        def to_dynamics_backend_options(self) -> dict[str, Any]:
            options_dict = self.to_dict()
            options_dict.update(seed_simulator=options_dict["seed"])
            del options_dict["seed"]
            del options_dict["dt"]
            del options_dict["channel_carrier_freqs"]
            del options_dict["rotating_frame"]
            del options_dict["evaluation_mode"]
            del options_dict["rwa_cutoff_freq"]
            return options_dict

    @dataclass
    class QiskitRunOptions(PulseBackend.RunOptions):
        pass

    @classmethod
    @trace()
    def from_backend(
        cls,
        backend: Union[BackendV1, BackendV2],
        qubits: Optional[list[int]] = None,
        options: Optional[QiskitOptions] = None
    ) -> QiskitPulseBackend:
        """Construct a QiskitPulseBackend instance from an existing backend instance.

        Args:
            backend: The ``Backend`` instance to build the :class:`.DynamicsBackend` from.
            qubits: List of qubits to include from the backend.
            options: Qiskit pulse backend options.

        Returns:
            QiskitPulseBackend
        """
        backend_characteristics = BackendCharacteristics(backend)
        qubits = [0] if qubits is None else qubits
        (
            static_hamiltonian,
            operators,
            channels,
            qubit_dims
        ) = parse_backend_hamiltonian_dict(backend_characteristics.hamiltonian, qubits)
        options = QiskitPulseBackend.QiskitOptions() if options is None else options
        if options.dt is None:
            options.dt = backend_characteristics.dt
        if options.control_channel_map is None:
            options.control_channel_map = backend_characteristics.get_control_channel_map(channels)
        if options.channel_carrier_freqs is None:
            options.channel_carrier_freqs = backend_characteristics.get_channel_frequencies(channels)
        if options.rotating_frame is None:
            if options.evaluation_mode is QiskitPulseBackend.EvaluationMode.SPARSE:
                options.rotating_frame = np.diag(static_hamiltonian)
            else:
                options.rotating_frame = static_hamiltonian
        options = QiskitPulseBackend.QiskitOptions() if options is None else options
        return cls(
            hamiltonian_dict=backend_characteristics.hamiltonian,
            qubits=qubits,
            options=options
        )

    @trace()
    def __init__(
        self,
        hamiltonian_dict: dict,
        qubits: Optional[list[int]] = None,
        options: Optional[QiskitOptions] = None
    ):
        """Instantiate :class:`~casq.QiskitPulseBackend`.

        Args:
            hamiltonian_dict: Pulse backend Hamiltonian dictionary.
            qubits: List of qubits to include from the backend.
            options: Qiskit pulse backend options.
        """
        options = QiskitPulseBackend.QiskitOptions() if options is None else options
        super().__init__(PulseBackend.NativeBackendType.QISKIT, hamiltonian_dict, qubits, options)
        self.options: QiskitPulseBackend.QiskitOptions = self.options

    @trace()
    @timer()
    def run(
        self,
        run_input: list[Union[PulseCircuit, QuantumCircuit, Schedule, ScheduleBlock]],
        run_options: Optional[QiskitPulseBackend.QiskitRunOptions] = None
    ) -> dict[str, PulseSolution]:
        """QiskitPulseBackend.run."""
        result = self._native_backend.run(run_input=run_input, **run_options.to_dict())
        return PulseSolution.from_qiskit(result.result())

    @trace()
    @timer()
    def _get_native_backend(self) -> DynamicsBackendPatch:
        """QiskitPulseBackend._get_native_backend."""
        if self.options.evaluation_mode is None:
            evaluation_mode = "dense"
        else:
            evaluation_mode = self.options.evaluation_mode.name.lower()
        solver = Solver(
            static_hamiltonian=self._hamiltonian.static,
            hamiltonian_operators=self._hamiltonian.operators,
            hamiltonian_channels=self._hamiltonian.channels,
            channel_carrier_freqs=self.options.channel_carrier_freqs,
            dt=self.options.dt,
            rotating_frame=self.options.rotating_frame,
            evaluation_mode=evaluation_mode,
            rwa_cutoff_freq=self.options.rwa_cutoff_freq,
        )
        options_dict = self.options.to_dynamics_backend_options()
        return DynamicsBackendPatch(solver, **options_dict)

    @trace()
    def _parse_hamiltonian_dict(self) -> tuple[PulseBackend.Hamiltonian, list[int]]:
        """QiskitPulseBackend._parse_hamiltonian_dict."""
        (
            static_hamiltonian,
            operators,
            channels,
            qubit_dims
        ) = parse_backend_hamiltonian_dict(self._hamiltonian_dict, self.qubits)
        hamiltonian = PulseBackend.Hamiltonian(static_hamiltonian, operators, channels)
        qubit_dims = [qubit_dims[idx] for idx in self.qubits]
        return hamiltonian, qubit_dims
