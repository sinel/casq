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
from typing import Any, Callable, Optional, Self, Union

from loguru import logger
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import BackendV1, BackendV2
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit_dynamics import RotatingFrame
from qiskit_dynamics.array import Array
from qiskit_dynamics.backend import parse_backend_hamiltonian_dict
from qiskit_dynamics.solvers import Solver

from casq.backends.pulse_backend import PulseBackend
from casq.backends.pulse_solution import PulseSolution
from casq.backends.qiskit.backend_characteristics import BackendCharacteristics
from casq.backends.qiskit.dynamics_backend_patch import DynamicsBackendPatch
from casq.backends.qiskit.helpers import get_experiment_result
from casq.common import timer, trace
from casq.gates.pulse_circuit import PulseCircuit


class QiskitPulseBackend(PulseBackend):
    """QiskitPulseBackend class."""

    class EvaluationMode(Enum):
        """Qiskit pulse solver evaluation mode options."""

        DENSE = 0
        SPARSE = 1

    @dataclass
    class QiskitOptions(PulseBackend.Options):
        """Qiskit pulse backend options."""

        dt: Optional[float] = None
        channel_carrier_freqs: Optional[dict] = None
        control_channel_map: Optional[dict] = None
        rotating_frame: Optional[Union[Array, RotatingFrame]] = None
        evaluation_mode: Optional[QiskitPulseBackend.EvaluationMode] = None
        rwa_cutoff_freq: Optional[float] = None
        experiment_result_function: Callable = get_experiment_result

        def to_native_options(self) -> dict[str, Any]:
            """Converts to native options."""
            options_dict = self.to_dict()
            if "seed" in options_dict:
                options_dict.update(seed_simulator=options_dict["seed"])
            options_dict.pop("seed", None)
            options_dict.pop("dt", None)
            options_dict.pop("channel_carrier_freqs", None)
            options_dict.pop("rotating_frame", None)
            options_dict.pop("evaluation_mode", None)
            options_dict.pop("rwa_cutoff_freq", None)
            return options_dict

    @dataclass
    class QiskitRunOptions(PulseBackend.RunOptions):
        """Qiskit pulse backend run options."""

        def to_native_options(self) -> dict[str, Any]:
            """Converts to native options."""
            options_dict = self.to_dict()
            return options_dict

    @classmethod
    @trace()
    def from_backend(
        cls,
        backend: Union[BackendV1, BackendV2],
        qubits: Optional[list[int]] = None,
        options: Optional[QiskitOptions] = None,
    ) -> Self:
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
            qubit_dims,
        ) = parse_backend_hamiltonian_dict(backend_characteristics.hamiltonian, qubits)
        options = QiskitPulseBackend.QiskitOptions() if options is None else options
        if isinstance(options, PulseBackend.Options):
            options = QiskitPulseBackend.QiskitOptions(**options.to_dict())
        if options.dt is None:
            options.dt = backend_characteristics.dt
        if options.control_channel_map is None:
            options.control_channel_map = (
                backend_characteristics.get_control_channel_map(channels)
            )
        if options.channel_carrier_freqs is None:
            options.channel_carrier_freqs = (
                backend_characteristics.get_channel_frequencies(channels)
            )
        if options.rotating_frame is None:
            if options.evaluation_mode is QiskitPulseBackend.EvaluationMode.SPARSE:
                options.rotating_frame = np.diag(static_hamiltonian)
            else:
                options.rotating_frame = static_hamiltonian
        options = QiskitPulseBackend.QiskitOptions() if options is None else options
        instance: Self = cls(
            hamiltonian_dict=backend_characteristics.hamiltonian,
            qubits=qubits,
            options=options,
        )
        return instance

    @trace()
    def __init__(
        self, hamiltonian_dict: dict, qubits: list[int], options: QiskitOptions
    ):
        """Instantiate :class:`~casq.QiskitPulseBackend`.

        Args:
            hamiltonian_dict: Pulse backend Hamiltonian dictionary.
            qubits: List of qubits to include from the backend.
            options: Qiskit pulse backend options.
        """
        options = QiskitPulseBackend.QiskitOptions() if options is None else options
        super().__init__(
            PulseBackend.NativeBackendType.QISKIT, hamiltonian_dict, qubits, options
        )
        self.options: QiskitPulseBackend.QiskitOptions = (
            QiskitPulseBackend.QiskitOptions(**self.options.to_dict())
        )

    @trace()
    @timer()
    def run(
        self,
        run_input: list[Union[PulseCircuit, QuantumCircuit, Schedule, ScheduleBlock]],
        run_options: Optional[QiskitPulseBackend.QiskitRunOptions] = None,
    ) -> dict[str, PulseSolution]:
        """QiskitPulseBackend.run."""
        run_options = (
            QiskitPulseBackend.QiskitRunOptions()
            if run_options is None
            else run_options
        )
        if isinstance(run_options, PulseBackend.RunOptions):
            run_options = QiskitPulseBackend.QiskitRunOptions(**run_options.to_dict())
        if run_options.initial_state is None:
            run_options.initial_state = "ground_state"
        result = self._native_backend.run(
            run_input=run_input, **run_options.to_native_options()
        ).result()
        result.header = {"casq": True}
        results: dict[str, PulseSolution] = PulseSolution.from_qiskit(result)
        return results

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
        options_dict = self.options.to_native_options()
        return DynamicsBackendPatch(solver, **options_dict)

    @trace()
    def _parse_hamiltonian_dict(self) -> tuple[PulseBackend.Hamiltonian, list[int]]:
        """QiskitPulseBackend._parse_hamiltonian_dict."""
        (
            static_hamiltonian,
            operators,
            channels,
            qubit_dims,
        ) = parse_backend_hamiltonian_dict(self._hamiltonian_dict, self.qubits)
        hamiltonian = PulseBackend.Hamiltonian(static_hamiltonian, operators, channels)
        qubit_dims = [qubit_dims[idx] for idx in self.qubits]
        return hamiltonian, qubit_dims
