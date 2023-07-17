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

from typing import Any, Optional, Self, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import BackendV1, BackendV2
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.quantum_info import DensityMatrix, Statevector
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

    @classmethod
    @trace()
    def from_backend(
        cls,
        backend: Union[BackendV1, BackendV2],
        qubits: Optional[list[int]] = None,
        rotating_frame: Union[Array, RotatingFrame, str] = "auto",
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Self:
        """Construct a QiskitPulseBackend instance from an existing backend instance.

        Args:
            backend: The ``Backend`` instance to build the :class:`.DynamicsBackend` from.
            qubits: List of qubits to include from the backend.
            rotating_frame: Rotating frame argument for the internal :class:`.Solver`. Defaults to
                ``"auto"``, allowing this method to pick a rotating frame.
            evaluation_mode: Evaluation mode argument for the internal :class:`.Solver`.
            rwa_cutoff_freq: Rotating wave approximation argument for the internal :class:`.Solver`.
            seed: Seed to use in random sampling. Defaults to None.

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
        instance: Self = cls(
            hamiltonian_dict=backend_characteristics.hamiltonian,
            qubits=qubits,
            dt=backend_characteristics.dt,
            channel_carrier_freqs=backend_characteristics.get_channel_frequencies(
                channels
            ),
            control_channel_map=backend_characteristics.get_control_channel_map(
                channels
            ),
            rotating_frame=rotating_frame,
            evaluation_mode=evaluation_mode,
            rwa_cutoff_freq=rwa_cutoff_freq,
            seed=seed,
        )
        return instance

    @trace()
    def __init__(
        self,
        hamiltonian_dict: dict,
        qubits: list[int],
        dt: Optional[float] = None,
        channel_carrier_freqs: Optional[dict] = None,
        control_channel_map: Optional[dict] = None,
        rotating_frame: Union[Array, RotatingFrame, str] = "auto",
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """Instantiate :class:`~casq.QiskitPulseBackend`.

        Args:
            hamiltonian_dict: Pulse backend Hamiltonian dictionary.
            qubits: List of qubits to include from the backend.
            dt: Sampling interval.
            channel_carrier_freqs: Dictionary mapping channel names to frequencies.
            control_channel_map: A dictionary mapping control channel labels to indices.
            rotating_frame: Rotating frame argument for the internal :class:`.Solver`. Defaults to
                ``"auto"``, allowing this method to pick a rotating frame.
            evaluation_mode: Evaluation mode argument for the internal :class:`.Solver`.
            rwa_cutoff_freq: Rotating wave approximation argument for the internal :class:`.Solver`.
            seed: Seed to use in random sampling. Defaults to None.
        """
        self._dt = dt
        self._channel_carrier_freqs = channel_carrier_freqs
        self._control_channel_map = control_channel_map
        self._rotating_frame = rotating_frame
        self._evaluation_mode = evaluation_mode
        self._rwa_cutoff_freq = rwa_cutoff_freq
        super().__init__(
            PulseBackend.NativeBackendType.QISKIT, hamiltonian_dict, qubits, seed
        )

    @trace()
    @timer()
    def run(
        self,
        run_input: list[Union[PulseCircuit, QuantumCircuit, Schedule, ScheduleBlock]],
        method: PulseBackend.ODESolverMethod,
        initial_state: Optional[Union[DensityMatrix, Statevector]] = None,
        shots: int = 1024,
        steps: Optional[int] = None,
    ) -> dict[str, PulseSolution]:
        """QiskitPulseBackend.run."""
        options = DynamicsBackendPatch.Options(
            initial_state="ground_state" if initial_state is None else initial_state,
            experiment_result_function=get_experiment_result,
            shots=shots,
            solver_options={"method": method.value},
        )
        self._native_backend.steps = steps
        result = self._native_backend.run(
            run_input=run_input, **options.to_dict()
        ).result()
        result.header = {"casq": True}
        results: dict[str, PulseSolution] = PulseSolution.from_qiskit(result)
        return results

    @trace()
    @timer()
    def _get_native_backend(self) -> DynamicsBackendPatch:
        """QiskitPulseBackend._get_native_backend."""
        if self._rotating_frame == "auto":
            if "dense" in self._evaluation_mode:
                self._rotating_frame = self._hamiltonian.static
            else:
                self._rotating_frame = np.diag(self._hamiltonian.static)
        solver = Solver(
            static_hamiltonian=self._hamiltonian.static,
            hamiltonian_operators=self._hamiltonian.operators,
            hamiltonian_channels=self._hamiltonian.channels,
            channel_carrier_freqs=self._channel_carrier_freqs,
            dt=self._dt,
            rotating_frame=self._rotating_frame,
            evaluation_mode=self._evaluation_mode,
            rwa_cutoff_freq=self._rwa_cutoff_freq,
        )
        options = DynamicsBackendPatch.Options(
            control_channel_map=self._control_channel_map,
            seed_simulator=self._seed,
            experiment_result_function=get_experiment_result,
        )
        return DynamicsBackendPatch(solver, **options.to_dict())

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
