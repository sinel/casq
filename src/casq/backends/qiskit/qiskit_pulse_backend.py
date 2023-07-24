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

import numpy.typing as npt
from qiskit.providers import Backend
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit_dynamics import RotatingFrame
from qiskit_dynamics.array import Array
from qiskit_dynamics.solvers import Solver

from casq.backends.pulse_backend import PulseBackend
from casq.backends.pulse_solution import PulseSolution
from casq.backends.qiskit.backend_characteristics import BackendCharacteristics
from casq.backends.qiskit.dynamics_backend_patch import DynamicsBackendPatch
from casq.backends.qiskit.helpers import get_experiment_result
from casq.common.decorators import timer, trace
from casq.gates.pulse_circuit import PulseCircuit
from casq.models.hamiltonian_model import HamiltonianModel
from casq.models.control_model import ControlModel
from casq.models.noise_model import NoiseModel


class QiskitPulseBackend(PulseBackend):
    """QiskitPulseBackend class."""

    @classmethod
    @trace()
    def from_backend(
        cls,
        backend: Backend,
        qubits: Optional[list[int]] = None,
        rotating_frame: Optional[Union[Array, RotatingFrame]] = None,
        in_frame_basis: bool = False,
        evaluation_mode: Optional[HamiltonianModel.EvaluationMode] = None,
        rwa_cutoff_freq: Optional[float] = None,
        rwa_carrier_freqs: Optional[
            Union[npt.NDArray, tuple[npt.NDArray, npt.NDArray]]
        ] = None,
        seed: Optional[int] = None,
    ) -> Self:
        """Construct a QiskitPulseBackend instance from an existing backend instance.

        Args:
            backend: The ``Backend`` instance to build the :class:`.DynamicsBackend` from.
            qubits: List of qubits to include from the backend.
            rotating_frame: Rotating frame argument for the internal :class:`.Solver`.
                    Defaults to None, allowing this method to pick a rotating frame.
            in_frame_basis: Whether to represent the model in the basis in which
                            the rotating frame operator is diagonalized.
            evaluation_mode: Evaluation mode to use by solver.
            rwa_cutoff_freq: Rotating wave approximation cutoff frequency.
                            If None, no approximation is made.
            rwa_carrier_freqs: Carrier frequencies to use for rotating wave approximation.
            seed: Seed to use in random sampling. Defaults to None.

        Returns:
            QiskitPulseBackend
        """
        backend_characteristics = BackendCharacteristics(backend)
        hamiltonian = HamiltonianModel(
            hamiltonian_dict=backend_characteristics.hamiltonian,
            extracted_qubits=qubits,
            rotating_frame=rotating_frame,
            in_frame_basis=in_frame_basis,
            evaluation_mode=evaluation_mode,
            rwa_cutoff_freq=rwa_cutoff_freq,
            rwa_carrier_freqs=rwa_carrier_freqs,
        )
        control = ControlModel(
            dt=backend_characteristics.dt,
            channel_carrier_freqs=backend_characteristics.get_channel_frequencies(
                hamiltonian.channels
            ),
            control_channel_map=backend_characteristics.get_control_channel_map(
                hamiltonian.channels
            ),
        )
        instance: Self = cls(hamiltonian=hamiltonian, control=control, seed=seed)
        return instance

    @trace()
    def __init__(
        self,
        hamiltonian: HamiltonianModel,
        control: ControlModel,
        noise: Optional[NoiseModel] = None,
        seed: Optional[int] = None,
    ):
        """Instantiate :class:`~casq.QiskitPulseBackend`.

        Args:
            hamiltonian: Hamiltonian model.
            control: Control model.
            noise: Noise model.
            seed: Seed to use in random sampling. Defaults to None.
        """
        super().__init__(hamiltonian=hamiltonian, control=control, noise=noise, seed=seed)

    @trace()
    @timer(unit="sec")
    def run(
        self,
        circuit: PulseCircuit,
        method: PulseBackend.ODESolverMethod,
        initial_state: Optional[Union[DensityMatrix, Statevector]] = None,
        shots: int = 1024,
        steps: Optional[int] = None,

        run_options: Optional[dict[str, Any]] = None
    ) -> PulseSolution:
        """QiskitPulseBackend.run."""
        if run_options:
            run_options.update(method=method.value)
        else:
            run_options = {"method": method.value},
        options = DynamicsBackendPatch.Options(
            initial_state="ground_state" if initial_state is None else initial_state,
            experiment_result_function=get_experiment_result,
            shots=shots,
            solver_options=run_options
        )
        self._native_backend.steps = steps
        result = (
            self._native_backend.run(run_input=circuit, **options.to_dict())
            .result()
            .results[0]
        )
        solution: PulseSolution = PulseSolution.from_qiskit(result)
        return solution

    @trace()
    @timer()
    def _get_native_backend(self) -> DynamicsBackendPatch:
        """QiskitPulseBackend._get_native_backend."""
        if self.noise:
            solver = Solver(
                static_hamiltonian=self.hamiltonian.static_operator,
                hamiltonian_operators=self.hamiltonian.operators,
                hamiltonian_channels=self.hamiltonian.channels,
                static_dissipators=Array(self.noise.static_dissipators),
                dissipator_operators=self.noise.dissipator_operators,
                dissipator_channels=self.noise.dissipator_channels,
                channel_carrier_freqs=self.control.channel_carrier_freqs,
                dt=self.control.dt,
                rotating_frame=self.hamiltonian.rotating_frame,
                evaluation_mode=self.hamiltonian.evaluation_mode.name.lower(),
                rwa_cutoff_freq=self.hamiltonian.rwa_cutoff_freq,
            )
        else:
            solver = Solver(
                static_hamiltonian=self.hamiltonian.static_operator,
                hamiltonian_operators=self.hamiltonian.operators,
                hamiltonian_channels=self.hamiltonian.channels,
                channel_carrier_freqs=self.control.channel_carrier_freqs,
                dt=self.control.dt,
                rotating_frame=self.hamiltonian.rotating_frame,
                evaluation_mode=self.hamiltonian.evaluation_mode.name.lower(),
                rwa_cutoff_freq=self.hamiltonian.rwa_cutoff_freq,
            )
        options = DynamicsBackendPatch.Options(
            control_channel_map=self.control.control_channel_map,
            seed_simulator=self._seed,
            experiment_result_function=get_experiment_result,
        )
        return DynamicsBackendPatch(solver, **options.to_dict())
