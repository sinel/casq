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
"""Patched Qiskit dynamics backend."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
from qiskit.providers import BackendV1, BackendV2
from qiskit.providers.models import PulseBackendConfiguration, PulseDefaults
from qiskit.pulse import Schedule
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.result import Result
from qiskit.transpiler import Target
from qiskit_dynamics import RotatingFrame, Solver
from qiskit_dynamics.array import Array
from qiskit_dynamics.backend import DynamicsBackend, default_experiment_result_function


class DynamicsBackendPatch(DynamicsBackend):
    """DynamicsBackend patch class.

    This class extends :class:`~qiskit.qiskit_dynamics.DynamicsBackend`
    to provide the following improvements.
    - Solutions are calculated for each t_eval point,
        however intermediate solutions are not stored in the result object.
        This bug is now fixed.
    - t_eval must be provided without knowing the internally calculated t_span range.
        Furthermore, manually entering t_span causes an error. As a result,
        providing a t_eval range is awkward at best and error-prone.
        As an alternative, automatic calculation of t_eval range
        based on a steps argument is now provided.

    Args:
        solver: Solver instance configured for pulse simulation.
        target: Target object.
        steps: Number of steps at which to solve the system.
            Used to automatically calculate an evenly-spaced t_eval range.
        options: Additional configuration options for the backend.

    Raises:
        If any instantiation arguments fail validation checks,
        then raises :py:class:`~qiskit.exceptions.QiskitError`.
    """

    @dataclass
    class Options:
        """Qiskit dynamics backend options."""

        shots: int = 1024
        solver: Optional[Solver] = None
        solver_options: dict[str, Any] = field(default_factory=dict)
        subsystem_labels: Optional[list[int]] = None
        subsystem_dims: Optional[list[int]] = None
        meas_map: Optional[dict] = None
        control_channel_map: Optional[dict] = None
        normalize_states: bool = True
        initial_state: Union[str, DensityMatrix, Statevector] = "ground_state"
        meas_level: MeasLevel = MeasLevel.CLASSIFIED
        meas_return: MeasReturnType = MeasReturnType.AVERAGE
        iq_centers: Optional[list[list[list[float]]]] = None
        iq_width: float = 0.2
        max_outcome_level: Optional[int] = 1
        memory: bool = True
        seed_simulator: Optional[int] = None
        experiment_result_function: Callable = default_experiment_result_function
        configuration: Optional[PulseBackendConfiguration] = None
        defaults: Optional[PulseDefaults] = None

        def to_dict(self) -> dict[str, Any]:
            """Converts to dict.

            Returns:
                Dictionary.
            """
            return asdict(
                self, dict_factory=lambda opt: {k: v for (k, v) in opt if v is not None}
            )

    @classmethod
    def from_backend(
        cls,
        backend: Union[BackendV1, BackendV2],
        qubits: Optional[list[int]] = None,
        rotating_frame: Union[Array, RotatingFrame, str] = "auto",
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        **options: Any,
    ) -> DynamicsBackendPatch:
        """Construct a DynamicsBackendPatch instance from an existing Backend instance.

        Args:
            backend: The ``Backend`` instance to build the :class:`.DynamicsBackend` from.
            qubits: List of qubits to include from the backend.
            rotating_frame: Rotating frame argument for the internal :class:`.Solver`. Defaults to
                ``"auto"``, allowing this method to pick a rotating frame.
            evaluation_mode: Evaluation mode argument for the internal :class:`.Solver`.
            rwa_cutoff_freq: Rotating wave approximation argument for the internal :class:`.Solver`.
            options: Additional configuration options for the backend.

        Returns:
            DynamicsBackendPatch

        Raises:
            If any required parameters are missing from the passed backend,
            then raises :py:class:`~qiskit.exceptions.QiskitError`.
        """
        dynamics_backend = DynamicsBackend.from_backend(
            backend=backend,
            subsystem_list=qubits,
            rotating_frame=rotating_frame,
            evaluation_mode=evaluation_mode,
            rwa_cutoff_freq=rwa_cutoff_freq,
            **options,
        )
        options = {
            k: v for k, v in dynamics_backend.options.__dict__.items() if v is not None
        }
        dynamics_backend_patch: DynamicsBackendPatch = DynamicsBackendPatch(**options)
        return dynamics_backend_patch

    def __init__(
        self,
        solver: Solver,
        target: Optional[Target] = None,
        **options: Any,
    ):
        """Initialize :class:`~casq.DynamicsBackendPatch`.

        Extends instantiation of :class:`~qiskit.qiskit_dynamics.DynamicsBackend`
        with additional 'steps' argument.
        """
        super().__init__(solver, target, **options)
        self.steps: Optional[int] = None

    def _run(
        self,
        job_id: str,
        t_span: Union[list[tuple[float, float]], list[list[float]]],
        schedules: list[Schedule],
        measurement_subsystems_list: list[list[int]],
        memory_slot_indices_list: list[list[int]],
        num_memory_slots_list: list[int],
    ) -> Result:
        """Run a list of simulations.

        Args:
            job_id: Job identifier.
            t_span: Tuple or list of initial and final time.
            schedules: List of schedules.
            measurement_subsystems_list: List of measurement subsystems.
            memory_slot_indices_list: List of memory slot indices.
            num_memory_slots_list: List of numbers of memory slots.

        Returns:
            ExperimentResult object.
        """
        if self.steps:
            auto_t_eval = np.linspace(t_span[0][0], t_span[0][1], self.steps)
            auto_t_eval[0] = t_span[0][0]
            auto_t_eval[-1] = t_span[0][1]
            self.options.solver_options.update(t_eval=auto_t_eval)
        return super()._run(
            job_id,
            t_span,
            schedules,
            measurement_subsystems_list,
            memory_slot_indices_list,
            num_memory_slots_list,
        )
