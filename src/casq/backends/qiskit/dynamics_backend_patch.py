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

from loguru import logger
import numpy as np
from qiskit.providers import BackendV1, BackendV2
from qiskit.pulse import Schedule
from qiskit.result import Result
from qiskit.transpiler import Target
from qiskit_dynamics import RotatingFrame, Solver
from qiskit_dynamics.array import Array
from qiskit_dynamics.backend import DynamicsBackend


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
    """

    @classmethod
    def from_backend(
        cls,
        backend: Union[BackendV1, BackendV2],
        subsystem_list: Optional[list[int]] = None,
        rotating_frame: Optional[Union[Array, RotatingFrame, str]] = "auto",
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        steps: Optional[int] = None,
        **options
    ) -> DynamicsBackendPatch:
        """Construct a DynamicsBackendPatch instance from an existing Backend instance.

        Args:
            backend: The ``Backend`` instance to build the :class:`.DynamicsBackend` from.
            subsystem_list: The list of qubits in the backend to include in the model.
            rotating_frame: Rotating frame argument for the internal :class:`.Solver`. Defaults to
                ``"auto"``, allowing this method to pick a rotating frame.
            evaluation_mode: Evaluation mode argument for the internal :class:`.Solver`.
            rwa_cutoff_freq: Rotating wave approximation argument for the internal :class:`.Solver`.
            steps: Number of steps at which to solve the system.
                Used to automatically calculate an evenly-spaced t_eval range.
            **options: Additional options to be applied in construction of the
                :class:`.DynamicsBackend`.

        Returns:
            DynamicsBackendPatch

        Raises:
            QiskitError: If any required parameters are missing from the passed backend.
        """
        dynamics_backend = DynamicsBackend.from_backend(
            backend=backend,
            subsystem_list=subsystem_list,
            rotating_frame=rotating_frame,
            evaluation_mode=evaluation_mode,
            rwa_cutoff_freq=rwa_cutoff_freq,
            **options
        )
        dynamics_backend_patch: DynamicsBackendPatch = DynamicsBackendPatch(
            solver=dynamics_backend.options.solver,
            rwa_cutoff_freq=rwa_cutoff_freq,
            steps=steps,
            **options
        )
        return dynamics_backend_patch

    def __init__(
        self,
        solver: Solver,
        target: Optional[Target] = None,
        rwa_cutoff_freq: Optional[float] = None,
        steps: Optional[int] = None,
        **options
    ):
        """Instantiate :class:`~casq.DynamicsBackendPatch`.

        Extends instantiation of :class:`~qiskit.qiskit_dynamics.DynamicsBackend`
        with additional 'steps' argument.

        Args:
            solver: Solver instance configured for pulse simulation.
            target: Target object.
            rwa_cutoff_freq: Rotating wave approximation argument for the internal :class:`.Solver`.
            steps: Number of steps at which to solve the system.
                Used to automatically calculate an evenly-spaced t_eval range.
            options: Additional configuration options for the simulator.

        Raises:
            QiskitError: If any instantiation arguments fail validation checks.
        """
        options.update(steps=steps)
        options.update(rwa_cutoff_freq=rwa_cutoff_freq)
        super().__init__(solver, target, **options)

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
        auto_t_eval = None
        if self.options.steps:
            auto_t_eval = np.linspace(t_span[0][0], t_span[0][1], self.options.steps)
            auto_t_eval[0] = t_span[0][0]
            auto_t_eval[-1] = t_span[0][1]
        if "solver_options" in self.options:
            t_eval = self.options.solver_options.get("t_eval", None)
            if t_eval is None:
                self.options.solver_options["t_eval"] = auto_t_eval
            else:
                self.options.solver_options["t_eval"] = t_eval
        else:
            self.options.solver_options = {"t_eval": auto_t_eval}
        logger.debug(f"t_eval = {self.options.solver_options['t_eval']}")
        return super()._run(
            job_id,
            t_span,
            schedules,
            measurement_subsystems_list,
            memory_slot_indices_list,
            num_memory_slots_list,
        )
