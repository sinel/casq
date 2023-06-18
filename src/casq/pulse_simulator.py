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
from __future__ import annotations

import datetime
from typing import Optional, Union
from uuid import uuid4

from loguru import logger
import numpy as np
import numpy.typing as nt
from qiskit import QuantumCircuit, transpile, schedule
from qiskit.result import Result
from qiskit.providers import BackendV1, BackendV2
from qiskit_dynamics import DynamicsBackend
from qiskit_dynamics.backend.dynamics_backend import (
    _get_backend_channel_freqs, _get_acquire_instruction_timings, _get_dressed_state_decomposition,
    _to_schedule_list, default_experiment_result_function
)
from qiskit_dynamics.backend.dynamics_job import DynamicsJob
from qiskit_dynamics.backend.backend_string_parser import parse_backend_hamiltonian_dict
from qiskit_dynamics.solvers.solver_classes import Solver
from qiskit.quantum_info import Statevector
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit import schedule as build_schedule
from scipy.integrate._ivp.ivp import OdeResult

from casq.common.decorators import trace
from casq.common.exceptions import CasqError
from casq.common.helpers import dbid, ufid
from casq.gates.pulse_gate import PulseGate


class PulseSimulator:
    """PulseSimulator class.

    Wraps and extends :py:class:`qiskit_dynamics.DynamicsBackend`.

    Args:
        backend: Qiskit backend  to use for extracting simulator model.
        (:py:class:`qiskit.providers.BackendV1`
        or :py:class:`qiskit.providers.BackendV2`)
    """

    @trace()
    def __init__(
            self,
            backend: Union[BackendV1, BackendV2],
            qubit_subset: list[int],
            name: Optional[str] = None,
            seed: Optional[int] = 280728
    ) -> None:
        """Initialize PulseSimulator."""
        self.dbid = dbid()
        self.ufid = name if name else ufid(self)
        self.backend = backend
        self.qubit_subset = qubit_subset
        self.rng = np.random.default_rng(seed)
        self.solver, self._dressed_evals, self._dressed_states = self._build_solver()

    def run(
        self, run_input: list[Union[QuantumCircuit, Schedule, ScheduleBlock]]
    ) -> Union[OdeResult, list[OdeResult]]:
        """PulseGate.to_circuit method.

        Builds simple circuit for solitary usage or testing of pulse gate.

        Args:
            run_input: List of schedules, schedule blocks, or quantum circuits to execute.

        Returns:
            :py:class:`qiskit_dynamics.backend.dynamics_job.DynamicsJob`
        """
        schedules, num_memory_slots_list = _to_schedule_list(run_input, backend=self.backend)
        (
            t_span,
            measurement_subsystems_list,
            memory_slot_indices_list,
        ) = _get_acquire_instruction_timings(
            schedules, list(range(self.solver.model.dim)), self.backend.configuration().dt
        )
        t_eval = np.linspace(0., t_span[0][-1], 100)
        y0 = Statevector(self._dressed_states[:, 0])
        return self.solver.solve(t_span=t_span, y0=y0, t_eval=t_eval, signals=schedules)

    def _build_solver(self):
        backend_target = getattr(self.backend, "target", None)
        backend_config = self.backend.configuration()
        backend_defaults = None
        if backend_target is not None and backend_target.dt is not None:
            dt = backend_target.dt
        else:
            dt = backend_config.dt
        if hasattr(self.backend, "defaults"):
            backend_defaults = self.backend.defaults()
        (
            static_hamiltonian,
            hamiltonian_operators,
            hamiltonian_channels,
            subsystem_dims,
        ) = parse_backend_hamiltonian_dict(backend_config.hamiltonian, self.qubit_subset)
        channel_freqs = _get_backend_channel_freqs(
            backend_target=backend_target,
            backend_config=backend_config,
            backend_defaults=backend_defaults,
            channels=hamiltonian_channels,
        )
        dressed_evals, dressed_states = _get_dressed_state_decomposition(static_hamiltonian)
        solver = Solver(
            static_hamiltonian=static_hamiltonian,
            hamiltonian_operators=hamiltonian_operators,
            hamiltonian_channels=hamiltonian_channels,
            channel_carrier_freqs=channel_freqs,
            dt=dt,
        )
        return solver, dressed_evals, dressed_states
