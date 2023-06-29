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

import copy

from typing import NamedTuple, Optional, Union

from qiskit import QuantumCircuit
from qiskit.circuit.quantumcircuit import InstructionSet
from qiskit.circuit import Bit, Register
from qiskit.circuit.parameterexpression import ParameterValueType

from jax import core
import jax.numpy as jnp
from loguru import logger
import numpy as np
import numpy.typing as npt
from qiskit_ibm_provider import IBMProvider
from qiskit.providers.models import PulseBackendConfiguration
from qiskit_dynamics import RotatingFrame
from qiskit_dynamics.array import Array
from qiskit_dynamics.backend import parse_backend_hamiltonian_dict
from qiskit_dynamics import Solver
from qiskit_dynamics.solvers.solver_classes import t_span_to_list, _y0_to_list, _signals_to_list

from qiskit import QuantumCircuit
from qiskit.pulse import Acquire, Schedule, ScheduleBlock
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit import schedule as build_schedule
from qiskit.scheduler.config import ScheduleConfig
from qiskit.scheduler.schedule_circuit import schedule_circuit
from qiskit.transpiler import Target
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData, QobjExperimentHeader
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.qobj.common import QobjHeader
from qiskit import QiskitError

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit_dynamics.signals import Signal
# noinspection PyProtectedMember
from scipy.integrate._ivp.ivp import OdeResult
from qiskit.providers import BackendV1
from qiskit.quantum_info import partial_trace, Statevector, DensityMatrix
# noinspection PyProtectedMember
from qiskit_dynamics.backend.backend_utils import (
    _get_dressed_state_decomposition,
    _get_memory_slot_probabilities,
    _sample_probability_dict,
    _get_counts_from_samples,
    _get_iq_data
)
from qiskit_dynamics.solvers.solver_utils import setup_args_lists
from qiskit_dynamics.solvers.solver_functions import _is_diffrax_method
from qiskit_dynamics import DynamicsBackend
from qiskit_dynamics.backend import DynamicsBackend, default_experiment_result_function

from casq import PulseBackendProperties
from casq.common import trace, CasqError
from casq.common import plot, plot_bloch, plot_signal, LineStyle, LineType, LineConfig, LegendStyle, MarkerStyle
from casq.gates import PulseGate
from casq.helpers import discretize, get_channel_frequencies


class PulseSimulator(DynamicsBackend):
    """PulseSimulator class.

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

    class Solution(ExperimentResult):

        def __init__(
            self, shots: Union[int, tuple[int, int]], success: bool,
            data: ExperimentResultData, status: Optional[str] = None,
            seed: Optional[int] = None, header: Optional[QobjExperimentHeader] = None
        ):
            """Instantiate :class:`~casq.PulseSimulator`.

            Extends instantiation of :class:`~qiskit.qiskit_dynamics.DynamicsBackend`
            with additional 'steps' argument.

            Args:
                shots(int or tuple): if an integer the number of shots or if a
                    tuple the starting and ending shot for this data
                success (bool): True if the experiment was successful
                data (ExperimentResultData): The data for the experiment's
                    result
                status (str): The status of the experiment
                seed (int): The seed used for simulation (if run on a simulator)
                header (qiskit.qobj.QobjExperimentHeader): A free form dictionary
                    header for the experiment
            """
            super().__init__(
                shots=shots, success=success,
                data=data, meas_level=MeasLevel.CLASSIFIED,
                status=status, seed=seed, header=header
            )

        def plot_population(self):
            populations = {}
            for key in self.populations[-1].keys():
                populations[key] = []
            for p in self.populations:
                for key in populations.keys():
                    value = p.get(key, 0)
                    populations[key].append(value)
            configs = []
            for key, value in populations.items():
                config = LineConfig(
                    x=self.times, y=value, label=f"Population in |{key}>",
                    line_style=LineStyle(), xtitle="Time (ns)", ytitle="Population"
                )
                configs.append(config)
            plot(data=configs, legend_style=LegendStyle())

        def plot_iq(self, time_index: Optional[int] = None):
            t = time_index if time_index else -1
            x = []
            y = []
            for iq in self.iq_data[t]:
                x.append(iq[0][0])
                y.append(iq[0][1])
            config = LineConfig(x=x, y=y, marker_style=MarkerStyle(), xtitle="I", ytitle="Q")
            plot(data=[config])

        def plot_iq_trajectory(self):
            x = []
            y = []
            for iq in self.avg_iq_data:
                x.append(iq[0][0])
                y.append(iq[0][1])
            config = LineConfig(
                x=x, y=y, marker_style=MarkerStyle(), xtitle="I", ytitle="Q"
            )
            plot(data=[config])

        def plot_trajectory(self, qubit: Optional[int] = None):
            x, y, z = self._xyz()
            if len(self.qubits) > 1:
                if qubit:
                    x, y, z = x[qubit], y[qubit], z[qubit]
                else:
                    raise CasqError(
                        "Cannot plot Bloch trajectory when qubit is not specified for a multi-qubit system."
                    )
            x_config = LineConfig(
                x=self.times, y=x, line_style=LineStyle(), label="$\\langle X \\rangle$", xtitle="$t$"
            )
            y_config = LineConfig(
                x=self.times, y=y, line_style=LineStyle(), label="$\\langle Y \\rangle$", xtitle="$t$"
            )
            z_config = LineConfig(
                x=self.times, y=z, line_style=LineStyle(), label="$\\langle Z \\rangle$", xtitle="$t$"
            )
            plot(data=[x_config, y_config, z_config], legend_style=LegendStyle())

        def plot_bloch_trajectory(self, qubit: Optional[int] = None):
            x, y, z = self._xyz()
            if len(self.qubits) > 1:
                if qubit:
                    x, y, z = x[qubit], y[qubit], z[qubit]
                else:
                    raise CasqError(
                        "Cannot plot Bloch trajectory when qubit is not specified for a multi-qubit system."
                    )
            plot_bloch(x, y, z)

        def _xyz(
            self
        ) -> Union[
            tuple[list[float], list[float], list[float]],
            tuple[dict[int, list[float]], dict[int, list[float]], dict[int, list[float]]]
        ]:
            if len(self.qubits) > 1:
                x = {}
                y = {}
                z = {}
                for q in self.qubits:
                    x[q] = []
                    y[q] = []
                    z[q] = []
                    for sv in self.statevectors:
                        traced_sv = self._trace(sv, q)
                        xp, yp, zp = traced_sv.data.real
                        x[q].append(xp)
                        y[q].append(yp)
                        z[q].append(zp)
                return x, y, z
            else:
                x = []
                y = []
                z = []
                for sv in self.statevectors:
                    xp, yp, zp = sv.data.real
                    x.append(xp)
                    y.append(yp)
                    z.append(zp)
                return x, y, z

        def _trace(self, state: Statevector, qubit: int) -> Statevector:
            traced_over_qubits = self.qubits
            traced_over_qubits.remove(qubit)
            partial_density_matrix = partial_trace(state, traced_over_qubits)
            return partial_density_matrix.to_statevector()

    @staticmethod
    def get_experiment_result(
            experiment_name: str,
            solver_result: OdeResult,
            measurement_subsystems: list[int],
            memory_slot_indices: list[int],
            num_memory_slots: Union[None, int],
            backend: DynamicsBackend,
            seed: Optional[int] = None,
            metadata: Optional[dict] = None,
    ) -> Solution:
        """Generates ExperimentResult objects from solver result.

        Args:
            experiment_name: Name of experiment.
            solver_result: Result object from :class:`Solver.solve`.
            measurement_subsystems: Labels of subsystems in the model being measured.
            memory_slot_indices: Indices of memory slots to store the results in for each subsystem.
            num_memory_slots: Total number of memory slots in the returned output. If ``None``,
                ``max(memory_slot_indices)`` will be used.
            backend: The backend instance that ran the simulation. Various options and properties
                are utilized.
            seed: Seed for any random number generation involved (e.g. when computing outcome samples).
            metadata: Metadata to add to the header of the
                :class:`~qiskit.result.models.ExperimentResult` object.

        Returns:
            :class:`~qiskit.result.models.ExperimentResult`

        Raises:
            QiskitError: If a specified option is unsupported.
        """
        counts = []
        samples = []
        populations = []
        iq_data = []
        avg_iq_data = []
        quantum_states = []
        for t, y in zip(solver_result.t, solver_result.y):
            # Take state out of frame, put in dressed basis, and normalize
            if isinstance(y, Statevector):
                # noinspection PyTypeChecker
                y = np.array(backend.options.solver.model.rotating_frame.state_out_of_frame(t=t, y=y))
                y = backend._dressed_states_adjoint @ y
                y = Statevector(y, dims=backend.options.subsystem_dims)
                if backend.options.normalize_states:
                    y = y / np.linalg.norm(y.data)
            elif isinstance(y, DensityMatrix):
                # noinspection PyTypeChecker
                y = np.array(
                    backend.options.solver.model.rotating_frame.operator_out_of_frame(t=t, operator=y)
                )
                y = backend._dressed_states_adjoint @ y @ backend._dressed_states
                y = DensityMatrix(y, dims=backend.options.subsystem_dims)
                if backend.options.normalize_states:
                    y = y / np.diag(y.data).sum()
            else:
                y = Statevector(y, dims=backend.options.subsystem_dims)
            quantum_states.append(y)
            # compute probabilities for measurement slot values
            measurement_subsystems = [
                backend.options.subsystem_labels.index(x) for x in measurement_subsystems
            ]
            populations_step = _get_memory_slot_probabilities(
                probability_dict=y.probabilities_dict(qargs=measurement_subsystems),
                memory_slot_indices=memory_slot_indices,
                num_memory_slots=num_memory_slots,
                max_outcome_value=1,
            )
            populations.append(populations_step)
            # sample
            samples_step = _sample_probability_dict(populations_step, shots=backend.options.shots, seed=seed)
            samples.append(samples_step)
            counts.append(_get_counts_from_samples(samples_step))
            # Default iq_centers
            iq_centers = []
            for sub_dim in backend.options.subsystem_dims:
                theta = 2 * jnp.pi / sub_dim
                iq_centers.append(
                    [[jnp.cos(idx * theta), jnp.sin(idx * theta)] for idx in range(sub_dim)]
                )
            # generate IQ
            iq_data_step = _get_iq_data(
                y,
                measurement_subsystems=measurement_subsystems,
                iq_centers=iq_centers,
                iq_width=0.2,
                shots=backend.options.shots,
                memory_slot_indices=memory_slot_indices,
                num_memory_slots=num_memory_slots,
                seed=seed,
            )
            iq_data.append(iq_data_step)
            avg_iq_data_step = jnp.average(iq_data_step, axis=0)
            avg_iq_data.append(avg_iq_data_step)
        data = ExperimentResultData(
            counts=counts, memory=samples,
            qubits=backend.options.subsystem_labels, times=solver_result.t,
            statevectors=quantum_states, populations=populations,
            iq_data=iq_data, avg_iq_data=avg_iq_data
        )
        return PulseSimulator.Solution(
            shots=backend.options.shots, success=True, data=data,
            seed=seed, header=QobjExperimentHeader(name=experiment_name, metadata=metadata)
        )

    @trace()
    def __init__(
        self,
        solver: Solver,
        target: Optional[Target] = None,
        steps: Optional[int] = None,
        **options,
    ):
        """Instantiate :class:`~casq.PulseSimulator`.

        Extends instantiation of :class:`~qiskit.qiskit_dynamics.DynamicsBackend`
        with additional 'steps' argument.

        Args:
            solver: Solver instance configured for pulse simulation.
            target: Target object.
            options: Additional configuration options for the simulator.
            steps: Number of steps at which to solve the system.
                Used to automatically calculate an evenly-spaced t_eval range.

        Raises:
            QiskitError: If any instantiation arguments fail validation checks.
        """
        super().__init__(solver, target, **options)
        self.steps = steps

    def _run(
            self,
            job_id,
            t_span,
            schedules,
            measurement_subsystems_list,
            memory_slot_indices_list,
            num_memory_slots_list,
    ) -> Result:
        auto_t_eval = None
        if self.steps:
            auto_t_eval = jnp.linspace(t_span[0], t_span[1], self.steps)
            auto_t_eval[0] = t_span[0]
            auto_t_eval[-1] = t_span[1]
        if "solver_options" in self.options:
            t_eval = self.options.solver_options.get("t_eval", None)
            if t_eval is None:
                self.options.solver_options["t_eval"] = auto_t_eval
            else:
                self.options.solver_options["t_eval"] = t_eval
        else:
            self.options.solver_options = {"t_eval": auto_t_eval}
        return super()._run(
            job_id, t_span, schedules,
            measurement_subsystems_list, memory_slot_indices_list,
            num_memory_slots_list
        )
