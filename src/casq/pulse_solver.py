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

from casq import PulseBackendProperties
from casq.common import trace, CasqError
from casq.common import plot, plot_bloch, plot_signal, LineStyle, LineType, LineConfig, LegendStyle, MarkerStyle
from casq.gates import PulseGate
from casq.helpers import discretize, get_channel_frequencies


class PulseSolver(Solver):
    """PulseSolver class.

    Extends Qiskit Solver class
    with automated calculation of time interval and evaluation steps.

    Args:
        name: Optional user-friendly name for pulse gate.
    """

    class Solution(NamedTuple):
        qubits: list[int]
        times: list[float]
        statevectors: list[Statevector]
        counts: list[dict]
        samples: list[Union[list, npt.NDArray]]
        populations: list[dict]
        iq_data: list[Union[list, npt.NDArray]]
        avg_iq_data: list[Union[list, npt.NDArray]]

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

    @classmethod
    @trace()
    def from_backend(
        cls,
        backend_name: str,
        qubits: Optional[list[int]] = None,
        rotating_frame: Optional[Union[Array, RotatingFrame, str]] = "auto",
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        seed: Optional[int] = None
    ) -> PulseSolver:
        """Build PulseSolver from IBMQ backend.

        Args:
            backend_name: IBMQ backend identifier.
            qubits: Qubits from the backend to include in the model.
            rotating_frame: Rotating frame argument for :py:class:`qiskit_dynamics.Solver`.
                Defaults to ``"auto"``, allowing this method to pick a rotating frame.
            evaluation_mode: Evaluation mode argument for :py:class:`qiskit_dynamics.Solver`.
            rwa_cutoff_freq: Rotating wave approximation argument for :py:class:`qiskit_dynamics.Solver`.

        Returns:
            :py:class:`qiskit_dynamics.Solver`
        """
        props = PulseBackendProperties(PulseBackendProperties.get_backend(backend_name))
        if qubits is None:
            qubits = list(range(props.num_qubits))
        else:
            qubits = sorted(qubits)
        if qubits[-1] >= props.num_qubits:
            raise CasqError(
                f"Selected qubit {qubits[-1]} is out of bounds "
                f"for backend {props.backend} with {props.num_qubits} qubits."
            )
        schedule_config = ScheduleConfig(
            inst_map=props.defaults.instruction_schedule_map,
            meas_map=[qubits],
            dt=props.dt
        )
        # hamiltonian
        (
            static_hamiltonian,
            hamiltonian_operators,
            hamiltonian_channels,
            qubit_dims,
        ) = parse_backend_hamiltonian_dict(props.hamiltonian, qubits)
        # channel frequencies
        channel_freqs = get_channel_frequencies(hamiltonian_channels, props)
        # rotating frame
        if rotating_frame == "auto":
            if "dense" in evaluation_mode:
                rotating_frame = static_hamiltonian
            else:
                rotating_frame = jnp.diag(static_hamiltonian)
        return cls(
            static_hamiltonian=Array(static_hamiltonian),
            hamiltonian_operators=Array(hamiltonian_operators),
            hamiltonian_channels=hamiltonian_channels,
            channel_carrier_freqs=channel_freqs,
            rotating_frame=rotating_frame,
            evaluation_mode=evaluation_mode,
            rwa_cutoff_freq=rwa_cutoff_freq,
            dt=props.dt,
            backend=props.backend,
            qubits=qubit_dims,
            schedule_config=schedule_config,
            seed=seed
        )

    @trace()
    def __init__(
        self,
        static_hamiltonian: Optional[Array] = None,
        hamiltonian_operators: Optional[Array] = None,
        static_dissipators: Optional[Array] = None,
        dissipator_operators: Optional[Array] = None,
        hamiltonian_channels: Optional[list[str]] = None,
        dissipator_channels: Optional[list[str]] = None,
        channel_carrier_freqs: Optional[dict] = None,
        dt: Optional[float] = None,
        rotating_frame: Optional[Union[Array, RotatingFrame]] = None,
        in_frame_basis: bool = False,
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        rwa_carrier_freqs: Optional[Union[Array, tuple[Array, Array]]] = None,
        validate: bool = True,
        backend: Optional[BackendV1] = None,
        qubits: Optional[dict[int, int]] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        seed: Optional[int] = None
    ):
        super().__init__(
            static_hamiltonian, hamiltonian_operators, static_dissipators, dissipator_operators,
            hamiltonian_channels, dissipator_channels, channel_carrier_freqs, dt,
            rotating_frame, in_frame_basis, evaluation_mode,
            rwa_cutoff_freq, rwa_carrier_freqs, validate
        )
        self._backend = backend
        self._backend_props = PulseBackendProperties(backend)
        self.qubits = qubits
        self.schedule_config = schedule_config
        self.seed = seed
        self._hamiltonian_channels = hamiltonian_channels
        self._channel_carrier_freqs = channel_carrier_freqs
        self._dressed_evals, self._dressed_states = _get_dressed_state_decomposition(static_hamiltonian.data)
        self._dressed_states_adjoint = self._dressed_states.conj().transpose()

    def solve(
        self,
        run_input: Union[list[Union[QuantumCircuit, Schedule, ScheduleBlock]]],
        initial_state: Optional[Union[Union[Array, QuantumState, BaseOperator]]] = None,
        convert_results: bool = True,
        steps: Optional[int] = None,
        normalize_states: bool = True,
        shots: int = 1024,
        **kwargs,
    ) -> PulseSolver.Solution:
        num_memory_slots = None
        if isinstance(run_input, ScheduleBlock):
            schedule = block_to_schedule(run_input)
        elif isinstance(run_input, Schedule):
            schedule = run_input
            signals = discretize(
                schedule, self._backend_props.dt,
                get_channel_frequencies(list(schedule.channels), self._backend_props)
            )
            schedule = [schedule]
        elif isinstance(run_input, QuantumCircuit):
            # raise CasqError(
            #     "QuantumCircuit is not supported because of 'https://github.com/Qiskit/qiskit-terra/issues/9488'"
            # )
            num_memory_slots = run_input.cregs[0].size
            schedule = schedule_circuit(run_input, self.schedule_config, method="alap")
            # # schedule = build_schedule(run_input, self._backend, dt=self._dt)
        else:
            raise CasqError(f"Type {type(run_input)} cannot be converted to Schedule.")
        schedule_acquires = []
        schedule_acquire_times = []
        for start_time, inst in schedule.instructions:
            # only track acquires saving in a memory slot
            if isinstance(inst, Acquire) and inst.mem_slot is not None:
                schedule_acquires.append(inst)
                schedule_acquire_times.append(start_time)
        if len(schedule_acquire_times) == 0:
            raise CasqError(
                "At least one measurement saving to a MemorySlot required in each schedule."
            )
        for acquire_time in schedule_acquire_times[1:]:
            if acquire_time != schedule_acquire_times[0]:
                raise CasqError("Only support for measurements at one time.")
        t_span = [0.0, self._dt * schedule_acquire_times[0]]
        measurement_subsystems = []
        memory_slot_indices = []
        for inst in schedule_acquires:
            if inst.channel.index in list(self.qubits.keys()):
                measurement_subsystems.append(inst.channel.index)
            else:
                raise CasqError(
                    f"Attempted to measure subsystem {inst.channel.index}, but it is not in subsystem_list.")
            memory_slot_indices.append(inst.mem_slot.index)
        if not initial_state:
            initial_state = Statevector(self._dressed_states[:, 0])
            # initial_state = np.array(self._dressed_states[:, 0])
        if steps:
            t_eval = jnp.linspace(t_span[0], t_span[1], steps)
            t_eval[0] = t_span[0]
            t_eval[-1] = t_span[1]
        else:
            t_eval = None
        result = super().solve(
            t_span=t_span, y0=initial_state, signals=[schedule],
            convert_results=convert_results, t_eval=t_eval, **kwargs
        )
        return self._build_solution(
            result[0], measurement_subsystems, memory_slot_indices, num_memory_slots, normalize_states, shots
        )

    def _build_solution(
        self, result: OdeResult,
        measurement_subsystems, memory_slot_indices, num_memory_slots,
        normalize_states: bool = True, shots: int = 1024
    ) -> PulseSolver.Solution:
        qubit_labels = list(self.qubits.keys())
        qubit_dims = list(self.qubits.values())
        counts = []
        samples = []
        populations = []
        iq_data = []
        avg_iq_data = []
        quantum_states = []
        for t, y in zip(result.t, result.y):
            # Take state out of frame, put in dressed basis, and normalize
            if isinstance(y, Statevector):
                # noinspection PyTypeChecker
                y = np.array(self.model.rotating_frame.state_out_of_frame(t=t, y=y))
                y = self._dressed_states_adjoint @ y
                y = Statevector(y, dims=qubit_dims)
                if normalize_states:
                    y = y / np.linalg.norm(y.data)
            elif isinstance(y, DensityMatrix):
                # noinspection PyTypeChecker
                y = np.array(
                    self.model.rotating_frame.operator_out_of_frame(t=t, operator=y)
                )
                y = self._dressed_states_adjoint @ y @ self._dressed_states
                y = DensityMatrix(y, dims=qubit_dims)
                if normalize_states:
                    y = y / np.diag(y.data).sum()
            else:
                y = Statevector(y, dims=qubit_dims)
            quantum_states.append(y)
            # compute probabilities for measurement slot values
            measurement_subsystems = [
                qubit_labels.index(x) for x in measurement_subsystems
            ]
            populations_step = _get_memory_slot_probabilities(
                probability_dict=y.probabilities_dict(qargs=measurement_subsystems),
                memory_slot_indices=memory_slot_indices,
                num_memory_slots=num_memory_slots,
                max_outcome_value=1,
            )
            populations.append(populations_step)
            # sample
            samples_step = _sample_probability_dict(populations_step, shots=shots, seed=self.seed)
            samples.append(samples_step)
            counts.append(_get_counts_from_samples(samples_step))
            # Default iq_centers
            iq_centers = []
            for sub_dim in qubit_dims:
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
                shots=shots,
                memory_slot_indices=memory_slot_indices,
                num_memory_slots=num_memory_slots,
                seed=self.seed,
            )
            iq_data.append(iq_data_step)
            avg_iq_data_step = jnp.average(iq_data_step, axis=0)
            avg_iq_data.append(avg_iq_data_step)
        return PulseSolver.Solution(
            qubits=qubit_labels, times=result.t,
            statevectors=quantum_states, counts=counts, samples=samples,
            populations=populations, iq_data=iq_data, avg_iq_data=avg_iq_data
        )
