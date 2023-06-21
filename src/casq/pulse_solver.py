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

import numpy as np
import numpy.typing as npt
from qiskit_ibm_provider import IBMProvider
from qiskit.providers.models import PulseBackendConfiguration
from qiskit_dynamics import RotatingFrame
from qiskit_dynamics.array import Array
from qiskit_dynamics.backend import parse_backend_hamiltonian_dict
from qiskit_dynamics import Solver

from qiskit import QuantumCircuit
from qiskit.pulse import Acquire, Schedule, ScheduleBlock
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit import schedule as build_schedule

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit_dynamics.signals import Signal
from scipy.integrate._ivp.ivp import OdeResult
from qiskit.providers import BackendV1
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_dynamics.backend.backend_utils import (
    _get_dressed_state_decomposition,
    _get_memory_slot_probabilities,
    _sample_probability_dict,
    _get_counts_from_samples,
    _get_iq_data
)

from casq.common import trace, CasqError


from casq.common import trace, dbid, ufid
from casq.gates import PulseGate


class PulseSolver(Solver):
    """PulseSolver class.

    Extends Qiskit Solver class
    with automated calculation of time inetrval and evaluation steps.

    Args:
        name: Optional user-friendly name for pulse gate.
    """

    class Solution(NamedTuple):
        time: list[float]
        states: list[Union[Statevector, DensityMatrix]]
        counts: list[dict]
        samples: list[Union[list, npt.NDArray]]
        populations: list[dict]
        iq_data: list[Union[list, npt.NDArray]]
        avg_iq_data: list[ Union[list, npt.NDArray]]

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
        provider = IBMProvider()
        backend = provider.get_backend(backend_name)
        config = backend.configuration()
        if not isinstance(config, PulseBackendConfiguration):
            raise CasqError(
                f"Backend configuration must be of type 'qiskit.providers.models.PulseBackendConfiguration'."
            )
        num_qubits = config.num_qubits
        hamiltonian = config.hamiltonian
        dt = config.dt
        if qubits is None:
            qubits = list(range(num_qubits))
        else:
            qubits = sorted(qubits)
        if qubits[-1] >= num_qubits:
            raise CasqError(
                f"Selected qubit {qubits[-1]} is out of bounds for backend {backend} with {num_qubits} qubits."
            )
        # hamiltonian
        (
            static_hamiltonian,
            hamiltonian_operators,
            hamiltonian_channels,
            qubit_dims,
        ) = parse_backend_hamiltonian_dict(hamiltonian, qubits)
        # channel frequencies
        drive_channels = []
        control_channels = []
        measure_channels = []
        for channel in hamiltonian_channels:
            if channel[0] == "d":
                drive_channels.append(channel)
            elif channel[0] == "u":
                control_channels.append(channel)
            elif channel[0] == "m":
                measure_channels.append(channel)
            else:
                raise CasqError(f"Unrecognized channel type {channel[0]} requested.")
        channel_freqs = {}
        defaults = backend.defaults() if hasattr(backend, "defaults") else None
        if defaults is None:
            raise CasqError("DriveChannels in model but frequencies not available in target or defaults.")
        else:
            drive_frequencies = defaults.qubit_freq_est
        for channel in drive_channels:
            idx = int(channel[1:])
            if idx >= len(drive_frequencies):
                raise CasqError(f"DriveChannel index {idx} is out of bounds.")
            channel_freqs[channel] = drive_frequencies[idx]
        control_channel_lo = config.u_channel_lo
        for channel in control_channels:
            idx = int(channel[1:])
            if idx >= len(control_channel_lo):
                raise CasqError(f"ControlChannel index {idx} is out of bounds.")
            freq = 0.0
            for channel_lo in control_channel_lo[idx]:
                freq += drive_frequencies[channel_lo.q] * channel_lo.scale
            channel_freqs[channel] = freq
        if measure_channels:
            if defaults is None:
                raise CasqError("MeasureChannels in model but frequencies not available in target or defaults.")
            else:
                measure_frequencies = defaults.meas_freq_est
                for channel in measure_channels:
                    idx = int(channel[1:])
                    if idx >= len(measure_frequencies):
                        raise CasqError(f"MeasureChannel index {idx} is out of bounds.")
                    channel_freqs[channel] = measure_frequencies[idx]
        for channel in hamiltonian_channels:
            if channel not in channel_freqs:
                raise CasqError(f"No carrier frequency found for channel {channel}.")
        # rotating frame
        if rotating_frame == "auto":
            if "dense" in evaluation_mode:
                rotating_frame = static_hamiltonian
            else:
                rotating_frame = np.diag(static_hamiltonian)
        return cls(
            static_hamiltonian=Array(static_hamiltonian),
            hamiltonian_operators=Array(hamiltonian_operators),
            hamiltonian_channels=hamiltonian_channels,
            channel_carrier_freqs=channel_freqs,
            rotating_frame=rotating_frame,
            evaluation_mode=evaluation_mode,
            rwa_cutoff_freq=rwa_cutoff_freq,
            dt=dt,
            backend=backend,
            qubits=qubit_dims,
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
        seed: Optional[int] = None
    ):
        super().__init__(
            static_hamiltonian, hamiltonian_operators, static_dissipators, dissipator_operators,
            hamiltonian_channels, dissipator_channels, channel_carrier_freqs, dt,
            rotating_frame, in_frame_basis, evaluation_mode,
            rwa_cutoff_freq, rwa_carrier_freqs, validate
        )
        self._backend = backend
        self.qubits = qubits
        self.seed = seed
        # self.rng = np.random.default_rng(seed)
        self._dressed_evals, self._dressed_states = _get_dressed_state_decomposition(static_hamiltonian.data)
        self._dressed_states_adjoint = self._dressed_states.conj().transpose()

    def solve(
        self,
        run_input: Union[QuantumCircuit, Schedule, ScheduleBlock],
        initial_state: Optional[Union[Statevector, DensityMatrix]] = None,
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
        elif isinstance(run_input, QuantumCircuit):
            num_memory_slots = run_input.cregs[0].size
            schedule = build_schedule(run_input, self._backend, dt=self._dt)
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
        t_span = (0.0, self._dt * schedule_acquire_times[0])
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
        if steps:
            t_eval = np.linspace(t_span[0], t_span[1], steps)
            t_eval[0] = t_span[0]
            t_eval[-1] = t_span[1]
        else:
            t_eval = None
        result = super().solve(Array(t_span), initial_state, [schedule], convert_results, t_eval=t_eval, **kwargs)[0]
        return self._build_solution(
            result, measurement_subsystems, memory_slot_indices, num_memory_slots, normalize_states, shots
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
        for t, y in zip(result.t, result.y):
            # Take state out of frame, put in dressed basis, and normalize
            if isinstance(y, Statevector):
                y = np.array(self.model.rotating_frame.state_out_of_frame(t=t, y=y))
                y = self._dressed_states_adjoint @ y
                y = Statevector(y, dims=qubit_dims)
                if normalize_states:
                    y = y / np.linalg.norm(y.data)
            elif isinstance(y, DensityMatrix):
                y = np.array(
                    self.model.rotating_frame.operator_out_of_frame(t=t, operator=y)
                )
                y = self._dressed_states_adjoint @ y @ self._dressed_states
                y = DensityMatrix(y, dims=qubit_dims)
                if normalize_states:
                    y = y / np.diag(y.data).sum()
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
                theta = 2 * np.pi / sub_dim
                iq_centers.append(
                    [[np.cos(idx * theta), np.sin(idx * theta)] for idx in range(sub_dim)]
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
            avg_iq_data_step = np.average(iq_data_step, axis=0)
            avg_iq_data.append(avg_iq_data_step)
        return PulseSolver.Solution(
            time=result.t, states=result.y, counts=counts, samples=samples,
            populations=populations, iq_data=iq_data, avg_iq_data=avg_iq_data
        )
