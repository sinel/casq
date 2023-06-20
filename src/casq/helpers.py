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

from typing import Optional, Union


import numpy as np
from qiskit_ibm_provider import IBMProvider
from qiskit.providers.models import PulseBackendConfiguration
from qiskit_dynamics import RotatingFrame
from qiskit_dynamics.array import Array
from qiskit_dynamics.backend import parse_backend_hamiltonian_dict
from qiskit_dynamics import Solver

from qiskit import QiskitError, QuantumCircuit
from qiskit.pulse import Acquire, Schedule, ScheduleBlock
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit import schedule as build_schedule

from casq.common import trace, CasqError


@trace()
def solver_from_backend(
    backend_name: str,
    qubits: Optional[list[int]] = None,
    rotating_frame: Optional[Union[Array, RotatingFrame, str]] = "auto",
    evaluation_mode: str = "dense",
    rwa_cutoff_freq: Optional[float] = None,
) -> Solver:
    """Build solver from IBMQ backend.

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
            raise QiskitError(f"DriveChannel index {idx} is out of bounds.")
        channel_freqs[channel] = drive_frequencies[idx]
    control_channel_lo = config.u_channel_lo
    for channel in control_channels:
        idx = int(channel[1:])
        if idx >= len(control_channel_lo):
            raise QiskitError(f"ControlChannel index {idx} is out of bounds.")
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
                    raise QiskitError(f"MeasureChannel index {idx} is out of bounds.")
                channel_freqs[channel] = measure_frequencies[idx]
    for channel in hamiltonian_channels:
        if channel not in channel_freqs:
            raise QiskitError(f"No carrier frequency found for channel {channel}.")
    # rotating frame
    if rotating_frame == "auto":
        if "dense" in evaluation_mode:
            rotating_frame = static_hamiltonian
        else:
            rotating_frame = np.diag(static_hamiltonian)
    return Solver(
        static_hamiltonian=Array(static_hamiltonian),
        hamiltonian_operators=Array(hamiltonian_operators),
        hamiltonian_channels=hamiltonian_channels,
        channel_carrier_freqs=channel_freqs,
        rotating_frame=rotating_frame,
        evaluation_mode=evaluation_mode,
        rwa_cutoff_freq=rwa_cutoff_freq,
        dt=dt
    )


@trace()
def time_interval(
    run_input: Union[QuantumCircuit, Schedule, ScheduleBlock],
    backend_name: str,
    dt: float
) -> list[float]:
    provider = IBMProvider()
    backend = provider.get_backend(backend_name)
    if isinstance(run_input, ScheduleBlock):
        schedule = block_to_schedule(run_input)
    elif isinstance(run_input, Schedule):
        schedule = run_input
    elif isinstance(run_input, QuantumCircuit):
        schedule = build_schedule(run_input, backend, dt=dt)
    else:
        raise QiskitError(f"Type {type(run_input)} cannot be converted to Schedule.")
    schedule_acquire_times = []
    for start_time, inst in schedule.instructions:
        # only track acquires saving in a memory slot
        if isinstance(inst, Acquire) and inst.mem_slot is not None:
            schedule_acquire_times.append(start_time)
    if len(schedule_acquire_times) == 0:
        raise CasqError(
            "At least one measurement saving to a MemorySlot required in each schedule."
        )
    for acquire_time in schedule_acquire_times[1:]:
        if acquire_time != schedule_acquire_times[0]:
            raise CasqError("Only support for measurements at one time.")
    return [0.0, dt * schedule_acquire_times[0]]
