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
"""Pulse gate."""
from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Union

from qiskit.circuit import Gate
from qiskit.providers import BackendV1
from qiskit.pulse import DriveChannel, Schedule, align_sequential, build, measure, play
from qiskit.pulse.library import Pulse
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit_dynamics import Signal

from casq.backends.qiskit.backend_characteristics import BackendCharacteristics
from casq.common.decorators import trace
from casq.common.exceptions import CasqError
from casq.common.helpers import dbid, discretize, ufid
from casq.common.plotting import plot_signal


class PulseGate(Gate):
    """PulseGate class.

    Abstract base class for all pulse gates.
    Note: Currently only single qubit gates are supported.

    Args:
        num_qubits: The number of qubits the gate acts on.
        duration: Pulse length in terms of the sampling period dt.
        name: Optional display name for the pulse gate.
    """

    @trace()
    def __init__(
        self,
        num_qubits: int,
        duration: int,
        name: Optional[str] = None,
    ) -> None:
        """Initialize PulseGate."""
        self.dbid = dbid()
        self.ufid = name if name else ufid(self)
        super().__init__(self.ufid, num_qubits, [], self.ufid)
        self.duration = duration

    @abstractmethod
    def pulse(self) -> Pulse:
        """PulseGate.pulse method.

        Builds pulse for pulse gate.

        Returns:
            :py:class:`qiskit.pulse.library.Pulse`
        """

    def schedule(
        self,
        qubit: int,
        backend: Optional[BackendV1] = None,
        dt: Optional[float] = None,
        channel_frequencies: Optional[dict[str, float]] = None,
        measured: bool = False,
        discretized: bool = False,
    ) -> Union[Schedule, list[Signal]]:
        """PulseGate.schedule method.

        Builds schedule to run pulse gate for testing or solitary optimization.

        Args:
            qubit: Qubit to attach gate instruction to.
            backend: Optional IBMQ backend. Required if building a measured schedule.
            dt: Optional time interval.
            channel_frequencies: Optional channel frequencies.
            measured: If True, convert schedule into discretized list of signals.
            discretized: If True, convert schedule into discretized list of signals.

        Returns:
            :py:class:`qiskit.pulse.Schedule`
            or list of :py:class:`qiskit_dynamics.signals.Signal`
        """
        schedule_name = f"{self.name}Schedule"
        if measured:
            if backend:
                with build(backend=backend, name=schedule_name) as sb:
                    with align_sequential():
                        play(self.pulse(), DriveChannel(qubit))
                        measure(qubit)
            else:
                raise CasqError(
                    "Backend is required for building schedules with measurements."
                )
        else:
            with build(name=schedule_name) as sb:
                play(self.pulse(), DriveChannel(qubit))
        sched = block_to_schedule(sb)
        if discretized:
            if backend:
                props = BackendCharacteristics(backend)
                channel_frequencies_from_backend: dict[
                    str, float
                ] = props.get_channel_frequencies(list(sched.channels))
                return discretize(sched, props.dt, channel_frequencies_from_backend)
            elif dt and channel_frequencies:
                return discretize(sched, dt, channel_frequencies)
            else:
                raise CasqError(
                    "Cannot discretize pulse schedule if neither backend "
                    "nor required properties (dt and channel frequencies) are provided."
                )
        else:
            return sched

    @trace()
    def draw_signal(
        self,
        qubit: int,
        dt: float,
        carrier_frequency: float,
        filename: Optional[str] = None,
        hidden: bool = False,
    ) -> None:
        """PulseGate.draw_signal method.

        Draws pulse gate signal.

        Args:
            qubit: Qubit to attach gate to.
            dt: Sample time length.
            carrier_frequency: Carrier frequency.
            filename: Saves figure to specified path if provided.
            hidden: Does not show figure if True.

        Returns:
            :py:class:`qiskit.QuantumCircuit`
        """
        plot_signal(
            self.schedule(qubit),
            dt,
            f"d{qubit}",
            carrier_frequency,
            self.duration,
            filename=filename,
            hidden=hidden,
        )
