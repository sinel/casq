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
from typing import Any, Optional

import numpy.typing as npt
from qiskit.circuit import Gate
from qiskit.pulse import DriveChannel, Schedule, build, play
from qiskit.pulse.library import ScalableSymbolicPulse
from qiskit.pulse.transforms import block_to_schedule

from casq.common.decorators import trace
from casq.common.helpers import dbid, ufid


class PulseGate(Gate):
    """PulseGate class.

    Abstract base class for all pulse gates.
    Note: Currently only single qubit gates are supported.

    Args:
        num_qubits: The number of qubits the gate acts on.
        duration: Pulse length in terms of the sampling period dt.
        amplitude: The magnitude of the amplitude of the pulse.
        angle: The angle of the complex amplitude of the pulse. Default value 0.
        limit_amplitude: If ``True``, then limit the amplitude of the waveform to 1.
            The default is ``True`` and the amplitude is constrained to 1.
        name: Optional display name for the pulse gate.
    """

    @trace()
    def __init__(
        self,
        num_qubits: int,
        duration: int,
        amplitude: float,
        angle: float = 0,
        limit_amplitude: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """Initialize PulseGate."""
        self.dbid = dbid()
        self.ufid = name if name else ufid(self)
        super().__init__(self.ufid, num_qubits, [], self.ufid)
        self.duration = duration
        self.amplitude = amplitude
        self.angle = angle
        self.limit_amplitude = limit_amplitude

    @abstractmethod
    def pulse(self, params: dict[str, Any]) -> ScalableSymbolicPulse:
        """PulseGate.pulse method.

        Builds pulse for pulse gate.

        Returns:
            :py:class:`qiskit.pulse.library.ScalableSymbolicPulse`
        """

    @abstractmethod
    def to_parameters_dict(self, parameters: npt.NDArray) -> dict[str, Any]:
        """GaussianSquarePulseGate.to_parameters_dict method.

        Builds parameter dictionary from array of parameter values.

        Args:
            parameters: Array of pulse parameter values in order [sigma, width].

        Returns:
            Dictionary of parameters.
        """

    def schedule(self, parameters: dict[str, Any], qubit: int) -> Schedule:
        """PulseGate.schedule method.

        Builds schedule to run pulse gate for testing or solitary optimization.

        Args:
            parameters: Dictionary of pulse parameters that defines the pulse envelope.
            qubit: Qubit to attach gate instruction to.

        Returns:
            :py:class:`qiskit.pulse.Schedule`
        """
        schedule_name = f"{self.name}Schedule"
        with build(name=schedule_name) as sb:
            play(self.pulse(parameters), DriveChannel(qubit))
        return block_to_schedule(sb)
