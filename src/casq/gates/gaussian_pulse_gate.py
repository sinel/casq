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

from typing import Optional

from qiskit import pulse

from casq.common.decorators import trace
from casq.gates.pulse_gate import PulseGate


class GaussianPulseGate(PulseGate):
    """GaussianPulseGate class.

    Args:
        name: Optional user-friendly name for pulse gate.
    """

    @trace()
    def __init__(
        self,
        duration: int,
        amplitude: float,
        sigma: float,
        name: Optional[str] = None,
    ) -> None:
        """Initialize GaussianPulseGate."""
        super().__init__(name)
        self.duration = duration
        self.amplitude = amplitude
        self.sigma = sigma

    @trace()
    def schedule(self, qubit: int) -> pulse.ScheduleBlock:
        """GaussianPulseGate.schedule method.

        Builds schedule block for pulse gate.

        Args:
            qubit: Index of qubit to drive.

        Returns:
            :py:class:`qiskit.pulse.ScheduleBlock`
        """
        if self._schedule:
            return self._schedule
        else:
            with pulse.build() as sb:
                pulse.play(
                    pulse.library.Gaussian(
                        duration=self.duration,
                        amp=self.amplitude,
                        sigma=self.sigma,
                        name=self.ufid,
                    ),
                    pulse.DriveChannel(qubit),
                )
            self._schedule = sb
            return sb
