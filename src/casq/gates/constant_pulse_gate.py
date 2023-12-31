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
"""Constant pulse gate."""
from __future__ import annotations

from typing import Any, Optional

import numpy.typing as npt
from qiskit.pulse.library import Constant, ScalableSymbolicPulse

from casq.common.decorators import trace
from casq.gates.pulse_gate import PulseGate


class ConstantPulseGate(PulseGate):
    """ConstantPulseGate class.

    Note: Currently only single qubit gates are supported.

    Args:
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
        duration: int,
        amplitude: float,
        angle: float = 0,
        limit_amplitude: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """Initialize ConstantPulseGate."""
        super().__init__(1, duration, amplitude, angle, limit_amplitude, name)

    @trace()
    def pulse(self, params: Optional[dict[str, Any]] = None) -> ScalableSymbolicPulse:
        """ConstantPulseGate.pulse method.

        Builds pulse for pulse gate.

        Returns:
            :py:class:`qiskit.pulse.library.ScalableSymbolicPulse`
        """
        return Constant(
            duration=self.duration,
            amp=self.amplitude,
            angle=self.angle,
            limit_amplitude=self.limit_amplitude,
            name=self.name,
        )

    @trace()
    def to_parameters_dict(
        self, parameters: Optional[npt.NDArray] = None
    ) -> Optional[dict[str, Any]]:
        """GaussianSquarePulseGate.to_parameters_dict method.

        Builds parameter dictionary from array of parameter values.

        Args:
            parameters: Array of pulse parameter values in order [sigma, width].

        Returns:
            Dictionary of parameters.
        """
        return None
