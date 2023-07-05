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
"""Gaussian square pulse gate."""
from __future__ import annotations

from typing import Optional

from loguru import logger
from qiskit.pulse import GaussianSquare
from qiskit.pulse.library import Pulse

# noinspection PyProtectedMember
from qiskit.pulse.library.symbolic_pulses import ScalableSymbolicPulse, _lifted_gaussian
import sympy as sym

from casq.circuit.pulse_gate import PulseGate
from casq.common import trace


class GaussianSquarePulseGate(PulseGate):
    """GaussianSquarePulseGate class.

    Note: Currently only single qubit gates are supported.

    Args:
        duration: Pulse length in terms of the sampling period dt.
        amplitude: The magnitude of the amplitude of the Gaussian and square pulse.
        sigma: A measure of how wide or narrow the Gaussian risefall is,
            i.e. its standard deviation.
        width: The duration of the embedded square pulse.
        angle: The angle of the complex amplitude of the pulse. Default value 0.
        risefall_sigma_ratio: The ratio of each risefall duration to sigma.
        limit_amplitude: If True, then limit the amplitude of the waveform to 1.
            The default is True and the amplitude is constrained to 1.
        jax: If True, use JAX-enabled implementation.
        name: Optional display name for the pulse gate.
    """

    @trace()
    def __init__(
        self,
        duration: int,
        amplitude: float,
        sigma: float,
        width: float,
        angle: float = 0,
        risefall_sigma_ratio: Optional[float] = None,
        limit_amplitude: bool = True,
        jax: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """Initialize GaussianPulseGate."""
        super().__init__(1, duration, jax, name)
        self.amplitude = amplitude
        self.sigma = sigma
        self.width = width
        self.angle = angle
        self.risefall_sigma_ratio = risefall_sigma_ratio
        self.limit_amplitude = limit_amplitude

    @trace()
    def pulse(self) -> Pulse:
        """GaussianPulseGate.pulse method.

        Builds pulse for pulse gate.

        Returns:
            :py:class:`qiskit.pulse.library.Pulse`
        """
        if self.jax:
            _t, _duration, _amp, _sigma, _width, _angle = sym.symbols(
                "t, duration, amp, sigma, width, angle"
            )
            _center = _duration / 2
            _sq_t0 = _center - _width / 2
            _sq_t1 = _center + _width / 2
            _gaussian_ledge = _lifted_gaussian(_t, _sq_t0, -1, _sigma)
            _gaussian_redge = _lifted_gaussian(_t, _sq_t1, _duration + 1, _sigma)
            envelope_expr = (
                _amp
                * sym.exp(sym.I * _angle)
                * sym.Piecewise(
                    (_gaussian_ledge, _t <= _sq_t0),
                    (_gaussian_redge, _t >= _sq_t1),
                    (1, True),
                )
            )
            # noinspection PyTypeChecker
            # Suppress warning for constraints argument in ScalableSymbolicPulse
            return ScalableSymbolicPulse(
                pulse_type="GaussianSquare",
                duration=self.duration,
                amp=self.amplitude,
                angle=self.angle,
                parameters={"sigma": self.sigma, "width": self.width},
                envelope=envelope_expr,
                constraints=sym.And(_sigma > 0, _width >= 0, _duration >= _width),
                valid_amp_conditions=sym.Abs(_amp) <= 1.0,
                name=self.name,
            )
        else:
            return GaussianSquare(
                duration=self.duration,
                amp=self.amplitude,
                sigma=self.sigma,
                width=self.width,
                angle=self.angle,
                risefall_sigma_ratio=self.risefall_sigma_ratio,
                limit_amplitude=self.limit_amplitude,
                name=self.name,
            )
