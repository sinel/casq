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
"""Drag pulse gate."""
from __future__ import annotations

from typing import Optional

from qiskit.pulse import Drag
from qiskit.pulse.library import Pulse

# from qiskit.pulse.library.symbolic_pulses import ScalableSymbolicPulse
import sympy as sym

from casq.common import trace
from casq.gates.pulse_gate import PulseGate


class DragPulseGate(PulseGate):
    """DragPulseGate class.

    Note: Currently only single qubit gates are supported.

    Args:
        duration: Pulse length in terms of the sampling period dt.
        amplitude: The magnitude of the amplitude of the Gaussian and square pulse.
        sigma: A measure of how wide or narrow the Gaussian risefall is,
            i.e. its standard deviation.
        beta: The correction amplitude.
        angle: The angle of the complex amplitude of the pulse. Default value 0.
        limit_amplitude: If True, then limit the amplitude of the waveform to 1.
            The default is True and the amplitude is constrained to 1.
        jax: If True, use JAX-enabled implementation.
        name: Optional display name for the pulse gate.
    """

    # Helper function that returns a lifted Gaussian symbolic equation.
    @staticmethod
    def lifted_drag(
        t: sym.Symbol, center: sym.Symbol, t_zero: sym.Symbol, sigma: sym.Symbol
    ) -> sym.Expr:
        """Helper function that returns a lifted Gaussian drag symbolic equation.

        Args:
            t: Symbol object representing time.
            center: Symbol or expression representing the middle point of the samples.
            t_zero: The value of t at which the pulse is lowered to 0.
            sigma: Symbol or expression representing Gaussian sigma.

        Returns:
            Symbolic equation.
        """
        t_shifted = (t - center).expand()
        t_offset = (t_zero - center).expand()
        gauss = sym.exp(-((t_shifted / sigma) ** 2) / 2)
        offset = sym.exp(-((t_offset / sigma) ** 2) / 2)
        expression: sym.Expr = (gauss - offset) / (1 - offset)
        return expression

    @trace()
    def __init__(
        self,
        duration: int,
        amplitude: float,
        sigma: float,
        beta: float,
        angle: Optional[float] = None,
        limit_amplitude: bool = True,
        jax: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """Initialize DragPulseGate."""
        super().__init__(1, duration, jax, name)
        self.amplitude = amplitude
        self.sigma = sigma
        self.beta = beta
        self.angle = angle
        self.limit_amplitude = limit_amplitude

    @trace()
    def pulse(self) -> Pulse:
        """GaussianPulseGate.pulse method.

        Builds pulse for pulse gate.

        Returns:
            :py:class:`qiskit.pulse.library.Pulse`
        """
        return Drag(
            duration=self.duration,
            amp=self.amplitude,
            sigma=self.sigma,
            beta=self.beta,
            angle=self.angle,
            limit_amplitude=self.limit_amplitude,
            name=self.name,
        )
