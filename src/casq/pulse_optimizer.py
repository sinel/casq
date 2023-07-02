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
"""Pulse circuit."""
from __future__ import annotations

from enum import Enum
from functools import partial
from typing import cast, Any, Callable, NamedTuple, Optional, Union

from jax import jit, value_and_grad
from loguru import logger
import numpy.typing as npt
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.quantum_info.analysis import hellinger_fidelity
from scipy.optimize import minimize

from casq.pulse_circuit import PulseCircuit
from casq.pulse_simulator import PulseSimulator
from casq.common import trace, CasqError
from casq.gates import PulseGate, DragPulseGate, GaussianPulseGate, GaussianSquarePulseGate


class PulseOptimizer:
    """PulseOptimizer class."""

    class PulseType(Enum):
        DRAG = 0
        GAUSSIAN = 1
        GAUSSIAN_SQUARE = 2

    class FidelityType(Enum):
        COUNTS = 0

    class Solution(NamedTuple):
        num_iterations: int
        parameters: list[float]
        fidelity: float
        message: str

    @trace()
    def __init__(
            self,
            pulse_type: PulseType,
            pulse_arguments: dict[str, Any],
            simulator: PulseSimulator,
            target_fidelity: Union[dict[str, float], DensityMatrix, Statevector],
            fidelity_type: Optional[FidelityType] = None,
            target_qubit: Optional[int] = None,
            jax: bool = False,
            jit: bool = False
    ):
        """Instantiate :class:`~casq.PulseOptimizer`.

        Args:
            pulse_type: Pulse type.
            pulse_arguments: Dict containing pulse arguments.
                Use None values for parameters,
                and actual values for fixed arguments.
            simulator: Pulse simulator.
            target_fidelity: Fidelity target.
            fidelity_type: Fidelity type. Defaults to FidelityType.COUNTS.
            target_qubit: Qubit to drive with pulse. Defaults to first qubit in simulator.
            jax: If True, then ODE solver method must be jax-compatible
                and jax-compatible pulse is constructed.
            jit: If True, then jit and value_and_grad is applied to objective function.
        """
        self.pulse_type = pulse_type
        self.pulse_arguments = pulse_arguments
        self.simulator = simulator
        self.target_fidelity = target_fidelity
        self.fidelity_type = PulseOptimizer.FidelityType.COUNTS if fidelity_type is None else fidelity_type
        self.target_qubit = target_qubit if target_qubit else self.simulator.qubits[0]
        self.jax = jax
        self.jit = jit
        if self.jax and self.simulator.method not in [
            PulseSimulator.ODESolverMethod.QISKIT_DYNAMICS_JAX_RK4,
            PulseSimulator.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT
        ]:
            raise CasqError(f"If 'jax' is enabled, a jax-compatible ODE solver method is required.")
        self.pulse_function = self._build_pulse_function()
        self.objective_function = self._build_objective_function()

    def optimize(self, params: npt.NDArray) -> PulseOptimizer.Solution:
        """PulseOptimizer.optimize method.

        Optimize pulse.

        Args:
            params: Pulse parameters.

        Returns:
            :py:class:`casq.PulseOptimizer.Solution`
        """
        opt_results = minimize(
            fun=self.objective_function, x0=params, jac=False, method="Nelder-Mead"
        )
        return PulseOptimizer.Solution(
            num_iterations=opt_results.nfev, parameters=opt_results.x,
            fidelity=opt_results.fun, message=opt_results.message
        )

    def _build_objective_function(self) -> Callable[[npt.NDArray], float]:
        """PulseOptimizer._build_objective_function method.

        Build objective function to minimize.

        Returns:
            Objective function.
        """
        def objective(params: npt.NDArray):
            p = self.pulse_function(params)
            circuit = PulseCircuit.from_pulse(
                p, self.simulator.backend, self.target_qubit
            )
            result = (
                self.simulator.run(
                    run_input=[circuit],
                ).result().results[-1]
            )
            logger.debug(f"Result = {result}")
            counts = result.data.counts[-1]
            fidelity = hellinger_fidelity(self.target_fidelity, counts)
            logger.debug(f"Counts = {counts} -> Fidelity = {fidelity}")
            return 1.0 - fidelity

        if self.jit:
            return jit(value_and_grad(objective))
        else:
            return objective

    def _build_pulse_function(self) -> Callable[[npt.NDArray], PulseGate]:
        """PulseOptimizer._build_pulse_function method.

        Build pulse function to construct pulse gate.

        Returns:
            Pulse function.
        """
        fixed = {}
        parameters = []
        for key, value in self.pulse_arguments.items():
            if value is None:
                parameters.append(key)
            else:
                fixed[key] = value
        fixed.update(jax=self.jax)
        logger.debug(f"Building pulse with parameters = {parameters} and fixed arguments = {fixed}")
        if self.pulse_type is PulseOptimizer.PulseType.DRAG:
            gate = partial(DragPulseGate, **fixed)
        elif self.pulse_type is PulseOptimizer.PulseType.GAUSSIAN_SQUARE:
            gate = partial(GaussianSquarePulseGate, **fixed)
        else:
            gate = partial(GaussianPulseGate, **fixed)

        def pulse(params: npt.NDArray):
            args = {}
            for i, name in enumerate(parameters):
                args[name] = params[i]
            return gate(**args)

        return pulse
