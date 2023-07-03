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
from typing import Any, Callable, NamedTuple, Optional, Union

from jax import jit, value_and_grad
from loguru import logger
import numpy.typing as npt
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.quantum_info.analysis import hellinger_fidelity
from scipy.optimize import (
    minimize, Bounds, HessianUpdateStrategy, LinearConstraint, NonlinearConstraint
)

from casq.pulse_simulator import PulseSimulator
from casq.common import trace, CasqError
from casq.circuit import (
    PulseCircuit, PulseGate,
    DragPulseGate, GaussianPulseGate, GaussianSquarePulseGate
)


class PulseOptimizer:
    """PulseOptimizer class."""

    class PulseType(Enum):
        DRAG = 0
        GAUSSIAN = 1
        GAUSSIAN_SQUARE = 2

    class FidelityType(Enum):
        COUNTS = 0

    class OptimizationMethod(str, Enum):
        BFGS = "BFGS"
        CG = "CG"
        COBYLA = "COBYLA"
        DOGLEG = "dogleg"
        L_BFGS_B = "L-BFGS-B"
        NEWTON_CG = "Newton-CG"
        NELDER_MEAD = "Nelder-Mead"
        POWELL = "Powell"
        SLSQP = "SLSQP"
        TNC = "TNC"
        TRUST_CONSTR = "trust-constr"
        TRUST_EXACT = "trust-exact"
        TRUST_KRYLOV = "trust-krylov"
        TRUST_NCG = "trust-ncg"

    class FiniteDifferenceScheme(str, Enum):
        CS = "cs"
        TWO_POINT = "2-point"
        THREE_POINT = "3-point"

    class Solution(NamedTuple):
        num_iterations: int
        parameters: list[float]
        measurement: Union[dict[str, float], DensityMatrix, Statevector]
        fidelity: float
        gate: PulseGate
        circuit: PulseCircuit
        message: str

    @trace()
    def __init__(
            self,
            pulse_type: PulseType,
            pulse_arguments: dict[str, Any],
            simulator: PulseSimulator,
            target_measurement: Union[dict[str, float], DensityMatrix, Statevector],
            fidelity_type: Optional[FidelityType] = None,
            target_qubit: Optional[int] = None,
            use_jax: bool = False,
            use_jit: bool = False
    ):
        """Instantiate :class:`~casq.PulseOptimizer`.

        Args:
            pulse_type: Pulse type.
            pulse_arguments: Dict containing pulse arguments.
                Use None values for parameters,
                and actual values for fixed arguments.
            simulator: Pulse simulator.
            target_measurement: Target measurement against which fidelity will be calculated.
            fidelity_type: Fidelity type. Defaults to FidelityType.COUNTS.
            target_qubit: Qubit to drive with pulse. Defaults to first qubit in simulator.
            use_jax: If True, then ODE solver method must be jax-compatible
                and jax-compatible pulse is constructed.
            use_jit: If True, then jit and value_and_grad is applied to objective function.
        """
        self.pulse_type = pulse_type
        self.pulse_arguments = pulse_arguments
        self.simulator = simulator
        self.target_measurement = target_measurement
        self.fidelity_type = PulseOptimizer.FidelityType.COUNTS if fidelity_type is None else fidelity_type
        self.target_qubit = target_qubit if target_qubit else self.simulator.qubits[0]
        self.use_jax = use_jax
        self.use_jit = use_jit
        if self.use_jax and self.simulator.method not in [
            PulseSimulator.ODESolverMethod.QISKIT_DYNAMICS_JAX_RK4,
            PulseSimulator.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT
        ]:
            raise CasqError(f"If 'jax' is enabled, a jax-compatible ODE solver method is required.")
        self.pulse_function = self._build_pulse_function()
        self.objective_function = self._build_objective_function()

    def optimize(
        self,
        params: npt.NDArray,
        method: OptimizationMethod,
        jac: Optional[Union[bool, FiniteDifferenceScheme, Callable]] = None,
        hess: Optional[Union[FiniteDifferenceScheme, HessianUpdateStrategy, Callable]] = None,
        hessp: Optional[Callable] = None,
        bounds: Optional[Union[list, Bounds]] = None,
        constraints: Optional[
            Union[
                dict, list[dict],
                LinearConstraint, list[LinearConstraint],
                NonlinearConstraint, list[NonlinearConstraint]
            ]
        ] = None,
        tol: Optional[float] = None,
        maxiter: Optional[int] = None,
        verbose: bool = True,
        callback: Optional[Callable] = None
    ) -> PulseOptimizer.Solution:
        """PulseOptimizer.optimize method.

        Optimize pulse.
        This is basically a wrapper around scipy.optimize.minimize.
        For more details, see
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        Args:
            params: Pulse parameters.
            method: Optimization method.
            jac: Method for computing the gradient vector.
                Only for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP,
                dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr.
            hess: Method for computing the Hessian matrix.
                Only for Newton-CG, dogleg, trust-ncg, trust-krylov,
                trust-exact and trust-constr.
            hessp: Hessian of objective function times an arbitrary vector p.
                Only for Newton-CG, trust-ncg, trust-krylov, trust-constr.
                Only one of hessp or hess needs to be given.
                If hess is provided, then hessp will be ignored.
            bounds: Bounds on variables for Nelder-Mead, L-BFGS-B, TNC,
                SLSQP, Powell, trust-constr, and COBYLA methods.
            constraints: Constraints definition.
                Only for COBYLA, SLSQP and trust-constr.
            tol: Tolerance for termination.
            maxiter: Maximum number of iterations to perform.
            verbose: If True, print convergence messages.
            callback: A callable called after each iteration.

        Returns:
            :py:class:`casq.PulseOptimizer.Solution`
        """
        options = {"disp": verbose}
        if maxiter:
            options.update(maxiter=maxiter)
        opt_results = minimize(
            fun=self.objective_function, x0=params,
            method=PulseOptimizer.OptimizationMethod.NELDER_MEAD,
            jac=jac, hess=hess, hessp=hessp,
            bounds=bounds, constraints=constraints, tol=tol,
            options=options, callback=callback
        )
        gate = self.pulse_function(opt_results.x)
        circuit = PulseCircuit.from_pulse(gate, self.simulator.backend, self.target_qubit)
        counts = self.simulator.run([circuit]).result().results[-1].data.counts[-1]
        return PulseOptimizer.Solution(
            num_iterations=opt_results.nfev, parameters=opt_results.x,
            measurement=counts, fidelity=1-opt_results.fun,
            gate=gate, circuit=circuit, message=opt_results.message
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
                ).result().results[0]
            )
            counts = result.data.counts[-1]
            fidelity = hellinger_fidelity(self.target_measurement, counts)
            logger.debug(f"PARAMETERS: {params} TARGET: {counts} OBJECTIVE: {1.0 - fidelity}")
            return 1.0 - fidelity

        if self.use_jit:
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
        fixed.update(jax=self.use_jax)
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