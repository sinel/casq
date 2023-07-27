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
from typing import Any, Callable, NamedTuple, Optional, Union

from jax import jit, value_and_grad
from loguru import logger
import numpy.typing as npt
from qiskit.pulse import ScalableSymbolicPulse
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.quantum_info.analysis import hellinger_fidelity
from scipy.optimize import (
    Bounds,
    HessianUpdateStrategy,
    LinearConstraint,
    NonlinearConstraint,
    minimize,
)

from casq.backends.pulse_backend import PulseBackend
from casq.common import CasqError, is_jax_enabled, timer, trace
from casq.gates import PulseCircuit, PulseGate


class PulseOptimizer:
    """PulseOptimizer class."""

    class PulseType(Enum):
        """Pulse type."""

        DRAG = 0
        GAUSSIAN = 1
        GAUSSIAN_SQUARE = 2

    class FidelityType(Enum):
        """Fidelity type."""

        COUNTS = 0

    class OptimizationMethod(str, Enum):
        """Optimization methods."""

        SCIPY_BFGS = "BFGS"
        SCIPY_CG = "CG"
        SCIPY_COBYLA = "COBYLA"
        SCIPY_DOGLEG = "dogleg"
        SCIPY_L_BFGS_B = "L-BFGS-B"
        SCIPY_NEWTON_CG = "Newton-CG"
        SCIPY_NELDER_MEAD = "Nelder-Mead"
        SCIPY_POWELL = "Powell"
        SCIPY_SLSQP = "SLSQP"
        SCIPY_TNC = "TNC"
        SCIPY_TRUST_CONSTR = "trust-constr"
        SCIPY_TRUST_EXACT = "trust-exact"
        SCIPY_TRUST_KRYLOV = "trust-krylov"
        SCIPY_TRUST_NCG = "trust-ncg"

    class FiniteDifferenceScheme(str, Enum):
        """Finite difference scheme."""

        CS = "cs"
        TWO_POINT = "2-point"
        THREE_POINT = "3-point"

    class Solution(NamedTuple):
        """Pulse optimizer solution."""

        initial_parameters: dict[str, Any]
        initial_pulse: ScalableSymbolicPulse
        num_iterations: int
        parameters: dict[str, Any]
        measurement: Union[dict[str, float], DensityMatrix, Statevector]
        fidelity: float
        pulse: ScalableSymbolicPulse
        gate: PulseGate
        circuit: PulseCircuit
        message: str

    @trace()
    def __init__(
        self,
        pulse_gate: PulseGate,
        pulse_backend: PulseBackend,
        method: PulseBackend.ODESolverMethod,
        target_measurement: Union[dict[str, float], DensityMatrix, Statevector],
        method_options: Optional[dict[str, Any]] = None,
        fidelity_type: Optional[FidelityType] = None,
        target_qubit: Optional[int] = None,
        use_jit: bool = False,
    ):
        """Instantiate :class:`~casq.PulseOptimizer`.

        Args:
            pulse_gate: Pulse gate.
            pulse_backend: Pulse backend.
            method: ODE solver method.
            target_measurement: Target measurement against which fidelity will be calculated.
            method_options: Options specific to method.
            fidelity_type: Fidelity type. Defaults to FidelityType.COUNTS.
            target_qubit: Qubit to drive with pulse. Defaults to first qubit in simulator.
            use_jit: If True, then jit and value_and_grad is applied to objective function.
        """
        self.pulse_gate = pulse_gate
        self.pulse_backend = pulse_backend
        self.method = method
        self.target_measurement = target_measurement
        self.method_options = method_options if method_options else {}
        self.fidelity_type = (
            PulseOptimizer.FidelityType.COUNTS
            if fidelity_type is None
            else fidelity_type
        )
        self.target_qubit = (
            target_qubit
            if target_qubit
            else self.pulse_backend.hamiltonian.extracted_qubits[0]
        )
        self.use_jit = use_jit
        if self.use_jit:
            logger.warning(
                f"Due to issue ..., PulseOptimizer does not currently support jit."
            )
            self.use_jit = False
        if not is_jax_enabled() and self.method in [
            PulseBackend.ODESolverMethod.QISKIT_DYNAMICS_JAX_RK4,
            PulseBackend.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT,
        ]:
            raise CasqError(
                f"Jax must be enabled for ODE solver method: {self.method.name}."
            )
        self.objective_function = self._build_objective_function()

    @timer(unit="sec")
    def solve(
        self,
        initial_params: npt.NDArray,
        method: OptimizationMethod,
        jac: Optional[Union[bool, FiniteDifferenceScheme, Callable]] = None,
        hess: Optional[
            Union[FiniteDifferenceScheme, HessianUpdateStrategy, Callable]
        ] = None,
        hessp: Optional[Callable] = None,
        bounds: Optional[Union[list, Bounds]] = None,
        constraints: Optional[
            Union[
                dict,
                list[dict],
                LinearConstraint,
                list[LinearConstraint],
                NonlinearConstraint,
                list[NonlinearConstraint],
            ]
        ] = None,
        tol: Optional[float] = None,
        maxiter: Optional[int] = None,
        verbose: bool = True,
        callback: Optional[Callable] = None,
    ) -> PulseOptimizer.Solution:
        """PulseOptimizer.optimize method.

        Optimize pulse.
        This is basically a wrapper around scipy.optimize.minimize.
        For more details, see
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        Args:
            initial_params: Pulse parameters.
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
        options: dict = {"disp": verbose}
        if maxiter:
            options.update(maxiter=maxiter)
        if method not in [
            PulseOptimizer.OptimizationMethod.SCIPY_COBYLA,
            PulseOptimizer.OptimizationMethod.SCIPY_L_BFGS_B,
            PulseOptimizer.OptimizationMethod.SCIPY_NELDER_MEAD,
            PulseOptimizer.OptimizationMethod.SCIPY_POWELL,
            PulseOptimizer.OptimizationMethod.SCIPY_SLSQP,
            PulseOptimizer.OptimizationMethod.SCIPY_TNC,
            PulseOptimizer.OptimizationMethod.SCIPY_TRUST_CONSTR,
        ]:
            logger.warning(f"Method {method.name} does not support bounds.")
            bounds = None
        if method not in [
            PulseOptimizer.OptimizationMethod.SCIPY_COBYLA,
            PulseOptimizer.OptimizationMethod.SCIPY_SLSQP,
            PulseOptimizer.OptimizationMethod.SCIPY_TRUST_CONSTR,
        ]:
            logger.warning(f"Method {method.name} does not support constraints.")
            constraints = None
        opt_results = minimize(
            fun=self.objective_function,
            x0=initial_params,
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            options=options,
            callback=callback,
        )
        initial_parameters = self.pulse_gate.to_parameters_dict(initial_params)
        initial_pulse = self.pulse_gate.pulse(initial_parameters)
        parameters = self.pulse_gate.to_parameters_dict(opt_results.x)
        opt_pulse = self.pulse_gate.pulse(parameters)
        opt_circuit = PulseCircuit.from_pulse_gate(self.pulse_gate, parameters)
        counts = self.pulse_backend.solve(
            circuit=opt_circuit, method=self.method, run_options=self.method_options
        ).counts[-1]
        return PulseOptimizer.Solution(
            initial_parameters=initial_parameters,
            initial_pulse=initial_pulse,
            num_iterations=opt_results.nfev,
            parameters=parameters,
            measurement=counts,
            fidelity=1 - opt_results.fun,
            gate=self.pulse_gate,
            pulse=opt_pulse,
            circuit=opt_circuit,
            message=opt_results.message,
        )

    def _build_objective_function(
        self,
    ) -> Callable[[npt.NDArray], Union[float, tuple[float, float]]]:
        """PulseOptimizer._build_objective_function method.

        Build objective function to minimize.

        Returns:
            Objective function.
        """

        def objective(params: npt.NDArray) -> float:
            parameters = self.pulse_gate.to_parameters_dict(params)
            circuit = PulseCircuit.from_pulse_gate(self.pulse_gate, parameters)
            solution = self.pulse_backend.solve(
                circuit=circuit, method=self.method, run_options=self.method_options
            )
            counts = solution.counts[-1]
            fidelity = hellinger_fidelity(self.target_measurement, counts)
            infidelity = 1.0 - float(fidelity)
            logger.debug(
                f"PARAMETERS: {params} RESULT: {counts} OBJECTIVE: {infidelity}"
            )
            return infidelity

        if self.use_jit:
            jit_objective: Callable[[npt.NDArray], tuple[float, float]] = jit(
                value_and_grad(objective)
            )
            return jit_objective
        else:
            return objective
