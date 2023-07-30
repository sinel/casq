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
"""Pulse optimizer."""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Union

from jax import jit, value_and_grad
from loguru import logger
from matplotlib import colormaps
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
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
from casq.common.plotting import (
    LegendStyle,
    LineConfig,
    LineData,
    LineStyle,
    MarkerStyle,
    plot,
)
from casq.gates import PulseCircuit, PulseGate


class PulseOptimizer:
    """PulseOptimizer class.

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

    @dataclass
    class Solution:
        """Pulse optimizer solution.

        Args:
            initial_parameters: Initial parameters.
            initial_pulse: Initial pulse.
            num_iterations: Number of iterations.
            iterations: Iteration data.
            parameters: Optimum parameters.
            measurement: Optimum measurement.
            fidelity: Optimum fidelity.
            pulse: Optimum pulse.
            gate: Optimum gate.
            circuit: Optimum circuit.
            message: Solution message.
        """

        initial_parameters: dict[str, Any]
        initial_pulse: ScalableSymbolicPulse
        num_iterations: int
        iterations: list[PulseOptimizer.Iteration]
        parameters: dict[str, Any]
        measurement: Union[dict[str, float], DensityMatrix, Statevector]
        fidelity: float
        pulse: ScalableSymbolicPulse
        gate: PulseGate
        circuit: PulseCircuit
        message: str

        def plot_objective_history(
            self,
            filename: Optional[str] = None,
            hidden: bool = False,
        ) -> Axes:
            """PulseOptimizer.Solution.plot_objective_history method.

            Plots iteration history of objective.

            Args:
                filename: If filename is provided as path str, then figure is saved as png.
                hidden: If False, then plot is not displayed. Useful if method is used for saving only.

            Returns:
                :py:class:`matplotlib.axes.Axes`
            """
            x = []
            y = []
            for iteration in self.iterations:
                x.append(iteration.index)
                y.append(iteration.objective)
            config = LineConfig(
                data=LineData(x, y),
                line_style=LineStyle(),
                xtitle="Iteration",
                ytitle="Objective",
            )
            axes = plot(
                configs=[config],
                filename=filename,
                hidden=hidden,
            )
            return axes

        def plot_parameter_history(
            self,
            parameters: Optional[list[str]] = None,
            filename: Optional[str] = None,
            hidden: bool = False,
        ) -> Axes:
            """PulseOptimizer.Solution.plot_parameter_history method.

            Plots iteration history of parameters.

            Args:
                parameters: Parameters to plot.
                filename: If filename is provided as path str, then figure is saved as png.
                hidden: If False, then plot is not displayed. Useful if method is used for saving only.

            Returns:
                :py:class:`matplotlib.axes.Axes`
            """
            x = []
            parameter_data: dict[str, Any] = {}
            if parameters:
                for parameter in parameters:
                    parameter_data[parameter] = []
            else:
                for parameter in self.iterations[0].parameters.keys():
                    parameter_data[parameter] = []
            for iteration in self.iterations:
                x.append(iteration.index)
                for key, value in iteration.parameters.items():
                    if parameters:
                        if key in parameters:
                            parameter_data[key].append(value)
                    else:
                        parameter_data[key].append(value)
            configs = []
            for parameter, data in parameter_data.items():
                configs.append(
                    LineConfig(
                        data=LineData(x, data),
                        line_style=LineStyle(),
                        label=parameter,
                        xtitle="Iteration",
                        ytitle="Parameter",
                    )
                )
            axes = plot(
                configs=configs,
                legend_style=LegendStyle(),
                filename=filename,
                hidden=hidden,
            )
            return axes

        def plot_objective_by_parameter(
            self,
            parameters: list[str],
            filename: Optional[str] = None,
            hidden: bool = False,
        ) -> Axes:
            """PulseOptimizer.Solution.plot_trajectory method.

            Plots iteration history of parameters.

            Args:
                parameters: Parameters to plot.
                filename: If filename is provided as path str, then figure is saved as png.
                hidden: If False, then plot is not displayed. Useful if method is used for saving only.

            Returns:
                :py:class:`matplotlib.axes.Axes`
            """
            if len(parameters) == 1:
                x = []
                y = []
                for iteration in self.iterations:
                    x.append(iteration.parameters[parameters[0]])
                    y.append(iteration.objective)
                config = LineConfig(
                    data=LineData(x, y),
                    line_style=LineStyle(color="steelblue", size=0.75),
                    marker_style=MarkerStyle(color="steelblue", size=7.5),
                    xtitle=parameters[0].capitalize(),
                    ytitle="Objective",
                )
                axes = plot(
                    configs=[config],
                    filename=filename,
                    hidden=hidden,
                )
            elif len(parameters) == 2:
                x = []
                y = []
                z = []
                for iteration in self.iterations:
                    x.append(iteration.parameters[parameters[0]])
                    y.append(iteration.parameters[parameters[1]])
                    z.append(iteration.objective)
                figure = plt.figure()
                axes = figure.add_subplot(projection="3d")
                axes.set_xlabel(parameters[0].capitalize(), labelpad=20)
                axes.set_ylabel(parameters[1].capitalize(), labelpad=20)
                axes.set_zlabel("Objective", labelpad=20)
                axes.xaxis.pane.fill = False
                axes.yaxis.pane.fill = False
                axes.zaxis.pane.fill = False
                axes.set_box_aspect([1, 1, 1], zoom=0.8)
                axes.plot3D(x, y, z, "steelblue", linewidth=0.75)
                axes.scatter3D(x, y, z, "steelblue", s=15)
            elif len(parameters) == 3:
                x = []
                y = []
                z = []
                objective = []
                for iteration in self.iterations:
                    x.append(iteration.parameters[parameters[0]])
                    y.append(iteration.parameters[parameters[1]])
                    z.append(iteration.parameters[parameters[2]])
                    objective.append(iteration.objective)
                figure = plt.figure()
                axes = figure.add_subplot(projection="3d")
                axes.set_xlabel(parameters[0].capitalize(), labelpad=20)
                axes.set_ylabel(parameters[1].capitalize(), labelpad=20)
                axes.set_zlabel(parameters[2].capitalize(), labelpad=20)
                axes.xaxis.pane.fill = False
                axes.yaxis.pane.fill = False
                axes.zaxis.pane.fill = False
                axes.set_box_aspect([1, 1, 1], zoom=0.8)
                axes.plot3D(x, y, z, "steelblue", linewidth=0.75)
                s3d = axes.scatter3D(
                    x, y, z, s=15, c=objective, cmap=colormaps["inferno"].reversed()
                )
                clb = figure.colorbar(s3d)
                clb.ax.set_title("Objective")
            else:
                raise (
                    ValueError(
                        f"Cannot visualize objective by more than three parameters!"
                    )
                )
            return axes

    @dataclass
    class Iteration:
        """Pulse optimizer iteration."""

        index: int
        parameters: dict[str, Any]
        result: Any
        objective: float

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
        """Initialize PulseOptimizer."""
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
        self._counter: int = 0
        self._iterations: list[PulseOptimizer.Iteration] = []

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
        self._counter = 0
        self._iterations = []
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
            iterations=self._iterations,
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

        def objective(parameters: npt.NDArray) -> float:
            parameters_dict = self.pulse_gate.to_parameters_dict(parameters)
            circuit = self._build_circuit(parameters_dict)
            solution = self.pulse_backend.solve(
                circuit=circuit, method=self.method, run_options=self.method_options
            )
            counts = solution.counts[-1]
            fidelity = hellinger_fidelity(self.target_measurement, counts)
            infidelity = 1.0 - float(fidelity)
            self._objective_callback(parameters_dict, counts, infidelity)
            return infidelity

        if self.use_jit:
            jit_objective: Callable[[npt.NDArray], tuple[float, float]] = jit(
                value_and_grad(objective)
            )
            return jit_objective
        else:
            return objective

    def _objective_callback(
        self, parameters: dict[str, Any], result: Any, objective: float
    ) -> None:
        """PulseOptimizer._objective_callback method.

        Callback used by objective function.
        """
        self._counter += 1
        iteration = PulseOptimizer.Iteration(
            index=self._counter,
            parameters=parameters,
            result=result,
            objective=objective,
        )
        self._iterations.append(iteration)
        logger.debug(
            f"ITERATION: {iteration.index} PARAMETERS: {iteration.parameters} "
            f"RESULT: {iteration.result} OBJECTIVE: {iteration.objective}"
        )

    @abstractmethod
    def _build_circuit(self, parameters: dict[str, Any]) -> PulseCircuit:
        """PulseOptimizer._build_circuit method.

        Build pulse circuit for objective function.

        Returns:
            :py:class:`casq.gates.PulseCircuit`
        """
