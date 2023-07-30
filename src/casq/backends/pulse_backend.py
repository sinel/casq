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
"""Pulse simulator."""
from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

from casq.common.decorators import trace
from casq.common.plotting import (
    LegendStyle,
    LineConfig,
    LineData,
    LineStyle,
    MarkerStyle,
    plot,
    plot_bloch,
)
from casq.gates.pulse_circuit import PulseCircuit
from casq.models.control_model import ControlModel
from casq.models.hamiltonian_model import HamiltonianModel


class PulseBackend:
    """PulseBackend class."""

    class ODESolverMethod(str, Enum):
        """Solver methods."""

        QISKIT_DYNAMICS_RK4 = "RK4"
        QISKIT_DYNAMICS_JAX_RK4 = "jax_RK4"
        QISKIT_DYNAMICS_JAX_ODEINT = "jax_odeint"
        SCIPY_BDF = "BDF"
        SCIPY_DOP853 = "DOP853"
        SCIPY_LSODA = "LSODA"
        SCIPY_RADAU = "Radau"
        SCIPY_RK23 = "RK23"
        SCIPY_RK45 = "RK45"

    class Solution:
        """PulseBackend.Solution class."""

        @trace()
        def __init__(
            self,
            circuit_name: str,
            qubits: list[int],
            times: list[float],
            samples: list[list[int]],
            counts: list[dict[str, int]],
            populations: list[dict[str, float]],
            states: list[Union[DensityMatrix, Statevector]],
            iq_data: list[list[tuple[float, float]]],
            avg_iq_data: list[tuple[float, float]],
            shots: int = 1024,
            seed: Optional[int] = None,
            is_success: bool = True,
            timestamp: float = datetime.timestamp(datetime.now()),
        ) -> None:
            """Instantiate :class:`~casq.PulseSolution`.

            Args:
                circuit_name: Pulse circuit name.
                times: Time at which pulse backend was solved.
                qubits: Integer labels for selected qubits from system. Defaults to [0].
                times: ...
                samples: ...
                counts: ...
                populations: ...
                states: ...
                iq_data: ...
                avg_iq_data: ...
                shots: Number of shots per experiment. Defaults to 1024.
                seed: Seed to use in random sampling. Defaults to None.
                is_success: ...
                timestamp: Posix timestamp.
            """
            self.circuit_name = circuit_name
            self.times = times
            self.qubits = qubits
            self.samples = samples
            self.counts = counts
            self.populations = populations
            self.states = states
            self.iq_data = iq_data
            self.avg_iq_data = avg_iq_data
            self.shots = shots
            self.seed = seed
            self.is_success = is_success
            self.timestamp = timestamp

        def plot_population(
            self, filename: Optional[str] = None, hidden: bool = False
        ) -> Axes:
            """PulseBackend.Solution.plot_population method.

            Plots populations from result.

            Args:
                filename: If filename is provided as path str, then figure is saved as png.
                hidden: If False, then plot is not displayed. Useful if method is used for saving only.

            Returns:
                Matplotlib Axes.
            """
            pops: dict[str, list[float]] = {}
            for key in self.populations[-1].keys():
                pops[key] = []
            for p in self.populations:
                for key in pops.keys():
                    value = p.get(key, 0)
                    pops[key].append(value)
            configs = []
            for key, values in pops.items():
                config = LineConfig(
                    data=LineData(self.times, values),
                    label=f"Population in |{key}>",
                    line_style=LineStyle(),
                    xtitle="Time (ns)",
                    ytitle="Population",
                )
                configs.append(config)
            axes = plot(
                configs=configs,
                legend_style=LegendStyle(),
                filename=filename,
                hidden=hidden,
            )
            return axes

        def plot_iq(
            self,
            time_index: Optional[int] = None,
            filename: Optional[str] = None,
            hidden: bool = False,
        ) -> Axes:
            """PulseBackend.Solution.plot_iq method.

            Plots IQ points from result.

            Args:
                time_index: Time at which to plot IQ points.
                filename: If filename is provided as path str, then figure is saved as png.
                hidden: If False, then plot is not displayed. Useful if method is used for saving only.

            Returns:
                Matplotlib Axes.
            """
            i = time_index if time_index else -1
            x = []
            y = []
            for iq in self.iq_data[i]:
                x.append(iq[0])
                y.append(iq[1])
            config = LineConfig(
                data=LineData(x, y), marker_style=MarkerStyle(), xtitle="I", ytitle="Q"
            )
            axes = plot(configs=[config], filename=filename, hidden=hidden)
            return axes

        def plot_iq_trajectory(
            self, filename: Optional[str] = None, hidden: bool = False
        ) -> Axes:
            """PulseBackend.Solution.plot_iq_trajectory method.

            Plots trajectory of average IQ points from result.

            Args:
                filename: If filename is provided as path str, then figure is saved as png.
                hidden: If False, then plot is not displayed. Useful if method is used for saving only.

            Returns:
                Matplotlib Axes.
            """
            x = []
            y = []
            for iq in self.avg_iq_data:
                x.append(iq[0])
                y.append(iq[1])
            config = LineConfig(
                data=LineData(x, y), marker_style=MarkerStyle(), xtitle="I", ytitle="Q"
            )
            axes = plot(configs=[config], filename=filename, hidden=hidden)
            return axes

        def plot_trajectory(
            self,
            qubit: int = 0,
            filename: Optional[str] = None,
            hidden: bool = False,
        ) -> Axes:
            """PulseBackend.Solution.plot_trajectory method.

            Plots statevector trajectory from result.

            Args:
                qubit: Qubit to plot trajectory of.
                filename: If filename is provided as path str, then figure is saved as png.
                hidden: If False, then plot is not displayed. Useful if method is used for saving only.

            Returns:
                Matplotlib Axes.
            """
            x, y, z = self._xyz(qubit)
            x_config = LineConfig(
                data=LineData(self.times, x),
                line_style=LineStyle(),
                label="$\\langle X \\rangle$",
                xtitle="$t$",
            )
            y_config = LineConfig(
                data=LineData(self.times, y),
                line_style=LineStyle(),
                label="$\\langle Y \\rangle$",
                xtitle="$t$",
            )
            z_config = LineConfig(
                data=LineData(self.times, z),
                line_style=LineStyle(),
                label="$\\langle Z \\rangle$",
                xtitle="$t$",
            )
            axes = plot(
                configs=[x_config, y_config, z_config],
                legend_style=LegendStyle(),
                filename=filename,
                hidden=hidden,
            )
            return axes

        def plot_bloch_trajectory(
            self,
            qubit: int = 0,
            filename: Optional[str] = None,
            hidden: bool = False,
        ) -> Axes3D:
            """PulseBackend.Solution.plot_bloch_trajectory method.

            Plots statevector trajectory on Bloch sphere from result.

            Args:
                qubit: Qubit to plot trajectory of.
                filename: If filename is provided as path str, then figure is saved as png.
                hidden: If False, then plot is not displayed. Useful if method is used for saving only.

            Returns:
                Matplotlib Axes.
            """
            x, y, z = self._xyz(qubit)
            axes = plot_bloch(x, y, z, filename=filename, hidden=hidden)
            return axes

        def _xyz(self, qubit: int = 0) -> tuple[list[float], list[float], list[float]]:
            """PulseBackend.Solution._xyz method.

            Transforms statevectors into 3D trajectory from result.

            Returns:
                XYZ data lists or dict of lists.
            """
            if len(self.qubits) > 1:
                xq: dict[int, list[float]] = {}
                yq: dict[int, list[float]] = {}
                zq: dict[int, list[float]] = {}
                for q in self.qubits:
                    xq[q] = []
                    yq[q] = []
                    zq[q] = []
                    for sv in self.states:
                        traced_sv = self._trace(sv, q)
                        xp, yp, zp = traced_sv.data.real
                        xq[q].append(xp)
                        yq[q].append(yp)
                        zq[q].append(zp)
                return xq[qubit], yq[qubit], zq[qubit]
            else:
                xsv: list[float] = []
                ysv: list[float] = []
                zsv: list[float] = []
                for sv in self.states:
                    xp, yp, zp = sv.data.real
                    xsv.append(xp)
                    ysv.append(yp)
                    zsv.append(zp)
                return xsv, ysv, zsv

        def _trace(self, state: Statevector, qubit: int) -> Statevector:
            """PulseBackend.Solution._trace method.

            Generate partial trace of statevector for specified qubit.

            Args:
                state: System state given as statevector.
                qubit: Qubit to trace out.

            Returns:
                Reduced statevector.
            """
            traced_over_qubits = self.qubits
            traced_over_qubits.remove(qubit)
            partial_density_matrix = partial_trace(state, traced_over_qubits)
            return partial_density_matrix.to_statevector()

    def __init__(
        self,
        hamiltonian: HamiltonianModel,
        control: ControlModel,
        seed: Optional[int] = None,
    ):
        """Instantiate :class:`~casq.backends.PulseBackend`.

        Args:
            hamiltonian: Hamiltonian model.
            control: Control model.
            seed: Seed to use in random sampling. Defaults to None.
        """
        self.hamiltonian = hamiltonian
        self.control = control
        self._seed = seed
        self._native_backend = self._get_native_backend()

    @abstractmethod
    def solve(
        self,
        circuit: PulseCircuit,
        method: ODESolverMethod,
        initial_state: Optional[Union[DensityMatrix, Statevector]] = None,
        shots: int = 1024,
        steps: Optional[int] = None,
        run_options: Optional[dict[str, Any]] = None,
    ) -> PulseBackend.Solution:
        """PulseBackend.run.

        Args:
            circuit: Pulse circuit.
            method: ODE solving method to use.
            initial_state: Initial state for simulation,
                either None,
                indicating that the ground state for the system Hamiltonian should be used,
                or an arbitrary Statevector or DensityMatrix.
            shots: Number of shots per experiment. Defaults to 1024.
            steps: Number of steps at which to solve the system.
                Used to automatically calculate an evenly-spaced t_eval range.
            run_options: Options specific to native backend's run method.
        """

    @abstractmethod
    def _get_native_backend(self) -> Any:
        """PulseBackend._get_native_backend."""
