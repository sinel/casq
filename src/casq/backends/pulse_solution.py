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

from datetime import datetime
from typing import Optional, Union

from matplotlib.figure import Figure
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
from qiskit.result.models import ExperimentResult

from casq.common.decorators import trace
from casq.common.exceptions import CasqError
from casq.common.plotting import (
    LegendStyle,
    LineConfig,
    LineData,
    LineStyle,
    MarkerStyle,
    plot,
    plot_bloch,
)


class PulseSolution:
    """PulseSolution class."""

    @classmethod
    def from_qiskit(
        cls,
        result: ExperimentResult,
    ) -> PulseSolution:
        """PulseSolution.from_qiskit method.

        Transforms Qiskit result into pulse backend solution.

        Args:
            result: Qiskit experiment result.

        Returns:
            Pulse solution.
        """
        if result.header.metadata.get("casq", False):
            samples = []
            for item in result.data.samples:
                samples.append(item.tolist())
            iq_data = []
            for item in result.data.iq_data:
                iq_data_inner = []
                for item_inner in item:
                    iq_data_inner.append((item_inner[0][0], item_inner[0][1]))
                iq_data.append(iq_data_inner)
            avg_iq_data = []
            for item in result.data.avg_iq_data:
                avg_iq_data.append((item[0][0], item[0][1]))
            solution: PulseSolution = PulseSolution(
                circuit_name=result.header.name,
                qubits=result.data.qubits,
                times=result.data.times,
                samples=samples,
                counts=result.data.counts_list,
                populations=result.data.populations,
                states=result.data.states,
                iq_data=iq_data,
                avg_iq_data=avg_iq_data,
                shots=result.shots,
                seed=result.seed,
                is_success=result.success,
                timestamp=result.header.metadata.get("timestamp", None),
            )
            return solution
        else:
            raise CasqError(
                "PulseSolution.from_qiskit method requires an ExperimentResult instance generated by casq."
            )

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
    ) -> Figure:
        """PulseSolution.plot_population method.

        Plots populations from result.

        Args:
            filename: If filename is provided as path str, then figure is saved as png.
            hidden: If False, then plot is not displayed. Useful if method is used for saving only.

        Returns:
            :py:class:`matplotlib.figure.Figure`
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
        return plot(
            configs=configs,
            legend_style=LegendStyle(),
            filename=filename,
            hidden=hidden,
        )

    def plot_iq(
        self,
        time_index: Optional[int] = None,
        filename: Optional[str] = None,
        hidden: bool = False,
    ) -> Figure:
        """PulseSolution.plot_iq method.

        Plots IQ points from result.

        Args:
            time_index: Time at which to plot IQ points.
            filename: If filename is provided as path str, then figure is saved as png.
            hidden: If False, then plot is not displayed. Useful if method is used for saving only.

        Returns:
            :py:class:`matplotlib.figure.Figure`
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
        return plot(configs=[config], filename=filename, hidden=hidden)

    def plot_iq_trajectory(
        self, filename: Optional[str] = None, hidden: bool = False
    ) -> Figure:
        """PulseSolution.plot_iq_trajectory method.

        Plots trajectory of average IQ points from result.

        Args:
            filename: If filename is provided as path str, then figure is saved as png.
            hidden: If False, then plot is not displayed. Useful if method is used for saving only.

        Returns:
            :py:class:`matplotlib.figure.Figure`
        """
        x = []
        y = []
        for iq in self.avg_iq_data:
            x.append(iq[0])
            y.append(iq[1])
        config = LineConfig(
            data=LineData(x, y), marker_style=MarkerStyle(), xtitle="I", ytitle="Q"
        )
        return plot(configs=[config], filename=filename, hidden=hidden)

    def plot_trajectory(
        self,
        qubit: int = 0,
        filename: Optional[str] = None,
        hidden: bool = False,
    ) -> Figure:
        """PulseSolution.plot_trajectory method.

        Plots statevector trajectory from result.

        Args:
            qubit: Qubit to plot trajectory of.
            filename: If filename is provided as path str, then figure is saved as png.
            hidden: If False, then plot is not displayed. Useful if method is used for saving only.

        Returns:
            :py:class:`matplotlib.figure.Figure`
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
        return plot(
            configs=[x_config, y_config, z_config],
            legend_style=LegendStyle(),
            filename=filename,
            hidden=hidden,
        )

    def plot_bloch_trajectory(
        self,
        qubit: int = 0,
        filename: Optional[str] = None,
        hidden: bool = False,
    ) -> Figure:
        """PulseSolution.plot_bloch_trajectory method.

        Plots statevector trajectory on Bloch sphere from result.

        Args:
            qubit: Qubit to plot trajectory of.
            filename: If filename is provided as path str, then figure is saved as png.
            hidden: If False, then plot is not displayed. Useful if method is used for saving only.

        Returns:
            :py:class:`matplotlib.figure.Figure`
        """
        x, y, z = self._xyz(qubit)
        return plot_bloch(x, y, z, filename=filename, hidden=hidden)

    def _xyz(self, qubit: int = 0) -> tuple[list[float], list[float], list[float]]:
        """PulseSolution._xyz method.

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
        """PulseSolution._trace method.

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
