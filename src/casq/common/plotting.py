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

from enum import Enum
from typing import NamedTuple, Optional, Union

from loguru import logger
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy.typing as npt
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit_dynamics.pulse import InstructionToSignals
from qiskit.pulse.transforms import block_to_schedule
import qutip

plt.style.use("seaborn-v0_8-notebook")


class LineType(Enum):
    SOLID = 0
    DASHED = 1
    DOTTED = 2
    DASHDOT = 3


class LineStyle(NamedTuple):
    color: Optional[str] = None
    type: Optional[LineType] = None
    size: Optional[float] = None


class MarkerType(Enum):
    CIRCLE = 0
    DIAMOND = 1
    HEXAGON1 = 2
    HEXAGON2 = 3
    PENTAGON = 4
    PIXEL = 5
    PLUS = 6
    PLUS_FILLED = 7
    POINT = 8
    SQUARE = 9
    STAR = 10
    TRIANGLE_DOWN = 11
    TRIANGLE_LEFT = 12
    TRIANGLE_RIGHT = 13
    TRIANGLE_UP = 14
    X = 15
    X_FILLED = 16


class MarkerStyle(NamedTuple):
    color: Optional[str] = None
    type: Optional[MarkerType] = None
    size: Optional[float] = None


class LegendLocation(Enum):
    # Values should not be changed
    # since they correspond to matplotlib location codes
    BEST = 0
    CENTER = 10
    CENTER_LEFT = 6
    CENTER_RIGHT = 7
    LOWER_CENTER = 8
    LOWER_LEFT = 3
    LOWER_RIGHT = 4
    UPPER_CENTER = 9
    UPPER_LEFT = 2
    UPPER_RIGHT = 1


class LegendStyle(NamedTuple):
    location: LegendLocation = LegendLocation.BEST
    anchor: Optional[tuple[float, float]] = None


class LineConfig(NamedTuple):
    x: Union[list[float], npt.NDArray, list[list[float]], list[npt.NDArray]]
    y: Union[list[float], npt.NDArray, list[list[float]], list[npt.NDArray]]
    label: Optional[str] = None
    xtitle: Optional[str] = None
    ytitle: Optional[str] = None
    xlim: tuple[float, float] = None
    ylim: tuple[float, float] = None
    xticks: list[float] = None
    yticks: list[float] = None
    line_style: Optional[LineStyle] = None
    marker_style: Optional[MarkerStyle] = None
    ax: Optional[Axes] = None


def add_line(
        ax: Axes,
        x: Union[list[float], npt.NDArray], y: Union[list[float], npt.NDArray],
        label: Optional[str] = None,
        line_style: Optional[LineStyle] = None,
        marker_style: Optional[MarkerStyle] = None
) -> None:
    line_obj = ax.plot(x, y)[0]
    if label:
        line_obj.set_label(label)
    if line_style:
        if line_style.color:
            line_obj.set_color(line_style.color)
        if line_style.type:
            line_obj.set_linestyle(line_style.type.name.lower())
        if line_style.size:
            line_obj.set_linewidth(line_style.size)
    if marker_style:
        if marker_style.color:
            line_obj.set_markeredgecolor(marker_style.color)
            line_obj.set_markerfacecolor(marker_style.color)
        if marker_style.type:
            line_obj.set_marker(marker_style.type.name.lower())
        if marker_style.size:
            line_obj.set_markersize(marker_style.size)


def add_line_collection(
        ax: Axes,
        x: Union[list[list[float]], list[npt.NDArray]], y: Union[list[list[float]], list[npt.NDArray]],
        label: Optional[str] = None,
        line_style: Optional[LineStyle] = None,
        marker_style: Optional[MarkerStyle] = None
) -> None:
    collection = LineCollection([list(zip(xc, yc)) for xc, yc in zip(x, y)])
    # print(collection)
    line_obj = ax.add_collection(collection)
    if label:
        line_obj.set_label(label)
    if line_style:
        if line_style.color:
            line_obj.set_color(line_style.color)
        if line_style.type:
            line_obj.set_linestyle(line_style.type.name.lower())
        if line_style.size:
            line_obj.set_linewidth(line_style.size)
    if marker_style:
        if marker_style.color:
            line_obj.set_markeredgecolor(marker_style.color)
            line_obj.set_markerfacecolor(marker_style.color)
        if marker_style.type:
            line_obj.set_marker(marker_style.type.name.lower())
        if marker_style.size:
            line_obj.set_markersize(marker_style.size)


def add_horizontal_line(ax: Axes, y: float, label: Optional[str] = None,
                        line_style: Optional[LineStyle] = None) -> None:
    line_obj = ax.axhline(y)
    if label:
        line_obj.set_label(label)
    if line_style:
        if line_style.color:
            line_obj.set_color(line_style.color)
        if line_style.type:
            line_obj.set_linestyle(line_style.type.name.lower())
        if line_style.size:
            line_obj.set_linewidth(line_style.size)


def add_vertical_line(ax: Axes, x: float, label: Optional[str] = None, line_style: Optional[LineStyle] = None) -> None:
    line_obj = ax.axvline(x)
    if label:
        line_obj.set_label(label)
    if line_style:
        if line_style.color:
            line_obj.set_color(line_style.color)
        if line_style.type:
            line_obj.set_linestyle(line_style.type.name.lower())
        if line_style.size:
            line_obj.set_linewidth(line_style.size)


def plot(
        data: list[LineConfig],
        figure: Optional[Figure] = None,
        hlines: Optional[list[tuple[float, Optional[str], LineStyle, Optional[Axes]]]] = None,
        vlines: Optional[list[tuple[float, Optional[str], LineStyle, Optional[Axes]]]] = None,
        xticks: list[float] = None, yticks: list[float] = None,
        title: str = None,
        legend_style: Optional[LegendStyle] = None, show_grid: bool = False,
        save: Optional[str] = None, hidden: bool = False
) -> Figure:
    # TODO: Validate that all dimensions match.
    if figure is None:
        figure, ax = plt.subplots(1, 1)
    figure.set_constrained_layout(True)
    if title:
        figure.suptitle(title)
    for x, y, label, xtitle, ytitle, xlim, ylim, xticks, yticks, line_style, marker_style, ax in data:
        if ax is None:
            ax = figure.axes[0]
        if isinstance(x[0], (list, np.ndarray)):
            add_line_collection(ax, x, y, label, line_style, marker_style)
        else:
            add_line(ax, x, y, label, line_style, marker_style)
        if legend_style:
            ax.legend(loc=legend_style.location.value, bbox_to_anchor=legend_style.anchor)
        if xtitle:
            ax.set_xlabel(xtitle)
        if ytitle:
            ax.set_ylabel(ytitle)
        if xlim:
            ax.set_xlim(list(xlim))
        if ylim:
            ax.set_ylim(list(ylim))
        if xticks:
            ax.set_xticks(xticks)
        if yticks:
            ax.set_yticks(yticks)
    if hlines:
        for y, label, line_style, ax in hlines:
            if ax is None:
                ax = figure.axes[0]
            add_horizontal_line(ax, y, label, line_style)
            if legend_style:
                ax.legend(loc=legend_style.location.value, bbox_to_anchor=legend_style.anchor)
    if vlines:
        for x, label, line_style, ax in vlines:
            if ax is None:
                ax = figure.axes[0]
            add_horizontal_line(ax, x, label, line_style)
            if legend_style:
                ax.legend(loc=legend_style.location.value, bbox_to_anchor=legend_style.anchor)
    if show_grid:
        plt.grid()
    if save:
        plt.savefig(save)
    if not hidden:
        plt.show()
    return figure


def plot_bloch(
        x: Union[list[float], npt.NDArray],
        y: Union[list[float], npt.NDArray],
        z: Union[list[float], npt.NDArray],
        filename: Optional[str] = None, hidden: bool = False
) -> Figure:
    b = qutip.Bloch()
    b.add_points([x, y, z], meth="l")
    b.add_vectors([x[-1], y[-1], z[-1]])
    b.render()
    if filename:
        b.save(filename)
    if not hidden:
        b.show()
    return b.fig


def plot_signal(
        schedule: Union[Schedule, ScheduleBlock], dt: float, channel: str, carrier_frequency: float, duration: float,
        number_of_samples: int = 10000, save: Optional[str] = None, hidden: bool = False
) -> Figure:
    if isinstance(schedule, ScheduleBlock):
        sched = block_to_schedule(schedule)
    else:
        sched = schedule
    converter = InstructionToSignals(dt, carriers={channel: carrier_frequency})
    signal = converter.get_signals(sched)[0]
    signal_times = np.linspace(0, duration, number_of_samples)
    signal_samples = signal(signal_times)
    signal_envelope_abs = np.abs(signal.envelope(signal_times))
    signal_envelope_phase = np.angle(signal.envelope(signal_times))
    signal_envelope_real = np.real(signal.envelope(signal_times))
    signal_envelope_img = np.imag(signal.envelope(signal_times))
    amplitude_min = 1.1 * np.min(-signal_envelope_abs)
    amplitude_max = 1.1 * np.max(signal_envelope_abs)
    phase_min = 1.1 * np.min(-signal_envelope_phase)
    phase_max = 1.1 * np.max(signal_envelope_phase)
    figure, axs = plt.subplots(2, 2, sharex="all", sharey="all")
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    config1 = LineConfig(
        x=signal_times, y=signal_samples, label="Signal",
        xtitle="Time (ns)", ytitle="Amplitude", xlim=(0, duration), ylim=(amplitude_min, amplitude_max),
        line_style=LineStyle(size=0.1), ax=ax1
    )
    config2 = LineConfig(
        x=[signal_times, signal_times], y=[signal_envelope_abs, -signal_envelope_abs], label="Envelope",
        xtitle="Time (ns)", ytitle="Amplitude", xlim=(0, duration), ylim=(amplitude_min, amplitude_max),
        line_style=LineStyle(color="black", size=1.5), ax=ax1
    )
    config3 = LineConfig(
        x=signal_times, y=signal_envelope_real, label="Envelope (real)",
        xtitle="Time (ns)", ytitle="Amplitude", xlim=(0, duration), ylim=(amplitude_min, amplitude_max),
        line_style=LineStyle(size=1), ax=ax1
    )
    config4 = LineConfig(
        x=signal_times, y=signal_envelope_img, label="Envelope (imaginary)",
        xtitle="Time (ns)", ytitle="Amplitude", xlim=(0, duration), ylim=(amplitude_min, amplitude_max),
        line_style=LineStyle(size=1), ax=ax1
    )
    config5 = LineConfig(
        x=signal_times, y=signal_envelope_phase, label=None,
        xtitle="Time (ns)", ytitle="Phase (radians)", xlim=(0, duration), ylim=(phase_min, phase_max),
        line_style=LineStyle(color="black", size=1), ax=ax2
    )
    plot(
        data=[config1, config2, config3, config4, config5],
        figure=figure, legend_style=LegendStyle(),
        save=save, hidden=hidden
    )
    return figure
