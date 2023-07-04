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
"""Common plotting functions used by library."""
from __future__ import annotations

from enum import Enum
from typing import NamedTuple, Optional, Union

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.transforms import block_to_schedule
from qiskit_dynamics.pulse import InstructionToSignals
import qutip

plt.style.use("seaborn-v0_8-notebook")


class LineType(Enum):
    """Matplotlib line types.

    Keys (lowercase) correspond to matplotlib line type codes.
    """

    SOLID = 0
    DASHED = 1
    DOTTED = 2
    DASHDOT = 3


class LineStyle(NamedTuple):
    """Line style properties."""

    color: Optional[str] = None
    type: Optional[LineType] = LineType.SOLID
    size: Optional[float] = None


class MarkerType(Enum):
    """Matplotlib marker types.

    Values correspond to matplotlib marker type codes.
    """

    CIRCLE = "o"
    DIAMOND = "D"
    HEXAGON1 = "h"
    HEXAGON2 = "H"
    PENTAGON = "p"
    PIXEL = ","
    PLUS = "+"
    PLUS_FILLED = "P"
    POINT = "."
    SQUARE = "s"
    STAR = "*"
    TRIANGLE_DOWN = "v"
    TRIANGLE_LEFT = "<"
    TRIANGLE_RIGHT = ">"
    TRIANGLE_UP = "^"
    X = "x"
    X_FILLED = "X"


class MarkerStyle(NamedTuple):
    """Marker style properties."""

    color: Optional[str] = None
    type: Optional[MarkerType] = MarkerType.CIRCLE
    size: Optional[float] = None


class LegendLocation(Enum):
    """Matplotlib legend locations.

    Values correspond to matplotlib legend location codes.
    """

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
    """Legend style properties."""

    location: LegendLocation = LegendLocation.BEST
    anchor: Optional[tuple[float, float]] = None


class LineConfig(NamedTuple):
    """Line configuration."""

    x: Union[list[float], npt.NDArray, list[list[float]], list[npt.NDArray]]
    y: Union[list[float], npt.NDArray, list[list[float]], list[npt.NDArray]]
    label: Optional[str] = None
    xtitle: Optional[str] = None
    ytitle: Optional[str] = None
    xlim: Optional[tuple[float, float]] = None
    ylim: Optional[tuple[float, float]] = None
    xticks: Optional[list[float]] = None
    yticks: Optional[list[float]] = None
    line_style: Optional[LineStyle] = None
    marker_style: Optional[MarkerStyle] = None
    ax: Optional[Axes] = None


def add_line(
    ax: Axes,
    x: Union[list[float], npt.NDArray],
    y: Union[list[float], npt.NDArray],
    label: Optional[str] = None,
    line_style: Optional[LineStyle] = None,
    marker_style: Optional[MarkerStyle] = None,
) -> None:
    """Add line to Matplotlib axes.

    Args:
        ax: Matplotlib axes.
        x: X data.
        y: Y data.
        label: Label to be used for line in legend.
        line_style: Line style.
        marker_style: Marker style.
    """
    line_obj = ax.plot(x, y)[0]
    if label:
        line_obj.set_label(label)
    if line_style:
        if line_style.type:
            line_obj.set_linestyle(line_style.type.name.lower())
        if line_style.color:
            line_obj.set_color(line_style.color)
        if line_style.size:
            line_obj.set_linewidth(line_style.size)
    else:
        line_obj.set_linestyle("")
    if marker_style:
        if marker_style.type:
            line_obj.set_marker(marker_style.type.value)
        if marker_style.color:
            line_obj.set_markeredgecolor(marker_style.color)
            line_obj.set_markerfacecolor(marker_style.color)
        if marker_style.size:
            line_obj.set_markersize(marker_style.size)
    else:
        line_obj.set_marker("")


def add_line_collection(
    ax: Axes,
    x: Union[list[list[float]], list[npt.NDArray]],
    y: Union[list[list[float]], list[npt.NDArray]],
    label: Optional[str] = None,
    line_style: Optional[LineStyle] = None,
    marker_style: Optional[MarkerStyle] = None,
) -> None:
    """Add line collection to Matplotlib axes.

    Args:
        ax: Matplotlib axes.
        x: X data collection.
        y: Y data collection.
        label: Label to be used for line collection in legend.
        line_style: Line style.
        marker_style: Marker style.
    """
    if all(isinstance(item, np.ndarray) for item in x):
        collection = LineCollection(
            [np.column_stack((np.asarray(xc), np.asarray(yc))) for xc, yc in zip(x, y)]
        )
    else:
        collection = LineCollection([list(zip(xc, yc)) for xc, yc in zip(x, y)])
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
    else:
        line_obj.set_linestyle = None
    if marker_style:
        if marker_style.color:
            line_obj.set_markeredgecolor(marker_style.color)
            line_obj.set_markerfacecolor(marker_style.color)
        if marker_style.type:
            line_obj.set_marker(marker_style.type.name.lower())
        if marker_style.size:
            line_obj.set_markersize(marker_style.size)
    else:
        line_obj.set_marker = None


def add_horizontal_line(
    ax: Axes,
    y: float,
    label: Optional[str] = None,
    line_style: Optional[LineStyle] = None,
) -> None:
    """Add horizontal line to Matplotlib axes.

    Args:
        ax: Matplotlib axes.
        y: Constant Y value for horizontal line.
        label: Label to be used for line in legend.
        line_style: Line style.
    """
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


def add_vertical_line(
    ax: Axes,
    x: float,
    label: Optional[str] = None,
    line_style: Optional[LineStyle] = None,
) -> None:
    """Add vertical line to Matplotlib axes.

    Args:
        ax: Matplotlib axes.
        x: Constant X value for vertical line.
        label: Label to be used for line in legend.
        line_style: Line style.
    """
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
    hlines: Optional[
        list[tuple[float, Optional[str], LineStyle, Optional[Axes]]]
    ] = None,
    vlines: Optional[
        list[tuple[float, Optional[str], LineStyle, Optional[Axes]]]
    ] = None,
    title: Optional[str] = None,
    legend_style: Optional[LegendStyle] = None,
    show_grid: bool = False,
    filename: Optional[str] = None,
    hidden: bool = False,
) -> Figure:
    """Create and plot Matplotlib figure.

    Args:
        data: Line configurations.
        figure: Optional Matplotlib figure to use for plotting.
        hlines: Horizontal line configurations.
        vlines: Vertical line configurations.
        title: Figure title.
        legend_style: Legend style. If None, then legend is hidden.
        show_grid: If True, use grid in plots.
        filename: If filename is provided as path str, then figure is saved as png.
        hidden: If False, then plot is not displayed. Useful if method is used for saving only.

    Returns:
        Matplotlib Figure.
    """
    # TODO: Validate that all dimensions match.
    if figure is None:
        figure, ax = plt.subplots(1, 1)
    figure.set_layout_engine("constrained")
    if title:
        figure.suptitle(title)
    for (
        x,
        y,
        label,
        xtitle,
        ytitle,
        xlim,
        ylim,
        xticks,
        yticks,
        line_style,
        marker_style,
        ax,
    ) in data:
        if ax is None:
            ax = figure.axes[0]
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            add_line(ax, x, y, label, line_style, marker_style)
        elif all(isinstance(item, float) for item in x) and all(
            isinstance(item, float) for item in y
        ):
            add_line(ax, x, y, label, line_style, marker_style)  # type: ignore
        else:
            add_line_collection(ax, x, y, label, line_style, marker_style)  # type: ignore
        if label and legend_style:
            ax.legend(
                loc=legend_style.location.value, bbox_to_anchor=legend_style.anchor
            )
        if xtitle:
            ax.set_xlabel(xtitle)
        if ytitle:
            ax.set_ylabel(ytitle)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if xticks:
            ax.set_xticks(xticks)
        if yticks:
            ax.set_yticks(yticks)
    if hlines:
        for y, label, line_style, ax in hlines:  # type: ignore
            if ax is None:
                ax = figure.axes[0]
            add_horizontal_line(ax, y, label, line_style)  # type: ignore
            if label and legend_style:
                ax.legend(
                    loc=legend_style.location.value, bbox_to_anchor=legend_style.anchor
                )
    if vlines:
        for x, label, line_style, ax in vlines:  # type: ignore
            if ax is None:
                ax = figure.axes[0]
            add_vertical_line(ax, x, label, line_style)  # type: ignore
            if label and legend_style:
                ax.legend(
                    loc=legend_style.location.value, bbox_to_anchor=legend_style.anchor
                )
    if show_grid:
        plt.grid()
    if filename:
        plt.savefig(filename)
    if not hidden:
        plt.show()
    return figure


def plot_bloch(
    x: Union[list[float], npt.NDArray],
    y: Union[list[float], npt.NDArray],
    z: Union[list[float], npt.NDArray],
    filename: Optional[str] = None,
    hidden: bool = False,
) -> Figure:
    """Create and plot Matplotlib figure.

    Args:
        x: X points of trajectory.
        y: Y points of trajectory.
        z: Z points of trajectory.
        filename: If filename is provided as path str, then figure is saved as png.
        hidden: If False, then plot is not displayed. Useful if method is used for saving only.

    Returns:
        Matplotlib Figure.
    """
    b = qutip.Bloch()
    b.point_color = ["g", "#000"]
    b.point_size = [128]
    b.vector_color = "r"
    b.add_points([x[0], y[0], z[0]], meth="s")
    b.add_points([x, y, z], meth="l")
    b.add_vectors([x[-1], y[-1], z[-1]])
    b.render()
    if filename:
        b.save(filename)
    if not hidden:
        b.show()
    return b.fig


def plot_signal(
    schedule: Union[Schedule, ScheduleBlock],
    dt: float,
    channel: str,
    carrier_frequency: float,
    duration: float,
    number_of_samples: int = 10000,
    filename: Optional[str] = None,
    hidden: bool = False,
) -> Figure:
    """Create and plot Matplotlib figure.

    Args:
        schedule: Qiskit pulse schedule.
        dt: Sampling time interval.
        channel: Qiskit pulse channel.
        carrier_frequency: Carrier frequency used for signal.
        duration: Total signal duration used for plotting.
        number_of_samples: Number of samples to use for plotting.
        filename: If filename is provided as path str, then figure is saved as png.
        hidden: If False, then plot is not displayed. Useful if method is used for saving only.

    Returns:
        Matplotlib Figure.
    """
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
        x=signal_times,
        y=signal_samples,
        label="Signal",
        xtitle="Time (ns)",
        ytitle="Amplitude",
        xlim=(0, duration),
        ylim=(amplitude_min, amplitude_max),
        line_style=LineStyle(size=0.1),
        ax=ax1,
    )
    config2 = LineConfig(
        x=[signal_times, signal_times],
        y=[signal_envelope_abs, -signal_envelope_abs],
        label="Envelope",
        xtitle="Time (ns)",
        ytitle="Amplitude",
        xlim=(0, duration),
        ylim=(amplitude_min, amplitude_max),
        line_style=LineStyle(color="black", size=1.5),
        ax=ax1,
    )
    config3 = LineConfig(
        x=signal_times,
        y=signal_envelope_real,
        label="Envelope (real)",
        xtitle="Time (ns)",
        ytitle="Amplitude",
        xlim=(0, duration),
        ylim=(amplitude_min, amplitude_max),
        line_style=LineStyle(size=1),
        ax=ax1,
    )
    config4 = LineConfig(
        x=signal_times,
        y=signal_envelope_img,
        label="Envelope (imaginary)",
        xtitle="Time (ns)",
        ytitle="Amplitude",
        xlim=(0, duration),
        ylim=(amplitude_min, amplitude_max),
        line_style=LineStyle(size=1),
        ax=ax1,
    )
    ylim_phase = None if phase_min == phase_max else (phase_min, phase_max)
    config5 = LineConfig(
        x=signal_times,
        y=signal_envelope_phase,
        label=None,
        xtitle="Time (ns)",
        ytitle="Phase (radians)",
        xlim=(0, duration),
        ylim=ylim_phase,
        line_style=LineStyle(color="black", size=1),
        ax=ax2,
    )
    plot(
        data=[config1, config2, config3, config4, config5],
        figure=figure,
        legend_style=LegendStyle(),
        filename=filename,
        hidden=hidden,
    )
    return figure
