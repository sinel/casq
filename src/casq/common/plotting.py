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
from typing import NamedTuple, Optional, Sequence, Union

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.typing as npt
import qutip

from casq.common.helpers import SignalData, TimeUnit

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


class LineData(NamedTuple):
    """Line configuration."""

    x: Union[list[int], list[float], npt.NDArray]
    y: Union[list[int], list[float], npt.NDArray]
    z: Optional[Union[list[int], list[float], npt.NDArray]] = None


class LineConfig(NamedTuple):
    """Line configuration."""

    data: LineData
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


class LineCollectionConfig(NamedTuple):
    """Line collection configuration."""

    data: list[LineData]
    label: Optional[str] = None
    xtitle: Optional[str] = None
    ytitle: Optional[str] = None
    xlim: Optional[tuple[float, float]] = None
    ylim: Optional[tuple[float, float]] = None
    xticks: Optional[list[float]] = None
    yticks: Optional[list[float]] = None
    line_style: Optional[LineStyle] = None
    ax: Optional[Axes] = None


def add_line(
    ax: Axes,
    data: LineData,
    label: Optional[str] = None,
    line_style: Optional[LineStyle] = None,
    marker_style: Optional[MarkerStyle] = None,
) -> None:
    """Add line to Matplotlib axes.

    Args:
        ax: Matplotlib axes.
        data: Line data.
        label: Label to be used for line in legend.
        line_style: Line style.
        marker_style: Marker style.
    """
    line_obj = ax.plot(data.x, data.y)[0]
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
    data: list[LineData],
    label: Optional[str] = None,
    line_style: Optional[LineStyle] = None,
) -> None:
    """Add line collection to Matplotlib axes.

    Args:
        ax: Matplotlib axes.
        data: Line collection data.
        label: Label to be used for line collection in legend.
        line_style: Line style.
    """
    if isinstance(data[0].x, np.ndarray):
        collection = LineCollection(
            [np.column_stack((line_data.x, line_data.y)) for line_data in data]
        )
    else:
        collection = LineCollection(
            [list(zip(line_data.x, line_data.y)) for line_data in data]
        )
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
    configs: Sequence[Union[LineConfig, LineCollectionConfig]],
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
) -> Union[Axes, list[Axes]]:
    """Create and plot Matplotlib figure.

    Args:
        configs: Line configurations.
        figure: Optional Matplotlib figure to use for plotting.
        hlines: Horizontal line configurations.
        vlines: Vertical line configurations.
        title: Figure title.
        legend_style: Legend style. If None, then legend is hidden.
        show_grid: If True, use grid in plots.
        filename: If filename is provided as path str, then figure is saved as png.
        hidden: If False, then plot is not displayed. Useful if method is used for saving only.

    Returns:
        Matplotlib Axes.
    """
    # TODO: Validate that all dimensions match.
    if figure is None:
        figure, ax = plt.subplots(1, 1)
    figure.set_layout_engine("constrained")
    if title:
        figure.suptitle(title)
    for config in configs:
        if isinstance(config, LineConfig):
            (
                line_data,
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
            ) = config
            if ax is None:
                ax = figure.axes[0]
            add_line(ax, line_data, label, line_style, marker_style)
        else:
            (
                line_collection_data,
                label,
                xtitle,
                ytitle,
                xlim,
                ylim,
                xticks,
                yticks,
                line_style,
                ax,
            ) = config
            if ax is None:
                ax = figure.axes[0]
            add_line_collection(ax, line_collection_data, label, line_style)
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
        for y, label, line_style, ax in hlines:
            if ax is None:
                ax = figure.axes[0]
            add_horizontal_line(ax, y, label, line_style)
            if label and legend_style:
                ax.legend(
                    loc=legend_style.location.value, bbox_to_anchor=legend_style.anchor
                )
    if vlines:
        for x, label, line_style, ax in vlines:
            if ax is None:
                ax = figure.axes[0]
            add_vertical_line(ax, x, label, line_style)
            if label and legend_style:
                ax.legend(
                    loc=legend_style.location.value, bbox_to_anchor=legend_style.anchor
                )
    if show_grid:
        plt.grid()
    if filename:
        plt.savefig(filename)  # pragma: no cover
    if not hidden:
        figure.show()  # pragma: no cover
    return figure.axes


def plot_bloch(
    x: Union[list[float], npt.NDArray],
    y: Union[list[float], npt.NDArray],
    z: Union[list[float], npt.NDArray],
    filename: Optional[str] = None,
    hidden: bool = False,
) -> Axes3D:
    """Create and plot Matplotlib figure.

    Args:
        x: X points of trajectory.
        y: Y points of trajectory.
        z: Z points of trajectory.
        filename: If filename is provided as path str, then figure is saved as png.
        hidden: If False, then plot is not displayed. Useful if method is used for saving only.

    Returns:
        Matplotlib Axes.
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
        b.save(filename)  # pragma: no cover
    if not hidden:
        b.show()  # pragma: no cover
    return b.axes


def plot_signal(
    signal_data: SignalData,
    duration: Optional[int] = None,
    start: int = 0,
    number_of_samples: int = 1000,
    time_unit: TimeUnit = TimeUnit.NANO_SEC,
    filename: Optional[str] = None,
    hidden: bool = False,
) -> list[Axes]:
    """Create and plot Matplotlib figure.

    Args:
        signal_data: Signal data.
        duration: Signal duration in number of dt intervals.
        start: Start time in number of dt intervals.
        number_of_samples: Number of samples to use for plotting.
        time_unit: Time unit used for scaling x-axis.
        filename: If filename is provided as path str, then figure is saved as png.
        hidden: If False, then plot is not displayed. Useful if method is used for saving only.

    Returns:
        Matplotlib Axes.
    """
    duration = duration if duration else int(signal_data.duration)
    signal_times = np.linspace(
        start * signal_data.dt, (start + duration) * signal_data.dt, number_of_samples
    )
    signal_samples = signal_data.signal(signal_times)
    if time_unit is TimeUnit.PICO_SEC:
        plot_times = 1e12 * signal_times
        xtitle = "Time (ps)"
    elif time_unit is TimeUnit.NANO_SEC:
        plot_times = 1e9 * signal_times
        xtitle = "Time (ns)"
    elif time_unit is TimeUnit.MICRO_SEC:
        plot_times = 1e6 * signal_times
        xtitle = "Time (us)"
    elif time_unit is TimeUnit.MILLI_SEC:
        plot_times = 1e3 * signal_times
        xtitle = "Time (ms)"
    elif time_unit is TimeUnit.SEC:
        plot_times = signal_samples
        xtitle = "Time (sec)"
    else:
        plot_times = np.asarray(range(len(signal_times)))
        xtitle = "Time (samples)"
    start_time = plot_times[0]
    end_time = plot_times[-1]
    xlim = (start_time, end_time)
    signal_samples_phase = np.angle(signal_data.signal(signal_times))
    signal_envelope_abs = np.abs(signal_data.signal.envelope(signal_times))
    i_signal_samples = signal_data.i_signal(signal_times)
    q_signal_samples = signal_data.q_signal(signal_times)
    figure, axs = plt.subplots(3, 2, sharex="all", sharey="all")
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    config1 = LineConfig(
        data=LineData(plot_times, signal_samples),
        label="Signal",
        xtitle=xtitle,
        ytitle="Amplitude",
        xlim=xlim,
        line_style=LineStyle(color="r"),
        ax=ax1,
    )
    config2 = LineCollectionConfig(
        data=[
            LineData(plot_times, signal_envelope_abs),
            LineData(plot_times, -signal_envelope_abs),
        ],
        label="Envelope",
        xtitle=xtitle,
        ytitle="Amplitude",
        xlim=xlim,
        line_style=LineStyle(size=1),
        ax=ax1,
    )
    config3 = LineConfig(
        data=LineData(plot_times, signal_samples_phase),
        label=None,
        xtitle=xtitle,
        ytitle="Phase (radians)",
        xlim=xlim,
        line_style=LineStyle(size=1),
        ax=ax2,
    )
    config4 = LineConfig(
        data=LineData(plot_times, i_signal_samples),
        label="I Component",
        xtitle=xtitle,
        ytitle="Amplitude",
        xlim=xlim,
        line_style=LineStyle(size=1),
        ax=ax3,
    )
    config5 = LineConfig(
        data=LineData(plot_times, q_signal_samples),
        label="Q Component",
        xtitle=xtitle,
        ytitle="Amplitude",
        xlim=xlim,
        line_style=LineStyle(size=1),
        ax=ax3,
    )
    plot(
        configs=[config1, config2, config3, config4, config5],
        figure=figure,
        legend_style=LegendStyle(location=LegendLocation.UPPER_LEFT, anchor=(1, 1)),
        filename=filename,
        hidden=hidden,
    )
    axes_list: list[Axes] = figure.axes
    return axes_list
