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
"""Base object tests."""
from __future__ import annotations

from matplotlib.figure import Figure
from matplotlib.legend import Legend
import matplotlib.pyplot as plt
import numpy as np

from casq.common import (
    LegendLocation,
    LegendStyle,
    LineConfig,
    LineData,
    LineStyle,
    LineType,
    MarkerStyle,
    MarkerType,
    add_horizontal_line,
    add_line,
    add_line_collection,
    add_vertical_line,
    plot,
    plot_bloch,
    plot_signal,
)


def test_add_line() -> None:
    """Unit test for add_line."""
    figure, ax = plt.subplots(1, 1)
    line_style = LineStyle(type=LineType.DASHED, color="blue", size=5)
    marker_style = MarkerStyle(type=MarkerType.X, color="red", size=20)
    add_line(
        ax,
        LineData([0, 1], [0, 1]),
        label="hello",
        line_style=line_style,
        marker_style=marker_style,
    )
    assert len(ax.lines) == 1
    line = ax.lines[0]
    assert line.get_label() == "hello"
    assert line.get_linestyle() == "--"
    assert line.get_color() == "blue"
    assert line.get_linewidth() == 5
    assert line.get_marker() == "x"
    assert line.get_markeredgecolor() == "red"
    assert line.get_markerfacecolor() == "red"
    assert line.get_markersize() == 20


def test_add_line_collection_with_floats() -> None:
    """Unit test for add_line_collection."""
    figure, ax = plt.subplots(1, 1)
    line_style = LineStyle(type=LineType.DASHED, color="blue", size=5)
    data = [LineData([0, 1], [0, 1]), LineData([0, 1], [0, 2])]
    add_line_collection(ax, data, label="hello", line_style=line_style)
    assert len(ax.collections) == 1
    collection = ax.collections[0]
    assert len(collection.get_segments()) == 2
    assert collection.get_label() == "hello"
    assert collection.get_linestyle()[0][0] == 0
    assert len(collection.get_linestyle()[0][1]) == 2
    assert all(collection.get_color()[0] == [0, 0, 1, 1])
    assert collection.get_linewidth() == 5


def test_add_line_collection_with_ndarray() -> None:
    """Unit test for add_line_collection."""
    figure, ax = plt.subplots(1, 1)
    data = [
        LineData(np.asarray([0, 1]), np.asarray([0, 1])),
        LineData(np.asarray([0, 1]), np.asarray([0, 2])),
    ]
    add_line_collection(ax, data)
    assert len(ax.collections) == 1
    assert len(ax.collections[0].get_segments()) == 2


def test_add_horizontal_line() -> None:
    """Unit test for add_horizontal_line."""
    figure, ax = plt.subplots(1, 1)
    line_style = LineStyle(type=LineType.DASHED, color="blue", size=5)
    add_horizontal_line(ax, 1, label="hello", line_style=line_style)
    assert len(ax.lines) == 1
    assert len(ax.lines) == 1
    line = ax.lines[0]
    assert line.get_label() == "hello"
    assert line.get_linestyle() == "--"
    assert line.get_color() == "blue"
    assert line.get_linewidth() == 5


def test_add_vertical_line() -> None:
    """Unit test for add_vertical_line."""
    figure, ax = plt.subplots(1, 1)
    line_style = LineStyle(type=LineType.DASHED, color="blue", size=5)
    add_vertical_line(ax, 1, label="hello", line_style=line_style)
    assert len(ax.lines) == 1
    assert len(ax.lines) == 1
    line = ax.lines[0]
    assert line.get_label() == "hello"
    assert line.get_linestyle() == "--"
    assert line.get_color() == "blue"
    assert line.get_linewidth() == 5


def test_plot() -> None:
    """Unit test for plot."""
    config = LineConfig(
        LineData([0, 1], [0, 1]),
        xtitle="x_title",
        ytitle="y_title",
        xlim=(0, 1),
        ylim=(0, 1),
        xticks=[0, 0.5, 1],
        yticks=[0, 0.5, 1],
    )
    line_style = LineStyle(type=LineType.DASHED, color="blue", size=5)
    legend_style = LegendStyle(location=LegendLocation.CENTER, anchor=(0, 0))
    figure = plot(
        [config],
        hlines=[(1, "hline_label", line_style, None)],
        vlines=[(1, "vline_label", line_style, None)],
        legend_style=legend_style,
        show_grid=True,
        title="figure_title",
        hidden=True,
    )
    ax = figure.axes[0]
    legends = [child for child in ax.get_children() if isinstance(child, Legend)]
    hline_label = legends[0].get_texts()[0].get_text()
    vline_label = legends[0].get_texts()[1].get_text()
    assert isinstance(figure, Figure)
    assert figure._suptitle.get_text() == "figure_title"
    assert hline_label == "hline_label"
    assert vline_label == "vline_label"
    assert legends[0].get_alignment() == "center"
    assert all(legends[0].get_bbox_to_anchor()._bbox.get_points()[0] == [0, 0])
    assert any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())
    assert ax.get_xlabel() == "x_title"
    assert ax.get_ylabel() == "y_title"
    assert ax.get_xlim() == (0, 1)
    assert ax.get_ylim() == (0, 1)
    assert all(ax.get_xticks() == [0, 0.5, 1])
    assert all(ax.get_yticks() == [0, 0.5, 1])


def test_plot_bloch() -> None:
    """Unit test for plot_bloch."""
    vec = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    figure = plot_bloch([1, 0, 0], [0, 1, 0], [0, 0, 1], hidden=True)
    assert isinstance(figure, Figure)


def test_plot_signal(pulse_schedule) -> None:
    """Unit test for plot_signal."""
    figure = plot_signal(pulse_schedule, 1, "", 1, 1, hidden=True)
    assert isinstance(figure, Figure)
