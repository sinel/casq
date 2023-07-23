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
"""Pulse gate tests."""
from __future__ import annotations

from datetime import datetime

from matplotlib.figure import Figure
from qiskit.quantum_info import Statevector

from casq.backends.pulse_solution import PulseSolution


def mock_solution() -> PulseSolution:
    """Mock pulse backend solution."""
    instance: PulseSolution = PulseSolution(
        "test",
        [0],
        times=[0, 1],
        samples=[[0, 1, 1, 0, 0], [1, 0, 1, 0, 1]],
        counts=[{"0": 1024}, {"0": 500, "1": 524}],
        populations=[{"0": 1}, {"0": 0.49, "1": 0.51}],
        states=[
            Statevector([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            Statevector([0.5 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j]),
        ],
        iq_data=[
            [(0.4, 0.6), (0.6, 0.4), (0.6, 0.4), (0.4, 0.6), (0.4, 0.6)],
            [(0.6, 0.4), (0.4, 0.6), (0.6, 0.4), (0.4, 0.6), (0.6, 0.4)],
        ],
        avg_iq_data=[(0.5, 0.5), (0.5, 0.5)],
        shots=5,
        is_success=True,
        timestamp=datetime.timestamp(datetime(2000, 1, 1)),
    )
    return instance


def test_pulse_solution_init() -> None:
    """Unit test for PulseSolution initialization."""
    solution = mock_solution()
    assert solution.circuit_name == "test"
    assert solution.times == [0, 1]
    assert solution.qubits == [0]
    assert solution.is_success
    assert solution.samples == [[0, 1, 1, 0, 0], [1, 0, 1, 0, 1]]
    assert solution.counts == [{"0": 1024}, {"0": 500, "1": 524}]
    assert solution.populations == [{"0": 1}, {"0": 0.49, "1": 0.51}]
    assert all(solution.states[0].data == [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])
    assert all(solution.states[1].data == [0.5 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j])
    assert solution.iq_data == [
        [(0.4, 0.6), (0.6, 0.4), (0.6, 0.4), (0.4, 0.6), (0.4, 0.6)],
        [(0.6, 0.4), (0.4, 0.6), (0.6, 0.4), (0.4, 0.6), (0.6, 0.4)],
    ]
    assert solution.avg_iq_data == [(0.5, 0.5), (0.5, 0.5)]
    assert solution.shots == 5
    assert solution.seed is None
    assert solution.timestamp == datetime.timestamp(datetime(2000, 1, 1))


def test_plot_population() -> None:
    """Unit test for PulseSolution.plot_population."""
    solution = mock_solution()
    solution.plot_population(hidden=True)


def test_plot_iq() -> None:
    """Unit test for PulseSolution.plot_iq."""
    solution = mock_solution()
    solution.plot_iq(hidden=True)


def test_plot_iq_trajectory() -> None:
    """Unit test for PulseSolution.plot_iq_trajectory."""
    solution = mock_solution()
    solution.plot_iq_trajectory(hidden=True)


def test_plot_trajectory() -> None:
    """Unit test for PulseSolution.plot_trajectory."""
    solution = mock_solution()
    solution.plot_trajectory(hidden=True)


def test_plot_bloch_trajectory() -> None:
    """Unit test for PulseSolution.plot_bloch_trajectory."""
    solution = mock_solution()
    solution.plot_bloch_trajectory(hidden=True)
