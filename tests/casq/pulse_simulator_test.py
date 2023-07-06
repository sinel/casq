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

from matplotlib.figure import Figure
from qiskit.result.models import ExperimentResult, ExperimentResultData

from casq import PulseSimulator


def test_from_backend() -> None:
    """Unit test for PulseSimulator initialization from backend."""
    simulator = PulseSimulator.from_backend("ibmq_manila")
    assert isinstance(simulator, PulseSimulator)


def test_from_backend_with_options() -> None:
    """Unit test for PulseSimulator initialization from backend with options."""
    method = PulseSimulator.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT
    solver_options = {"atol": 1}
    simulator = PulseSimulator.from_backend(
        "ibmq_manila", method=method, solver_options=solver_options
    )
    assert simulator.method == PulseSimulator.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT
    assert (
        simulator.options["solver_options"]["method"]
        == PulseSimulator.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT.value
    )


def test_steps(pulse_schedule) -> None:
    """Unit test for PulseSimulator steps."""
    simulator = PulseSimulator.from_backend("ibmq_manila", qubits=[0], steps=10)
    result = simulator.run([pulse_schedule]).result().results[0]
    assert isinstance(result, ExperimentResult)
    assert isinstance(result.data, ExperimentResultData)
    assert result.shots == simulator.shots
    assert result.data.counts[0] == {"0": 1024}
    assert hasattr(result.data, "qubits")
    assert result.data.qubits == [0]
    assert hasattr(result.data, "times")
    assert result.data.times[0] == 0.
    assert result.data.times[-1] == 256*simulator.dt
    assert hasattr(result.data, "statevectors")
    assert (result.data.statevectors[0].data == [1., 0., 0.]).all()
    assert hasattr(result.data, "populations")
    assert len(result.data.populations) == 10
    assert result.data.populations[0] == {"0": 1.}
    assert hasattr(result.data, "iq_data")
    assert hasattr(result.data, "avg_iq_data")


def test_plot_population(pulse_schedule) -> None:
    """Unit test for PulseSimulator.plot_population."""
    simulator = PulseSimulator.from_backend("ibmq_manila", qubits=[0], steps=10)
    result = simulator.run([pulse_schedule]).result().results[0]
    figure = PulseSimulator.plot_population(result, hidden=True)
    assert isinstance(figure, Figure)
