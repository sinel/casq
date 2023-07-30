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

from loguru import logger
import numpy as np
import pytest
from qiskit.providers import Backend
from qiskit_dynamics.array import Array

from casq.backends.helpers import build_from_backend
from casq.backends.pulse_backend import PulseBackend
from casq.common.decorators import timer
from casq.common.exceptions import CasqError
from casq.common.helpers import initialize_jax
from casq.gates.constant_pulse_gate import ConstantPulseGate
from casq.gates.gaussian_pulse_gate import GaussianPulseGate
from casq.optimizers.pulse_optimizer import PulseOptimizer
from casq.optimizers.single_qubit_gates.x_gate_optimizer import XGateOptimizer

initialize_jax()


def test_init(backend: Backend) -> None:
    """Unit test for PulseOptimizer initialization."""
    optimizer = PulseOptimizer(
        pulse_gate=ConstantPulseGate(duration=1, amplitude=1),
        pulse_backend=build_from_backend(backend),
        method=PulseBackend.ODESolverMethod.SCIPY_DOP853,
        target_measurement={"0": 0, "1": 1024},
    )
    assert isinstance(optimizer, PulseOptimizer)


def test_init_jax_enabled(backend: Backend) -> None:
    """Unit test for PulseOptimizer initialization with jax."""
    optimizer = PulseOptimizer(
        pulse_gate=ConstantPulseGate(duration=1, amplitude=1),
        pulse_backend=build_from_backend(backend),
        method=PulseBackend.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT,
        target_measurement={"0": 0, "1": 1024},
    )
    assert isinstance(optimizer, PulseOptimizer)


def test_init_jax_disabled(backend: Backend) -> None:
    """Unit test for PulseOptimizer initialization from backend."""
    Array.set_default_backend("numpy")
    with pytest.raises(CasqError) as e:
        optimizer = PulseOptimizer(
            pulse_gate=ConstantPulseGate(duration=1, amplitude=1),
            pulse_backend=build_from_backend(backend),
            method=PulseBackend.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT,
            target_measurement={"0": 0, "1": 1024},
        )
    assert isinstance(e.value, CasqError)
    assert (
        e.value.message
        == "Jax must be enabled for ODE solver method: QISKIT_DYNAMICS_JAX_ODEINT."
    )
    Array.set_default_backend("jax")


@timer(unit="sec")
def test_optimize(backend: Backend) -> None:
    """Unit test for PulseOptimizer initialization from backend."""
    optimizer = XGateOptimizer(
        pulse_gate=GaussianPulseGate(duration=4, amplitude=1),
        pulse_backend=build_from_backend(backend),
        method=PulseBackend.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT,
        fidelity_type=PulseOptimizer.FidelityType.COUNTS,
    )
    solution = optimizer.solve(
        initial_params=np.array([1.0]),
        method=PulseOptimizer.OptimizationMethod.SCIPY_NELDER_MEAD,
        constraints=[
            {"type": "ineq", "fun": lambda x: x[0]},
            {"type": "ineq", "fun": lambda x: 4 - x[0]},
        ],
        tol=1,
        maxiter=10,
    )
    assert isinstance(solution, PulseOptimizer.Solution)
    solution.plot_objective_history(hidden=True)
    solution.plot_parameter_history(hidden=True)
