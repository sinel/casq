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

from casq.backends.pulse_backend import PulseBackend
from casq.backends.qiskit.qiskit_pulse_backend import QiskitPulseBackend
from casq.common.decorators import timer
from casq.common.exceptions import CasqError
from casq.pulse_optimizer import PulseOptimizer


def test_init_gaussian(backend: Backend) -> None:
    """Unit test for PulseSimulator initialization from backend."""
    pulse_backend = QiskitPulseBackend.from_backend(backend)
    optimizer = PulseOptimizer(
        pulse_type=PulseOptimizer.PulseType.GAUSSIAN,
        pulse_arguments={
            "duration": 230,
            "amplitude": 1,
            "name": "x",
            "sigma": None,
        },
        backend=pulse_backend,
        method=PulseBackend.ODESolverMethod.SCIPY_DOP853,
        target_measurement={"0": 0, "1": 1024},
    )
    assert isinstance(optimizer, PulseOptimizer)


def test_init_gaussian_square(backend: Backend) -> None:
    """Unit test for PulseSimulator initialization from backend."""
    pulse_backend = QiskitPulseBackend.from_backend(backend)
    optimizer = PulseOptimizer(
        pulse_type=PulseOptimizer.PulseType.GAUSSIAN_SQUARE,
        pulse_arguments={
            "duration": 230,
            "amplitude": 1,
            "name": "x",
            "sigma": None,
            "width": None,
        },
        backend=pulse_backend,
        method=PulseBackend.ODESolverMethod.SCIPY_DOP853,
        target_measurement={"0": 0, "1": 1024},
    )
    assert isinstance(optimizer, PulseOptimizer)


def test_init_drag(backend: Backend) -> None:
    """Unit test for PulseSimulator initialization from backend."""
    pulse_backend = QiskitPulseBackend.from_backend(backend)
    optimizer = PulseOptimizer(
        pulse_type=PulseOptimizer.PulseType.DRAG,
        pulse_arguments={
            "duration": 230,
            "amplitude": 1,
            "name": "x",
            "sigma": None,
            "beta": None,
        },
        backend=pulse_backend,
        method=PulseBackend.ODESolverMethod.SCIPY_DOP853,
        target_measurement={"0": 0, "1": 1024},
    )
    assert isinstance(optimizer, PulseOptimizer)


def test_init_jit(backend: Backend) -> None:
    """Unit test for PulseSimulator initialization from backend."""
    pulse_backend = QiskitPulseBackend.from_backend(backend)
    optimizer = PulseOptimizer(
        pulse_type=PulseOptimizer.PulseType.GAUSSIAN_SQUARE,
        pulse_arguments={
            "duration": 230,
            "amplitude": 1,
            "name": "x",
            "sigma": None,
            "width": None,
        },
        backend=pulse_backend,
        method=PulseBackend.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT,
        target_measurement={"0": 0, "1": 1024},
        use_jax=True,
        use_jit=True,
    )
    assert isinstance(optimizer, PulseOptimizer)


def test_jax_with_invalid_method(backend: Backend) -> None:
    """Unit test for PulseSimulator initialization from backend."""
    pulse_backend = QiskitPulseBackend.from_backend(backend)
    with pytest.raises(CasqError) as e:
        PulseOptimizer(
            pulse_type=PulseOptimizer.PulseType.GAUSSIAN_SQUARE,
            pulse_arguments={
                "duration": 230,
                "amplitude": 1,
                "name": "x",
                "sigma": None,
                "width": None,
            },
            backend=pulse_backend,
            method=PulseBackend.ODESolverMethod.SCIPY_DOP853,
            target_measurement={"0": 0, "1": 1024},
            use_jax=True,
        )
    assert isinstance(e.value, CasqError)
    assert (
        e.value.message
        == "If 'jax' is enabled, a jax-compatible ODE solver method is required."
    )


@timer(unit="sec")
def test_optimize(backend: Backend) -> None:
    """Unit test for PulseSimulator initialization from backend."""
    pulse_backend = QiskitPulseBackend.from_backend(backend)
    optimizer = PulseOptimizer(
        pulse_type=PulseOptimizer.PulseType.GAUSSIAN_SQUARE,
        pulse_arguments={
            "duration": 4,
            "amplitude": 1,
            "name": "x",
            "sigma": None,
            "width": None,
        },
        backend=pulse_backend,
        method=PulseBackend.ODESolverMethod.SCIPY_DOP853,
        target_measurement={"0": 0, "1": 1024},
    )
    initial_params = np.array([1.0, 1.0])
    solution = optimizer.optimize(
        initial_params,
        method=PulseOptimizer.OptimizationMethod.SCIPY_NELDER_MEAD,
        tol=1.0,
        maxiter=10,
    )
    logger.debug(solution)
    assert isinstance(solution, PulseOptimizer.Solution)
