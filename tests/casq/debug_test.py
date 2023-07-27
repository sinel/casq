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
"""Tests used for debugging."""
from __future__ import annotations

import numpy as np
from qiskit.providers.fake_provider import FakeManila

from casq import PulseOptimizer
from casq.backends.helpers import build_from_backend
from casq.backends.pulse_backend import PulseBackend
from casq.common.decorators import timer
from casq.common.helpers import initialize_jax
from casq.gates import GaussianSquarePulseGate

initialize_jax()


@timer(unit="sec")
def exclude_test_debug() -> None:
    """Unit test for debugging."""
    backend = FakeManila()
    dt = backend.configuration().dt
    pulse_backend = build_from_backend(backend, [0])
    pulse_gate = GaussianSquarePulseGate(duration=256, amplitude=1, name="x")
    optimizer = PulseOptimizer(
        pulse_gate=pulse_gate,
        pulse_backend=pulse_backend,
        method=PulseBackend.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT,
        method_options={"method": "jax_odeint", "atol": 1e-6, "rtol": 1e-8, "hmax": dt},
        target_measurement={"0": 0, "1": 1024},
        fidelity_type=PulseOptimizer.FidelityType.COUNTS,
        target_qubit=0,
    )
    solution = optimizer.run(
        initial_params=np.array([10.0, 10.0]),
        method=PulseOptimizer.OptimizationMethod.SCIPY_NELDER_MEAD,
        bounds=[(0, 256), (0, 256)],
        constraints=[
            {"type": "ineq", "fun": lambda x: x[0]},
            {"type": "ineq", "fun": lambda x: 256 - x[0]},
            {"type": "ineq", "fun": lambda x: x[1]},
            {"type": "ineq", "fun": lambda x: 256 - x[1]},
        ],
        tol=1e-2,
    )
    print(
        "================================================================================"
    )
    print("OPTIMIZED PULSE")
    print(
        "================================================================================"
    )
    print(f"ITERATIONS: {solution.num_iterations}")
    print(f"OPTIMIZED PARAMETERS: {solution.parameters}")
    print(f"MEASUREMENT: {solution.measurement}")
    print(f"FIDELITY: {solution.fidelity}")
    print(f"MESSAGE: {solution.message}")
    print(
        "================================================================================"
    )
