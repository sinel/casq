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

import jax
from qiskit.providers import BackendV1
from qiskit_dynamics.array import Array

from casq.backends.helpers import build_from_backend
from casq.backends.pulse_backend import PulseBackend
from casq.backends.qiskit.qiskit_pulse_backend import QiskitPulseBackend
from casq.common.decorators import timer
from casq.gates.gaussian_pulse_gate import GaussianPulseGate
from casq.gates.pulse_circuit import PulseCircuit

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
Array.set_default_backend("jax")


def test_from_backend(backend: BackendV1) -> None:
    """Unit test for QiskitPulseBackend initialization from backend."""
    pulse_backend = QiskitPulseBackend.from_backend(backend)
    assert isinstance(pulse_backend, QiskitPulseBackend)


def test_seed_option(backend: BackendV1) -> None:
    """Unit test for PulseBackend initialization from backend."""
    pulse_backend = QiskitPulseBackend.from_backend(backend, seed=1)
    assert pulse_backend._seed == 1
    assert pulse_backend._native_backend.options.seed_simulator == 1


@timer(unit="sec")
def test_run(backend: BackendV1) -> None:
    """Unit test for QiskitPulseBackend run."""
    pulse_backend = build_from_backend(backend, extracted_qubits=[0])
    gate = GaussianPulseGate(16, 1)
    circuit = PulseCircuit.from_pulse_gate(gate, {"sigma": 8})
    solution = pulse_backend.solve(
        circuit, method=PulseBackend.ODESolverMethod.SCIPY_DOP853
    )
    assert isinstance(solution, PulseBackend.PulseSolution)


@timer(unit="sec")
def test_jax_run(backend: BackendV1) -> None:
    """Unit test for QiskitPulseBackend run using jax."""
    pulse_backend = build_from_backend(backend, extracted_qubits=[0])
    gate = GaussianPulseGate(16, 1)
    circuit = PulseCircuit.from_pulse_gate(gate, {"sigma": 8})
    solution = pulse_backend.solve(
        circuit, method=PulseBackend.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT
    )
    assert isinstance(solution, PulseBackend.PulseSolution)
