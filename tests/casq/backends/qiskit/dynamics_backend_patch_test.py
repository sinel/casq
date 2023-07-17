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
from qiskit.providers import BackendV1
from qiskit.pulse import Schedule
from qiskit_dynamics.backend.dynamics_backend import DynamicsJob

from casq.backends import DynamicsBackendPatch, PulseSolution, QiskitPulseBackend
from casq.common import timer


def test_options_to_dict() -> None:
    """Unit test for PulseSimulator initialization from backend."""
    options = DynamicsBackendPatch.Options()
    options_dict = options.to_dict()
    assert options_dict.get("shots", None) == 1024
    assert not ("configuration" in options_dict)


def test_from_backend(backend: BackendV1) -> None:
    """Unit test for PulseSimulator initialization from backend."""
    dynamics_backend = DynamicsBackendPatch.from_backend(backend)
    assert isinstance(dynamics_backend, DynamicsBackendPatch)


@timer(unit="sec")
def test_run(backend: BackendV1, pulse_schedule: Schedule) -> None:
    """Unit test for PulseSimulator initialization from backend."""
    dynamics_backend = DynamicsBackendPatch.from_backend(backend)
    dynamics_backend.set_options(shots=5)
    dynamics_backend.steps = 10
    result = dynamics_backend.run([pulse_schedule])
    assert isinstance(result, DynamicsJob)
