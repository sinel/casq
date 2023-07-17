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

from casq.backends import PulseBackend, PulseSolution, QiskitPulseBackend
from casq.common import timer


def test_from_backend(backend: BackendV1) -> None:
    """Unit test for PulseSimulator initialization from backend."""
    pulse_backend = QiskitPulseBackend.from_backend(backend)
    assert isinstance(pulse_backend, QiskitPulseBackend)


def test_seed_option(backend: BackendV1) -> None:
    """Unit test for PulseSimulator initialization from backend."""
    pulse_backend = QiskitPulseBackend.from_backend(backend, seed=1)
    assert pulse_backend._seed == 1
    assert pulse_backend._native_backend.options.seed_simulator == 1


@timer(unit="sec")
def test_run(backend: BackendV1, pulse_schedule: Schedule) -> None:
    """Unit test for PulseSimulator initialization from backend."""
    pulse_backend = QiskitPulseBackend.from_backend(backend)
    solution = pulse_backend.run(
        [pulse_schedule], method=PulseBackend.ODESolverMethod.SCIPY_DOP853, shots=5
    )["test"]
    assert isinstance(solution, PulseSolution)
