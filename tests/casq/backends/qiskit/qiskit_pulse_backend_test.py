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

from casq.backends.qiskit.backend_characteristics import BackendCharacteristics
from casq.backends.qiskit.dynamics_backend_patch import DynamicsBackendPatch
from casq.backends.qiskit.qiskit_pulse_backend import QiskitPulseBackend


def test_from_backend() -> None:
    """Unit test for PulseSimulator initialization from backend."""
    qiskit_dynamics_backend = DynamicsBackendPatch.from_backend(
        BackendCharacteristics.get_backend("ibmq_manila")
    )
    backend = QiskitPulseBackend.from_backend(qiskit_dynamics_backend)
    assert isinstance(backend, QiskitPulseBackend)
