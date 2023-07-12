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
"""BackendCharacteristics tests."""
from __future__ import annotations

from pytest import LogCaptureFixture
from qiskit.providers.models import (
    BackendProperties,
    PulseBackendConfiguration,
    PulseDefaults,
)
from qiskit.pulse.channels import (
    ControlChannel,
    DriveChannel,
    MeasureChannel,
    MemorySlot,
)

from casq.backends.qiskit.backend_characteristics import BackendCharacteristics


def test_get_backend() -> None:
    """Unit test for BackendCharacteristics.get_backend."""
    backend = BackendCharacteristics.get_backend("ibmq_manila")
    assert backend.name == "ibmq_manila"


def test_backend_properties_init() -> None:
    """Unit test for BackendCharacteristics initialization."""
    backend_characteristics = BackendCharacteristics("ibmq_manila")
    assert isinstance(backend_characteristics.config, PulseBackendConfiguration)
    assert isinstance(backend_characteristics.properties, BackendProperties)
    assert isinstance(backend_characteristics.defaults, PulseDefaults)


def test_get_qubit_properties() -> None:
    """Unit test for BackendCharacteristics.get_qubit_properties."""
    backend_characteristics = BackendCharacteristics("ibmq_manila")
    qubit_properties = backend_characteristics.get_qubit_properties(0)
    assert isinstance(qubit_properties, BackendCharacteristics.QubitProperties)


def test_get_channel_frequencies_by_letter() -> None:
    """Unit test for BackendCharacteristics.get_channel_frequencies by letter."""
    backend_characteristics = BackendCharacteristics("ibmq_manila")
    freqs = backend_characteristics.get_channel_frequencies(["d0", "u0", "m0"])
    assert isinstance(freqs["d0"], float)
    assert isinstance(freqs["u0"], float)
    assert isinstance(freqs["m0"], float)


def test_get_channel_frequencies_by_letter_warning(loguru_caplog: LogCaptureFixture) -> None:
    """Unit test for BackendCharacteristics.get_channel_frequencies by letter with unrecognized letter warning."""
    backend_characteristics = BackendCharacteristics("ibmq_manila")
    backend_characteristics.get_channel_frequencies(["x0"])
    assert loguru_caplog.records[0].levelname == "WARNING"
    assert "Unrecognized channel [x0] requested." in loguru_caplog.records[0].message


def test_get_channel_frequencies_by_channel() -> None:
    """Unit test for BackendCharacteristics.get_channel_frequencies by channel."""
    backend_characteristics = BackendCharacteristics("ibmq_manila")
    freqs = backend_characteristics.get_channel_frequencies(
        [DriveChannel(0), ControlChannel(0), MeasureChannel(0)]
    )
    assert isinstance(freqs["d0"], float)
    assert isinstance(freqs["u0"], float)
    assert isinstance(freqs["m0"], float)


def test_get_channel_frequencies_by_channel_warning(loguru_caplog: LogCaptureFixture) -> None:
    """Unit test for BackendCharacteristics.get_channel_frequencies by channel with unrecognized channel warning."""
    backend_characteristics = BackendCharacteristics("ibmq_manila")
    backend_characteristics.get_channel_frequencies([MemorySlot(0)])
    assert loguru_caplog.records[0].levelname == "WARNING"
    assert (
        "Unrecognized channel [MemorySlot(0)] requested."
        in loguru_caplog.records[0].message
    )
