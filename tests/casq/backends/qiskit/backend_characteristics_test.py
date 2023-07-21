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

import pytest
from qiskit.providers import BackendV1
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
from casq.common.exceptions import CasqError


def test_backend_properties_init(backend: BackendV1) -> None:
    """Unit test for BackendCharacteristics initialization."""
    backend_characteristics = BackendCharacteristics(backend)
    assert isinstance(backend_characteristics.config, PulseBackendConfiguration)
    assert isinstance(backend_characteristics.properties, BackendProperties)
    assert isinstance(backend_characteristics.defaults, PulseDefaults)


def test_get_qubit_properties(backend: BackendV1) -> None:
    """Unit test for BackendCharacteristics.get_qubit_properties."""
    backend_characteristics = BackendCharacteristics(backend)
    qubit_properties = backend_characteristics.get_qubit_properties(0)
    assert isinstance(qubit_properties, BackendCharacteristics.QubitProperties)


def test_get_gate_properties(backend: BackendV1) -> None:
    """Unit test for BackendCharacteristics.get_qubit_properties."""
    backend_characteristics = BackendCharacteristics(backend)
    gate_properties = backend_characteristics.get_gate_properties("x")
    assert isinstance(gate_properties, BackendCharacteristics.GateProperties)


def test_get_channel_frequencies_by_letter(backend: BackendV1) -> None:
    """Unit test for BackendCharacteristics.get_channel_frequencies by letter."""
    backend_characteristics = BackendCharacteristics(backend)
    freqs = backend_characteristics.get_channel_frequencies(["d0", "u0", "m0"])
    assert isinstance(freqs["d0"], float)
    assert isinstance(freqs["u0"], float)
    assert isinstance(freqs["m0"], float)


def test_get_channel_frequencies_by_letter_warning(
    backend: BackendV1, loguru_caplog: pytest.LogCaptureFixture
) -> None:
    """Unit test for BackendCharacteristics.get_channel_frequencies by letter with unrecognized letter warning."""
    backend_characteristics = BackendCharacteristics(backend)
    backend_characteristics.get_channel_frequencies(["x0"])
    assert loguru_caplog.records[0].levelname == "WARNING"
    assert "Unrecognized channel [x0] requested." in loguru_caplog.records[0].message


def test_get_channel_frequencies_by_channel(backend: BackendV1) -> None:
    """Unit test for BackendCharacteristics.get_channel_frequencies by channel."""
    backend_characteristics = BackendCharacteristics(backend)
    freqs = backend_characteristics.get_channel_frequencies(
        [DriveChannel(0), ControlChannel(0), MeasureChannel(0)]
    )
    assert isinstance(freqs["d0"], float)
    assert isinstance(freqs["u0"], float)
    assert isinstance(freqs["m0"], float)


def test_get_channel_frequencies_by_channel_warning(
    backend: BackendV1, loguru_caplog: pytest.LogCaptureFixture
) -> None:
    """Unit test for BackendCharacteristics.get_channel_frequencies by channel with unrecognized channel warning."""
    backend_characteristics = BackendCharacteristics(backend)
    backend_characteristics.get_channel_frequencies([MemorySlot(0)])
    assert loguru_caplog.records[0].levelname == "WARNING"
    assert (
        "Unrecognized channel [MemorySlot(0)] requested."
        in loguru_caplog.records[0].message
    )


def test_get_channel_frequencies_with_too_many_drive_channels(
    backend: BackendV1,
) -> None:
    """Unit test for BackendCharacteristics.get_channel_frequencies by letter."""
    backend_characteristics = BackendCharacteristics(backend)
    with pytest.raises(CasqError) as e:
        backend_characteristics.get_channel_frequencies(
            ["d0", "d1", "d2", "d3", "d4", "d5"]
        )
    assert isinstance(e.value, CasqError)
    assert e.value.message == "DriveChannel index 5 is out of bounds."


def test_get_channel_frequencies_with_too_many_control_channels(
    backend: BackendV1,
) -> None:
    """Unit test for BackendCharacteristics.get_channel_frequencies by letter."""
    backend_characteristics = BackendCharacteristics(backend)
    with pytest.raises(CasqError) as e:
        backend_characteristics.get_channel_frequencies(
            ["u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]
        )
    assert isinstance(e.value, CasqError)
    assert e.value.message == "ControlChannel index 8 is out of bounds."


def test_get_channel_frequencies_with_too_many_measure_channels(
    backend: BackendV1,
) -> None:
    """Unit test for BackendCharacteristics.get_channel_frequencies by letter."""
    backend_characteristics = BackendCharacteristics(backend)
    with pytest.raises(CasqError) as e:
        backend_characteristics.get_channel_frequencies(
            ["m0", "m1", "m2", "m3", "m4", "m5"]
        )
    assert isinstance(e.value, CasqError)
    assert e.value.message == "MeasureChannel index 5 is out of bounds."
