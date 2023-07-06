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
"""Unit test configuration."""
from __future__ import annotations

import logging
from typing import Generator

from loguru import logger
import pytest
from qiskit import pulse

from casq.common import PulseBackendProperties


@pytest.fixture
def backend():
    return PulseBackendProperties.get_backend("ibmq_manila")


@pytest.fixture
def backend_properties():
    return PulseBackendProperties("ibmq_manila")


@pytest.fixture
def pulse_schedule():
    gaussian = pulse.library.Gaussian(256, 1, 128, name="Gaussian")
    with pulse.build() as schedule:
        d0 = pulse.DriveChannel(0)
        m0 = pulse.MemorySlot(0)
        with pulse.align_sequential():
            pulse.play(gaussian, d0)
            pulse.acquire(duration=1, qubit_or_channel=0, register=m0)
    return schedule


@pytest.fixture
def loguru_caplog(
    caplog: pytest.LogCaptureFixture,
) -> Generator[pytest.LogCaptureFixture, None, None]:
    """Fixture for capturing loguru logging output via ptest.

    Since pytest links to the standard library’s logging module,
    it is necessary to add a sink that propagates Loguru to the caplog handler.
    This is done by overriding the caplog fixture to capture its handler.
    See:
    https://loguru.readthedocs.io/en/stable/resources/migration.html#replacing-caplog-fixture-from-pytest-library

    Args:
        caplog: The pytest caplog fixture
        which captures logging output so that it can be tested against.
    """

    class PropagateHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropagateHandler(), format="{message}")
    yield caplog
    logger.remove(handler_id)
