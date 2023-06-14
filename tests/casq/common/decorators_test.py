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
from __future__ import annotations

from pytest import LogCaptureFixture

from casq.common.decorators import timer, trace


def test_trace_timer_suppressed(loguru_caplog: LogCaptureFixture) -> None:
    """Unit test for trace and timer decorators when suppressed due to log level."""

    @trace()
    @timer()
    def dummy_function() -> None:
        pass

    dummy_function()
    assert len(loguru_caplog.records) == 0


def test_trace_timer(loguru_caplog: LogCaptureFixture) -> None:
    """Unit test for trace and timer decorators."""

    @trace(level="DEBUG")
    @timer(level="DEBUG")
    def dummy_function() -> None:
        pass

    dummy_function()
    assert (
        loguru_caplog.records[0].msg == "Entering [dummy_function(args=[], kwargs=[])]."
    )
    assert "Executed [dummy_function] in" in loguru_caplog.records[1].msg
    assert "milliseconds" in loguru_caplog.records[1].msg
    assert (
        loguru_caplog.records[2].msg == "Exiting [dummy_function] with result [None]."
    )


def test_trace_timer_sec(loguru_caplog: LogCaptureFixture) -> None:
    """Unit test for trace and timer decorators."""

    @trace(level="DEBUG")
    @timer(level="DEBUG", unit="sec")
    def dummy_function() -> None:
        pass

    dummy_function()
    assert (
        loguru_caplog.records[0].msg == "Entering [dummy_function(args=[], kwargs=[])]."
    )
    assert "Executed [dummy_function] in" in loguru_caplog.records[1].msg
    assert "seconds" in loguru_caplog.records[1].msg
    assert (
        loguru_caplog.records[2].msg == "Exiting [dummy_function] with result [None]."
    )
