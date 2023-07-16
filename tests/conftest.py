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

from datetime import datetime
import logging
from typing import Generator

from loguru import logger
import pytest
from qiskit import pulse
from qiskit.providers.fake_provider import FakeManila
from qiskit.providers.ibmq import IBMQBackend
from qiskit.providers.models import (
    BackendProperties,
    PulseBackendConfiguration,
    PulseDefaults,
    UchannelLO
)
from qiskit.providers.models.backendproperties import Gate, Nduv
from qiskit.pulse import ControlChannel, InstructionScheduleMap, Schedule, ScheduleBlock
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from unittest.mock import MagicMock


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


@pytest.fixture
def backend_configuration():
    mock_obj = MagicMock(spec=PulseBackendConfiguration)
    mock_obj.dt = 2.2222222222222221e-10
    mock_obj.dtm = 2.2222222222222221e-10
    mock_obj.max_shots = 100000
    mock_obj.num_qubits = 5
    mock_obj.hamiltonian = {
        "description": "Hamiltonian description.",
        "h_latex":
            "\\begin{align} \\mathcal{H}/\\hbar = "
            "& \\sum_{i=0}^{4}\\left(\\frac{\\omega_{q,i}}{2}(\\mathbb{I}-\\sigma_i^{z})+"
            "\\frac{\\Delta_{i}}{2}(O_i^2-O_i)+"
            "\\Omega_{d,i}D_i(t)\\sigma_i^{X}\\right) \\\\ & +"
            "J_{0,1}(\\sigma_{0}^{+}\\sigma_{1}^{-}+\\sigma_{0}^{-}\\sigma_{1}^{+}) +"
            "J_{1,2}(\\sigma_{1}^{+}\\sigma_{2}^{-}+\\sigma_{1}^{-}\\sigma_{2}^{+}) +"
            "J_{2,3}(\\sigma_{2}^{+}\\sigma_{3}^{-}+\\sigma_{2}^{-}\\sigma_{3}^{+}) +"
            "J_{3,4}(\\sigma_{3}^{+}\\sigma_{4}^{-}+\\sigma_{3}^{-}\\sigma_{4}^{+}) \\\\ & +"
            "\\Omega_{d,0}(U_{0}^{(0,1)}(t))\\sigma_{0}^{X} +"
            "\\Omega_{d,1}(U_{1}^{(1,0)}(t)+U_{2}^{(1,2)}(t))\\sigma_{1}^{X} \\\\ & +"
            "\\Omega_{d,2}(U_{3}^{(2,1)}(t)+U_{4}^{(2,3)}(t))\\sigma_{2}^{X} +"
            "\\Omega_{d,3}(U_{6}^{(3,4)}(t)+U_{5}^{(3,2)}(t))\\sigma_{3}^{X} \\\\ & +"
            "\\Omega_{d,4}(U_{7}^{(4,3)}(t))\\sigma_{4}^{X} \\\\ \\end{align}",
        "h_str": [
            "_SUM[i,0,4,wq{i}/2*(I{i}-Z{i})]",
            "_SUM[i,0,4,delta{i}/2*O{i}*O{i}]",
            "_SUM[i,0,4,-delta{i}/2*O{i}]",
            "_SUM[i,0,4,omegad{i}*X{i}||D{i}]",
            "jq0q1*Sp0*Sm1",
            "jq0q1*Sm0*Sp1",
            "jq1q2*Sp1*Sm2",
            "jq1q2*Sm1*Sp2",
            "jq2q3*Sp2*Sm3",
            "jq2q3*Sm2*Sp3",
            "jq3q4*Sp3*Sm4",
            "jq3q4*Sm3*Sp4",
            "omegad1*X0||U0",
            "omegad0*X1||U1",
            "omegad2*X1||U2",
            "omegad1*X2||U3",
            "omegad3*X2||U4",
            "omegad4*X3||U6",
            "omegad2*X3||U5",
            "omegad3*X4||U7"
        ],
        "osc": {},
        "qub": {"0": 3, "1": 3, "2": 3, "3": 3, "4": 3},
        "vars": {
            "delta0": -2165345334.8252344,
            "delta1": -2169482392.6367006,
            "delta2": -2152313197.3287387,
            "delta3": -2158766696.6684937,
            "delta4": -2149525690.7311115,
            "jq0q1": 11845444.218797993,
            "jq1q2": 11967839.68906386,
            "jq2q3": 12402113.956012368,
            "jq3q4": 12186910.37040823,
            "omegad0": 927399441.9102196,
            "omegad1": 893467356.9563578,
            "omegad2": 927717495.8913018,
            "omegad3": 922663167.1827058,
            "omegad4": 1149699656.8472202,
            "wq0": 31178989972.422466,
            "wq1": 30397257218.264915,
            "wq2": 31649947796.41772,
            "wq3": 31107815232.243824,
            "wq4": 31825133813.261204
        }
    }
    mock_obj.u_channel_lo = [[UchannelLO(1, (1+0j))], [UchannelLO(0, (1+0j))], [UchannelLO(2, (1+0j))], [UchannelLO(1, (1+0j))], [UchannelLO(3, (1+0j))], [UchannelLO(2, (1+0j))], [UchannelLO(4, (1+0j))], [UchannelLO(3, (1+0j))]]
    mock_obj.control_channels = {(0, 1): [ControlChannel(0)], (1, 0): [ControlChannel(1)], (1, 2): [ControlChannel(2)], (2, 1): [ControlChannel(3)], (2, 3): [ControlChannel(4)], (3, 2): [ControlChannel(5)], (3, 4): [ControlChannel(6)], (4, 3): [ControlChannel(7)]}
    mock_obj.meas_map = [[0, 1, 2, 3, 4]]
    return mock_obj


@pytest.fixture
def backend_properties():
    gate_error = Nduv(datetime.now(), "gate_error", "", 0.00019195390414599716)
    gate_length = Nduv(datetime.now(), "gate_length", "ns", 35.55555555555556)
    gate_parameters = [gate_error, gate_length]
    gates = []
    for i in range(5):
        gates.append(Gate([i], "id", gate_parameters, name=f"id{i}"))
        gates.append(Gate([i], "x", gate_parameters, name=f"x{i}"))
        gates.append(Gate([i], "reset", gate_parameters, name=f"reset{i}"))
    mock_obj = MagicMock(spec=BackendProperties)
    mock_obj.gates = gates
    mock_obj.gate_error.return_value = 0.00019195390414599716
    mock_obj.gate_length.return_value = 3.5555555555555554e-08
    mock_obj.is_gate_operational.return_value = True
    mock_obj.frequency.return_value = 4962290374.723673
    mock_obj.readout_error.return_value = 0.020399999999999974
    mock_obj.readout_length.return_value = 5.35111111111111e-06
    mock_obj.t1.return_value = 0.00015951171612765792
    mock_obj.t2.return_value = 0.00011354821551695982
    mock_obj.is_qubit_operational.return_value = True
    return mock_obj


@pytest.fixture
def backend_defaults():
    mock_obj = MagicMock(spec=PulseDefaults)
    mock_obj.qubit_freq_est = [
        4.962290374723673, 4.837873742722657, 5.03724564040032, 4.95096256300096, 5.065127360941541
    ]
    mock_obj.meas_freq_est = [
        7.163170819, 7.283276284, 7.218945583, 7.110101402, 7.346892709
    ]
    mock_obj.instruction_schedule_map = InstructionScheduleMap()
    return mock_obj


@pytest.fixture
def backend(backend_configuration, backend_properties, backend_defaults):
    # mock_obj = MagicMock(spec=IBMQBackend)
    # mock_obj.configuration.return_value = backend_configuration
    # mock_obj.properties.return_value = backend_properties
    # mock_obj.defaults.return_value = backend_defaults
    # return mock_obj
    return FakeManila()


@pytest.fixture
def pulse_schedule_block(backend) -> ScheduleBlock:
    """Fixture for building a test pulse schedule."""
    gaussian = pulse.library.Gaussian(4, 1, 2, name="Gaussian")
    with pulse.build(backend, name="test") as sb:
        d0 = pulse.DriveChannel(0)
        with pulse.align_sequential():
            pulse.play(gaussian, d0)
            pulse.measure(0)
    return sb


@pytest.fixture
def pulse_schedule(pulse_schedule_block) -> Schedule:
    """Fixture for building a test pulse schedule."""
    return block_to_schedule(pulse_schedule_block)
