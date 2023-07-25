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
from loguru import logger
from qiskit.providers import BackendV1
from qiskit_dynamics.array import Array

from casq.backends.pulse_backend import PulseBackend
from casq.backends.pulse_solution import PulseSolution
from casq.backends.qiskit.qiskit_pulse_backend import QiskitPulseBackend
from casq.common.decorators import timer
from casq.gates.gaussian_pulse_gate import GaussianPulseGate
from casq.gates.drag_pulse_gate import DragPulseGate
from casq.gates.pulse_circuit import PulseCircuit
from casq.backends.helpers import build_from_backend
from casq.models.transmon_model import TransmonModel
from casq.models.transmon_noise_model import TransmonNoiseModel
from casq.backends.helpers import build, BackendLibrary
from casq.models.control_model import ControlModel

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
Array.set_default_backend("jax")

qubit_map = {
    0: TransmonModel.TransmonProperties(
        frequency=31179079102.853794,
        anharmonicity=-2165345334.8252344,
        drive=926545606.6640488
    ),
    1: TransmonModel.TransmonProperties(
        frequency=30397743782.610542,
        anharmonicity=-2169482392.6367006,
        drive=892870223.8110852
    ),
    2: TransmonModel.TransmonProperties(
        frequency=31649945798.50227,
        anharmonicity=-2152313197.3287387,
        drive=927794953.0001632
    ),
    3: TransmonModel.TransmonProperties(
        frequency=31107813662.24873,
        anharmonicity=-2158766696.6684937,
        drive=921439621.8693779
    ),
    4: TransmonModel.TransmonProperties(
        frequency=31825180853.3539,
        anharmonicity=-2149525690.7311115,
        drive=1150709205.1097605
    ),
}
coupling_map = {
    (0, 1): 11845444.218797993,
    (1, 2): 11967839.68906386,
    (2, 3): 12402113.956012368,
    (3, 4): 12186910.37040823,
}
hamiltonian = TransmonModel(
    qubit_map=qubit_map,
    coupling_map=coupling_map,
    extracted_qubits=[0]
)
qubit_map = {
    0: TransmonNoiseModel.TransmonNoiseProperties(
        t1=0.00010918719287058488,
        t2=5.077229750099717e-06
    ),
    1: TransmonNoiseModel.TransmonNoiseProperties(
        t1=5.753535189181149e-05,
        t2=6.165015600725496e-05
    ),
    2: TransmonNoiseModel.TransmonNoiseProperties(
        t1=0.00018344197711073844,
        t2=2.512378482362435e-05
    ),
    3: TransmonNoiseModel.TransmonNoiseProperties(
        t1=0.00010961657783040683,
        t2=5.7120186456626996e-05
    ),
    4: TransmonNoiseModel.TransmonNoiseProperties(
        t1=0.00010247738825319845,
        t2=3.722985261736209e-05
    )
}
noise = TransmonNoiseModel(qubit_map=qubit_map)
control = ControlModel(
    dt=2.2222222222222221e-10,
    channel_carrier_freqs={
        "d0": 4962770879.920025,
        "d1": 4838412258.764764,
        "d2": 5036989248.286842,
        "d3": 4951300212.210368,
        "d4": 5066350584.469812,
        "u0": 4838412258.764764,
        "u1": 4962770879.920025,
        "u2": 5036989248.286842,
        "u3": 4838412258.764764,
        "u4": 4951300212.210368
    }
)


def test_from_backend(backend: BackendV1) -> None:
    """Unit test for PulseBackend initialization from backend."""
    pulse_backend = QiskitPulseBackend.from_backend(backend)
    assert isinstance(pulse_backend, QiskitPulseBackend)


def test_seed_option(backend: BackendV1) -> None:
    """Unit test for PulseBackend initialization from backend."""
    pulse_backend = QiskitPulseBackend.from_backend(backend, seed=1)
    assert pulse_backend._seed == 1
    assert pulse_backend._native_backend.options.seed_simulator == 1


@timer(unit="sec")
def test_run(backend: BackendV1) -> None:
    """Unit test for PulseBackend initialization from backend."""
    pulse_backend = build_from_backend(backend, qubits=[0])
    gate = DragPulseGate(256, 1, 128, 2)
    circuit = PulseCircuit.from_pulse(gate, 0)
    solution = pulse_backend.run(
        circuit, method=PulseBackend.ODESolverMethod.SCIPY_DOP853
    )
    assert isinstance(solution, PulseSolution)
    logger.debug(solution.counts[-1])


@timer(unit="sec")
def test_jax_run(backend: BackendV1) -> None:
    """Unit test for PulseBackend initialization from backend."""
    pulse_backend = build_from_backend(backend, qubits=[0])
    gate = DragPulseGate(256, 1, 128, 2)
    circuit = PulseCircuit.from_pulse(gate, 0)
    solution = pulse_backend.run(
        circuit, method=PulseBackend.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT
    )
    assert isinstance(solution, PulseSolution)
    logger.debug(solution.counts[-1])


@timer(unit="sec")
def test_transmon_run(backend: BackendV1) -> None:
    """Unit test for PulseBackend initialization from backend."""
    pulse_backend = build(
        backend_library=BackendLibrary.QISKIT,
        hamiltonian=hamiltonian,
        control=control,
        noise=noise
    )
    gate = DragPulseGate(256, 1, 128, 2)
    circuit = PulseCircuit.from_pulse(gate, 0)
    solution = pulse_backend.run(
        circuit, method=PulseBackend.ODESolverMethod.SCIPY_DOP853
    )
    assert isinstance(solution, PulseSolution)
    logger.debug(solution.counts[-1])


@timer(unit="sec")
def test_transmon_jax_run(backend: BackendV1) -> None:
    """Unit test for PulseBackend initialization from backend."""
    pulse_backend = build(
        backend_library=BackendLibrary.QISKIT,
        hamiltonian=hamiltonian,
        control=control,
        noise=noise
    )
    gate = DragPulseGate(256, 1, 128, 2)
    circuit = PulseCircuit.from_pulse(gate, 0)
    solution = pulse_backend.run(
        circuit, method=PulseBackend.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT
    )
    assert isinstance(solution, PulseSolution)
    logger.debug(solution.counts[-1])
