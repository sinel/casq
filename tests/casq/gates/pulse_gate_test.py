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

from matplotlib.figure import Figure
from qiskit import pulse

from casq.gates.pulse_gate import PulseGate


class DummyPulseGate(PulseGate):
    """DummyPulseGate class."""
    def schedule(self, qubit: int) -> pulse.ScheduleBlock:
        """GaussianPulseGate.schedule method.

        Args:
            qubit: Index of qubit to drive.

        Returns:
            :py:class:`qiskit.pulse.ScheduleBlock`
        """
        with pulse.build() as sb:
            pulse.play(
                pulse.library.Constant(1, 1, name=self.ufid),
                pulse.DriveChannel(qubit),
            )
        self._schedule = sb
        return sb


def test_circuit() -> None:
    """Unit test for pulse gate circuit."""
    dummy = DummyPulseGate()
    circuit = dummy.circuit(0)
    operation = circuit.data[0].operation
    assert operation.name.endswith("DummyPulseGate")
    assert operation.num_qubits == 1


def test_circuit_lazy() -> None:
    """Unit test for pulse gate circuit lazy loading."""
    dummy = DummyPulseGate()
    circuit1 = dummy.circuit(0)
    circuit2 = dummy.circuit(0)
    assert id(circuit1) == id(circuit2)


def test_draw_schedule() -> None:
    """Unit test for draw_schedule method."""
    dummy = DummyPulseGate()
    figure = dummy.draw_schedule(hidden=True)
    assert isinstance(figure, Figure)


def test_draw_circuit() -> None:
    """Unit test for draw_circuit method."""
    dummy = DummyPulseGate()
    figure = dummy.draw_circuit(hidden=True)
    assert isinstance(figure, Figure)
