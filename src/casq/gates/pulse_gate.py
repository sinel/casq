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

from abc import abstractmethod
from typing import Optional

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate

from casq.casq_object import CasqObject
from casq.common.decorators import trace


class PulseGate(CasqObject):
    """PulseGate class.

    Abstract base class for all pulse gates.

    Args:
        name: Optional user-friendly name for pulse gate.
    """

    @trace()
    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize PulseGate."""
        super().__init__(name)
        self._schedule: Optional[pulse.ScheduleBlock] = None
        self._circuit: Optional[QuantumCircuit] = None

    @abstractmethod
    def schedule(self, qubit: int) -> pulse.ScheduleBlock:  # pragma: no cover
        """PulseGate.schedule method.

        Builds schedule block for pulse gate.
        This is an abstract method which children of this base class must override.

        Args:
            qubit: Index of qubit to drive.

        Returns:
            :py:class:`qiskit.pulse.ScheduleBlock`
        """
        pass

    @trace()
    def circuit(self, qubit: int) -> QuantumCircuit:
        """PulseGate.to_circuit method.

        Builds simple circuit for solitary usage or testing of pulse gate.

        Args:
            qubit: Index of qubit to drive.

        Returns:
            :py:class:`matplotlib.figure.Figure`
        """
        if self._circuit:
            return self._circuit
        else:
            circuit = QuantumCircuit(1, 1)
            custom_gate = Gate(self.ufid, 1, [])
            circuit.append(custom_gate, [qubit])
            circuit.measure([qubit], [qubit])
            circuit.add_calibration(self.ufid, [qubit], self.schedule)
            self._circuit = circuit
            return circuit

    @trace()
    def draw_schedule(self, path: Optional[str] = None, hidden: bool = False) -> Figure:
        """PulseGate.draw_schedule method.

        Draws pulse gate schedule.

        Args:
            path: Saves figure to specified path if provided.
            hidden: Does not show figure if True.

        Returns:
            :py:class:`matplotlib.figure.Figure`
        """
        # noinspection PyUnresolvedReferences
        figure = self.schedule(0).draw()
        if not hidden:
            plt.show()
        if path:
            plt.savefig(path)
        return figure

    @trace()
    def draw_circuit(self, path: Optional[str] = None, hidden: bool = False) -> Figure:
        """PulseGate.draw_circuit method.

        Draws pulse gate circuit.

        Args:
            path: Saves figure to specified path if provided.
            hidden: Does not show figure if True.

        Returns:
            :py:class:`qiskit.QuantumCircuit`
        """
        # noinspection PyUnresolvedReferences
        figure = self.circuit(0).draw("mpl")
        if not hidden:
            plt.show()
        if path:
            plt.savefig(path)
        return figure
