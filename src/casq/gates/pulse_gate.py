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

from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate
from qiskit.pulse.transforms import block_to_schedule
from qiskit_dynamics.pulse import InstructionToSignals

from casq.common.decorators import trace
from casq.common.helpers import dbid, ufid
from casq.common.plotting import plot, plot_signal, LineStyle


class PulseGate(Gate):
    """PulseGate class.

    Abstract base class for all pulse gates.
    Note: Currently only single qubit gates are supported.

    Args:
        num_qubits: Number of qubits that pulse gate acts on.
    """

    @trace()
    def __init__(self, num_qubits: int, duration: int) -> None:
        """Initialize PulseGate."""
        self.dbid = dbid()
        self.ufid = ufid(self)
        super().__init__(self.ufid, num_qubits, [], None)
        self.duration = duration

    @abstractmethod
    def instruction(self, qubit: int) -> pulse.Instruction:
        """PulseGate.instruction method.

        Builds instruction for pulse gate.

        Args:
            qubit: Qubit to attach gate instruction to.

        Returns:
            :py:class:`qiskit.pulse.Instruction`
        """
        pass

    def schedule(self, qubit: int) -> pulse.Schedule:
        """PulseGate.schedule method.

        Builds schedule block for pulse gate.

        Args:
            qubit: Qubit to attach gate instruction to.

        Returns:
            :py:class:`qiskit.pulse.Schedule`
        """
        with pulse.build() as sb:
            self.instruction(qubit)
        return block_to_schedule(sb)

    @trace()
    def circuit(self, qubit: int) -> QuantumCircuit:
        """PulseGate.circuit method.

        Builds simple circuit for solitary usage or testing of pulse gate.

        Args:
            qubit: Qubit to attach gate to.

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
            circuit.add_calibration(self.ufid, [qubit], self.schedule(qubit))
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

    @trace()
    def draw_signal(
        self, qubit: int, dt: float, carrier_frequency: float, duration: float,
        save: Optional[str] = None, hidden: bool = False
    ) -> Figure:
        """PulseGate.draw_signal method.

        Draws pulse gate signal.

        Args:
            qubit: Qubit to attach gate to.
            dt: Sample time length.
            carrier_frequency: Carrier frequency.
            duration: Duration to plot signal.
            save: Saves figure to specified path if provided.
            hidden: Does not show figure if True.

        Returns:
            :py:class:`qiskit.QuantumCircuit`
        """
        figure = plot_signal(
            self.schedule(qubit), dt, f"d{qubit}", carrier_frequency, duration,
            save=save, hidden=hidden
        )
        return figure
