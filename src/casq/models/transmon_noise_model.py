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
"""Transmon noise model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import Operator, Pauli

from casq.common.decorators import trace
from casq.models.noise_model import NoiseModel


class TransmonNoiseModel(NoiseModel):
    """TransmonNoiseModel class."""

    @dataclass
    class TransmonNoiseProperties:
        """Transmon qubit noise properties."""

        t1: float
        t2: float

    @trace()
    def __init__(self, qubit_map: dict[int, TransmonNoiseProperties]) -> None:
        """Initialize TransmonNoiseModel.

        Args:
            qubit_map: Dictionary mapping qubit indices to noise properties.
        """
        # TO-DO: Temporary placeholder implementation as extended copy-paste from qiskit-dynamics tutorial.
        self.qubit_map = qubit_map
        qubits = qubit_map.keys()
        num_qubits = len(qubits)
        zeros = Operator(np.zeros((2 ** num_qubits, 2 ** num_qubits)))
        static_dissipators = []
        for q in qubits:
            sigma_x = zeros + Pauli("X")(q)
            sigma_y = zeros + Pauli("Y")(q)
            sigma_z = zeros + Pauli("Z")(q)
            sigma_p = 0.5 * (sigma_x + 1j * sigma_y)
            op1 = np.sqrt(1.0 / qubit_map[q].t1) * sigma_p
            op2 = np.sqrt(1.0 / qubit_map[q].t2) * sigma_z
            static_dissipators.append(np.asarray([op1, op2]))
        super().__init__(static_dissipators=np.asarray(static_dissipators))
