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

import numpy as np
from qiskit.quantum_info import Operator

from casq.common import trace
from casq.models.noise_model import NoiseModel


class TransmonNoiseModel(NoiseModel):
    """TransmonNoiseModel class."""

    @trace()
    def __init__(self, t1: float, t2: float) -> None:
        """Initialize TransmonNoiseModel.

        Args:
            t1: T1 relaxation time.
            t2: T2 relaxation time.
        """
        # TO-DO: Temporary placeholder implementation as copy-paste from qiskit-dynamics tutorial.
        self.t1 = t1
        self.t2 = t2
        sigma_x = Operator.from_label("X")
        sigma_y = Operator.from_label("Y")
        sigma_z = Operator.from_label("Z")
        sigma_p = 0.5 * (sigma_x + 1j * sigma_y)
        super().__init__(
            static_dissipators=np.asarray(
                [np.sqrt(1.0 / t1) * sigma_p, np.sqrt(1.0 / t2) * sigma_z]
            )
        )
