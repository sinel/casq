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
"""Pulse simulator."""
from __future__ import annotations

from enum import Enum
from typing import Optional, Self

import numpy as np
import numpy.typing as npt
from qiskit_dynamics.backend import parse_backend_hamiltonian_dict

from casq.common import trace


class HamiltonianModel:
    """HamiltonianModel class."""

    class EvaluationMode(Enum):
        """Evaluation mode."""

        DENSE = 0
        SPARSE = 1

    @trace()
    def __init__(
        self,
        hamiltonian_dict: dict,
        qubits: Optional[list[int]] = None,
        rotating_frame: Optional[npt.NDArray] = None,
        in_frame_basis: bool = False,
        evaluation_mode: EvaluationMode = EvaluationMode.DENSE,
        rwa_cutoff_freq: Optional[float] = None,
    ) -> None:
        """Initialize HamiltonianModel.

        Args:
            hamiltonian_dict: Dictionary representing Hamiltonian in string specification.
            qubits: List of qubits to extract from the Hamiltonian.
            rotating_frame: Rotating frame operator.
                            If specified with a 1d array, it is interpreted as the
                            diagonal of a diagonal matrix. Assumed to store
                            the anti-hermitian matrix F = -iH.
            in_frame_basis: Whether to represent the model in the basis in which
                            the rotating frame operator is diagonalized.
            evaluation_mode: Evaluation mode to use.
            rwa_cutoff_freq: Rotating wave approximation argument.
        """
        self.hamiltonian_dict = hamiltonian_dict
        self.qubits = [0] if qubits is None else qubits
        self.in_frame_basis = in_frame_basis
        self.evaluation_mode = (
            HamiltonianModel.EvaluationMode.DENSE
            if evaluation_mode is None
            else evaluation_mode
        )
        self.rwa_cutoff_freq = rwa_cutoff_freq
        (
            self.static_operator,
            self.operators,
            self.channels,
            self.qubit_dims,
        ) = parse_backend_hamiltonian_dict(hamiltonian_dict, qubits)
        if rotating_frame is None:
            if evaluation_mode is HamiltonianModel.EvaluationMode.DENSE:
                self.rotating_frame = self.static_operator
            else:
                self.rotating_frame = np.diag(self.static_operator)
        else:
            self.rotating_frame = rotating_frame
