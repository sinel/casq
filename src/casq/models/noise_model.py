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
"""Noise model."""
from __future__ import annotations

from typing import Optional

import numpy.typing as npt
from qiskit_dynamics.array import Array

from casq.common.decorators import trace


class NoiseModel:
    """NoiseModel class."""

    @trace()
    def __init__(
        self,
        static_dissipators: Array,
        dissipator_operators: Optional[Array] = None,
        dissipator_channels: Optional[list[str]] = None,
    ) -> None:
        """Initialize NoiseModel.

        Args:
            static_dissipators: Constant dissipation operators.
            dissipator_operators: Dissipation operators with time-dependent coefficients.
            dissipator_channels: List of channel names in pulse schedules
                corresponding to dissipator operators.
        """
        # TO-DO: Basically copy-paste from solver arguments for Lindblad model.
        # Need to understand better and relate to noise/decoherence terms.
        self.static_dissipators = static_dissipators
        self.dissipator_operators = dissipator_operators
        self.dissipator_channels = dissipator_channels
