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

from typing import Optional

from casq.common import trace
from casq.models.hamiltonian_model import HamiltonianModel


class PulseBackendModel:
    """PulseBackendModel class."""

    @trace()
    def __init__(
        self,
        hamiltonian: HamiltonianModel,
        dt: Optional[float] = None,
        channel_carrier_freqs: Optional[dict] = None,
        control_channel_map: Optional[dict] = None,
    ) -> None:
        """Instantiate :class:`~casq.PulseSolution`.

        Args:
            hamiltonian: Pulse circuit name.
            dt: Sampling interval.
            channel_carrier_freqs: Dictionary mapping channel names to frequencies.
            control_channel_map: A dictionary mapping control channel labels to indices.
        """
        self.hamiltonian = hamiltonian
        self.dt = dt
        self.channel_carrier_freqs = channel_carrier_freqs
        self.control_channel_map = control_channel_map
