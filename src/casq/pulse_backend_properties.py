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
"""Pulse-specific helper functions."""
from __future__ import annotations

from typing import NamedTuple, Union

from qiskit.providers import BackendV1
from qiskit.providers.models import (
    BackendProperties,
    PulseBackendConfiguration,
    PulseDefaults,
)
from qiskit_ibm_provider import IBMProvider

from casq.common import CasqError, trace


class PulseBackendProperties:
    """PulseBackendProperties class.

    Extracts pulse backend properties needed by various casq classes and methods.

    Args:
        backend: IBMQ backend.
    """

    class QubitProperties(NamedTuple):
        """Qubit properties."""

        frequency: float
        readout_error: float
        readout_length: float
        t1: float
        t2: float
        is_operational: bool

    @staticmethod
    def get_backend(name: str) -> BackendV1:  # pragma: no cover
        """PulseBackendProperties._get_backend method.

        Returns:
            :py:class:`qiskit.providers.Backend`
        """
        provider = IBMProvider()
        return provider.get_backend(name)

    @trace()
    def __init__(self, backend: Union[str, BackendV1]):
        """Initialize PulseBackendProperties."""
        if isinstance(backend, str):
            self.backend = PulseBackendProperties.get_backend(backend)
        else:
            self.backend = backend
        self.config = self._get_config()
        self.defaults = self._get_defaults()
        self.properties = self._get_properties()
        self.dt = self.config.dt
        self.dtm = self.config.dtm
        self.max_shots = self.config.max_shots
        self.num_qubits = self.config.num_qubits
        self.hamiltonian = self.config.hamiltonian
        self.control_channel_lo = self.config.u_channel_lo
        self.qubit_frequencies = self.defaults.qubit_freq_est
        self.measurement_frequencies = self.defaults.meas_freq_est
        self.qubits = self.properties.qubits
        self.gates = self.properties.gates

    def get_qubit_properties(self, qubit: int) -> QubitProperties:
        """PulseBackendProperties.get_qubit_properties method.

        Args:
            qubit: Qubit to attach gate to.

        Returns:
            :py:class:`casq.PulseBackendProperties.QubitProperties`
        """
        return PulseBackendProperties.QubitProperties(
            frequency=self.properties.frequency(qubit),
            readout_error=self.properties.readout_error(qubit),
            readout_length=self.properties.readout_length(qubit),
            t1=self.properties.t1(qubit),
            t2=self.properties.t2(qubit),
            is_operational=self.properties.is_operational(qubit),
        )

    def _get_config(self) -> PulseBackendConfiguration:  # pragma: no cover
        """PulseBackendProperties._get_config method.

        Returns:
            :py:class:`qiskit.providers.models.PulseBackendConfiguration`
        """
        if hasattr(self.backend, "configuration"):
            config = self.backend.configuration()
            if isinstance(config, PulseBackendConfiguration):
                return config
            else:
                raise CasqError("Backend configuration must support OpenPulse.")
        else:
            raise CasqError(
                "Backend must have a configuration method which returns "
                "a 'qiskit.providers.models.PulseBackendConfiguration' instance."
            )

    def _get_defaults(self) -> PulseDefaults:  # pragma: no cover
        """PulseBackendProperties._get_defaults method.

        Returns:
            :py:class:`qiskit.providers.models.PulseDefaults`
        """
        if hasattr(self.backend, "defaults"):
            return self.backend.defaults()
        else:
            raise CasqError(
                "Backend must have a defaults method "
                "which returns a 'qiskit.providers.models.PulseDefaults' instance."
            )

    def _get_properties(self) -> BackendProperties:  # pragma: no cover
        """PulseBackendProperties._get_properties method.

        Returns:
            :py:class:`qiskit.providers.models.BackendProperties`
        """
        if hasattr(self.backend, "properties"):
            return self.backend.properties()
        else:
            raise CasqError(
                "Backend must have a properties method "
                "which returns a 'qiskit.providers.models.BackendProperties' instance."
            )
