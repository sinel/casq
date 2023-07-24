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
"""BackendCharacteristics."""
from __future__ import annotations

from typing import Any, NamedTuple, Union

from loguru import logger
import numpy as np
from qiskit.providers import Backend
from qiskit.providers.models import (
    BackendProperties,
    PulseBackendConfiguration,
    PulseDefaults,
)
from qiskit.pulse.channels import Channel, ControlChannel, DriveChannel, MeasureChannel
from qiskit_ibm_provider import IBMProvider

from casq.common.decorators import trace
from casq.common.exceptions import CasqError


class BackendCharacteristics:
    """BackendCharacteristics class.

    Extracts IBMQ backend characteristics needed by various casq classes and methods.
    Requires PulseBackendConfiguration with valid configuration, properties, and defaults.

    BackendV2 currently lacks all necessary characteristics
    that may be utilized by a PulseBackend.
    However, this may change in future Qiskit versions.

    Args:
        backend: IBMQ backend compatible with BackendV1.
    """

    class GateProperties(NamedTuple):
        """Gate properties."""

        gate_error: float
        gate_length: float
        qubits: list[int]
        parameters: dict[str, Any]
        is_operational: bool

    class QubitProperties(NamedTuple):
        """Qubit properties."""

        frequency: float
        readout_error: float
        readout_length: float
        t1: float
        t2: float
        is_operational: bool

    @staticmethod
    def get_backend(name: str) -> Backend:  # pragma: no cover
        """BackendCharacteristics._get_backend method.

        Returns:
            :py:class:`qiskit.providers.Backend`
        """
        provider = IBMProvider()
        return provider.get_backend(name)

    @trace()
    def __init__(self, backend: Union[str, Backend]):
        """Initialize BackendCharacteristics."""
        if isinstance(backend, str):
            self.backend = BackendCharacteristics.get_backend(backend)
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
        self.inst_map = self.defaults.instruction_schedule_map
        self.meas_map = self.config.meas_map

    def get_qubit_properties(self, qubit: int) -> QubitProperties:
        """BackendCharacteristics.get_qubit_properties method.

        Args:
            qubit: Qubit to attach gate to.

        Returns:
            :py:class:`casq.backends.qiskit.BackendCharacteristics.QubitProperties`
        """
        return BackendCharacteristics.QubitProperties(
            frequency=self.properties.frequency(qubit),
            readout_error=self.properties.readout_error(qubit),
            readout_length=self.properties.readout_length(qubit),
            t1=self.properties.t1(qubit),
            t2=self.properties.t2(qubit),
            is_operational=self.properties.is_qubit_operational(qubit),
        )

    def get_gate_properties(self, gate_name: str) -> GateProperties:
        """BackendCharacteristics.get_gate_properties method.

        Args:
            gate_name: Gate name.

        Returns:
            :py:class:`casq.backends.qiskit.BackendCharacteristics.GateProperties`
        """
        gate = next(g for g in self.properties.gates if g.gate == gate_name)
        gate_dict = gate.to_dict()
        qubits = gate_dict["qubits"]
        return BackendCharacteristics.GateProperties(
            gate_error=self.properties.gate_error(gate_name, qubits),
            gate_length=self.properties.gate_length(gate_name, qubits),
            qubits=qubits,
            parameters=gate_dict["parameters"],
            is_operational=self.properties.is_gate_operational(gate_name, qubits),
        )

    def _get_config(self) -> PulseBackendConfiguration:  # pragma: no cover
        """BackendCharacteristics._get_config method.

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
        """BackendCharacteristics._get_defaults method.

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
        """BackendCharacteristics._get_properties method.

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

    def get_channel_frequencies(
        self, channels: Union[list[str], list[Channel]]
    ) -> dict[str, float]:
        """Discretizes pulse schedule into signals.

        Args:
            channels: List of channel names or channel instances.

        Returns:
            List of :py:class:`qiskit_dynamics.signals.Signal`
        """
        drive_channels = []
        control_channels = []
        measure_channels = []
        if isinstance(channels[0], str):
            for channel in channels:
                if channel[0] == "d":
                    drive_channels.append(DriveChannel(int(channel[1:])))
                elif channel[0] == "u":
                    control_channels.append(ControlChannel(int(channel[1:])))
                elif channel[0] == "m":
                    measure_channels.append(MeasureChannel(int(channel[1:])))
                else:
                    logger.warning(f"Unrecognized channel [{channel}] requested.")
        else:
            for channel in channels:
                if isinstance(channel, DriveChannel):
                    drive_channels.append(channel)
                elif isinstance(channel, ControlChannel):
                    control_channels.append(channel)
                elif isinstance(channel, MeasureChannel):
                    measure_channels.append(channel)
                else:
                    logger.warning(f"Unrecognized channel [{channel}] requested.")
        channel_freqs = {}
        drive_frequencies = self.qubit_frequencies
        for channel in drive_channels:
            if channel.index >= len(drive_frequencies):
                raise CasqError(f"DriveChannel index {channel.index} is out of bounds.")
            channel_freqs[channel.name] = drive_frequencies[channel.index]
        for channel in control_channels:
            if channel.index >= len(self.control_channel_lo):
                raise CasqError(
                    f"ControlChannel index {channel.index} is out of bounds."
                )
            freq = 0.0
            for channel_lo in self.control_channel_lo[channel.index]:
                # TO-DO: channel_lo.scale is complex.
                # So resulting frequency may be complex?
                freq += drive_frequencies[channel_lo.q] * channel_lo.scale
            if np.imag(freq) == 0:
                channel_freqs[channel.name] = np.real(freq)
            else:
                channel_freqs[channel.name] = freq  # pragma: no cover
        if measure_channels:
            measure_frequencies = self.measurement_frequencies
            for channel in measure_channels:
                if channel.index >= len(measure_frequencies):
                    raise CasqError(
                        f"MeasureChannel index {channel.index} is out of bounds."
                    )
                channel_freqs[channel.name] = measure_frequencies[channel.index]
        return channel_freqs

    def get_control_channel_map(
        self, channels_filter: list[str]
    ) -> dict[tuple[int, ...], int]:
        """Get control channel map from backend configuration.

        Args:
            channels_filter: List of channel names to filter by.

        Returns:
            Dictionary mapping qubits to control channels.
        """
        control_channel_map = {}
        for qubits in self.config.control_channels:
            index = self.config.control_channels[qubits][0].index
            if f"u{index}" in channels_filter:
                control_channel_map[qubits] = index
        return control_channel_map
