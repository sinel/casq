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

from typing import Optional, Union

from loguru import logger
from qiskit.providers import BackendV1
from qiskit.pulse import Schedule
from qiskit.pulse.channels import DriveChannel, ControlChannel, MeasureChannel, Channel
from qiskit_dynamics import Signal
from qiskit_dynamics.pulse import InstructionToSignals

from casq import PulseBackendProperties
from casq.common import CasqError


def discretize(schedule: Schedule, dt: float, channel_frequencies: dict[str, float]) -> list[Signal]:
    """Discretizes pulse schedule into signals.

    Args:
        schedule: Pulse schedule.
        dt: Time interval.
        channel_frequencies: Channel frequencies.

    Returns:
        List of :py:class:`qiskit_dynamics.signals.Signal`
    """
    converter = InstructionToSignals(dt, carriers=channel_frequencies, channels=list(channel_frequencies.keys()))
    return converter.get_signals(schedule)


def get_channel_frequencies(
    channels: Union[list[str], list[Channel]], props: PulseBackendProperties
) -> dict[str, float]:
    """Discretizes pulse schedule into signals.

    Args:
        channels: List of channel names or channel instances.
        props: Backend properties.

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
    drive_frequencies = props.qubit_frequencies
    for channel in drive_channels:
        if channel.index >= len(drive_frequencies):
            raise CasqError(f"DriveChannel index {channel.index} is out of bounds.")
        channel_freqs[channel.name] = drive_frequencies[channel.index]
    for channel in control_channels:
        if channel.index >= len(props.control_channel_lo):
            raise CasqError(f"ControlChannel index {channel.index} is out of bounds.")
        freq = 0.0
        for channel_lo in props.control_channel_lo[channel.index]:
            freq += drive_frequencies[channel_lo.q] * channel_lo.scale
        channel_freqs[channel.name] = freq
    if measure_channels:
        measure_frequencies = props.measurement_frequencies
        for channel in measure_channels:
            if channel.index >= len(measure_frequencies):
                raise CasqError(f"MeasureChannel index {channel.index} is out of bounds.")
            channel_freqs[channel.name] = measure_frequencies[channel.index]
    return channel_freqs
