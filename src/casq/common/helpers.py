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
"""Common helper functions used by library."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import jax
from qiskit.pulse import Schedule
from qiskit_dynamics import Signal
from qiskit_dynamics.array import Array
from qiskit_dynamics.pulse import InstructionToSignals
from wonderwords import RandomWord


class TimeUnit(Enum):
    SAMPLE = 0  # Sampling intervals
    PICO_SEC = 1  # 10^-12
    NANO_SEC = 2  # 10^-9
    MICRO_SEC = 3  # 10^-6
    MILLI_SEC = 4  # 10^-3
    SEC = 5


def dbid() -> str:
    """Generates database identifier for object.

    Returns:
        Database identifier.
    """
    return str(uuid4())


def ufid(obj: Any) -> str:
    """Generates user-friendly identifier for object.

    Args:
        obj: Object.

    Returns:
        User-friendly identifier.
    """
    random_word = RandomWord()
    name = obj.__class__.__name__
    if not name[0].isupper():
        name = name.capitalize()
    return f"{random_word.word(include_categories=['adjective'])}{name}"


def initialize_jax() -> None:
    """Initializes jax to use CPU in 64-bit mode."""
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    Array.set_default_backend("jax")


@dataclass
class SignalData:
    name: str
    dt: float
    duration: float
    signal: Signal
    i_signal: Signal
    q_signal: Signal
    carrier: float


def discretize(
    schedule: Schedule, dt: float, channel_frequencies: dict[str, float], carrier_frequency: Optional[float] = None
) -> list[SignalData]:
    """Discretizes pulse schedule into signals.

    Args:
        schedule: Pulse schedule.
        dt: Time interval.
        channel_frequencies: Channel frequencies.
        carrier_frequency: Carrier frequency used for calculating IQ signal components.
            If None, then corresponding channel frequencies are used.

    Returns:
        List of SignalData.
    """
    signals = []
    converter = InstructionToSignals(
        dt, carriers=channel_frequencies, channels=list(channel_frequencies.keys())
    )
    schedule_signals = converter.get_signals(schedule)
    for signal in schedule_signals:
        carrier = carrier_frequency if carrier_frequency else channel_frequencies[signal.name]
        iq_signals = converter.get_awg_signals([signal], carrier)
        signals.append(SignalData(signal.name, dt, signal.duration, signal, iq_signals[0], iq_signals[1], carrier))
    return signals
