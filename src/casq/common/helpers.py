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

from typing import Any
from uuid import uuid4

import jax
from qiskit_dynamics.array import Array
from wonderwords import RandomWord


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
    return (
        f"{random_word.word(include_categories=['adjective'])}"
        f"{obj.__class__.__name__}"
    )


def initialize_jax() -> None:
    """Initializes jax to use CPU in 64-bit mode."""
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    Array.set_default_backend("jax")
