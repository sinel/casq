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
"""Base object tests."""
from __future__ import annotations

from typing import Any
from uuid import UUID

from qiskit_dynamics.array import Array

from casq.common import dbid, ufid, initialize_jax


def test_dbid() -> None:
    """Unit test for dbid."""
    assert UUID(dbid(), version=4)


def test_ufid() -> None:
    """Unit test for ufid."""
    obj: dict[str, Any] = {}
    name = ufid(obj)
    assert isinstance(name, str)
    assert name.endswith("Dict")


def test_initialize_jax() -> None:
    """Unit test for initializing jax."""
    initialize_jax()
    assert Array.default_backend() == "jax"
