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

from uuid import UUID

from casq.base_object import BaseObject


def test_database_id() -> None:
    """Unit test for database id."""

    class Dummy(BaseObject):
        pass

    dummy = Dummy()
    assert UUID(dummy.dbid, version=4)


def test_user_friendly_id() -> None:
    """Unit test for user-friendly id."""

    class Dummy(BaseObject):
        pass

    dummy = Dummy()
    print(dummy.ufid)
    assert isinstance(dummy.ufid, str)
    assert dummy.ufid.endswith("Dummy")
