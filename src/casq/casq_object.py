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

from abc import ABC
from typing import Optional
from uuid import uuid4

from wonderwords import RandomWord

from casq.common.decorators import trace


class CasqObject(ABC):
    """CasqObject class.

    Base object for all casq classes.
    Ensures that every object has a user-friendly text id and database uuid(v4).

    Args:
        name: User-friendly name for object.
        Default is auto-generated using a random adjective plus child classname.
    """

    @trace()
    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize CasqObject."""
        self.dbid = str(uuid4())
        if name is None:
            random_word = RandomWord()
            self.ufid = (
                f"{random_word.word(include_categories=['adjective'])}"
                f"{self.__class__.__name__}"
            )
