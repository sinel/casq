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
"""Command line output of information about Casq."""


def about() -> None:
    """Provides information about Casq."""
    print(
        "================================================================================"
    )
    print("Casq: Python Toolkit for Control and Analysis of Superconducting Qubits")
    print(
        "================================================================================"
    )
    print("Copyright (c) 2023 Sinan Inel <sinan.inel@aalto.fi>.")
    print("")
    print(f"Source:                    https://github.com/sinel/casq")
    print("Version:                    0.1.0")
    print("Python Version:             3.11")
    print("Qiskit Version:             0.43.2")
    print("Qiskit Dynamics Version:    0.4.1")
    print("QuTiP Version:              4.7.2")
    print("C3 Version:                 1.4")
    print(
        "================================================================================"
    )


if __name__ == "__main__":
    about()
