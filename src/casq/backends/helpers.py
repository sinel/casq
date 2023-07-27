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
"""Backend helper functions used by library."""
from __future__ import annotations

from enum import Enum
from typing import Optional, Union

import numpy.typing as npt
from qiskit.providers import Backend
from qiskit_dynamics import RotatingFrame
from qiskit_dynamics.array import Array

from casq.backends.qiskit.qiskit_pulse_backend import QiskitPulseBackend
from casq.models.control_model import ControlModel
from casq.models.hamiltonian_model import HamiltonianModel


class BackendLibrary(Enum):
    """Backend library."""

    C3 = 0
    QISKIT = 1
    QUTIP = 2


def build(
    backend_library: BackendLibrary,
    hamiltonian: HamiltonianModel,
    control: ControlModel,
    seed: Optional[int] = None,
) -> QiskitPulseBackend:
    """Build PulseBackend.

        Currently, only supports Qiskit.

    Args:
        backend_library: Backend library.
        hamiltonian: Hamiltonian model.
        control: Control model.
        seed: Seed to use in random sampling. Defaults to None.

    Returns:
        QiskitPulseBackend
    """
    if backend_library is BackendLibrary.QISKIT:
        pulse_backend: QiskitPulseBackend = QiskitPulseBackend(
            hamiltonian=hamiltonian, control=control, seed=seed
        )
        return pulse_backend
    else:
        raise ValueError(f"Unknown backend library: {backend_library.name}.")


def build_from_backend(
    backend: Backend,
    extracted_qubits: Optional[list[int]] = None,
    rotating_frame: Optional[Union[Array, RotatingFrame]] = None,
    in_frame_basis: bool = False,
    evaluation_mode: Optional[HamiltonianModel.EvaluationMode] = None,
    rwa_cutoff_freq: Optional[float] = None,
    rwa_carrier_freqs: Optional[
        Union[npt.NDArray, tuple[npt.NDArray, npt.NDArray]]
    ] = None,
    seed: Optional[int] = None,
) -> QiskitPulseBackend:
    """Build PulseBackend from library-specific backend.

    Currently, only supports Qiskit.

    Args:
        backend: The ``Backend`` instance to build the :class:`.DynamicsBackend` from.
        extracted_qubits: List of qubits to extract from the Hamiltonian.
        rotating_frame: Rotating frame argument for the internal :class:`.Solver`.
                Defaults to None, allowing this method to pick a rotating frame.
        in_frame_basis: Whether to represent the model in the basis in which
                        the rotating frame operator is diagonalized.
        evaluation_mode: Evaluation mode to use by solver.
        rwa_cutoff_freq: Rotating wave approximation cutoff frequency.
                        If None, no approximation is made.
        rwa_carrier_freqs: Carrier frequencies to use for rotating wave approximation.
        seed: Seed to use in random sampling. Defaults to None.

    Returns:
        QiskitPulseBackend
    """
    if isinstance(backend, Backend):
        pulse_backend: QiskitPulseBackend = QiskitPulseBackend.from_backend(
            backend=backend,
            extracted_qubits=extracted_qubits,
            rotating_frame=rotating_frame,
            in_frame_basis=in_frame_basis,
            evaluation_mode=evaluation_mode,
            rwa_cutoff_freq=rwa_cutoff_freq,
            rwa_carrier_freqs=rwa_carrier_freqs,
            seed=seed,
        )
        return pulse_backend
    else:
        raise ValueError(f"Unknown backend: {repr(backend)}.")
