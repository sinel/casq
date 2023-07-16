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
"""Qiskit helper functions used by library."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.result.models import (
    ExperimentResult,
    ExperimentResultData,
    QobjExperimentHeader,
)
from qiskit_dynamics.backend import DynamicsBackend

# noinspection PyProtectedMember
from qiskit_dynamics.backend.backend_utils import (
    _get_counts_from_samples,
    _get_iq_data,
    _get_memory_slot_probabilities,
    _sample_probability_dict,
)

# noinspection PyProtectedMember
from scipy.integrate._ivp.ivp import OdeResult


def get_experiment_result(
    experiment_name: str,
    solver_result: OdeResult,
    measurement_subsystems: list[int],
    memory_slot_indices: list[int],
    num_memory_slots: Union[None, int],
    backend: DynamicsBackend,
    seed: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> ExperimentResult:
    """Generates ExperimentResult objects from solver result.

    Args:
        experiment_name: Name of experiment.
        solver_result: Result object from :class:`Solver.solve`.
        measurement_subsystems: Labels of subsystems in the model being measured.
        memory_slot_indices: Indices of memory slots
            to store the results in for each subsystem.
        num_memory_slots: Total number of memory slots in the returned output.
            If ``None``, ``max(memory_slot_indices)`` will be used.
        backend: The backend instance that ran the simulation.
            Various options and properties are utilized.
        seed: Seed for any random number generation involved
            (e.g. when computing outcome samples).
        metadata: Metadata to add to the header of the
            :class:`~qiskit.result.models.ExperimentResult` object.

    Returns:
        :class:`~qiskit.result.models.ExperimentResult`

    Raises:
        QiskitError: If a specified option is unsupported.
    """
    counts = []
    samples = []
    populations = []
    iq_data = []
    avg_iq_data = []
    quantum_states = []
    for t, y in zip(solver_result.t, solver_result.y):
        # Take state out of frame, put in dressed basis, and normalize
        if isinstance(y, Statevector):
            # noinspection PyTypeChecker
            y = np.array(
                backend.options.solver.model.rotating_frame.state_out_of_frame(t=t, y=y)
            )
            # noinspection PyProtectedMember
            y = backend._dressed_states_adjoint @ y
            y = Statevector(y, dims=backend.options.subsystem_dims)
            if backend.options.normalize_states:
                y = y / np.linalg.norm(y.data)
        elif isinstance(y, DensityMatrix):
            # noinspection PyTypeChecker
            y = np.array(
                backend.options.solver.model.rotating_frame.operator_out_of_frame(
                    t=t, operator=y
                )
            )
            # noinspection PyProtectedMember
            y = backend._dressed_states_adjoint @ y @ backend._dressed_states
            y = DensityMatrix(y, dims=backend.options.subsystem_dims)
            if backend.options.normalize_states:
                y = y / np.diag(y.data).sum()
        else:
            y = Statevector(y, dims=backend.options.subsystem_dims)
        quantum_states.append(y)
        # compute probabilities for measurement slot values
        measurement_subsystems = [
            backend.options.subsystem_labels.index(x) for x in measurement_subsystems
        ]
        populations_step = _get_memory_slot_probabilities(
            probability_dict=y.probabilities_dict(qargs=measurement_subsystems),
            memory_slot_indices=memory_slot_indices,
            num_memory_slots=num_memory_slots,
            max_outcome_value=1,
        )
        populations.append(populations_step)
        # sample
        samples_step = _sample_probability_dict(
            populations_step, shots=backend.options.shots, seed=seed
        )
        samples.append(samples_step)
        counts.append(_get_counts_from_samples(samples_step))
        # Default iq_centers
        iq_centers = []
        for sub_dim in backend.options.subsystem_dims:
            theta = 2 * np.pi / sub_dim
            iq_centers.append(
                [[np.cos(idx * theta), np.sin(idx * theta)] for idx in range(sub_dim)]
            )
        # generate IQ
        iq_data_step = _get_iq_data(
            y,
            measurement_subsystems=measurement_subsystems,
            iq_centers=iq_centers,
            iq_width=0.2,
            shots=backend.options.shots,
            memory_slot_indices=memory_slot_indices,
            num_memory_slots=num_memory_slots,
            seed=seed,
        )
        iq_data.append(iq_data_step)
        avg_iq_data_step = np.average(iq_data_step, axis=0)
        avg_iq_data.append(avg_iq_data_step)
    # noinspection PyTypeChecker
    data = ExperimentResultData(
        counts=counts,
        memory=samples,
        qubits=backend.options.subsystem_labels,
        times=solver_result.t,
        states=quantum_states,
        populations=populations,
        iq_data=iq_data,
        avg_iq_data=avg_iq_data,
    )
    metadata.update(casq=True)
    return ExperimentResult(
        shots=backend.options.shots,
        success=True,
        data=data,
        seed=seed,
        header=QobjExperimentHeader(name=experiment_name, metadata=metadata),
    )
