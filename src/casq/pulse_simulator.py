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

import copy
import datetime
from typing import Any, Optional, Union
from uuid import uuid4


from loguru import logger
import numpy as np
from qiskit import QiskitError, QuantumCircuit
from qiskit.providers import BackendV1, BackendV2
from qiskit.providers.models import UchannelLO
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit.transpiler import Target
from qiskit_dynamics import DynamicsBackend, RotatingFrame, Signal
from qiskit_dynamics.array import Array
from qiskit_dynamics.backend import parse_backend_hamiltonian_dict
from qiskit_dynamics.backend.dynamics_backend import (
    _get_acquire_instruction_timings, _to_schedule_list, _validate_run_input
)
from qiskit_dynamics.backend.dynamics_job import DynamicsJob
from qiskit_dynamics.solvers.solver_classes import Solver
from qiskit_ibm_runtime import IBMBackend

from casq.common import trace, timer, CasqError


class PulseSimulator(DynamicsBackend):
    """PulseSimulator class.

    Wraps and extends :py:class:`qiskit_dynamics.DynamicsBackend`.
    Currently, only extension is allowing access to solver instance.

    Args:
        solver: Solver instance configured for pulse simulation.
        target: Target object.
        options: Additional configuration options for the simulator.

    Raises:
        QiskitError: If any instantiation arguments fail validation checks.
    """

    @trace()
    def __init__(
            self,
            solver: Solver,
            target: Optional[Target] = None,
            **options,
    ) -> None:
        """Initialize PulseSimulator."""
        super().__init__(solver, target, **options)
        self.solver = solver
        self.ground_state = Statevector(self._dressed_states[:, 0])

    @trace()
    @timer(unit="sec")
    def run(
            self,
            run_input: list[Union[QuantumCircuit, Schedule, ScheduleBlock]],
            steps: Optional[int] = None,
            validate: Optional[bool] = True,
            **options,
    ) -> DynamicsJob:
        """Run a list of simulations.

        Args:
            run_input: A list of simulations, specified by ``QuantumCircuit``, ``Schedule``, or
                ``ScheduleBlock`` instances.
            steps: Number of time steps at which to return the solution.
            validate: Whether to run validation checks on the input.
            **options: Additional run options to temporarily override current backend options.

        Returns:
            DynamicsJob object containing results and status.

        Raises:
            QiskitError: If invalid options are set.
        """
        if validate:
            _validate_run_input(run_input, accept_list=False)
        try:
            if options["solver_options"]["t_span"]:
                raise CasqError(f"Option 't_span' not supported by PulseSimulator.run.")
        except KeyError:
            pass
        try:
            if options["solver_options"]["t_eval"]:
                raise CasqError(f"Option 't_eval' not supported by PulseSimulator.run.")
        except KeyError:
            pass
        # Configure run options for simulation
        if options:
            backend = copy.deepcopy(self)
            backend.set_options(**options)
        else:
            backend = self

        schedules, num_memory_slots_list = _to_schedule_list(run_input, backend=backend)

        # get the acquires sample times and subsystem measurement information
        (
            t_span,
            measurement_subsystems_list,
            memory_slot_indices_list,
        ) = _get_acquire_instruction_timings(
            schedules, backend.options.subsystem_labels, backend.options.solver._dt
        )
        logger.debug(f"Computed time interval to solve over is {t_span}.")
        if steps:
            t_eval = np.linspace(t_span[0][0], t_span[0][1], steps)
            t_eval[0] = t_span[0][0]
            t_eval[-1] = t_span[0][1]
        else:
            t_eval = None
        backend.options["solver_options"]["t_eval"] = t_eval

        # Build and submit job
        job_id = str(uuid4())
        dynamics_job = DynamicsJob(
            backend=backend,
            job_id=job_id,
            fn=backend._run,
            fn_kwargs={
                "t_span": t_span,
                "schedules": schedules,
                "measurement_subsystems_list": measurement_subsystems_list,
                "memory_slot_indices_list": memory_slot_indices_list,
                "num_memory_slots_list": num_memory_slots_list,
            },
        )
        dynamics_job.submit()

        return dynamics_job

    def _run(
        self,
        job_id,
        t_span,
        schedules,
        measurement_subsystems_list,
        memory_slot_indices_list,
        num_memory_slots_list,
    ) -> Result:
        """Simulate a list of schedules."""

        # simulate all schedules
        y0 = self.options.initial_state
        if y0 == "ground_state":
            y0 = Statevector(self._dressed_states[:, 0])
        # logger.debug(t_span)
        # logger.debug(self.options.solver_options["t_eval"])
        # logger.debug(y0)
        # logger.debug(schedules)

        solver_results = self.options.solver.solve(
            t_span=t_span, y0=y0, signals=schedules, **self.options.solver_options
        )
        # logger.debug(solver_results)

        # compute results for each experiment
        experiment_names = [schedule.name for schedule in schedules]
        experiment_metadatas = [schedule.metadata for schedule in schedules]
        rng = np.random.default_rng(self.options.seed_simulator)
        experiment_results = []
        for (
            experiment_name,
            solver_result,
            measurement_subsystems,
            memory_slot_indices,
            num_memory_slots,
            experiment_metadata,
        ) in zip(
            experiment_names,
            solver_results,
            measurement_subsystems_list,
            memory_slot_indices_list,
            num_memory_slots_list,
            experiment_metadatas,
        ):
            experiment_results.append(
                self.options.experiment_result_function(
                    experiment_name,
                    solver_result,
                    measurement_subsystems,
                    memory_slot_indices,
                    num_memory_slots,
                    self,
                    seed=rng.integers(low=0, high=9223372036854775807),
                    metadata=experiment_metadata,
                )
            )

        logger.debug(len(experiment_results))
        logger.debug(experiment_results[0])
        # Construct full result object
        return Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            qobj_id="",
            job_id=job_id,
            success=True,
            results=experiment_results,
            date=datetime.datetime.now().isoformat(),
        )

    @classmethod
    def from_backend(
        cls,
        backend: Union[BackendV1, BackendV2, IBMBackend],
        subsystem_list: Optional[list[int]] = None,
        rotating_frame: Optional[Union[Array, RotatingFrame, str]] = "auto",
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        **options,
    ) -> PulseSimulator:
        """Construct a PulseSimulator instance from an existing Backend instance.

        Based on copy-paste from
        https://github.com/Qiskit-Extensions/qiskit-dynamics/blob/37d12ac33fffd57daea3c9d5f9cda3f00acc37ea/qiskit_dynamics/backend/dynamics_backend.py#L545
        with following modifications:
        - Fix support for fake backends.
        BackendV2 does not have configuration method,
        hence it will always fail according to current implementation.
        Actually it should fail because it doesn't have hamiltonian information.
        Also, the new IBMBackend in qiskit-ibm-runtime
        has all required attributes/methods in one place.
        So raising error for BackendV2,
        and added support for IBMBackend.

        Note: https://github.com/Qiskit/qiskit-terra/issues/8712

        Args:
            backend: The ``Backend`` instance to build the :class:`.PulseSimulator` from.
            subsystem_list: The list of qubits in the backend to include in the model.
            rotating_frame: Rotating frame argument for the internal :class:`.Solver`. Defaults to
                ``"auto"``, allowing this method to pick a rotating frame.
            evaluation_mode: Evaluation mode argument for the internal :class:`.Solver`.
            rwa_cutoff_freq: Rotating wave approximation argument for the internal :class:`.Solver`.
            **options: Additional options to be applied in construction of the
                :class:`.PulseSimulator`.

        Returns:
            PulseSimulator

        Raises:
            QiskitError: If any required parameters are missing from the passed backend.
        """
        if isinstance(backend, IBMBackend):
            return PulseSimulator._from_backend_ibm(
                backend, subsystem_list, rotating_frame, evaluation_mode, rwa_cutoff_freq, **options
            )
        elif isinstance(backend, BackendV1):
            return PulseSimulator._from_backend_v1(
                backend, subsystem_list, rotating_frame, evaluation_mode, rwa_cutoff_freq, **options
            )
        else:  # BackendV2
            return PulseSimulator._from_backend_v2(
                backend, subsystem_list, rotating_frame, evaluation_mode, rwa_cutoff_freq, **options
            )

    @classmethod
    def _from_backend_ibm(
        cls,
        backend: IBMBackend,
        subsystem_list: Optional[list[int]] = None,
        rotating_frame: Optional[Union[Array, RotatingFrame, str]] = "auto",
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        **options,
    ) -> PulseSimulator:
        """Construct a PulseSimulator instance from an existing IBMBackend instance.

        Args:
            backend: The ``IBMBackend`` instance to build the :class:`.PulseSimulator` from.
            subsystem_list: The list of qubits in the backend to include in the model.
            rotating_frame: Rotating frame argument for the internal :class:`.Solver`. Defaults to
                ``"auto"``, allowing this method to pick a rotating frame.
            evaluation_mode: Evaluation mode argument for the internal :class:`.Solver`.
            rwa_cutoff_freq: Rotating wave approximation argument for the internal :class:`.Solver`.
            **options: Additional options to be applied in construction of the
                :class:`.PulseSimulator`.

        Returns:
            PulseSimulator

        Raises:
            QiskitError: If any required parameters are missing from the passed backend.
        """
        config = backend.configuration()
        defaults = backend.defaults() if hasattr(backend, "defaults") else None
        num_qubits = config.num_qubits
        dt = config.dt
        if config.hamiltonian is None:
            raise QiskitError(
                "DynamicsBackend.from_backend requires that backend.configuration() has a hamiltonian."
            )
        else:
            hamiltonian = config.hamiltonian
        u_channel_lo = config.u_channel_lo
        control_channels = config.control_channels
        if defaults is None:
            drive_frequencies = None
        else:
            drive_frequencies = defaults.qubit_freq_est
        if defaults is None:
            measure_frequencies = None
        else:
            measure_frequencies = defaults.meas_freq_est
        return PulseSimulator._from_backend_parameters(
            num_qubits, dt, hamiltonian, u_channel_lo,
            control_channels, drive_frequencies, measure_frequencies,
            subsystem_list, rotating_frame, evaluation_mode, rwa_cutoff_freq, **options
        )

    @classmethod
    def _from_backend_v1(
        cls,
        backend: BackendV1,
        subsystem_list: Optional[list[int]] = None,
        rotating_frame: Optional[Union[Array, RotatingFrame, str]] = "auto",
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        **options,
    ) -> PulseSimulator:
        """Construct a PulseSimulator instance from an existing BackendV1 instance.

        Args:
            backend: The ``BackendV1`` instance to build the :class:`.PulseSimulator` from.
            subsystem_list: The list of qubits in the backend to include in the model.
            rotating_frame: Rotating frame argument for the internal :class:`.Solver`. Defaults to
                ``"auto"``, allowing this method to pick a rotating frame.
            evaluation_mode: Evaluation mode argument for the internal :class:`.Solver`.
            rwa_cutoff_freq: Rotating wave approximation argument for the internal :class:`.Solver`.
            **options: Additional options to be applied in construction of the
                :class:`.PulseSimulator`.

        Returns:
            PulseSimulator

        Raises:
            QiskitError: If any required parameters are missing from the passed backend.
        """
        config = backend.configuration()
        defaults = backend.defaults() if hasattr(backend, "defaults") else None
        num_qubits = config.num_qubits
        dt = config.dt
        if config.hamiltonian is None:
            raise QiskitError(
                "DynamicsBackend.from_backend requires that backend.configuration() has a hamiltonian."
            )
        else:
            hamiltonian = config.hamiltonian
        u_channel_lo = config.u_channel_lo
        if hasattr(config, "control_channels"):
            control_channels = config.control_channels
        else:
            raise QiskitError(
                "DynamicsBackend.from_backend requires that backend.configuration() has control_channels."
            )
        if defaults is None:
            drive_frequencies = None
        else:
            drive_frequencies = defaults.qubit_freq_est
        if defaults is None:
            measure_frequencies = None
        else:
            measure_frequencies = defaults.meas_freq_est
        return PulseSimulator._from_backend_parameters(
            num_qubits, dt, hamiltonian, u_channel_lo,
            control_channels, drive_frequencies, measure_frequencies,
            subsystem_list, rotating_frame, evaluation_mode, rwa_cutoff_freq, **options
        )

    @classmethod
    def _from_backend_v2(
        cls,
        backend: BackendV2,
        subsystem_list: Optional[list[int]] = None,
        rotating_frame: Optional[Union[Array, RotatingFrame, str]] = "auto",
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        **options,
    ) -> PulseSimulator:
        """Construct a PulseSimulator instance from an existing BackendV2 instance.

        Args:
            backend: The ``BackendV2`` instance to build the :class:`.PulseSimulator` from.
            subsystem_list: The list of qubits in the backend to include in the model.
            rotating_frame: Rotating frame argument for the internal :class:`.Solver`. Defaults to
                ``"auto"``, allowing this method to pick a rotating frame.
            evaluation_mode: Evaluation mode argument for the internal :class:`.Solver`.
            rwa_cutoff_freq: Rotating wave approximation argument for the internal :class:`.Solver`.
            **options: Additional options to be applied in construction of the
                :class:`.PulseSimulator`.

        Returns:
            PulseSimulator

        Raises:
            QiskitError: If any required parameters are missing from the passed backend.
        """
        raise QiskitError(
            "BackendV2 cannot be used as a backend argument for the DynamicsBackend.from_backend method."
        )

    @classmethod
    def _from_backend_parameters(
        cls,
        num_qubits: int,
        dt: float,
        hamiltonian: dict[str, Any],
        u_channel_lo: list[list[UchannelLO]],
        control_channels: Optional[dict[tuple[int, ...], list]] = None,
        drive_frequencies: Optional[list[float]] = None,
        measure_frequencies: Optional[list[float]] = None,
        subsystem_list: Optional[list[int]] = None,
        rotating_frame: Optional[Union[Array, RotatingFrame, str]] = "auto",
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        **options,
    ) -> PulseSimulator:
        """Construct a PulseSimulator instance from an existing BackendV1 instance.

        Args:
            num_qubits: Number of qubits.
            dt: Qubit drive channel timestep in nanoseconds.
            hamiltonian: Dictionary with fields characterizing the system hamiltonian.
            control_channels: Dictionary with fields characterizing the control channels.
            drive_frequencies: Drive channel frequencies.
            measure_frequencies: Measure channel frequencies.
            u_channel_lo: U-channel relationship on device los.
            subsystem_list: The list of qubits in the backend to include in the model.
            rotating_frame: Rotating frame argument for the internal :class:`.Solver`. Defaults to
                ``"auto"``, allowing this method to pick a rotating frame.
            evaluation_mode: Evaluation mode argument for the internal :class:`.Solver`.
            rwa_cutoff_freq: Rotating wave approximation argument for the internal :class:`.Solver`.
            **options: Additional options to be applied in construction of the
                :class:`.PulseSimulator`.

        Returns:
            PulseSimulator

        Raises:
            QiskitError: If any required parameters are missing from the passed backend.
        """
        if subsystem_list is not None:
            subsystem_list = sorted(subsystem_list)
            if subsystem_list[-1] >= num_qubits:
                raise QiskitError(
                    f"subsystem_list contained {subsystem_list[-1]}, which is out of bounds for "
                    f"backend with {num_qubits} qubits."
                )
        else:
            subsystem_list = list(range(num_qubits))

        (
            static_hamiltonian,
            hamiltonian_operators,
            hamiltonian_channels,
            subsystem_dims,
        ) = parse_backend_hamiltonian_dict(hamiltonian, subsystem_list)
        subsystem_dims = [subsystem_dims[idx] for idx in subsystem_list]

        # construct model frequencies dictionary from backend
        # partition types of channels
        drive_channels = []
        meas_channels = []
        u_channels = []
        for channel in hamiltonian_channels:
            if channel[0] == "d":
                drive_channels.append(channel)
            elif channel[0] == "m":
                meas_channels.append(channel)
            elif channel[0] == "u":
                u_channels.append(channel)
            else:
                raise QiskitError("Unrecognized channel type requested.")
        # extract and validate channel frequency parameters
        if drive_channels and drive_frequencies is None:
            raise QiskitError(
                "DriveChannels in model but frequencies not available in target or defaults."
            )
        if meas_channels and measure_frequencies is None:
            raise QiskitError("MeasureChannels in model but defaults does not have meas_freq_est.")
        # populate frequencies
        channel_freqs = {}
        for channel in drive_channels:
            idx = int(channel[1:])
            if idx >= len(drive_frequencies):
                raise QiskitError(f"DriveChannel index {idx} is out of bounds.")
            channel_freqs[channel] = drive_frequencies[idx]
        for channel in meas_channels:
            idx = int(channel[1:])
            if idx >= len(measure_frequencies):
                raise QiskitError(f"MeasureChannel index {idx} is out of bounds.")
            channel_freqs[channel] = measure_frequencies[idx]
        for channel in u_channels:
            idx = int(channel[1:])
            if idx >= len(u_channel_lo):
                raise QiskitError(f"ControlChannel index {idx} is out of bounds.")
            freq = 0.0
            for channel_lo in u_channel_lo[idx]:
                freq += drive_frequencies[channel_lo.q] * channel_lo.scale
            channel_freqs[channel] = freq
        # validate that all channels have frequencies
        for channel in hamiltonian_channels:
            if channel not in channel_freqs:
                raise QiskitError(f"No carrier frequency found for channel {channel}.")

        # Add control_channel_map from backend (only if not specified before by user)
        if "control_channel_map" not in options:
            if control_channels:
                control_channel_map_backend = {
                    qubits: control_channels[qubits][0].index for qubits in control_channels
                }
            else:
                control_channel_map_backend = {}
            # Reduce control_channel_map based on which channels are in the model
            if bool(control_channel_map_backend):
                control_channel_map = {}
                for label, idx in control_channel_map_backend.items():
                    if f"u{idx}" in hamiltonian_channels:
                        control_channel_map[label] = idx
                options["control_channel_map"] = control_channel_map

        # build the solver
        if rotating_frame == "auto":
            if "dense" in evaluation_mode:
                rotating_frame = static_hamiltonian
            else:
                rotating_frame = np.diag(static_hamiltonian)

        solver = Solver(
            static_hamiltonian=Array(static_hamiltonian),
            hamiltonian_operators=Array(hamiltonian_operators),
            hamiltonian_channels=hamiltonian_channels,
            channel_carrier_freqs=channel_freqs,
            dt=dt,
            rotating_frame=rotating_frame,
            evaluation_mode=evaluation_mode,
            rwa_cutoff_freq=rwa_cutoff_freq,
        )

        return cls(
            solver=solver,
            target=Target(dt=dt),
            subsystem_labels=subsystem_list,
            subsystem_dims=subsystem_dims,
            **options,
        )
