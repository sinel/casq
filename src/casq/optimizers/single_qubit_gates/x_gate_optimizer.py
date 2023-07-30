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
"""Pulse circuit."""
from __future__ import annotations

from typing import Any, Optional

from casq.backends.pulse_backend import PulseBackend
from casq.common import trace
from casq.gates import PulseGate
from casq.optimizers.single_qubit_gates.single_qubit_gate_optimizer import (
    SingleQubitGateOptimizer,
)


class XGateOptimizer(SingleQubitGateOptimizer):
    """XGateOptimizer class.

    Args:
        pulse_gate: Pulse gate.
        pulse_backend: Pulse backend.
        method: ODE solver method.
        method_options: Options specific to method.
        fidelity_type: Fidelity type. Defaults to FidelityType.COUNTS.
        use_jit: If True, then jit and value_and_grad is applied to objective function.
    """

    @trace()
    def __init__(
        self,
        pulse_gate: PulseGate,
        pulse_backend: PulseBackend,
        method: PulseBackend.ODESolverMethod,
        method_options: Optional[dict[str, Any]] = None,
        fidelity_type: Optional[XGateOptimizer.FidelityType] = None,
        use_jit: bool = False,
    ):
        """Initialize XGateOptimizer."""
        target_measurement = {"0": 0, "1": 1024}
        super().__init__(
            pulse_gate,
            pulse_backend,
            method,
            target_measurement,
            method_options,
            fidelity_type,
            use_jit,
        )
