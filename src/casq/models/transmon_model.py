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
"""Transmon model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy.typing as npt

from casq.common.decorators import trace
from casq.models.hamiltonian_model import HamiltonianModel


class TransmonModel(HamiltonianModel):
    """TransmonModel class."""

    MANILA = {
        "description": "Qubits are modeled as Duffing oscillators. "
        "In this case, the system includes higher energy states, i.e. not just |0> and |1>. "
        "The Pauli operators are generalized via the following set of transformations:"
        "\n\n$(\\mathbb{I}-\\sigma_{i}^z)/2 \\rightarrow O_i \\equiv b^\\dagger_{i} b_{i}$,\n\n$\\sigma_{+} "
        "\\rightarrow b^\\dagger$,\n\n$\\sigma_{-} \\rightarrow b$,\n\n$\\sigma_{i}^X \\rightarrow b^\\dagger_{i} + "
        "b_{i}$.\n\nQubits are coupled through resonator buses. "
        "The provided Hamiltonian has been projected into the zero excitation subspace of the resonator buses "
        "leading to an effective qubit-qubit flip-flop interaction. "
        "The qubit resonance frequencies in the Hamiltonian are the cavity dressed frequencies "
        "and not exactly what is returned by the backend defaults, "
        "which also includes the dressing due to the qubit-qubit interactions."
        "\n\nQuantities are returned in angular frequencies, with units 2*pi*GHz."
        "\n\nWARNING: Currently not all system Hamiltonian information is available to the public, "
        "missing values have been replaced with 0.\n",
        "h_latex": "\\begin{align} \\mathcal{H}/\\hbar = & \\sum_{i=0}^{4}\\left(\\frac{\\omega_{q,i}}{2}(\\mathbb{I}-\\sigma_i^{z})+\\frac{\\Delta_{i}}{2}(O_i^2-O_i)+\\Omega_{d,i}D_i(t)\\sigma_i^{X}\\right) \\\\ & + J_{0,1}(\\sigma_{0}^{+}\\sigma_{1}^{-}+\\sigma_{0}^{-}\\sigma_{1}^{+}) + J_{1,2}(\\sigma_{1}^{+}\\sigma_{2}^{-}+\\sigma_{1}^{-}\\sigma_{2}^{+}) + J_{2,3}(\\sigma_{2}^{+}\\sigma_{3}^{-}+\\sigma_{2}^{-}\\sigma_{3}^{+}) + J_{3,4}(\\sigma_{3}^{+}\\sigma_{4}^{-}+\\sigma_{3}^{-}\\sigma_{4}^{+}) \\\\ & + \\Omega_{d,0}(U_{0}^{(0,1)}(t))\\sigma_{0}^{X} + \\Omega_{d,1}(U_{1}^{(1,0)}(t)+U_{2}^{(1,2)}(t))\\sigma_{1}^{X} \\\\ & + \\Omega_{d,2}(U_{3}^{(2,1)}(t)+U_{4}^{(2,3)}(t))\\sigma_{2}^{X} + \\Omega_{d,3}(U_{6}^{(3,4)}(t)+U_{5}^{(3,2)}(t))\\sigma_{3}^{X} \\\\ & + \\Omega_{d,4}(U_{7}^{(4,3)}(t))\\sigma_{4}^{X} \\\\ \\end{align}",
        "h_str": [
            "_SUM[i,0,4,wq{i}/2*(I{i}-Z{i})]",
            "_SUM[i,0,4,delta{i}/2*O{i}*O{i}]",
            "_SUM[i,0,4,-delta{i}/2*O{i}]",
            "_SUM[i,0,4,omegad{i}*X{i}||D{i}]",
            "jq0q1*Sp0*Sm1",
            "jq0q1*Sm0*Sp1",
            "jq1q2*Sp1*Sm2",
            "jq1q2*Sm1*Sp2",
            "jq2q3*Sp2*Sm3",
            "jq2q3*Sm2*Sp3",
            "jq3q4*Sp3*Sm4",
            "jq3q4*Sm3*Sp4",
            "omegad1*X0||U0",
            "omegad0*X1||U1",
            "omegad2*X1||U2",
            "omegad1*X2||U3",
            "omegad3*X2||U4",
            "omegad4*X3||U6",
            "omegad2*X3||U5",
            "omegad3*X4||U7",
        ],
        "osc": {},
        "qub": {"0": 3, "1": 3, "2": 3, "3": 3, "4": 3},
        "vars": {
            "delta0": -2165345334.8252344,
            "delta1": -2169482392.6367006,
            "delta2": -2152313197.3287387,
            "delta3": -2158766696.6684937,
            "delta4": -2149525690.7311115,
            "jq0q1": 11845444.218797993,
            "jq1q2": 11967839.68906386,
            "jq2q3": 12402113.956012368,
            "jq3q4": 12186910.37040823,
            "omegad0": 926545606.6640488,
            "omegad1": 892870223.8110852,
            "omegad2": 927794953.0001632,
            "omegad3": 921439621.8693779,
            "omegad4": 1150709205.1097605,
            "wq0": 31179079102.853794,
            "wq1": 30397743782.610542,
            "wq2": 31649945798.50227,
            "wq3": 31107813662.24873,
            "wq4": 31825180853.3539,
        },
    }

    @dataclass
    class TransmonProperties:
        """Transmon qubit properties."""

        frequency: float
        anharmonicity: float
        drive: float

    @trace()
    def __init__(
        self,
        qubit_map: dict[int, TransmonProperties],
        coupling_map: dict[tuple[int, int], float],
        extracted_qubits: Optional[list[int]] = None,
        rotating_frame: Optional[npt.NDArray] = None,
        in_frame_basis: bool = False,
        evaluation_mode: HamiltonianModel.EvaluationMode = HamiltonianModel.EvaluationMode.DENSE,
        rwa_cutoff_freq: Optional[float] = None,
        rwa_carrier_freqs: Optional[
            Union[npt.NDArray, tuple[npt.NDArray, npt.NDArray]]
        ] = None,
    ) -> None:
        """Initialize HamiltonianModel.

        Args:
            qubit_map: Dictionary mapping qubit indices to properties.
            coupling_map: Dictionary mapping qubit couplings to coupling strength.
            extracted_qubits: List of qubits to extract from the Hamiltonian.
            rotating_frame: Rotating frame operator.
                            If specified with a 1d array, it is interpreted as the
                            diagonal of a diagonal matrix. Assumed to store
                            the anti-hermitian matrix F = -iH.
            in_frame_basis: Whether to represent the model in the basis in which
                            the rotating frame operator is diagonalized.
            evaluation_mode: Evaluation mode to use.
            rwa_cutoff_freq: Rotating wave approximation cutoff frequency.
                            If None, no approximation is made.
            rwa_carrier_freqs: Carrier frequencies to use for rotating wave approximation.
        """
        # TO-DO: Need more understanding of how Hamiltonian interacts with choices for rotating frame and RWA.
        # And what about resonator? What does following comment for ibmq_manila hamiltonian mean?
        # "The provided Hamiltonian has been projected into "'"
        # "'"the zero excitation subspace of the resonator buses "'"
        # "'"leading to an effective qubit-qubit flip-flop interaction."
        q_max = len(qubit_map.keys()) - 1
        h_str = [
            f"_SUM[i,0,{q_max},wq{{i}}/2*(I{{i}}-Z{{i}})]",
            f"_SUM[i,0,{q_max},delta{{i}}/2*O{{i}}*O{{i}}]",
            f"_SUM[i,0,{q_max},-delta{{i}}/2*O{{i}}]",
            f"_SUM[i,0,{q_max},omegad{{i}}*X{{i}}||D{{i}}]",
        ]
        h_qub = {}
        h_vars = {}
        for q, props in qubit_map.items():
            h_qub[str(q)] = 3
            h_vars[f"wq{q}"] = props.frequency
            h_vars[f"delta{q}"] = props.anharmonicity
            h_vars[f"omegad{q}"] = props.drive
        for coupling, coupling_strength in coupling_map.items():
            q0 = coupling[0]
            q1 = coupling[1]
            h_vars[f"jq{q0}q{q1}"] = coupling_strength
            h_str.append(f"jq{q0}q{q1}*Sp{q0}*Sm{q1}")
            h_str.append(f"jq{q0}q{q1}*Sp{q1}*Sm{q0}")
            h_str.append(f"omegad{q1}*X{q0}||U{q0}")
            h_str.append(f"omegad{q0}*X{q1}||U{q1}")
        super().__init__(
            hamiltonian_dict={"h_str": h_str, "qub": h_qub, "vars": h_vars},
            extracted_qubits=extracted_qubits,
            rotating_frame=rotating_frame,
            in_frame_basis=in_frame_basis,
            evaluation_mode=evaluation_mode,
            rwa_cutoff_freq=rwa_cutoff_freq,
            rwa_carrier_freqs=rwa_carrier_freqs,
        )
