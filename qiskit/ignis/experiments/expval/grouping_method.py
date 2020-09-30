# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Snapshot Expectation Value Experiment.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.ignis.experiments.base import ConstantGenerator
from qiskit.quantum_info import Operator, SparsePauliOp

logger = logging.getLogger(__name__)


class GroupingExpvalGenerator(ConstantGenerator):
    """Snapshot expectation value Generator"""

    def __init__(
        self,
        observable: Union[SparsePauliOp, Operator],
        qubits: Optional[List[int]] = None,
        strategy="nogrouping",
    ):
        """Initialize generator"""
        if isinstance(observable, SparsePauliOp):
            self._op = observable
        else:
            self._op = SparsePauliOp.from_operator(observable)

        # Get snapshot params for operator
        params = [[coeff, pauli] for pauli, coeff in self._op.label_iter()]

        num_qubits = self._op.num_qubits
        snapshot_qubits = list(range(num_qubits))

        circuit = QuantumCircuit(num_qubits)
        for param in params:
            circuit.snapshot(
                "expval_" + param[1],
                snapshot_type="expectation_value_pauli",
                qubits=snapshot_qubits,
                params=[[1, param[1]]],
            )

        super().__init__(
            "expval",
            [circuit],
            [{"method": "snapshot_nogrouping", "params": params}],
            qubits=qubits,
        )


def grouping_analysis_fn(
    data: List[Dict], metadata: List[Dict[str, any]], mitigator: Optional = None
):
    """Fit expectation value from snapshots."""
    if mitigator is not None:
        logger.warning(
            "Error mitigation cannot be used with the snapshot"
            " expectation value method."
        )

    if len(data) != 1:
        raise QiskitError("Invalid list of data")

    snapshots = data[0]
    meta = metadata[0]

    params = meta["params"]
    coeffs = {param[1]: param[0][0] + param[0][1] * 1j for param in params}

    expval = 0
    variance = 0
    for key, value in snapshots.items():
        prefix = key.split("_")[0]
        if prefix == "expval":
            paulis = key.split("_")[1]
            expval_paulis = value[0]['value']
            expval += coeffs[paulis] * expval_paulis
            variance += abs(coeffs[paulis])**2 * (1-expval_paulis**2)

    # Convert to real if imaginary part is zero
    if np.isclose(expval.imag, 0):
        expval = expval.real
    if np.isclose(variance.imag, 0):
        variance = variance.real

    # Get shots
    if "shots" in meta:
        shots = meta["shots"]
    else:
        shots = snapshots.get("shots", 1)

    stderror = np.sqrt(variance / shots)

    return expval, stderror
