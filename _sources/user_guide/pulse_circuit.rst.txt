.. _pulse-circuit:

################################################################################
Pulse Circuit
################################################################################

Similar to `PulseGate <../autoapi/casq/gates/pulse_gate/index.html>`_ extending Qiskit `Gate <https://qiskit.org/documentation/stubs/qiskit.circuit.Gate.html#qiskit.circuit.Gate>`_, the `PulseCircuit <../autoapi/casq/gates/pulse_circuit/index.html>`_ class extends the Qiskit `QuantumCircuit <https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html>`_ class to allow user-friendly construction of circuits using pulse gates.

Adding a pulse gate
================================================================================

Using the ``pulse_gate`` method, adding a pulse gate to a circuit is as simple and intuitive as:

.. jupyter-execute::

    from casq.gates import DragPulseGate, PulseCircuit

    gate = DragPulseGate(duration=256, amplitude=1)
    circuit = PulseCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.pulse_gate(gate, {"sigma": 128, "beta": 2}, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    circuit.draw('mpl')

Building a single-gate pulse circuit
================================================================================

For optimizing single-gate pulses, we only need a simple circuit consisting of the target pulse gate. In this case, the ``from_pulse_gate`` helper method can be used to create a circuit on the fly:

.. jupyter-execute::

    circuit = PulseCircuit.from_pulse_gate(gate, {"sigma": 128, "beta": 2})
    circuit.draw('mpl')
