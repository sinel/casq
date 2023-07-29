.. _helpers:

################################################################################
Convert pulse circuit into schedule
################################################################################

Converting a circuit into a schedule with measurement is a common operation done for visualization or computational reasons. The ``to_schedule`` helper method accomplishes this in a single line:

.. jupyter-execute::

    from qiskit.providers.fake_provider import FakeManila

    from casq.backends.qiskit.helpers import to_schedule
    from casq.gates import DragPulseGate, PulseCircuit

    gate = DragPulseGate(duration=256, amplitude=1)
    circuit = PulseCircuit.from_pulse_gate(gate, {"sigma": 128, "beta": 2})
    schedule = to_schedule(circuit=circuit, backend=FakeManila())
    schedule.draw()
