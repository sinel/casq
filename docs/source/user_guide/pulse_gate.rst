.. _pulse-gate:

################################################################################
Pulse Gate
################################################################################

Since Qiskit does not support circuit-level pulse gates yet, the `PulseGate <../autoapi/casq/gates/pulse_gate/index.html>`_ class extends the Qiskit `Gate <https://qiskit.org/documentation/stubs/qiskit.circuit.Gate.html#qiskit.circuit.Gate>`_ class to support this feature. It is simply a thin wrapper around Qiskit `Gate <https://qiskit.org/documentation/stubs/qiskit.circuit.Gate.html#qiskit.circuit.Gate>`_ to allow user-friendly construction of pulse gates using standard waveforms and IDE-friendly keyword arguments for parameters.

Accessing underlying waveform
================================================================================

The ``pulse`` method returns a parametric representation of the underlying waveform corresponding to the `PulseGate <../autoapi/casq/gates/pulse_gate/index.html>`_ type. Currently, the following waveforms from the Qiskit pulse library are supported:

* `Constant <https://qiskit.org/documentation/stubs/qiskit.pulse.library.Constant.html#qiskit.pulse.library.Constant>`_
* `Gaussian <https://qiskit.org/documentation/stubs/qiskit.pulse.library.Gaussian.html#qiskit.pulse.library.Gaussian>`_
* `GaussianSquare <https://qiskit.org/documentation/stubs/qiskit.pulse.library.GaussianSquare.html#qiskit.pulse.library.GaussianSquare>`_
* `Drag <https://qiskit.org/documentation/stubs/qiskit.pulse.library.Drag.html#qiskit.pulse.library.Drag>`_
* `GaussianSquareDrag <https://qiskit.org/documentation/stubs/qiskit.pulse.library.GaussianSquareDrag.html#qiskit.pulse.library.GaussianSquareDrag>`_

The remainder of the Qiskit pulse library will be supported in the near future. Since Qiskit is used as the common API for pulse-, gate- and circuit-level objects, conversion methods from Qiskit to other supported libraries (e.g. Qutip, C3) will also be provided in the future.

Furthermore, each parametric pulse representation within a `PulseGate <../autoapi/casq/gates/pulse_gate/index.html>`_ is constructed as a `ScalableSymbolicPulse <https://github.com/Qiskit/qiskit-terra/blob/0.24.2/qiskit/pulse/library/symbolic_pulses.py#L573>`_ instance with support for `JAX <https://jax.readthedocs.io/en/latest/>`_.

For example, creating a `DragPulseGate <../autoapi/casq/gates/drag_pulse_gate/index.html>`_ and viewing the corresponding parametric waveform is as simple as:

.. jupyter-execute::

    from casq.gates import DragPulseGate

    gate = DragPulseGate(duration=256, amplitude=1)
    gate.pulse({"sigma": 128, "beta": 2}).draw()

Wrapping within a schedule
================================================================================

The ``schedule`` method embeds the pulse gate within a schedule. It is embedded by attaching and playing it on the drive channel of the specified qubit.

.. jupyter-execute::

    schedule = gate.schedule({"sigma": 128, "beta": 2}, qubit=0)
    schedule.draw()
