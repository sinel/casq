.. _pulse-gate:

Pulse Gate
================================================================================

Since Qiskit does not support circuit-level pulse gates yet, the `PulseGate <../autoapi/casq/gates/pulse_gate/index.html>`_ class extends the Qiskit `Gate <https://qiskit.org/documentation/stubs/qiskit.circuit.Gate.html#qiskit.circuit.Gate>`_ class to support this feature. It is the basic building block for creating circuits which support custom gates with low-level pulse definitions.

Parametric Representation
--------------------------------------------------------------------------------

The ``pulse`` method returns a parametric representation of the underlying waveform corresponding to the `PulseGate <../autoapi/casq/gates/pulse_gate/index.html>`_ type. Currently, the following parametric pulse representations from the Qiskit pulse library are supported:

* `Gaussian <https://qiskit.org/documentation/stubs/qiskit.pulse.library.Gaussian_class.rst.html#qiskit.pulse.library.Gaussian>`_
* `GaussianSquare <https://qiskit.org/documentation/stubs/qiskit.pulse.library.GaussianSquare.html#qiskit.pulse.library.GaussianSquare>`_
* `Drag <https://qiskit.org/documentation/stubs/qiskit.pulse.library.Drag_class.rst.html#qiskit.pulse.library.Drag>`_

The remainder of the Qiskit pulse library will be supported in the near future. Since Qiskit is used as the common API for pulse-, gate- and circuit-level objects, conversion methods from Qiskit to other supported libraries (e.g. Qutip, C3) will also be provided in the future.

Furthermore, each parametric pulse representation within a `PulseGate <../autoapi/casq/gates/pulse_gate/index.html>`_ is constructed as a `ScalableSymbolicPulse <https://github.com/Qiskit/qiskit-terra/blob/0.24.2/qiskit/pulse/library/symbolic_pulses.py#L573>`_ instance with support for `JAX <https://jax.readthedocs.io/en/latest/>`_.

Example
--------------------------------------------------------------------------------

For example, creating a Drag pulse gate and viewing the corresponding parametric waveform is as simple as:

.. jupyter-execute::

    from casq.gates import DragPulseGate

    gate = DragPulseGate(duration=256, amplitude=1)
    gate.pulse({"sigma": 128, "beta": 2}).draw()

The pulse gate can easily be used as part of a Qiskit QuantumGate.

.. jupyter-execute::

    from qiskit.pulse import build, play, DriveChannel
    from qiskit.pulse.transforms import block_to_schedule
    from casq.common import discretize, plot_signal
    from casq.gates import DragPulseGate

    gate = DragPulseGate(duration=256, amplitude=1)
    with build() as sb:
        play(gate.pulse({"sigma": 128, "beta": 2}), DriveChannel(0))
    schedule = block_to_schedule(sb)
    signals = discretize(schedule, dt=0.22e-9, channel_frequencies={"d0": 5e9})
    plot_signal(signals[0], number_of_samples=100)

A low number of samples were used for plotting the discretized signal in order to clearly view the waveform within the envelope of the pulse.
