.. _pulse-gate:

Pulse Gate
================================================================================

Since Qiskit does not yet support circuit-level pulse gates, the `PulseGate <../autoapi/casq/gates/pulse_gate/index.html>`_ class extends the Qiskit `Gate <https://qiskit.org/documentation/stubs/qiskit.circuit.Gate.html#qiskit.circuit.Gate>`_ class to support this feature as well as adding several helper methods. As a result, it is the basic object used in Casq for building circuits supporting custom gates with low-level pulse definitions.

Parametric Pulse Representations
--------------------------------------------------------------------------------

The ``pulse`` method returns a parametric pulse representation based on `PulseGate <../autoapi/casq/gates/pulse_gate/index.html>`_ type. Currently, the following parametric pulse representations from the Qiskit pulse library are supported:

* `Gaussian <https://qiskit.org/documentation/stubs/qiskit.pulse.library.Gaussian_class.rst.html#qiskit.pulse.library.Gaussian>`_
* `GaussianSquare <https://qiskit.org/documentation/stubs/qiskit.pulse.library.GaussianSquare.html#qiskit.pulse.library.GaussianSquare>`_
* `Drag <https://qiskit.org/documentation/stubs/qiskit.pulse.library.Drag_class.rst.html#qiskit.pulse.library.Drag>`_

The remainder of the Qiskit pulse library will be supported in the near future. Since Qiskit is used as the common API for pulse-, gate- and circuit-level objects, conversion methods from Qiskit to other supported libraries (e.g. Qutip, C3) will also be provided in the future.

Furthermore, each parametric pulse representation within a `PulseGate <../autoapi/casq/gates/pulse_gate/index.html>`_ is constructed as a `ScalableSymbolicPulse <https://github.com/Qiskit/qiskit-terra/blob/0.24.2/qiskit/pulse/library/symbolic_pulses.py#L573>`_ instance with support for `JAX <https://jax.readthedocs.io/en/latest/>`_.

.. jupyter-execute::

    from casq.gates import DragPulseGate

    gate = DragPulseGate(duration=256, amplitude=1, sigma=128, beta=2)
    gate.draw_signal(qubit=0, dt=0.22e-9, carrier_frequency=5e8)

Helper Methods
--------------------------------------------------------------------------------

The `PulseGate <../autoapi/casq/gates/pulse_gate/index.html>`_ class provides two helper methods.

The ``schedule`` helper method allows one to convert a pulse gate into a Qiskit `Schedule <https://qiskit.org/documentation/stubs/qiskit.pulse.Schedule.html#qiskit.pulse.Schedule>`_.

If the ``measured`` argument is True, then the resulting `Schedule <https://qiskit.org/documentation/stubs/qiskit.pulse.Schedule.html#qiskit.pulse.Schedule>`_ includes measurement of the specified qubit. Measurement requires specification of an appropriate Qiskit `Backend <https://qiskit.org/documentation/stubs/qiskit.providers.Backend.html#qiskit.providers.Backend>`_ instance.

If the ``discretized`` argument is True, then the method will discretize the pulse representation of the gate, and return a list of `Signal <https://qiskit.org/ecosystem/dynamics/stubs/qiskit_dynamics.signals.Signal.html>`_ instances. This method requires either an appropriate Qiskit `Backend <https://qiskit.org/documentation/stubs/qiskit.providers.Backend.html#qiskit.providers.Backend>`_ instance, or arguments specifying sampling time interval and channel carrier frequencies.

The other ``draw_signal`` helper method allows one to plot the discretized pulse representation of the gate.
