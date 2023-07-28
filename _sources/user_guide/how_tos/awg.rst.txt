.. _awg:

################################################################################
Generate AWG signals from pulse gate
################################################################################

It is simple to generate and plot the signals that are used with IQ mixers, corresponding to the output ports of an Arbitrary Waveform Generator (AWG), from a pulse gate.

Let's first define a pulse gate.

.. jupyter-execute::

    from casq.gates import DragPulseGate

    gate = DragPulseGate(duration=256, amplitude=1)
    gate.pulse({"sigma": 128, "beta": 2}).draw()

We can then use the ``discretize`` helper function to get the discrete signals for the corresponding schedule of this pulse gate. The ``discretize`` function returns a list of ``SignalData`` instances. Each instance contains the discretized signal and corresponding IQ signals for each instruction in the schedule. For the current case, there is only one ``play`` instruction.

.. jupyter-execute::

    from casq.common import discretize, plot_signal

    schedule = gate.schedule({"sigma": 128, "beta": 2}, qubit=0)
    signals = discretize(schedule, dt=0.22e-9, channel_frequencies={"d0": 5e9})
    plot_signal(signals[0], number_of_samples=100)

A low number of samples were used for plotting the discretized signal in order to clearly view the waveform within the envelope of the pulse.
