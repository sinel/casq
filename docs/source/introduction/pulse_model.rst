.. _pulse-model:

################################################################################
Pulse Backend
################################################################################

The pulse backend is modeled in three parts: the `HamiltonianModel <../autoapi/casq/models/hamiltonian_model/index.html>`_ characterizing the quantum system, the `NoiseModel <../autoapi/casq/models/noise_model/index.html>`_ characterizing the interaction of the quantum system with the environment, and the `ControlModel <../autoapi/casq/models/control_model/index.html>`_ characterizing the physical system used to control the quantum system.

Hamiltonian model
================================================================================

The `HamiltonianModel <../autoapi/casq/models/hamiltonian_model/index.html>`_ contains all the static and time-dependent operators defining the Hamiltonian to be used for the Schrödinger equation defining the behavior of the quantum system as well as parameters specifying the rotating frame transformation and rotating wave approximation.

There will be a growing library of pre-defined Hamiltonian models, but currently only the `TransmonModel <../autoapi/casq/models/transmon_model/index.html>`_ is provided to model superconducting quantum circuits consisting of **transmon** qubits. Hamiltonian models for **flux** and **unimon** qubits are planned in the near future.

For example, building a Hamiltonian model of a 5-qubit transmon quantum processor is as simple as:

.. jupyter-execute::

    import json
    from casq.models import TransmonModel

    qubit_map = {
        0: TransmonModel.TransmonProperties(
            frequency=31179079102.853794,
            anharmonicity=-2165345334.8252344,
            drive=926545606.6640488
        ),
        1: TransmonModel.TransmonProperties(
            frequency=31179079102.853794,
            anharmonicity=-2165345334.8252344,
            drive=926545606.6640488
        ),
        2: TransmonModel.TransmonProperties(
            frequency=31179079102.853794,
            anharmonicity=-2165345334.8252344,
            drive=926545606.6640488
        ),
        3: TransmonModel.TransmonProperties(
            frequency=31179079102.853794,
            anharmonicity=-2165345334.8252344,
            drive=926545606.6640488
        ),
        4: TransmonModel.TransmonProperties(
            frequency=31179079102.853794,
            anharmonicity=-2165345334.8252344,
            drive=926545606.6640488
        ),
    }
    coupling_map = {
        (0, 1): 11845444.218797993,
        (1, 2): 11845444.218797993,
        (2, 3): 11845444.218797993,
        (3, 4): 11845444.218797993,
    }
    hamiltonian = TransmonModel(
        qubit_map=qubit_map,
        coupling_map=coupling_map
    )
    # View resulting Hamiltonian dictionary
    print(json.dumps(hamiltonian.hamiltonian_dict, indent=2, default=str))

Noise model
================================================================================

The `NoiseModel <../autoapi/casq/models/noise_model/index.html>`_ is still under early development, and more work is required to shape into a more user-friendly structure. Currently, it simply consists of a few fields for defining the Lindblad master equation in terms of static and time-dependent operators.

A library of pre-defined noise models matching each pre-defined Hamiltonian model is also under construction with the `TransmonNoiseModel <../autoapi/casq/models/transmon_noise_model/index.html>`_ as its first member. For example, adding noise to the model of a 5-qubit transmon quantum processor is as simple as:

.. jupyter-execute::

    from casq.models import TransmonNoiseModel

    qubit_map = {
        0: TransmonNoiseModel.TransmonNoiseProperties(
            t1=0.00010918719287058488,
            t2=5.077229750099717e-06
        ),
        1: TransmonNoiseModel.TransmonNoiseProperties(
            t1=5.753535189181149e-05,
            t2=6.165015600725496e-05
        ),
        2: TransmonNoiseModel.TransmonNoiseProperties(
            t1=0.00018344197711073844,
            t2=2.512378482362435e-05
        ),
        3: TransmonNoiseModel.TransmonNoiseProperties(
            t1=0.00010961657783040683,
            t2=5.7120186456626996e-05
        ),
        4: TransmonNoiseModel.TransmonNoiseProperties(
            t1=0.00010247738825319845,
            t2=3.722985261736209e-05
        )
    }
    noise = TransmonNoiseModel(qubit_map=qubit_map)
    # View resulting static dissipators for Lindblad equation
    print(noise.static_dissipators)

Control model
================================================================================

The `ControlModel <../autoapi/casq/models/control_model/index.html>`_ defines the relevant properties of the physical system used for controlling the quantum system, such as the sampling interval used for digitizing microwave pulses, or channel frequencies used for applying drive, control, and measurement pulses.

Running a circuit on the backend
================================================================================

One can then proceed to build a pulse backend using the above models as follows:

.. jupyter-execute::

    from qiskit.providers.fake_provider import FakeManila
    from casq.backends.helpers import build, BackendLibrary
    from casq.models import ControlModel
    from casq.backends import BackendCharacteristics

    backend = FakeManila()
    characteristics = BackendCharacteristics(backend)
    dt = characteristics.dt
    print(dt)
    freqs = characteristics.get_channel_frequencies(
        channels=hamiltonian.channels
    )
    print(freqs)
    control = ControlModel(
        dt=dt,
        channel_carrier_freqs=freqs
    )
    backend = build(
        backend_library=BackendLibrary.QISKIT,
        hamiltonian=hamiltonian,
        control=control
    )

The resulting pulse backend can then be used to simulate the execution of a circuit as follows:

.. jupyter-execute::

    from casq.backends import PulseBackend
    from casq.gates import DragPulseGate, PulseCircuit

    gate = DragPulseGate(duration=256, amplitude=1, sigma=128, beta=2)
    circ = PulseCircuit.from_pulse(gate)
    solution = backend.run(circ, method=PulseBackend.ODESolverMethod.SCIPY_DOP853)
    print(solution.counts[-1])
