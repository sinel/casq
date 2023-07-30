.. _pulse-backend:

################################################################################
Pulse Backend
################################################################################

The pulse backend is modeled in three parts: the `HamiltonianModel <../autoapi/casq/models/hamiltonian_model/index.html>`_ characterizing the quantum system, the NoiseModel characterizing the interaction of the quantum system with the environment, and the `ControlModel <../autoapi/casq/models/control_model/index.html>`_ characterizing the physical system used to control the quantum system.

.. warning::
    The NoiseModel is under construction. Need to better understand how to use Lindblad operators to characterize different types of noise. Also, can qiskit-dynamics Lindblad solver be combined with other ways to characterize/simulate noise? If not, how can this be combined with noise simulation approaches by other libraries to provide a higher-level abstraction?

.. note::
    Let's not forget to configure the environment to use Jax.

    .. jupyter-execute::

        from casq.common import initialize_jax

        initialize_jax()

Hamiltonian model
================================================================================

The `HamiltonianModel <../autoapi/casq/models/hamiltonian_model/index.html>`_ contains all the static and time-dependent operators defining the Hamiltonian to be used for the Schrödinger equation defining the behavior of the quantum system as well as parameters specifying the rotating frame transformation and rotating wave approximation.

.. note::
    The Hamiltonian model is standardized around using the `Hamiltonian dictionary string specification by Qiskit <https://qiskit.org/ecosystem/dynamics/stubs/qiskit_dynamics.backend.parse_backend_hamiltonian_dict.html>`_. It is convenient to define a Hamiltonian in string format rather than Python code for intuitive input and construction of models. However, the Hamiltonian dictionary approach by Qiskit is useful yet awkward, and a more intuitive and easy-to-use DSL will be developed for specifying Hamiltonian's in the near future.

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
            frequency=30397743782.610542,
            anharmonicity=-2169482392.6367006,
            drive=892870223.8110852
        ),
        2: TransmonModel.TransmonProperties(
            frequency=31649945798.50227,
            anharmonicity=-2152313197.3287387,
            drive=927794953.0001632
        ),
        3: TransmonModel.TransmonProperties(
            frequency=31107813662.24873,
            anharmonicity=-2158766696.6684937,
            drive=921439621.8693779
        ),
        4: TransmonModel.TransmonProperties(
            frequency=31825180853.3539,
            anharmonicity=-2149525690.7311115,
            drive=1150709205.1097605
        ),
    }
    coupling_map = {
        (0, 1): 11845444.218797993,
        (1, 2): 11967839.68906386,
        (2, 3): 12402113.956012368,
        (3, 4): 12186910.37040823,
    }
    hamiltonian = TransmonModel(
        qubit_map=qubit_map,
        coupling_map=coupling_map,
        extracted_qubits=[0]
    )

Control model
================================================================================

The `ControlModel <../autoapi/casq/models/control_model/index.html>`_ defines the relevant properties of the physical system used for controlling the quantum system, such as the sampling interval used for digitizing microwave pulses, or channel frequencies used for applying drive, control, and measurement pulses.

.. jupyter-execute::

    from casq.models import ControlModel

    control = ControlModel(
        dt=2.2222222222222221e-10,
        channel_carrier_freqs={
            "d0": 4962770879.920025,
            "d1": 4838412258.764764,
            "d2": 5036989248.286842,
            "d3": 4951300212.210368,
            "d4": 5066350584.469812,
            "u0": 4838412258.764764,
            "u1": 4962770879.920025,
            "u2": 5036989248.286842,
            "u3": 4838412258.764764,
            "u4": 4951300212.210368
        }
    )

Running a circuit on the backend
================================================================================

One can then proceed to build a pulse backend using the above models as follows:

.. jupyter-execute::

    from casq.backends.helpers import build, BackendLibrary

    pulse_backend = build(
        backend_library=BackendLibrary.QISKIT,
        hamiltonian=hamiltonian,
        control=control
    )

The resulting pulse backend can be used to simulate the execution of a circuit as follows:

.. jupyter-execute::

    %%time

    from casq.backends import PulseBackend
    from casq.gates import DragPulseGate, PulseCircuit

    gate = DragPulseGate(duration=256, amplitude=1)
    circuit = PulseCircuit.from_pulse_gate(gate, {"sigma": 128, "beta": 2})
    solution = pulse_backend.solve(
        circuit,
        method=PulseBackend.ODESolverMethod.SCIPY_DOP853
    )
    print(solution.counts[-1])

Run performance will significantly improve if a JAX solver is used.

.. jupyter-execute::

    %%time

    solution = pulse_backend.solve(
        circuit,
        method=PulseBackend.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT
    )
    print(solution.counts[-1])

It is easy to visualize the results. For example, one can view the resulting IQ points as follows.

.. jupyter-execute::

    solution.plot_iq();

Using Qiskit backends
================================================================================

It is very simple to construct pulse backends based on model information provided by a Qiskit backend.

.. jupyter-execute::

    %%time

    from qiskit.providers.fake_provider import FakeManila
    from casq.backends import build_from_backend

    qiskit_pulse_backend = build_from_backend(
        backend=FakeManila(),
        extracted_qubits=[0]
    )
    solution = qiskit_pulse_backend.solve(
        circuit,
        method=PulseBackend.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT
    )
    print(solution.counts[-1])

.. warning::
    The parameters values for the pulse backend built via models are based on FakeManila. However, the resulting counts are significantly different than those obtained via building the pulse backend directly from FakeManila. Need to figure out why - is this a bug or is there some explanation for it?

Time evolution
================================================================================

Using the ``steps`` argument, the pulse backend can be executed to return the results of intermediate steps to solve for the time evolution of the circuit execution.

.. jupyter-execute::

    %%time

    solution = qiskit_pulse_backend.solve(
        circuit,
        method=PulseBackend.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT,
        steps=100
    )

Using the resulting solution, the time trajectory of the Pauli vectors can easily be viewed.

.. jupyter-execute::

    solution.plot_trajectory();

Similarly, the time trajectory of the Bloch vector can also be viewed.

.. jupyter-execute::

    solution.plot_bloch_trajectory();

Or how the qubit populations vary in time.

.. jupyter-execute::

    solution.plot_population();
