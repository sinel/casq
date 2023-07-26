.. _pulse-optimizer:

################################################################################
Pulse Optimizer
################################################################################

The pulse backend is modeled in three parts: the `HamiltonianModel <../autoapi/casq/models/hamiltonian_model/index.html>`_ characterizing the quantum system, the `NoiseModel <../autoapi/casq/models/noise_model/index.html>`_ characterizing the interaction of the quantum system with the environment, and the `ControlModel <../autoapi/casq/models/control_model/index.html>`_ characterizing the physical system used to control the quantum system.

.. note::
    Let's first configure the environment to use Jax.

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

    %%time

    import numpy as np
    from qiskit.providers.fake_provider import FakeManila

    from casq import PulseOptimizer
    from casq.backends import build_from_backend, PulseBackend

    pulse_backend = build_from_backend(
        backend=FakeManila(),
        extracted_qubits=[0]
    )
    optimizer = PulseOptimizer(
        pulse_type=PulseOptimizer.PulseType.GAUSSIAN_SQUARE,
        pulse_arguments={
            "duration": 256,
            "amplitude": 1,
            "name": "x",
            "sigma": None,
            "width": None,
        },
        backend=pulse_backend,
        method=PulseBackend.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT,
        target_measurement={"0": 0, "1": 1024},
        fidelity_type=PulseOptimizer.FidelityType.COUNTS,
        target_qubit=0,
        use_jax=True,
        # use_jit=True,
    )
    solution = optimizer.optimize(
        params=np.array([10.0, 10.0]),
        method=PulseOptimizer.OptimizationMethod.SCIPY_NELDER_MEAD,
        # jac=True,
        tol=1e-2,
        # maxiter=10
    )
    print(
        "================================================================================"
    )
    print("OPTIMIZED PULSE")
    print(
        "================================================================================"
    )
    print(f"ITERATIONS: {solution.num_iterations}")
    print(f"OPTIMIZED PARAMETERS: {solution.parameters}")
    print(f"MEASUREMENT: {solution.measurement}")
    print(f"FIDELITY: {solution.fidelity}")
    print(f"MESSAGE: {solution.message}")
    print(
        "================================================================================"
    )
