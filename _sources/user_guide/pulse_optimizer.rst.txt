.. _pulse-optimizer:

################################################################################
Pulse Optimizer
################################################################################

The pulse optimizer `PulseOptimizer <../autoapi/casq/optimizers/pulse_optimizer/index.html>`_ provides a high-level interface for optimizing pulse gates within a circuit. Its children provide further specialized instances for optimizing single- qubit and two-qubit gates.

.. note::
    Let's first configure the environment to use Jax.

    .. jupyter-execute::

        from casq.common import initialize_jax

        initialize_jax()

Single-qubit gates
================================================================================

While the `SingleQubitGateOptimizer <../autoapi/casq/optimizers/single_qubit_gates/SingleQubitGateOptimizer/index.html>`_ can be used to optimize any single-qubit pulse gate, specialized optimizers such as the `XGateOptimizer <../autoapi/casq/optimizers/single_qubit_gates/XGateOptimizer/index.html>`_ can be used to optimize specific gates with minimal input.

For example, an X-gate within a 5-qubit transmon quantum processor is as simple as:

.. jupyter-execute::

    %%time

    import numpy as np
    from qiskit.providers.fake_provider import FakeManila

    from casq.optimizers import XGateOptimizer
    from casq.backends import build_from_backend, PulseBackend
    from casq.gates import GaussianSquarePulseGate

    pulse_gate=GaussianSquarePulseGate(duration=256, amplitude=1, name="x")
    backend = FakeManila()
    dt = backend.configuration().dt
    pulse_backend = build_from_backend(backend, extracted_qubits=[0])
    optimizer = XGateOptimizer(
        pulse_gate=pulse_gate,
        pulse_backend=pulse_backend,
        method=PulseBackend.ODESolverMethod.QISKIT_DYNAMICS_JAX_ODEINT,
        method_options={"method": "jax_odeint", "atol": 1e-6, "rtol": 1e-8, "hmax": dt},
        fidelity_type=XGateOptimizer.FidelityType.COUNTS
    )
    solution = optimizer.solve(
        initial_params=np.array([10.0, 10.0]),
        method=XGateOptimizer.OptimizationMethod.SCIPY_NELDER_MEAD,
        constraints=[
            {"type": "ineq", "fun": lambda x: x[0]},
            {"type": "ineq", "fun": lambda x: 256 - x[0]},
            {"type": "ineq", "fun": lambda x: x[1]},
            {"type": "ineq", "fun": lambda x: 256 - x[1]},
        ],
        tol=1e-2
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

As a result, the initial pulse

.. jupyter-execute::

    solution.initial_pulse.draw()

has been optimized into:

.. jupyter-execute::

    solution.pulse.draw()
