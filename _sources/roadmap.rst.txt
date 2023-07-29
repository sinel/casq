.. _roadmap:

.. |check| raw:: html

    <input checked="" type="checkbox">

.. |uncheck| raw:: html

    <input type="checkbox">

################################################################################
Roadmap
################################################################################

v0.1.0 (prototype)
================================================================================

* |check| Skeleton
    * |check| Local testing, linting & formatting
    * |check| Documentation
    * |check| Continuous integration
* |check| Qiskit foundation
* |check| Support for Qiskit dynamics
* |check| Pulse gates & circuit
* |check| Pulse backend & models
* |check| Pulse optimizer
* |check| Single-qubit gate optimization

v0.2.0 (functionality & verification)
================================================================================

* |uncheck| Noise model
* |uncheck| Additional Hamiltonian models
* |uncheck| Model verification
* |uncheck| Model builder DSL - now or later?
* |uncheck| Two-qubit gate optimization
* |uncheck| Additional fidelity functions
* |uncheck| Improved plotting & reporting

v0.3.0 (library support: qutip)
================================================================================

* |uncheck| Support for QuTiP (qutip-qip, qutip-qtrl, qutip-jax, GRAPE, CRAB)
* |uncheck| Improve pulse backend & model abstraction
* |uncheck| Pulse verification via QuTiP

v0.4.0 (library support: c3)
================================================================================

* |uncheck| Support for C3 (CMA-ES)
* |uncheck| Improve pulse backend & model abstraction
* |uncheck| Pulse verification via C3

v0.5.0 (ode solvers)
================================================================================

* |uncheck| Support for diffrax (jax)
* |uncheck| Support for SciPy ODE solvers (jax?)
* |uncheck| Other alternatives?

v0.6.0 (optimization)
================================================================================

* |uncheck| Support for SciPy minimize (with jax)
* |uncheck| Support for Google jaxopt
* |uncheck| TensorFlow as alternative for performance and automatic differentiation?
* |uncheck| Bayesian algorithms (scikit-optimize, bayes_opt, hyperopt)
* |uncheck| Evolutionary algorithms (leap, pygad, pymoo, pycma)
* |uncheck| KROTOV (https://github.com/qucontrol/krotov)
* |uncheck| GOAT (https://github.com/pmarkus-github/goat-quantumcontrol)
* |uncheck| COCOA (code?)
* |uncheck| Grid sweep for landscape

v1.0.0 (ui & packaging)
================================================================================

* |uncheck| UI
* |uncheck| Configuration
* |uncheck| Batch jobs for sweeping and comparative analysis
* |uncheck| Semantic versioning
* |uncheck| PyPI

Down the road...
================================================================================

* |uncheck| Machine/reinforcement learning
* |uncheck| Waveform-independent optimization?
