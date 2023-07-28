.. _single_qubit:

################################################################################
Optimizing single-qubit gates
################################################################################

================================================================================
Pulse Optimization
================================================================================

In this tutorial, we will walk through a detailed example of optimizing a single-qubit gate using
``casq``.

We will optimize an X-gate on a model of a qubit system using the following steps:

#. Configure the environment to work with JAX.
#. Build a parameterized pulse gate to optimize over.
#. Build a pulse backend with the model of the system and ODE solver.
#. Build a pulse optimizer specifying search method, fidelity function, and search method.
#. Repeat the optimization to compare different pulse gates, methods, and fidelity functions.
