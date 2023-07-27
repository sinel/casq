.. _getting-started:

################################################################################
Getting Started
################################################################################

Casq is under development and it cannot yet be installed via PyPi. It has to installed from source, and the latest source code is available in the `Github repository <https://github.com/sinel/casq>`_.

Requirements
================================================================================

Casq depends on many requirements which are all specified and maintained by the ``pyproject.toml`` file in the root directory of its Github repository.

Installing from source
================================================================================

You may use pip (or, in a similar manner, any other dependency manager such as Poetry) to install Casq and use it as a library.

.. code-block:: python

    python -m pip install casq@git+https://github.com/sinel/casq

The installation can be verified using:

.. code-block:: python

    import casq

    casq.about()
