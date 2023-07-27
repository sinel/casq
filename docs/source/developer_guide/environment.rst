.. _environment:

################################################################################
Environment
################################################################################

Source Code
================================================================================

The library is hosted on `Github <https://github.com/sinel/casq>`_. For development, it should be cloned locally using

.. code-block::

    git clone https://github.com/sinel/casq.git

Since the library is in very early development stage, versioning/tagging has not been implemented and the ``main`` branch is being used for all development. Hence all pull and push requests are currently made to the ``main`` branch only.

Dependency Management
================================================================================

`Poetry <https://python-poetry.org/>`_ is used as virtual environment and for dependency management. It is required for development, and it should be installed according to instructions provided on its web site. Once installed, all required runtime and development dependencies can be added by simply running:

.. code-block::

    poetry update

Executing Python
================================================================================

After cloning and completing installation of `Poetry <https://python-poetry.org/>`_ and all Casq dependencies, Python files can be executed as follows:

.. code-block::

    poetry run python about.py
