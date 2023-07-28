.. _style:

################################################################################
Coding Style
################################################################################

All coding must follow `PEP8 <https://peps.python.org/pep-0008/>`_ style.

Automated Formatting
================================================================================

However PEP8 allows a significant amount of flexibility which leads to code that looks different from one developer to another and disrupts one's focus on the content. Hence code is further styled and automatically formatted according to the strict rules enforced by `Black <https://pypi.org/project/black/>`_ to ensure consistency.

Black can be run within a terminal window in the root directory using the following command:

.. code-block::

    poetry run black src tests

All configuration for `Black <https://pypi.org/project/black/>`_ can be found in ``pyproject.toml``.

Linting
================================================================================

We also use `Ruff <https://beta.ruff.rs/docs/>`_ for linting to enforce enforce Google-style docstring convention and some other linting features not covered by `Black <https://pypi.org/project/black/>`_. It can also be run in a similar manner using the following command:

.. code-block::

    poetry run ruff src tests

All configuration for `Ruff <https://beta.ruff.rs/docs/>`_ can be found in ``pyproject.toml``.

Sorting Imports
================================================================================

We use `isort <https://pycqa.github.io/isort/>`_ for automated sorting of imports. It can also be run in a similar manner using the following command:

.. code-block::

    poetry run isort src tests

All configuration for `isort <https://pycqa.github.io/isort/>`_ can be found in ``pyproject.toml``.
