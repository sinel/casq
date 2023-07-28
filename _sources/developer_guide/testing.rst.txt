.. _testing:

################################################################################
Testing
################################################################################

`Pytest <https://docs.pytest.org/en/7.4.x/>`_ is used for unit testing. `pytest-mock <https://pytest-mock.readthedocs.io/en/latest/>`_ is used for mocking, and `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ is used for coverage reports.

All configuration for unit testing can be found in ``pyproject.toml``. Hence all unit tests can be simply run using the following command:

.. code-block::

    poetry run pytest

Additionally, `tox <https://tox.wiki/en/4.6.4/>`_ is used to standardize the testing environment. All unit tests can also be executed using `tox <https://tox.wiki/en/4.6.4/>`_ as follows:

.. code-block::

    poetry run tox

Configuration for `tox <https://tox.wiki/en/4.6.4/>`_ is kept in a separate ``tox.ini`` file in the root directory.
