.. _ci:

################################################################################
Continuous Integration
################################################################################

Local
================================================================================

`pre-commit <https://pre-commit.com/>`_ is used locally to automatically run hooks on every commit and ensure that there are no issues related to formatting, linting, import sorting, type safety, and unit testing. Check their site for more details, but at a minimum, the following command must be run so that the hooks are setup for automatic execution.

.. code-block::

    pre-commit install

Server-side
================================================================================

Additionally, a Github Actions workflow is setup to run on the server-side following every push. In addition to performing all of the hooks executed by pre-commit, this workflow also runs all unit tests using ``tox`` as well as building and deploying the documentation for viewing at https://sinaninel.com/casq/.

It is highly recommended to use a tool such as `act <https://github.com/nektos/act>`_ to test your GitHub Actions workflow locally. Using ``tox`` locally may also help in identifying any potential server-side issues related to testing or building documentation.
