.. _documentation:

################################################################################
Documentation
################################################################################

The documentation is built using `Sphinx <https://www.sphinx-doc.org/en/master/>`_ plus some additional extensions including `pydata-sphinx-theme <https://pydata-sphinx-theme.readthedocs.io/en/stable/>`_ for the theme and `sphinx-autoapi <https://sphinx-autoapi.readthedocs.io/en/latest/>`_ for automated generation of API documentation based on docstrings.

Documentation can be built using the following command:

.. code-block::

    poetry run make clean html

The ``clean`` argument is not required since existing built content may be over-written instead of first being deleted, but it is highly recommended for consistent results (though it is much slower).

On pushing any changes to Github, documentation is automatically built server-side and deployed to the ``gh-pages`` branch. Documentation is then available for viewing at the following url: https://sinaninel.com/casq/.
