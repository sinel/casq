# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.insert(0, package_path)
os.environ["PYTHONPATH"] = ":".join((package_path, os.environ.get("PYTHONPATH", "")))

# -- Project information -----------------------------------------------------

project = "casq"
copyright = "2023, Sinan Inel"
author = "Sinan Inel"

# The full version, including alpha/beta/rc tags
release = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoapi.extension",
    "jupyter_sphinx",
    "sphinx_copybutton",
    "sphinx_favicon",
]
autoapi_dirs = ["../../src"]
autodoc_typehints = "description"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/sinel/casq",  # required
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
    "logo": {
        "alt_text": "casq",
        "image_light": "_static/logo_title_light_padding.png",
        "image_dark": "_static/logo_title_dark_padding.png",
    },
    "primary_sidebar_end": ["indices.html"],
    "navbar_align": "content",
    "header_links_before_dropdown": 6,
    "footer_start": ["copyright"],
    "footer_end": [],
}
html_show_sourcelink = False
html_title = "Casq v0.1.0 documentation"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
favicons = ["images/favicon.png"]
