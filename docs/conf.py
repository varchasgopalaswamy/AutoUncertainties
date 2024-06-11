from __future__ import annotations

import datetime
import importlib.metadata

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import sphinx_rtd_theme  # noqa: F401

# -- Path setup --------------------------------------------------------------


sys.path.insert(0, os.path.abspath("../../"))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "auto_uncertainties"
version = importlib.metadata.version(project)
release = version
this_year = datetime.date.today()
copyright = f"2021-{this_year:%Y}, Varchas Gopalaswamy"
author = "Varchas Gopalaswamy"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]


templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "setup.rst",
    "versioneer.rst",
    "tests*",
]


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

autodoc_default_options = {"class-doc-from": "__init__"}

add_function_parentheses = False
# -- Options for extensions ----------------------------------------------------
# napoleon
typehints_fully_qualified = False
typehints_defaults = "comma"
typehints_use_rtype = True
typehints_document_rtype = True
always_document_param_types = True
typehints_use_signature = True
typehints_use_signature_return = True
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_title = f"{project} v{version} Manual"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


default_role = "py:obj"


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "dask": ("https://docs.dask.org/en/latest", None),
    "sparse": ("https://sparse.pydata.org/en/latest/", None),
    "pint": ("https://pint.readthedocs.io/en/stable", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
