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
copyright = f"2021-{this_year:%Y} Varchas Gopalaswamy"
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
    "sphinx_design",
    "sphinx.ext.graphviz",
    "autoapi.extension",
]

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "setup.rst",
    "versioneer.rst",
    "tests*",
    "_autoapi_templates",
]

autoapi_ignore = ["tests*", "*_version.py"]

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = "sphinx"

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_last_updated_fmt = "%B %d, %Y at %I:%M %p"

html_title = f"{project} v{version} Manual"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

default_role = "py:obj"

# Avoids cluttered function signature docs
maximum_signature_line_length = 70

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "dask": ("https://docs.dask.org/en/latest", None),
    "sparse": ("https://sparse.pydata.org/en/latest/", None),
    "pint": ("https://pint.readthedocs.io/en/stable", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}


# ------------------------- Options for extensions -------------------------

# copybutton
copybutton_prompt_text = ">>> "

# type hints
typehints_defaults = "comma"
autodoc_typehints = "description"
typehints_use_rtype = True
typehints_document_rtype = True
always_document_param_types = True

# AutoAPI
autoapi_template_dir = "_autoapi_templates"
autoapi_dirs = ["../../auto_uncertainties"]
suppress_warnings = ["autoapi.python_import_resolution"]
autoapi_add_toctree_entry = False
autoapi_python_class_content = "class"
autoapi_root = "api"
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
    "show-inheritance-diagram",
]


# Custom jinja environment additions for AutoAPI
def autoapi_prepare_jinja_env(jinja_env):
    from docs.source.jinja_formatters import (
        format_alias,
        format_function_defaults,
        format_typevar,
    )

    # Add custom jinja filters and functions
    jinja_env.filters["format_alias"] = format_alias
    jinja_env.filters["format_typevar"] = format_typevar
    jinja_env.globals["format_function_defaults"] = format_function_defaults
