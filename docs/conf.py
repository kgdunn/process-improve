# Configuration file for the Sphinx documentation builder.

import os
import sys
from pathlib import Path

import tomllib

# Add project root to path so autodoc can find the package
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

_pyproject = tomllib.loads(Path(__file__).resolve().parent.parent.joinpath("pyproject.toml").read_text())
release = _pyproject["project"]["version"]

project = "process-improve"
copyright = "2010-2026, Kevin Dunn"  # noqa: A001
author = "Kevin Dunn"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Napoleon settings (NumPy docstrings) ------------------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autodoc settings -------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# -- Intersphinx mapping ----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- Nitpick ignore ---------------------------------------------------------
# Suppress cross-reference warnings from inherited sklearn docstrings that
# reference labels/terms defined only inside sklearn's own documentation.
nitpick_ignore = [
    ("std:term", "meta-estimator"),
    ("std:label", "metadata_routing"),
]

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/kgdunn/process_improve",
    "show_toc_level": 2,
    "navigation_with_keys": True,
}
