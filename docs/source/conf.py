# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Add src to path so Sphinx can import the package
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Forecast Evaluation"
copyright = "2026, Bank of England"
author = "James Hurley, Paul Labonne, Harry Li"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Pull from docstrings
    "sphinx.ext.napoleon",  # Support NumPy/Google style
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.autosummary",  # Generate summary tables
    "sphinx.ext.intersphinx",  # Link to other docs
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autosummary settings
autosummary_generate = True
autosummary_generate_overwrite = True

autodoc_default_options = {
    "members": True,
    "imported-members": True,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = ["api/generated/**"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = []
