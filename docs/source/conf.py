# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "WarpRec"
copyright = "2025, Marco Avolio, Potito Aghilar, Sabino Roccotelli, Vito Walter Anelli, Joseph Trotta"
author = (
    "Marco Avolio, Potito Aghilar, Sabino Roccotelli, Vito Walter Anelli, Joseph Trotta"
)
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Autodoc configuration ---------------------------------------------------

autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
    "torch_cluster",
    "torch_spline_conv",
    "ray",
    "narwhals",
    "codecarbon",
    "torchmetrics",
    "polars",
    "pandas",
    "scipy",
    "numpy",
    "joblib",
    "tabulate",
    "azure",
]

# -- Napoleon configuration --------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# -- Autosummary configuration -----------------------------------------------

autosummary_generate = True

# -- MathJax configuration ---------------------------------------------------
# Enable inline $...$ math delimiters (not enabled by default in MathJax 3)

mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    },
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = project
html_static_path = ["_static"]
