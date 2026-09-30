# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values see
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# Let autodoc import the package from the source tree, whether or not it is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# -- Project information -----------------------------------------------------

project = "s2conv"
copyright = "2026, Jason D. McEwen and contributors"
author = "Jason D. McEwen and contributors"

try:
    release = get_version("s2conv")
except PackageNotFoundError:
    # The package is not installed, so its version is not known.
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

# Document members in the order in which they appear in the source.
autodoc_member_order = "bysource"

# Docstrings use the Google style, with reStructuredText maths (:math:).
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

language = "en"

# Patterns, relative to this directory, of files and directories to ignore.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_align": "left",
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/astro-informatics/s2conv",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}
