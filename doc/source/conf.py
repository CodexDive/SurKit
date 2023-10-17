# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
os.environ["READTHEDOCS"] = "True"
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../surkit'))
from __info__ import __shortversion__, __version__


project = 'SurKit'

copyright = '2023, CodexDive'
author = 'Kejie Fu'

version = __shortversion__
release = __version__
# language = None

todo_include_todos = False

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


pygments_style = "sphinx"
html_static_path = []
# html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_path = []
# autodoc_typehints = 'description'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.ifconfig',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
]