# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('../../.'))

project = 'Promptzl'
copyright = '2024, Philipp Koch'
author = 'Philipp Koch'
release = '0.2.0'
version = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinxcontrib.jquery',
    'sphinx_datatables',
    ]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']


# Autodoc settings
autoclass_content = 'init'

# Mock imports for modules that are not available during the build process
autodoc_mock_imports = [
    'numpy', 'pandas', 'polars', 'torch', 'datasets', 'transformers', 'tqdm'
]

html_title = "Pr🥨mptzl"

# Data-Tables config
datatables_version = "1.13.4"

datatables_class = "sphinx-datatable"

# any custom options to pass to the DataTables constructor. Note that any
# options you set are used for all DataTables.
datatables_options = {}