import os
import sys

# Make it so modules are visible to autodoc
sys.path.insert(0, os.path.abspath(".."))

project = "SNNLib"
copyright = "2020, Matthew Dutson"
author = "Matthew Dutson"

html_theme = "sphinx_rtd_theme"
extensions = ["sphinx.ext.autodoc"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
