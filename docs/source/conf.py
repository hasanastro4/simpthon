# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import importlib
import inspect
import re
import subprocess
import os
import sys
sys.path.insert(0, os.path.abspath('..\simpthon'))


# -- Project information -----------------------------------------------------

project = 'simpthon'
copyright = '2022, Hasanuddin'
author = 'Hasanuddin'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [ 'sphinx.ext.mathjax', 'sphinx.ext.napoleon', 'sphinx.ext.autodoc'
]
# from jobovy/galpy via
# https://github.com/jobovy/galpy/blob/main/doc/source/conf.py
# from disnake via:
# https://twitter.com/readthedocs/status/1541830907082022913?s=20&t=eJ293FfjILT7sIxEyz834w
_simpthon_module_path = os.path.dirname(importlib.util.find_spec("simpthon").origin)
github_repo = "https://github.com/hasanastro4/simpthon"
# Current git reference. Uses branch/tag name if found, otherwise uses commit hash
def git(*args):
    return subprocess.check_output(["git", *args]).strip().decode()
git_ref= None
try:
    git_ref= git("name-rev", "--name-only", "--no-undefined", "HEAD")
    git_ref= re.sub(r"^(remotes/[^/]+|tags)/", "", git_ref)
except Exception:
    pass
# (if no name found or relative ref, use commit hash instead)
if not git_ref or re.search(r"[\^~]", git_ref):
    try:
        git_ref = git("rev-parse", "HEAD")
    except Exception:
        git_ref = "main"
def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    try:
        obj= sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj= getattr(obj, part)
        obj= inspect.unwrap(obj)

        if isinstance(obj, property):
            obj= inspect.unwrap(obj.fget)

        path= os.path.relpath(inspect.getsourcefile(obj),start=_simpthon_module_path)
        src, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None

    path = f"{path}#L{lineno}-L{lineno + len(src) - 1}"
    return f"{github_repo}/blob/{git_ref}/simpthon/{path}"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']