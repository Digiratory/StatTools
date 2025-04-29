# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import StatTools

release: str = StatTools.__version__
# for example take major/minor
version: str = ".".join(release.split(".")[:2])
# debug that building expected version
print(f"Building Documentation for FluctuationAnalysisTools: {StatTools.__version__}")

project = "Fluctuation Analysis Tools"
copyright = "2025, Aleksandr Sinitca, Alexandr Kuzmenko, Asya Lyanova"
author = "Aleksandr Sinitca, Alexandr Kuzmenko, Asya Lyanova"
release = "1.10.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.ifconfig",
    "sphinx_design",
    "sphinx_copybutton",
    #   'sphinx_tags',
    "sphinx.ext.napoleon",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

add_module_names = False


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "Pillow": ("https://pillow.readthedocs.io/en/stable/", None),
    "cycler": ("https://matplotlib.org/cycler/", None),
    "dateutil": ("https://dateutil.readthedocs.io/en/stable/", None),
    "ipykernel": ("https://ipykernel.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pytest": ("https://pytest.org/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "tornado": ("https://www.tornadoweb.org/en/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "meson-python": ("https://mesonbuild.com/meson-python/", None),
    "pip": ("https://pip.pypa.io/en/stable/", None),
}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

# Don't include link to doc source files
html_show_sourcelink = False

# Copies only relevant code, not the '>>>' prompt
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# Use typographic quote characters.
smartquotes = False
