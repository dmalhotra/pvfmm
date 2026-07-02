# Sphinx configuration for the PVFMM documentation (https://readthedocs.org).
#
# Build locally with:
#   pip install -r docs/requirements.txt
#   doxygen must be on PATH (Flatiron cluster: module load doxygen)
#   sphinx-build -W --keep-going -b html docs docs/_build/html

import os
import shutil
import subprocess

DOCS_DIR = os.path.abspath(os.path.dirname(__file__))

# -- Doxygen XML for the C/C++ API reference (consumed by breathe) -----------
# Run unconditionally at config time: parsing the handful of public headers
# takes ~1 s and always-fresh XML avoids stale-cache surprises.

def run_doxygen():
    if shutil.which("doxygen") is None:
        raise RuntimeError(
            "doxygen not found on PATH; it is required to build the API reference.\n"
            "  Flatiron cluster:  module load doxygen\n"
            "  Debian/Ubuntu:     sudo apt-get install doxygen\n"
            "  Read the Docs:     installed via build.apt_packages in .readthedocs.yaml"
        )
    result = subprocess.run(["doxygen", "Doxyfile"], cwd=DOCS_DIR,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"doxygen failed (exit {result.returncode}):\n{result.stderr}")

run_doxygen()

# -- Project information ------------------------------------------------------

project = "PVFMM"
author = "Dhairya Malhotra"
copyright = "2010-2026, Dhairya Malhotra"
# Single-sourced from version.txt (see scripts/release.sh), like the
# autotools and CMake builds.
with open(os.path.join(DOCS_DIR, "..", "version.txt")) as f:
    release = f.read().strip()
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_parser",
    "breathe",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]
myst_heading_anchors = 3

# Doxygen output of the root ./Doxyfile (html/, latex/) also lands in docs/;
# keep Sphinx away from it.
exclude_patterns = ["_build", "_doxygen", "html", "latex", "Thumbs.db", ".DS_Store"]

# -- Breathe ------------------------------------------------------------------

breathe_projects = {"pvfmm": os.path.join(DOCS_DIR, "_doxygen", "xml")}
breathe_default_project = "pvfmm"
# pvfmm.h is parsed by doxygen as C++ (extern "C"); render it in the C domain.
breathe_domain_by_extension = {"h": "c", "hpp": "cpp"}

# -- HTML output ---------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}

# Don't require the example/binding code blocks to be valid highlight targets.
highlight_language = "none"

# sphinx-copybutton: strip shell prompts when copying.
copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True
