"""The notebook has to run in a browser, which constrains what epkit may import.

Under Pyodide there is no openep-py, no VTK and no PyVista, and never will be.
These tests fail loudly if an innocuous-looking import creeps back in and breaks
the published page — a failure that would otherwise only show up in the browser
console after a deploy.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_runtime_stack_imports():
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    import plotly  # noqa: F401
    import scipy  # noqa: F401


def test_importing_epkit_does_not_pull_in_the_offline_stack():
    code = (
        "import sys; import epkit; "
        "forbidden = [m for m in ('openep', 'pyvista', 'vtk') if m in sys.modules]; "
        "print(forbidden)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        f"epkit imported {result.stdout.strip()}, which cannot exist in the browser"
    )


def test_notebook_is_a_valid_marimo_app():
    """Importing the notebook defines the app without executing any cell."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import marimo, ep_primer; assert isinstance(ep_primer.app, marimo.App)",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_notebook_declares_only_browser_installable_dependencies():
    """The PEP 723 header drives what micropip installs in the WASM build."""
    header = (ROOT / "ep_primer.py").read_text().split('"""')[0]
    assert "# /// script" in header
    for package in ("marimo", "numpy", "scipy", "matplotlib", "plotly"):
        assert f'"{package}"' in header
    for absent in ("openep", "pyvista", "vtk", "trame"):
        assert f'"{absent}"' not in header
