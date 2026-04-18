#!/usr/bin/env python3
"""Launcher for the local emission viewer app."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    """Parse launcher arguments."""
    parser = argparse.ArgumentParser(description="Launch the local emission viewer app.")
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Local Streamlit port.",
    )
    parser.add_argument(
        "--default-product",
        type=str,
        default="",
        help="Optional default HDF5 product path.",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Additional product search roots.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the Streamlit app with the local virtual environment."""
    arguments = parse_arguments()
    repository_root = Path(__file__).resolve().parents[1]
    python_path = repository_root / ".venv" / "bin" / "python"
    if not python_path.exists():
        python_path = Path(sys.executable)

    search_roots = [
        repository_root / "particleEmission" / "output",
        repository_root / "emissionValidation",
    ]
    search_roots.extend(Path(item).expanduser() for item in arguments.search_root)

    environment = dict(os.environ)
    environment["EMISSION_VIEWER_SEARCH_ROOTS"] = os.pathsep.join(str(path) for path in search_roots)
    if arguments.default_product:
        environment["EMISSION_VIEWER_DEFAULT_PRODUCT"] = str(Path(arguments.default_product).expanduser())

    command = [
        str(python_path),
        "-m",
        "streamlit",
        "run",
        str(repository_root / "particleEmission" / "emission_viewer_app.py"),
        "--server.port",
        str(arguments.port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    completed = subprocess.run(command, cwd=repository_root, env=environment, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
