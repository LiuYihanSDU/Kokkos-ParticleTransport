#!/usr/bin/env python3
"""Background plasma map readers used by emission synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - optional runtime dependency
    h5py = None

from .compact_field_io import coarse_average_2d


@dataclass
class EmissionBackgroundMaps:
    """Thermal/background maps on a 2D field-aligned image grid."""

    x_edges: np.ndarray
    y_edges: np.ndarray
    thermal_density_cm3: np.ndarray
    temperature_mk: np.ndarray
    source_path: str
    source_kind: str = "mhd_converted"


def _require_h5py() -> None:
    """Raise a clear error when h5py is unavailable."""
    if h5py is None:
        raise RuntimeError("background_io.py requires h5py to read background maps.")


def _read_required_dataset(handle: "h5py.File", name: str) -> np.ndarray:
    """Read one required dataset as a double-precision array."""
    if name not in handle:
        raise RuntimeError(f"background map file is missing dataset '{name}'")
    return np.asarray(handle[name][()], dtype=np.float64)


def read_emission_background_maps(path: str | Path) -> EmissionBackgroundMaps:
    """Read an Athena/MHD converted background-map HDF5 file."""
    _require_h5py()
    file_path = Path(path)
    with h5py.File(file_path, "r") as handle:
        x_edges = _read_required_dataset(handle, "x_edges")
        y_edges = _read_required_dataset(handle, "y_edges")
        thermal_density = _read_required_dataset(handle, "thermal_density_cm3")
        temperature = _read_required_dataset(handle, "temperature_mk")
        source_kind = str(handle.attrs.get("schema", "mhd_converted"))

    expected_shape = (y_edges.size - 1, x_edges.size - 1)
    if thermal_density.shape != expected_shape or temperature.shape != expected_shape:
        raise RuntimeError(
            "background map shape must match x_edges/y_edges cell dimensions"
        )
    if np.any(~np.isfinite(thermal_density)) or np.any(thermal_density < 0.0):
        raise RuntimeError("thermal_density_cm3 must be finite and non-negative")
    if np.any(~np.isfinite(temperature)) or np.any(temperature < 0.0):
        raise RuntimeError("temperature_mk must be finite and non-negative")

    return EmissionBackgroundMaps(
        x_edges=x_edges,
        y_edges=y_edges,
        thermal_density_cm3=thermal_density,
        temperature_mk=temperature,
        source_path=str(file_path),
        source_kind=source_kind,
    )


def coarse_background_maps(
    background: EmissionBackgroundMaps,
    image_nx: int,
    image_ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average thermal density and temperature maps onto an emission image grid."""
    density_map = coarse_average_2d(
        background.thermal_density_cm3,
        background.x_edges,
        background.y_edges,
        image_nx,
        image_ny,
    )
    temperature_map = coarse_average_2d(
        background.temperature_mk,
        background.x_edges,
        background.y_edges,
        image_nx,
        image_ny,
    )
    return density_map, temperature_map
