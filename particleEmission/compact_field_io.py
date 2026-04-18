#!/usr/bin/env python3
"""Compact field readers and coarse-grid helpers for particle emission."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CompactFieldSnapshot:
    """Compact 2D field snapshot used by the current example workflow."""

    nx: int
    ny: int
    x_edges: np.ndarray
    y_edges: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray

    @property
    def x_centers(self) -> np.ndarray:
        """Return cell-centered x coordinates."""
        return 0.5 * (self.x_edges[:-1] + self.x_edges[1:])

    @property
    def y_centers(self) -> np.ndarray:
        """Return cell-centered y coordinates."""
        return 0.5 * (self.y_edges[:-1] + self.y_edges[1:])

    @property
    def magnetic_field_magnitude(self) -> np.ndarray:
        """Return the magnetic-field magnitude on the native grid."""
        return np.sqrt(self.bx * self.bx + self.by * self.by + self.bz * self.bz)


def read_compact_field_snapshot(path: str | Path) -> CompactFieldSnapshot:
    """Read the compact 2D field format used by the example reconnection output."""
    file_path = Path(path)
    with file_path.open("rb") as handle:
        nx = struct.unpack("<i", handle.read(4))[0]
        ny = struct.unpack("<i", handle.read(4))[0]
        x_edges = np.frombuffer(handle.read(8 * (nx + 1)), dtype="<f8").copy()
        y_edges = np.frombuffer(handle.read(8 * (ny + 1)), dtype="<f8").copy()

        cell_count = nx * ny

        def read_component() -> np.ndarray:
            return np.frombuffer(handle.read(8 * cell_count), dtype="<f8").copy().reshape(
                ny, nx
            )

        bx = read_component()
        by = read_component()
        bz = read_component()
        vx = read_component()
        vy = read_component()
        vz = read_component()

    return CompactFieldSnapshot(nx, ny, x_edges, y_edges, bx, by, bz, vx, vy, vz)


def coarse_average_2d(
    values: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    image_nx: int,
    image_ny: int,
) -> np.ndarray:
    """Average one native-grid scalar field onto a coarse image grid."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("values must be a 2D array")

    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    x_index = np.clip(
        np.floor((x_centers - x_edges[0]) / (x_edges[-1] - x_edges[0]) * image_nx).astype(int),
        0,
        image_nx - 1,
    )
    y_index = np.clip(
        np.floor((y_centers - y_edges[0]) / (y_edges[-1] - y_edges[0]) * image_ny).astype(int),
        0,
        image_ny - 1,
    )

    coarse = np.zeros((image_ny, image_nx), dtype=np.float64)
    counts = np.zeros((image_ny, image_nx), dtype=np.float64)

    for native_y, coarse_y in enumerate(y_index):
        row_values = array[native_y, :]
        row_sum = np.bincount(x_index, weights=row_values, minlength=image_nx)
        row_count = np.bincount(x_index, minlength=image_nx).astype(np.float64)
        coarse[coarse_y, :] += row_sum
        counts[coarse_y, :] += row_count

    result = np.zeros_like(coarse)
    nonzero = counts > 0.0
    result[nonzero] = coarse[nonzero] / counts[nonzero]
    return result


def coarse_pixel_centers(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    image_nx: int,
    image_ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return coarse-grid cell centers over the native field domain."""
    x_edges_coarse = np.linspace(float(x_edges[0]), float(x_edges[-1]), image_nx + 1)
    y_edges_coarse = np.linspace(float(y_edges[0]), float(y_edges[-1]), image_ny + 1)
    x_centers = 0.5 * (x_edges_coarse[:-1] + x_edges_coarse[1:])
    y_centers = 0.5 * (y_edges_coarse[:-1] + y_edges_coarse[1:])
    return x_centers, y_centers

