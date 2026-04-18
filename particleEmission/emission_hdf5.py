#!/usr/bin/env python3
"""HDF5 product helpers for particle-emission post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - optional runtime dependency
    h5py = None


LIGHT_SPEED_CGS = 2.99792458e10
BOLTZMANN_CGS = 1.380649e-16


@dataclass
class EmissionHdf5Metadata:
    """Small metadata container for one emission HDF5 product."""

    frame_id: int = 0
    transport_model: str = "unknown"
    field_source_kind: str = "unknown"
    background_source_kind: str = "unknown"
    coordinate_unit: str = "code_length"
    specific_intensity_unit: str = "erg s^-1 cm^-2 Hz^-1 sr^-1"
    brightness_temperature_unit: str = "K"


def _require_h5py() -> None:
    """Raise a clear error when h5py is unavailable."""
    if h5py is None:
        raise RuntimeError(
            "emission_hdf5.py requires h5py to read or write emission products."
        )


def specific_intensity_to_brightness_temperature(
    specific_intensity: np.ndarray | float,
    frequency_hz: np.ndarray | float,
) -> np.ndarray:
    """Convert specific intensity to brightness temperature in the Rayleigh-Jeans limit."""
    intensity = np.asarray(specific_intensity, dtype=np.float64)
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(frequency <= 0.0) or not np.all(np.isfinite(frequency)):
        raise ValueError("frequency_hz must be positive and finite")
    return (LIGHT_SPEED_CGS * LIGHT_SPEED_CGS * intensity /
            (2.0 * BOLTZMANN_CGS * frequency * frequency))


def brightness_temperature_to_specific_intensity(
    brightness_temperature: np.ndarray | float,
    frequency_hz: np.ndarray | float,
) -> np.ndarray:
    """Convert brightness temperature to specific intensity in the Rayleigh-Jeans limit."""
    temperature = np.asarray(brightness_temperature, dtype=np.float64)
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(frequency <= 0.0) or not np.all(np.isfinite(frequency)):
        raise ValueError("frequency_hz must be positive and finite")
    return (2.0 * BOLTZMANN_CGS * temperature * frequency * frequency /
            (LIGHT_SPEED_CGS * LIGHT_SPEED_CGS))


def _write_node(group: "h5py.Group", key: str, value: Any) -> None:
    """Recursively write nested dictionaries into an HDF5 group."""
    if isinstance(value, dict):
        child = group.create_group(key)
        for child_key, child_value in value.items():
            _write_node(child, child_key, child_value)
        return

    if isinstance(value, str):
        group.create_dataset(key, data=np.bytes_(value))
        return

    if np.isscalar(value):
        group.create_dataset(key, data=value)
        return

    array = np.asarray(value)
    group.create_dataset(key, data=array)


def write_emission_product(
    path: str | Path,
    metadata: EmissionHdf5Metadata,
    payload: dict[str, Any],
) -> None:
    """Write one nested emission product to HDF5."""
    _require_h5py()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_payload = dict(payload)
    serializable_payload.setdefault(
        "meta",
        {
            "frame_id": metadata.frame_id,
            "transport_model": metadata.transport_model,
            "field_source_kind": metadata.field_source_kind,
            "background_source_kind": metadata.background_source_kind,
            "coordinate_unit": metadata.coordinate_unit,
            "specific_intensity_unit": metadata.specific_intensity_unit,
            "brightness_temperature_unit": metadata.brightness_temperature_unit,
        },
    )

    with h5py.File(output_path, "w") as handle:
        for key, value in serializable_payload.items():
            _write_node(handle, key, value)


def _read_node(node: "h5py.Dataset | h5py.Group") -> Any:
    """Recursively read one HDF5 node into Python-native containers."""
    if isinstance(node, h5py.Dataset):
        value = node[()]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.ndarray) and value.dtype.kind == "S":
            return value.astype(str)
        return value

    result: dict[str, Any] = {}
    for key, value in node.items():
        result[key] = _read_node(value)
    return result


def read_emission_product(path: str | Path) -> dict[str, Any]:
    """Read one emission HDF5 product into nested dictionaries and arrays."""
    _require_h5py()
    with h5py.File(Path(path), "r") as handle:
        return {key: _read_node(value) for key, value in handle.items()}


def find_frequency_index(
    frequency_hz: np.ndarray,
    target_frequency_hz: float,
    *,
    relative_tolerance: float = 1.0e-6,
) -> int:
    """Return the nearest frequency index with an optional tolerance check."""
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1 or frequency.size == 0:
        raise ValueError("frequency_hz must be a non-empty 1D array")
    if target_frequency_hz <= 0.0 or not np.isfinite(target_frequency_hz):
        raise ValueError("target_frequency_hz must be positive and finite")

    index = int(np.argmin(np.abs(frequency - target_frequency_hz)))
    reference = frequency[index]
    if abs(reference - target_frequency_hz) > relative_tolerance * target_frequency_hz:
        raise ValueError(
            f"no stored frequency matches {target_frequency_hz:.6e} Hz within tolerance"
        )
    return index


def extract_frequency_slice(
    image_cube: np.ndarray,
    frequency_hz: np.ndarray,
    target_frequency_hz: float,
) -> np.ndarray:
    """Extract one 2D image plane from a `(nfreq, ny, nx)` cube."""
    cube = np.asarray(image_cube)
    if cube.ndim != 3:
        raise ValueError("image_cube must have shape (nfreq, ny, nx)")
    index = find_frequency_index(np.asarray(frequency_hz), target_frequency_hz)
    return np.asarray(cube[index, :, :], dtype=np.float64)


def gaussian_beam_weights(
    shape: tuple[int, int],
    center_xy: tuple[float, float],
    fwhm_pixels: float,
) -> np.ndarray:
    """Return normalized Gaussian beam weights over one image plane."""
    if len(shape) != 2:
        raise ValueError("shape must be (ny, nx)")
    if fwhm_pixels <= 0.0 or not np.isfinite(fwhm_pixels):
        raise ValueError("fwhm_pixels must be positive and finite")

    ny, nx = shape
    center_x, center_y = center_xy
    sigma = fwhm_pixels / np.sqrt(8.0 * np.log(2.0))
    y, x = np.indices((ny, nx), dtype=np.float64)
    radius2 = (x - center_x) ** 2 + (y - center_y) ** 2
    weights = np.exp(-0.5 * radius2 / (sigma * sigma))
    weight_sum = weights.sum()
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        raise ValueError("beam weights are numerically invalid")
    return weights / weight_sum


def circular_roi_mask(
    shape: tuple[int, int],
    center_xy: tuple[float, float],
    radius_pixels: float,
) -> np.ndarray:
    """Return a boolean circular ROI mask."""
    if radius_pixels <= 0.0 or not np.isfinite(radius_pixels):
        raise ValueError("radius_pixels must be positive and finite")
    ny, nx = shape
    center_x, center_y = center_xy
    y, x = np.indices((ny, nx), dtype=np.float64)
    radius2 = (x - center_x) ** 2 + (y - center_y) ** 2
    return radius2 <= radius_pixels * radius_pixels


def integrate_roi_spectrum(
    image_cube: np.ndarray,
    roi_weights: np.ndarray,
) -> np.ndarray:
    """Integrate one `(nfreq, ny, nx)` cube over a weighted ROI."""
    cube = np.asarray(image_cube, dtype=np.float64)
    weights = np.asarray(roi_weights, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError("image_cube must have shape (nfreq, ny, nx)")
    if weights.shape != cube.shape[1:]:
        raise ValueError("roi_weights must match the spatial image shape")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("roi_weights must be finite and non-negative")
    return np.tensordot(cube, weights, axes=([1, 2], [0, 1]))


def _gaussian_kernel_1d(fwhm_pixels: float, truncate_sigma: float = 4.0) -> np.ndarray:
    """Build one normalized 1D Gaussian kernel."""
    if fwhm_pixels <= 0.0 or not np.isfinite(fwhm_pixels):
        raise ValueError("fwhm_pixels must be positive and finite")
    sigma = fwhm_pixels / np.sqrt(8.0 * np.log(2.0))
    radius = int(np.ceil(truncate_sigma * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * offsets * offsets / (sigma * sigma))
    return kernel / kernel.sum()


def convolve_image_with_gaussian_beam(
    image_plane: np.ndarray,
    fwhm_pixels: float,
) -> np.ndarray:
    """Apply a separable Gaussian beam convolution to one 2D image plane."""
    plane = np.asarray(image_plane, dtype=np.float64)
    if plane.ndim != 2:
        raise ValueError("image_plane must be 2D")
    kernel = _gaussian_kernel_1d(fwhm_pixels)

    padded_y = np.pad(plane, ((kernel.size // 2, kernel.size // 2), (0, 0)), mode="edge")
    smoothed_y = np.empty_like(plane)
    for ix in range(plane.shape[1]):
        smoothed_y[:, ix] = np.convolve(padded_y[:, ix], kernel, mode="valid")

    padded_x = np.pad(smoothed_y, ((0, 0), (kernel.size // 2, kernel.size // 2)), mode="edge")
    smoothed = np.empty_like(plane)
    for iy in range(plane.shape[0]):
        smoothed[iy, :] = np.convolve(padded_x[iy, :], kernel, mode="valid")

    return smoothed
