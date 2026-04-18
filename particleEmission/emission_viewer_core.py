#!/usr/bin/env python3
"""Core helpers for browsing emission HDF5 products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from particleEmission.emission_hdf5 import (
    circular_roi_mask,
    convolve_image_with_gaussian_beam,
    gaussian_beam_weights,
    read_emission_product,
)


@dataclass(frozen=True)
class EmissionDataEntry:
    """One discovered numeric entry inside an emission product."""

    path: str
    label: str
    unit: str
    values: np.ndarray


@dataclass(frozen=True)
class EmissionProductCatalog:
    """Flattened catalog of one emission product."""

    path: Path
    raw: dict[str, Any]
    frequency_hz: np.ndarray
    cubes: dict[str, EmissionDataEntry]
    maps: dict[str, EmissionDataEntry]
    spectra: dict[str, EmissionDataEntry]
    vectors: dict[str, EmissionDataEntry]
    scalars: dict[str, float | int]
    text: dict[str, str]
    metrics: dict[str, Any]


def discover_hdf5_products(search_roots: list[str | Path]) -> list[Path]:
    """Discover HDF5 emission products under one or more roots."""
    discovered: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        root_path = Path(root).expanduser()
        if root_path.is_file() and root_path.suffix.lower() in {".h5", ".hdf5"}:
            resolved = root_path.resolve()
            if resolved not in seen:
                discovered.append(resolved)
                seen.add(resolved)
            continue
        if not root_path.exists():
            continue
        for path in sorted(root_path.rglob("*.h5")):
            resolved = path.resolve()
            if resolved not in seen:
                discovered.append(resolved)
                seen.add(resolved)
        for path in sorted(root_path.rglob("*.hdf5")):
            resolved = path.resolve()
            if resolved not in seen:
                discovered.append(resolved)
                seen.add(resolved)
    return discovered


def _is_string_like(value: Any) -> bool:
    """Return whether one node is text-like."""
    if isinstance(value, (str, bytes)):
        return True
    if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S"}:
        return True
    return False


def _walk_nodes(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten a nested product dictionary into `(path, value)` pairs."""
    if isinstance(node, dict):
        pairs: list[tuple[str, Any]] = []
        for key in sorted(node):
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            pairs.extend(_walk_nodes(node[key], child_prefix))
        return pairs
    return [(prefix, node)]


def _path_label(path: str) -> str:
    """Return a short human-readable label."""
    return path.replace("/", " / ")


def _infer_unit(path: str) -> str:
    """Infer a unit string from the dataset path."""
    lower = path.lower()
    unit_map = [
        ("brightness_temperature", "K"),
        ("specific", "erg s^-1 cm^-2 Hz^-1 sr^-1"),
        ("flux_sfu", "sfu"),
        ("density_cm3", "cm^-3"),
        ("gauss", "G"),
        ("energy_ev", "eV"),
        ("energy_mev", "MeV"),
        ("freq_hz", "Hz"),
        ("frequencies_ghz", "GHz"),
        ("arcsec", "arcsec"),
        ("r_squared", ""),
        ("weight", "particle weight"),
        ("accepted", ""),
        ("index", ""),
    ]
    for token, unit in unit_map:
        if token in lower:
            return unit
    return ""


def _coerce_text_value(value: Any) -> str:
    """Convert one text-like node to a Python string."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.ndim == 0:
            return str(array.item())
        return np.array2string(array.astype(str), separator=", ")
    return str(value)


def _coerce_numeric_array(value: Any) -> np.ndarray:
    """Convert one numeric node to a NumPy array."""
    if np.isscalar(value):
        return np.asarray(value)
    return np.asarray(value)


def _looks_like_frequency_axis(path: str, values: np.ndarray) -> bool:
    """Return whether one numeric vector likely stores frequencies."""
    lower = path.lower()
    if "freq" in lower or "frequency" in lower:
        return values.ndim == 1 and values.size > 1 and np.all(np.isfinite(values)) and np.all(values > 0.0)
    return False


def _find_frequency_hz(raw: dict[str, Any]) -> np.ndarray:
    """Locate the primary frequency axis."""
    flattened = _walk_nodes(raw)
    best_match: np.ndarray | None = None
    for path, value in flattened:
        if _is_string_like(value):
            continue
        array = _coerce_numeric_array(value)
        if _looks_like_frequency_axis(path, array) and path.endswith("freq_hz"):
            return np.asarray(array, dtype=np.float64)
        if best_match is None and _looks_like_frequency_axis(path, array):
            best_match = np.asarray(array, dtype=np.float64)
    if best_match is None:
        raise RuntimeError("could not find a frequency axis inside the selected emission product")
    return best_match


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return `numerator / denominator` with finite handling."""
    ratio = np.full_like(np.asarray(numerator, dtype=np.float64), np.nan, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    valid = np.isfinite(denominator_array) & (np.abs(denominator_array) > 0.0)
    ratio[valid] = np.asarray(numerator, dtype=np.float64)[valid] / denominator_array[valid]
    return ratio


def _add_derived_cubes(
    cubes: dict[str, EmissionDataEntry],
    spectra: dict[str, EmissionDataEntry],
) -> tuple[dict[str, EmissionDataEntry], dict[str, EmissionDataEntry]]:
    """Add derived polarization-degree cubes and spectra where possible."""
    derived_cubes = dict(cubes)
    derived_spectra = dict(spectra)

    suffix_pairs = [
        ("I_flux_sfu", "V_flux_sfu", "circular_degree"),
        ("I_specific", "V_specific", "circular_degree"),
        ("I_brightness_temperature", "V_brightness_temperature", "circular_degree"),
        ("I_flux_sfu", "linear_flux_sfu", "linear_degree"),
        ("I_specific", "linear_specific", "linear_degree"),
        ("I_brightness_temperature", "linear_brightness_temperature", "linear_degree"),
    ]

    def add_from_collection(
        collection: dict[str, EmissionDataEntry],
        derived: dict[str, EmissionDataEntry],
    ) -> None:
        for intensity_suffix, polarized_suffix, derived_name in suffix_pairs:
            for path, entry in list(collection.items()):
                if not path.endswith(intensity_suffix):
                    continue
                polarized_path = path[: -len(intensity_suffix)] + polarized_suffix
                if polarized_path not in collection:
                    continue
                ratio_values = _safe_ratio(collection[polarized_path].values, entry.values)
                derived_path = path[: -len(intensity_suffix)] + derived_name
                derived[derived_path] = EmissionDataEntry(
                    path=derived_path,
                    label=_path_label(derived_path),
                    unit="fraction",
                    values=ratio_values,
                )

    add_from_collection(cubes, derived_cubes)
    add_from_collection(spectra, derived_spectra)
    return derived_cubes, derived_spectra


def load_emission_product_catalog(path: str | Path) -> EmissionProductCatalog:
    """Read one product and organize all datasets for browsing."""
    product_path = Path(path).expanduser().resolve()
    raw = read_emission_product(product_path)
    frequency_hz = _find_frequency_hz(raw)
    frequency_count = int(frequency_hz.size)

    cubes: dict[str, EmissionDataEntry] = {}
    maps: dict[str, EmissionDataEntry] = {}
    spectra: dict[str, EmissionDataEntry] = {}
    vectors: dict[str, EmissionDataEntry] = {}
    scalars: dict[str, float | int] = {}
    text: dict[str, str] = {}

    for node_path, value in _walk_nodes(raw):
        if _is_string_like(value):
            text[node_path] = _coerce_text_value(value)
            continue

        array = _coerce_numeric_array(value)
        if array.dtype.kind not in "iufb":
            continue

        entry = EmissionDataEntry(
            path=node_path,
            label=_path_label(node_path),
            unit=_infer_unit(node_path),
            values=np.asarray(array),
        )
        if _looks_like_frequency_axis(node_path, array):
            vectors[node_path] = entry
            continue
        if array.ndim == 3 and array.shape[0] == frequency_count:
            cubes[node_path] = entry
            continue
        if array.ndim == 2:
            maps[node_path] = entry
            continue
        if array.ndim == 1 and array.size == frequency_count:
            spectra[node_path] = entry
            continue
        if array.ndim == 0:
            item = array.item()
            scalars[node_path] = item
            continue
        if array.ndim == 1 and array.size == 1:
            item = array.reshape(-1)[0].item()
            scalars[node_path] = item
            continue
        vectors[node_path] = entry

    cubes, spectra = _add_derived_cubes(cubes, spectra)

    metrics = raw.get("metrics", {})
    return EmissionProductCatalog(
        path=product_path,
        raw=raw,
        frequency_hz=np.asarray(frequency_hz, dtype=np.float64),
        cubes=cubes,
        maps=maps,
        spectra=spectra,
        vectors=vectors,
        scalars=scalars,
        text=text,
        metrics=metrics if isinstance(metrics, dict) else {},
    )


def preferred_cube_path(catalog: EmissionProductCatalog) -> str:
    """Return a good default image cube path."""
    candidates = [
        "reconstruction/I_flux_sfu",
        "truth/I_flux_sfu",
        "image/full/I_flux_sfu",
        "image/beam/I_flux_sfu",
    ]
    for candidate in candidates:
        if candidate in catalog.cubes:
            return candidate
    return sorted(catalog.cubes)[0]


def preferred_map_path(catalog: EmissionProductCatalog) -> str:
    """Return a good default 2D map path."""
    candidates = [
        "reconstruction/nonthermal_density_cm3",
        "truth/nonthermal_density_cm3",
        "cell/nonthermal_density_cm3",
        "reconstruction/power_law_index",
        "truth/power_law_index",
        "cell/power_law_index",
    ]
    for candidate in candidates:
        if candidate in catalog.maps:
            return candidate
    return sorted(catalog.maps)[0]


def preferred_spectrum_path(catalog: EmissionProductCatalog) -> str:
    """Return a good default stored spectrum path."""
    candidates = [
        "reconstruction/integrated_spectrum_sfu",
        "truth/integrated_spectrum_sfu",
        "roi/full_image/I_flux_sfu",
        "roi/center_beam/I_flux_sfu",
    ]
    for candidate in candidates:
        if candidate in catalog.spectra:
            return candidate
    return sorted(catalog.spectra)[0]


def guess_comparison_path(selected_path: str, available_paths: list[str]) -> str | None:
    """Guess a useful comparison dataset for the current selection."""
    parts = selected_path.split("/")
    available = set(available_paths)
    if parts[0] == "reconstruction":
        candidate = "/".join(["truth"] + parts[1:])
        if candidate in available:
            return candidate
    if parts[0] == "truth":
        candidate = "/".join(["reconstruction"] + parts[1:])
        if candidate in available:
            return candidate
    if len(parts) >= 2 and parts[0] == "image" and parts[1] == "full":
        candidate = "/".join(["image", "beam"] + parts[2:])
        if candidate in available:
            return candidate
    if len(parts) >= 2 and parts[0] == "image" and parts[1] == "beam":
        candidate = "/".join(["image", "full"] + parts[2:])
        if candidate in available:
            return candidate

    leaf = parts[-1]
    matching = [path for path in available_paths if path != selected_path and path.endswith("/" + leaf)]
    if matching:
        return sorted(matching)[0]
    return None


def is_component_like_path(path: str) -> bool:
    """Return whether one dataset path looks like a component/reference curve."""
    lowered = path.lower()
    tokens = ("references/", "diagnostics/", "components/")
    return any(token in lowered for token in tokens)


def matching_leaf_paths(
    available_paths: list[str],
    selected_path: str,
    *,
    require_component_like: bool = False,
) -> list[str]:
    """Return paths that share the same leaf name as the selected path."""
    selected_leaf = selected_path.split("/")[-1]
    matches = []
    for path in available_paths:
        if path == selected_path:
            continue
        if path.split("/")[-1] != selected_leaf:
            continue
        if require_component_like and not is_component_like_path(path):
            continue
        matches.append(path)
    return sorted(matches)


def format_frequency_label(frequency_hz: float) -> str:
    """Format one frequency in GHz or MHz."""
    value_hz = float(frequency_hz)
    if value_hz < 1.0e9:
        return f"{value_hz / 1.0e6:.0f} MHz"
    return f"{value_hz / 1.0e9:.2f} GHz"


def effective_beam_fwhm_pixels(
    base_fwhm_pixels: float,
    frequency_hz: float,
    reference_frequency_hz: float,
    scaled_with_frequency: bool,
) -> float:
    """Return the effective beam FWHM in pixels."""
    if not scaled_with_frequency:
        return float(base_fwhm_pixels)
    if frequency_hz <= 0.0 or reference_frequency_hz <= 0.0:
        raise ValueError("frequency_hz and reference_frequency_hz must be positive")
    return float(base_fwhm_pixels) * float(reference_frequency_hz) / float(frequency_hz)


def apply_gaussian_beam_to_plane(
    plane: np.ndarray,
    *,
    enabled: bool,
    base_fwhm_pixels: float,
    frequency_hz: float,
    reference_frequency_hz: float,
    scaled_with_frequency: bool,
) -> tuple[np.ndarray, float]:
    """Optionally apply a Gaussian beam to one image plane."""
    effective_fwhm = effective_beam_fwhm_pixels(
        base_fwhm_pixels,
        frequency_hz,
        reference_frequency_hz,
        scaled_with_frequency,
    )
    if not enabled:
        return np.asarray(plane, dtype=np.float64), effective_fwhm
    if effective_fwhm <= 0.0:
        raise ValueError("effective beam FWHM must be positive")
    return convolve_image_with_gaussian_beam(plane, effective_fwhm), effective_fwhm


def roi_weights_for_plane(
    shape: tuple[int, int],
    *,
    roi_mode: str,
    center_xy: tuple[float, float],
    radius_pixels: float,
    beam_fwhm_pixels: float,
) -> np.ndarray:
    """Build ROI weights for one image plane."""
    if roi_mode == "full_image":
        return np.ones(shape, dtype=np.float64)
    if roi_mode == "circular_aperture":
        return circular_roi_mask(shape, center_xy, radius_pixels).astype(np.float64)
    if roi_mode == "gaussian_beam":
        return gaussian_beam_weights(shape, center_xy, beam_fwhm_pixels)
    raise ValueError(f"unsupported roi_mode: {roi_mode}")


def integrate_cube_over_roi(
    cube: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    roi_mode: str,
    center_xy: tuple[float, float],
    radius_pixels: float,
    base_beam_fwhm_pixels: float,
    scaled_with_frequency: bool,
    reference_frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a full cube over a fixed or frequency-scaled ROI."""
    array = np.asarray(cube, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("cube must have shape (nfreq, ny, nx)")
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1 or frequency.size != array.shape[0]:
        raise ValueError("frequency_hz must match the first cube dimension")

    spectrum = np.zeros(frequency.size, dtype=np.float64)
    effective_fwhm_pixels = np.full(frequency.size, np.nan, dtype=np.float64)
    for index, frequency_value in enumerate(frequency):
        effective_fwhm = effective_beam_fwhm_pixels(
            base_beam_fwhm_pixels,
            float(frequency_value),
            float(reference_frequency_hz),
            scaled_with_frequency,
        )
        effective_fwhm_pixels[index] = effective_fwhm
        weights = roi_weights_for_plane(
            array.shape[1:],
            roi_mode=roi_mode,
            center_xy=center_xy,
            radius_pixels=radius_pixels,
            beam_fwhm_pixels=effective_fwhm,
        )
        spectrum[index] = float(np.sum(array[index] * weights))
    return spectrum, effective_fwhm_pixels


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> list[dict[str, Any]]:
    """Flatten nested metric dictionaries into table rows."""
    rows: list[dict[str, Any]] = []
    for key in sorted(metrics):
        value = metrics[key]
        metric_path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(flatten_metrics(value, metric_path))
            continue
        rows.append({"metric": metric_path, "value": value})
    return rows


def save_array_csv(path: Path, array: np.ndarray) -> None:
    """Save one array to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, np.asarray(array, dtype=np.float64), delimiter=",")
