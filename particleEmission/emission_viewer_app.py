#!/usr/bin/env python3
"""Streamlit viewer for emission HDF5 products."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import numpy as np
import streamlit as st

from particleEmission.emission_viewer_core import (
    EmissionProductCatalog,
    apply_gaussian_beam_to_plane,
    discover_hdf5_products,
    effective_beam_fwhm_pixels,
    flatten_metrics,
    format_frequency_label,
    guess_comparison_path,
    integrate_cube_over_roi,
    is_component_like_path,
    load_emission_product_catalog,
    matching_leaf_paths,
    preferred_cube_path,
    preferred_map_path,
    preferred_spectrum_path,
    roi_weights_for_plane,
    save_array_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_ROOT = ROOT / "particleEmission" / "output" / "viewer_exports"
DEFAULT_SEARCH_ROOTS = [
    ROOT / "particleEmission" / "output",
    ROOT / "emissionValidation",
]


def _search_roots_from_environment() -> list[Path]:
    """Return search roots from the environment or the defaults."""
    env_value = os.environ.get("EMISSION_VIEWER_SEARCH_ROOTS", "").strip()
    if not env_value:
        return DEFAULT_SEARCH_ROOTS
    return [Path(item).expanduser() for item in env_value.split(os.pathsep) if item.strip()]


def _default_product_from_environment() -> str:
    """Return the optional default product path from the environment."""
    return os.environ.get("EMISSION_VIEWER_DEFAULT_PRODUCT", "").strip()


@st.cache_data(show_spinner=False)
def _cached_catalog(path: str) -> EmissionProductCatalog:
    """Load and cache one product catalog."""
    return load_emission_product_catalog(path)


def _figure_to_png_bytes(figure: plt.Figure) -> bytes:
    """Serialize one Matplotlib figure to PNG bytes."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def _spectrum_to_csv_bytes(
    frequency_hz: np.ndarray,
    series: list[tuple[str, np.ndarray]],
) -> bytes:
    """Serialize one multi-series spectrum table to CSV bytes."""
    header = ["frequency_hz"] + [name for name, _ in series]
    matrix = np.column_stack([frequency_hz] + [np.asarray(values, dtype=np.float64) for _, values in series])
    buffer = io.StringIO()
    buffer.write(",".join(header) + "\n")
    np.savetxt(buffer, matrix, delimiter=",")
    return buffer.getvalue().encode("utf-8")


def _export_directory(catalog: EmissionProductCatalog) -> Path:
    """Return the default export directory for the current product."""
    return DEFAULT_EXPORT_ROOT / catalog.path.stem


def _infer_display_defaults(catalog: EmissionProductCatalog) -> tuple[str, str | None]:
    """Pick default cube and comparison paths."""
    primary = preferred_cube_path(catalog)
    comparison = guess_comparison_path(primary, sorted(catalog.cubes))
    return primary, comparison


def _format_scalar(value: Any) -> str:
    """Format one scalar or short vector for display."""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    array = np.asarray(value)
    if array.ndim == 0:
        return _format_scalar(array.item())
    if array.size == 1:
        return _format_scalar(array.reshape(-1)[0].item())
    return np.array2string(array, precision=4, separator=", ")


def _pick_color_limits(primary: np.ndarray, comparison: np.ndarray | None, logarithmic: bool) -> tuple[float, float]:
    """Pick a shared color scale."""
    arrays = [np.asarray(primary, dtype=np.float64)]
    if comparison is not None:
        arrays.append(np.asarray(comparison, dtype=np.float64))
    finite_chunks = [array[np.isfinite(array)] for array in arrays if np.any(np.isfinite(array))]
    if not finite_chunks:
        return 0.0, 1.0
    finite = np.concatenate(finite_chunks)
    if finite.size == 0:
        return 0.0, 1.0
    if logarithmic:
        positive = finite[finite > 0.0]
        if positive.size == 0:
            return 1.0e-12, 1.0
        return float(np.percentile(positive, 2.0)), float(np.percentile(positive, 99.5))
    return float(np.percentile(finite, 2.0)), float(np.percentile(finite, 99.5))


def _render_overlay(
    axis: Any,
    shape: tuple[int, int],
    *,
    roi_mode: str,
    center_x: float,
    center_y: float,
    radius_pixels: float,
    beam_fwhm_pixels: float,
) -> None:
    """Draw the current ROI overlay on one axes."""
    axis.scatter([center_x], [center_y], marker="+", color="white", s=140, linewidths=1.2)
    if roi_mode == "circular_aperture":
        axis.add_patch(
            Circle(
                (center_x, center_y),
                radius_pixels,
                facecolor="none",
                edgecolor="white",
                linewidth=1.2,
                linestyle="--",
            )
        )
    elif roi_mode == "gaussian_beam":
        axis.add_patch(
            Circle(
                (center_x, center_y),
                0.5 * beam_fwhm_pixels,
                facecolor="none",
                edgecolor="white",
                linewidth=1.2,
                linestyle="--",
            )
        )
    axis.set_xlim(-0.5, shape[1] - 0.5)
    axis.set_ylim(-0.5, shape[0] - 0.5)


def _plot_image_panels(
    primary_plane: np.ndarray,
    primary_label: str,
    *,
    unit: str,
    comparison_plane: np.ndarray | None,
    comparison_label: str | None,
    show_fractional_difference: bool,
    logarithmic: bool,
    roi_mode: str,
    center_x: float,
    center_y: float,
    radius_pixels: float,
    beam_fwhm_pixels: float,
) -> plt.Figure:
    """Build the image comparison figure."""
    panel_count = 1 if comparison_plane is None else 2 + int(show_fractional_difference)
    figure, axes = plt.subplots(1, panel_count, figsize=(5.4 * panel_count, 4.8))
    axes_array = np.atleast_1d(axes)
    vmin, vmax = _pick_color_limits(primary_plane, comparison_plane, logarithmic)
    if logarithmic:
        image_norm = LogNorm(vmin=max(vmin, 1.0e-16), vmax=max(vmax, max(vmin, 1.0e-16) * 1.01))
    else:
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        image_norm = None

    for axis, plane, title in [(axes_array[0], primary_plane, primary_label)]:
        artist = axis.imshow(
            plane if not logarithmic else np.maximum(plane, max(vmin, 1.0e-16)),
            origin="lower",
            cmap="inferno",
            norm=image_norm,
            vmin=None if logarithmic else vmin,
            vmax=None if logarithmic else vmax,
        )
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        _render_overlay(
            axis,
            plane.shape,
            roi_mode=roi_mode,
            center_x=center_x,
            center_y=center_y,
            radius_pixels=radius_pixels,
            beam_fwhm_pixels=beam_fwhm_pixels,
        )
        figure.colorbar(artist, ax=axis, shrink=0.82, label=unit or None)

    if comparison_plane is not None:
        artist = axes_array[1].imshow(
            comparison_plane if not logarithmic else np.maximum(comparison_plane, max(vmin, 1.0e-16)),
            origin="lower",
            cmap="inferno",
            norm=image_norm,
            vmin=None if logarithmic else vmin,
            vmax=None if logarithmic else vmax,
        )
        axes_array[1].set_title(comparison_label or "Comparison")
        axes_array[1].set_xticks([])
        axes_array[1].set_yticks([])
        _render_overlay(
            axes_array[1],
            comparison_plane.shape,
            roi_mode=roi_mode,
            center_x=center_x,
            center_y=center_y,
            radius_pixels=radius_pixels,
            beam_fwhm_pixels=beam_fwhm_pixels,
        )
        figure.colorbar(artist, ax=axes_array[1], shrink=0.82, label=unit or None)

        if show_fractional_difference:
            difference = np.full_like(primary_plane, np.nan, dtype=np.float64)
            valid = np.isfinite(primary_plane) & (np.abs(primary_plane) > 0.0) & np.isfinite(comparison_plane)
            difference[valid] = (comparison_plane[valid] - primary_plane[valid]) / primary_plane[valid]
            limit = float(np.nanpercentile(np.abs(difference[valid]), 98.0)) if np.any(valid) else 1.0
            limit = max(limit, 1.0e-6)
            artist = axes_array[2].imshow(
                difference,
                origin="lower",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            axes_array[2].set_title("Fractional Difference")
            axes_array[2].set_xticks([])
            axes_array[2].set_yticks([])
            _render_overlay(
                axes_array[2],
                difference.shape,
                roi_mode=roi_mode,
                center_x=center_x,
                center_y=center_y,
                radius_pixels=radius_pixels,
                beam_fwhm_pixels=beam_fwhm_pixels,
            )
            figure.colorbar(artist, ax=axes_array[2], shrink=0.82, label="(comp - ref) / ref")

    figure.tight_layout()
    return figure


def _plot_map_panels(
    primary_map: np.ndarray,
    primary_label: str,
    *,
    unit: str,
    comparison_map: np.ndarray | None,
    comparison_label: str | None,
    logarithmic: bool,
    roi_mode: str,
    center_x: float,
    center_y: float,
    radius_pixels: float,
    beam_fwhm_pixels: float,
) -> plt.Figure:
    """Build the static map figure."""
    return _plot_image_panels(
        primary_map,
        primary_label,
        unit=unit,
        comparison_plane=comparison_map,
        comparison_label=comparison_label,
        show_fractional_difference=comparison_map is not None,
        logarithmic=logarithmic,
        roi_mode=roi_mode,
        center_x=center_x,
        center_y=center_y,
        radius_pixels=radius_pixels,
        beam_fwhm_pixels=beam_fwhm_pixels,
    )


def _plot_spectra(
    frequency_hz: np.ndarray,
    primary_spectrum: np.ndarray,
    primary_label: str,
    *,
    unit: str,
    comparison_spectrum: np.ndarray | None,
    comparison_label: str | None,
    overlays: list[tuple[str, np.ndarray]],
) -> plt.Figure:
    """Build the spectrum figure."""
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 7.4), sharex=True)
    axes_array = np.atleast_1d(axes)
    positive = np.all(primary_spectrum[np.isfinite(primary_spectrum)] > 0.0)
    if comparison_spectrum is not None:
        positive = positive and np.all(comparison_spectrum[np.isfinite(comparison_spectrum)] > 0.0)

    if positive:
        axes_array[0].loglog(frequency_hz, primary_spectrum, linewidth=2.2, label=primary_label)
        if comparison_spectrum is not None:
            axes_array[0].loglog(frequency_hz, comparison_spectrum, linewidth=2.2, label=comparison_label)
        for label, series in overlays:
            axes_array[0].loglog(frequency_hz, series, linewidth=1.5, linestyle="--", label=label)
    else:
        axes_array[0].semilogx(frequency_hz, primary_spectrum, linewidth=2.2, label=primary_label)
        if comparison_spectrum is not None:
            axes_array[0].semilogx(frequency_hz, comparison_spectrum, linewidth=2.2, label=comparison_label)
        for label, series in overlays:
            axes_array[0].semilogx(frequency_hz, series, linewidth=1.5, linestyle="--", label=label)

    axes_array[0].set_ylabel(unit or "value")
    axes_array[0].grid(True, which="both", alpha=0.28)
    axes_array[0].legend()

    if comparison_spectrum is not None:
        ratio = np.full_like(primary_spectrum, np.nan, dtype=np.float64)
        valid = np.isfinite(primary_spectrum) & (np.abs(primary_spectrum) > 0.0) & np.isfinite(comparison_spectrum)
        ratio[valid] = comparison_spectrum[valid] / primary_spectrum[valid]
        axes_array[1].semilogx(frequency_hz, ratio, linewidth=2.0, color="tab:red")
        axes_array[1].axhline(1.0, linewidth=1.0, linestyle="--", color="black")
        axes_array[1].set_ylabel("comp / ref")
    else:
        axes_array[1].semilogx(frequency_hz, primary_spectrum, linewidth=2.0, color="tab:blue")
        axes_array[1].set_ylabel(unit or "value")
    axes_array[1].set_xlabel("Frequency [Hz]")
    axes_array[1].grid(True, which="both", alpha=0.28)
    figure.tight_layout()
    return figure


def _plane_stats(plane: np.ndarray) -> dict[str, float]:
    """Return a compact summary for one image plane."""
    array = np.asarray(plane, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "sum": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "sum": float(np.sum(finite)),
        "mean": float(np.mean(finite)),
    }


def _current_frequency_key(frequency_hz: float, metrics: dict[str, Any]) -> str | None:
    """Find the nearest morphology metric key for the selected frequency."""
    morphology = metrics.get("morphology_metrics", {})
    if not isinstance(morphology, dict):
        return None
    target_ghz = float(frequency_hz) / 1.0e9
    best_key = None
    best_error = None
    for key in morphology:
        try:
            candidate_ghz = float(str(key).replace("GHz", ""))
        except ValueError:
            continue
        error = abs(candidate_ghz - target_ghz)
        if best_error is None or error < best_error:
            best_error = error
            best_key = key
    return best_key


def _save_bytes(path: Path, payload: bytes) -> None:
    """Save one byte payload to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)


def _render_product_selector() -> Path | None:
    """Render the product selector in the sidebar."""
    st.sidebar.header("Product")
    search_roots = _search_roots_from_environment()
    discovered = discover_hdf5_products(search_roots)
    default_path = _default_product_from_environment()
    option_labels = [str(path) for path in discovered]

    manual_path = st.sidebar.text_input(
        "HDF5 path",
        value=default_path if default_path else (option_labels[0] if option_labels else ""),
    ).strip()

    if option_labels:
        selected_option = st.sidebar.selectbox(
            "Discovered",
            options=option_labels,
            index=option_labels.index(default_path) if default_path in option_labels else 0,
        )
        if st.sidebar.button("Use discovered"):
            manual_path = selected_option

    if not manual_path:
        return None
    path = Path(manual_path).expanduser()
    if not path.exists():
        st.sidebar.error("File not found")
        return None
    return path.resolve()


def main() -> None:
    """Run the Streamlit emission viewer."""
    st.set_page_config(page_title="Emission Viewer", layout="wide")
    st.title("Emission Viewer")

    product_path = _render_product_selector()
    if product_path is None:
        st.info("Select one emission product to begin.")
        return

    try:
        catalog = _cached_catalog(str(product_path))
    except Exception as exc:  # pragma: no cover - interactive path
        st.error(f"Failed to load product: {exc}")
        return

    if not catalog.cubes:
        st.error("The selected product does not contain any spectral image cubes.")
        return

    default_cube, default_comparison = _infer_display_defaults(catalog)
    cube_paths = sorted(catalog.cubes)
    primary_cube_path = st.sidebar.selectbox(
        "Image cube",
        options=cube_paths,
        index=cube_paths.index(default_cube),
    )
    comparison_options = ["(none)"] + cube_paths
    comparison_default = default_comparison if default_comparison in cube_paths else "(none)"
    comparison_cube_path = st.sidebar.selectbox(
        "Comparison cube",
        options=comparison_options,
        index=comparison_options.index(comparison_default),
    )
    comparison_cube_path = None if comparison_cube_path == "(none)" else comparison_cube_path

    frequency_labels = [format_frequency_label(value) for value in catalog.frequency_hz]
    frequency_index = st.sidebar.select_slider(
        "Frequency",
        options=list(range(catalog.frequency_hz.size)),
        value=int(np.argmin(np.abs(catalog.frequency_hz - 3.0e9))),
        format_func=lambda index: frequency_labels[index],
    )

    primary_cube = np.asarray(catalog.cubes[primary_cube_path].values, dtype=np.float64)
    image_shape = primary_cube.shape[1:]
    center_x = st.sidebar.slider("Center x", 0.0, float(image_shape[1] - 1), 0.5 * (image_shape[1] - 1), 0.5)
    center_y = st.sidebar.slider("Center y", 0.0, float(image_shape[0] - 1), 0.5 * (image_shape[0] - 1), 0.5)

    roi_mode = st.sidebar.selectbox(
        "ROI mode",
        options=["full_image", "circular_aperture", "gaussian_beam"],
        format_func=lambda value: {
            "full_image": "Full image",
            "circular_aperture": "Circular aperture",
            "gaussian_beam": "Gaussian beam",
        }[value],
    )
    radius_pixels = st.sidebar.slider("Aperture radius [px]", 0.5, float(max(image_shape)), 6.0, 0.5)
    base_beam_fwhm_pixels = st.sidebar.slider("Beam FWHM @ ref [px]", 0.5, float(max(image_shape)), 3.0, 0.5)
    scaled_beam = st.sidebar.checkbox("Beam scales with frequency", value=False)
    reference_frequency_hz = st.sidebar.number_input(
        "Reference frequency [GHz]",
        min_value=0.03,
        max_value=20.0,
        value=3.0,
        step=0.1,
    ) * 1.0e9

    display_beam = st.sidebar.checkbox("Display beam smoothing", value=False)
    display_log = st.sidebar.checkbox("Log image scale", value=True)

    tabs = st.tabs(["Images", "Spectra", "Maps", "Metrics", "Metadata"])

    current_frequency_hz = float(catalog.frequency_hz[frequency_index])
    current_beam_fwhm = effective_beam_fwhm_pixels(
        base_beam_fwhm_pixels,
        current_frequency_hz,
        reference_frequency_hz,
        scaled_beam,
    )

    with tabs[0]:
        primary_plane_raw = primary_cube[frequency_index]
        primary_plane, _ = apply_gaussian_beam_to_plane(
            primary_plane_raw,
            enabled=display_beam,
            base_fwhm_pixels=base_beam_fwhm_pixels,
            frequency_hz=current_frequency_hz,
            reference_frequency_hz=reference_frequency_hz,
            scaled_with_frequency=scaled_beam,
        )

        comparison_plane = None
        comparison_label = None
        if comparison_cube_path is not None:
            comparison_cube = np.asarray(catalog.cubes[comparison_cube_path].values, dtype=np.float64)
            comparison_plane, _ = apply_gaussian_beam_to_plane(
                comparison_cube[frequency_index],
                enabled=display_beam,
                base_fwhm_pixels=base_beam_fwhm_pixels,
                frequency_hz=current_frequency_hz,
                reference_frequency_hz=reference_frequency_hz,
                scaled_with_frequency=scaled_beam,
            )
            comparison_label = catalog.cubes[comparison_cube_path].label

        image_figure = _plot_image_panels(
            primary_plane,
            catalog.cubes[primary_cube_path].label,
            unit=catalog.cubes[primary_cube_path].unit,
            comparison_plane=comparison_plane,
            comparison_label=comparison_label,
            show_fractional_difference=comparison_plane is not None,
            logarithmic=display_log,
            roi_mode=roi_mode,
            center_x=center_x,
            center_y=center_y,
            radius_pixels=radius_pixels,
            beam_fwhm_pixels=current_beam_fwhm,
        )
        st.pyplot(image_figure, width="stretch")

        stat_columns = st.columns(4)
        for column, (label, value) in zip(stat_columns, _plane_stats(primary_plane).items()):
            column.metric(label, f"{value:.6g}")
        st.caption(
            f"{format_frequency_label(current_frequency_hz)} | beam FWHM = {current_beam_fwhm:.3f} px"
        )

        image_png = _figure_to_png_bytes(
            _plot_image_panels(
                primary_plane,
                catalog.cubes[primary_cube_path].label,
                unit=catalog.cubes[primary_cube_path].unit,
                comparison_plane=comparison_plane,
                comparison_label=comparison_label,
                show_fractional_difference=comparison_plane is not None,
                logarithmic=display_log,
                roi_mode=roi_mode,
                center_x=center_x,
                center_y=center_y,
                radius_pixels=radius_pixels,
                beam_fwhm_pixels=current_beam_fwhm,
            )
        )
        export_basename = (
            f"{catalog.path.stem}_{primary_cube_path.replace('/', '_')}_{format_frequency_label(current_frequency_hz).replace(' ', '_')}"
        )
        st.download_button(
            "Download image PNG",
            data=image_png,
            file_name=f"{export_basename}.png",
            mime="image/png",
        )
        if st.button("Save image to output folder"):
            save_path = _export_directory(catalog) / f"{export_basename}.png"
            _save_bytes(save_path, image_png)
            st.success(f"Saved {save_path}")

    with tabs[1]:
        spectrum_cube_path = st.selectbox(
            "Spectrum cube",
            options=cube_paths,
            index=cube_paths.index(primary_cube_path),
        )
        spectrum_comparison_path = st.selectbox(
            "Spectrum comparison",
            options=["(none)"] + cube_paths,
            index=0 if comparison_cube_path is None else (["(none)"] + cube_paths).index(comparison_cube_path),
        )
        spectrum_comparison_path = None if spectrum_comparison_path == "(none)" else spectrum_comparison_path
        stored_overlay_paths = st.multiselect(
            "Stored spectra",
            options=sorted(catalog.spectra),
            default=[
                path
                for path in [preferred_spectrum_path(catalog), guess_comparison_path(preferred_spectrum_path(catalog), sorted(catalog.spectra))]
                if path is not None and path in catalog.spectra
            ],
        )
        component_cube_options = matching_leaf_paths(
            cube_paths,
            spectrum_cube_path,
            require_component_like=True,
        )
        preferred_component_paths = [
            path
            for path in [
                f"references/thermal_only/{spectrum_cube_path.split('/')[-1]}",
                f"references/no_razin_density_proxy/{spectrum_cube_path.split('/')[-1]}",
                f"references/optically_thin_proxy/{spectrum_cube_path.split('/')[-1]}",
                f"diagnostics/absorption_impact/{spectrum_cube_path.split('/')[-1]}",
                f"diagnostics/razin_impact/{spectrum_cube_path.split('/')[-1]}",
                f"references/background_only/{spectrum_cube_path.split('/')[-1]}",
            ]
            if path in component_cube_options
        ]
        component_cube_paths = st.multiselect(
            "Reference / diagnostic cubes",
            options=component_cube_options,
            default=preferred_component_paths,
        )

        primary_spectrum, effective_fwhm = integrate_cube_over_roi(
            catalog.cubes[spectrum_cube_path].values,
            catalog.frequency_hz,
            roi_mode=roi_mode,
            center_xy=(center_x, center_y),
            radius_pixels=radius_pixels,
            base_beam_fwhm_pixels=base_beam_fwhm_pixels,
            scaled_with_frequency=scaled_beam,
            reference_frequency_hz=reference_frequency_hz,
        )
        comparison_spectrum = None
        comparison_spectrum_label = None
        if spectrum_comparison_path is not None:
            comparison_spectrum, _ = integrate_cube_over_roi(
                catalog.cubes[spectrum_comparison_path].values,
                catalog.frequency_hz,
                roi_mode=roi_mode,
                center_xy=(center_x, center_y),
                radius_pixels=radius_pixels,
                base_beam_fwhm_pixels=base_beam_fwhm_pixels,
                scaled_with_frequency=scaled_beam,
                reference_frequency_hz=reference_frequency_hz,
            )
            comparison_spectrum_label = catalog.cubes[spectrum_comparison_path].label

        overlay_series = [
            (catalog.spectra[path].label, np.asarray(catalog.spectra[path].values, dtype=np.float64))
            for path in stored_overlay_paths
        ]
        for path in component_cube_paths:
            component_spectrum, _ = integrate_cube_over_roi(
                catalog.cubes[path].values,
                catalog.frequency_hz,
                roi_mode=roi_mode,
                center_xy=(center_x, center_y),
                radius_pixels=radius_pixels,
                base_beam_fwhm_pixels=base_beam_fwhm_pixels,
                scaled_with_frequency=scaled_beam,
                reference_frequency_hz=reference_frequency_hz,
            )
            overlay_series.append((catalog.cubes[path].label, component_spectrum))
        spectrum_figure = _plot_spectra(
            catalog.frequency_hz,
            primary_spectrum,
            catalog.cubes[spectrum_cube_path].label,
            unit=catalog.cubes[spectrum_cube_path].unit,
            comparison_spectrum=comparison_spectrum,
            comparison_label=comparison_spectrum_label,
            overlays=overlay_series,
        )
        st.pyplot(spectrum_figure, width="stretch")
        component_note = catalog.text.get("component_info/model_note")
        if component_note and (component_cube_paths or any(is_component_like_path(path) for path in stored_overlay_paths)):
            st.caption(component_note)

        roi_weights = roi_weights_for_plane(
            image_shape,
            roi_mode=roi_mode,
            center_xy=(center_x, center_y),
            radius_pixels=radius_pixels,
            beam_fwhm_pixels=current_beam_fwhm,
        )
        roi_columns = st.columns(4)
        roi_columns[0].metric("ROI weight sum", f"{float(np.sum(roi_weights)):.6g}")
        roi_columns[1].metric("ROI max weight", f"{float(np.max(roi_weights)):.6g}")
        roi_columns[2].metric("Current FWHM [px]", f"{current_beam_fwhm:.4g}")
        roi_columns[3].metric("ROI mode", {"full_image": "full", "circular_aperture": "aperture", "gaussian_beam": "beam"}[roi_mode])

        if scaled_beam and roi_mode == "gaussian_beam":
            beam_figure, beam_axis = plt.subplots(figsize=(6.8, 3.8))
            beam_axis.semilogx(catalog.frequency_hz, effective_fwhm, linewidth=2.0)
            beam_axis.set_xlabel("Frequency [Hz]")
            beam_axis.set_ylabel("FWHM [px]")
            beam_axis.grid(True, which="both", alpha=0.3)
            beam_axis.set_title("Frequency-Scaled Beam")
            beam_figure.tight_layout()
            st.pyplot(beam_figure, width="content")
            plt.close(beam_figure)

        spectrum_series = [(catalog.cubes[spectrum_cube_path].path, primary_spectrum)]
        if comparison_spectrum is not None and comparison_spectrum_label is not None:
            spectrum_series.append((catalog.cubes[spectrum_comparison_path].path, comparison_spectrum))
        for label, values in overlay_series:
            spectrum_series.append((label, values))
        spectrum_png = _figure_to_png_bytes(
            _plot_spectra(
                catalog.frequency_hz,
                primary_spectrum,
                catalog.cubes[spectrum_cube_path].label,
                unit=catalog.cubes[spectrum_cube_path].unit,
                comparison_spectrum=comparison_spectrum,
                comparison_label=comparison_spectrum_label,
                overlays=overlay_series,
            )
        )
        spectrum_csv = _spectrum_to_csv_bytes(catalog.frequency_hz, spectrum_series)
        spectrum_basename = (
            f"{catalog.path.stem}_{spectrum_cube_path.replace('/', '_')}_{roi_mode}_cx{center_x:.1f}_cy{center_y:.1f}"
        )
        download_columns = st.columns(2)
        download_columns[0].download_button(
            "Download spectrum PNG",
            data=spectrum_png,
            file_name=f"{spectrum_basename}.png",
            mime="image/png",
        )
        download_columns[1].download_button(
            "Download spectrum CSV",
            data=spectrum_csv,
            file_name=f"{spectrum_basename}.csv",
            mime="text/csv",
        )
        save_columns = st.columns(2)
        if save_columns[0].button("Save spectrum PNG"):
            save_path = _export_directory(catalog) / f"{spectrum_basename}.png"
            _save_bytes(save_path, spectrum_png)
            st.success(f"Saved {save_path}")
        if save_columns[1].button("Save spectrum CSV"):
            save_path = _export_directory(catalog) / f"{spectrum_basename}.csv"
            _save_bytes(save_path, spectrum_csv)
            st.success(f"Saved {save_path}")

    with tabs[2]:
        if catalog.maps:
            map_paths = sorted(catalog.maps)
            default_map = preferred_map_path(catalog)
            default_map_comparison = guess_comparison_path(default_map, map_paths)
            primary_map_path = st.selectbox(
                "Static map",
                options=map_paths,
                index=map_paths.index(default_map),
            )
            map_comparison_path = st.selectbox(
                "Map comparison",
                options=["(none)"] + map_paths,
                index=0 if default_map_comparison is None else (["(none)"] + map_paths).index(default_map_comparison),
            )
            map_comparison_path = None if map_comparison_path == "(none)" else map_comparison_path
            map_log = st.checkbox("Log map scale", value=False)

            primary_map = np.asarray(catalog.maps[primary_map_path].values, dtype=np.float64)
            comparison_map = (
                np.asarray(catalog.maps[map_comparison_path].values, dtype=np.float64)
                if map_comparison_path is not None
                else None
            )
            map_figure = _plot_map_panels(
                primary_map,
                catalog.maps[primary_map_path].label,
                unit=catalog.maps[primary_map_path].unit,
                comparison_map=comparison_map,
                comparison_label=catalog.maps[map_comparison_path].label if map_comparison_path else None,
                logarithmic=map_log,
                roi_mode=roi_mode,
                center_x=center_x,
                center_y=center_y,
                radius_pixels=radius_pixels,
                beam_fwhm_pixels=current_beam_fwhm,
            )
            st.pyplot(map_figure, width="stretch")

            map_png = _figure_to_png_bytes(
                _plot_map_panels(
                    primary_map,
                    catalog.maps[primary_map_path].label,
                    unit=catalog.maps[primary_map_path].unit,
                    comparison_map=comparison_map,
                    comparison_label=catalog.maps[map_comparison_path].label if map_comparison_path else None,
                    logarithmic=map_log,
                    roi_mode=roi_mode,
                    center_x=center_x,
                    center_y=center_y,
                    radius_pixels=radius_pixels,
                    beam_fwhm_pixels=current_beam_fwhm,
                )
            )
            map_csv_buffer = io.StringIO()
            np.savetxt(map_csv_buffer, primary_map, delimiter=",")
            map_basename = f"{catalog.path.stem}_{primary_map_path.replace('/', '_')}"
            download_map_columns = st.columns(2)
            download_map_columns[0].download_button(
                "Download map PNG",
                data=map_png,
                file_name=f"{map_basename}.png",
                mime="image/png",
            )
            download_map_columns[1].download_button(
                "Download map CSV",
                data=map_csv_buffer.getvalue().encode("utf-8"),
                file_name=f"{map_basename}.csv",
                mime="text/csv",
            )
            save_map_columns = st.columns(2)
            if save_map_columns[0].button("Save map PNG"):
                save_path = _export_directory(catalog) / f"{map_basename}.png"
                _save_bytes(save_path, map_png)
                st.success(f"Saved {save_path}")
            if save_map_columns[1].button("Save map CSV"):
                save_path = _export_directory(catalog) / f"{map_basename}.csv"
                save_array_csv(save_path, primary_map)
                st.success(f"Saved {save_path}")
        else:
            st.info("No 2D static maps were found in this product.")

    with tabs[3]:
        summary_rows = []
        for key, value in sorted(catalog.scalars.items()):
            summary_rows.append({"name": key, "value": _format_scalar(value)})
        if summary_rows:
            st.subheader("Scalars")
            st.dataframe(summary_rows, width="stretch", hide_index=True)

        if catalog.metrics:
            metric_rows = flatten_metrics(catalog.metrics)
            if metric_rows:
                st.subheader("Metrics")
                st.dataframe(metric_rows, width="stretch", hide_index=True)

            current_key = _current_frequency_key(current_frequency_hz, catalog.metrics)
            if current_key is not None:
                st.subheader(f"Morphology @ {current_key}")
                st.json(catalog.metrics["morphology_metrics"][current_key], expanded=True)
        else:
            st.info("No metrics group was found in this product.")

    with tabs[4]:
        meta_payload = {
            "path": str(catalog.path),
            "frequency_count": int(catalog.frequency_hz.size),
            "cube_count": len(catalog.cubes),
            "map_count": len(catalog.maps),
            "spectrum_count": len(catalog.spectra),
            "meta": {
                key: value
                for key, value in sorted(catalog.text.items())
                if key.startswith("meta/")
            },
        }
        st.json(meta_payload, expanded=True)

        if "config" in catalog.raw:
            st.subheader("Config")
            st.json(catalog.raw["config"], expanded=False)
        if "calibration" in catalog.raw:
            st.subheader("Calibration")
            st.json(catalog.raw["calibration"], expanded=False)
        if "component_info" in catalog.raw:
            st.subheader("Component Info")
            st.json(catalog.raw["component_info"], expanded=False)
        if catalog.vectors:
            st.subheader("Vectors")
            vector_rows = [
                {
                    "name": path,
                    "shape": list(entry.values.shape),
                    "unit": entry.unit,
                }
                for path, entry in sorted(catalog.vectors.items())
            ]
            st.dataframe(vector_rows, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
