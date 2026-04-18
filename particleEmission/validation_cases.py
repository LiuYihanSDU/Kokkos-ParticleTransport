#!/usr/bin/env python3
"""Synthetic validation cases for the particle-emission prototype."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from .emission_hdf5 import write_emission_product, EmissionHdf5Metadata
from .emission_references import evaluate_reference_bundle
from .microwave_backend import MicrowaveSourceParameters, MicrowaveSpectrumBackend
from .particle_io import fit_power_law_index_with_quality


SFU_CGS = 1.0e-19
ARCSEC_TO_RADIAN = np.pi / (180.0 * 3600.0)
ASTRONOMICAL_UNIT_CM = 1.495978707e13


@dataclass(frozen=True)
class ValidationCase:
    """One synthetic validation case definition."""

    name: str
    image_nx: int
    image_ny: int
    pixel_size_arcsec: float
    los_depth_arcsec: float
    viewing_angle_deg: float
    thermal_density_cm3: float
    temperature_mk: float
    minimum_energy_mev: float
    maximum_energy_mev: float
    macro_particle_electron_count: float
    morphology_frequencies_ghz: tuple[float, ...]
    spectrum_frequencies_ghz: tuple[float, ...]
    power_law_index_default: float
    power_law_index_bounds: tuple[float, float]
    minimum_power_law_fit_bins: int
    minimum_power_law_fit_r_squared: float


@dataclass(frozen=True)
class EvaluatedCubeBundle:
    """One full cube plus component-aware reference cubes."""

    intensity_cube: np.ndarray
    circular_cube: np.ndarray
    intensity_spectrum: np.ndarray
    circular_spectrum: np.ndarray
    references: dict[str, dict[str, np.ndarray]]
    diagnostics: dict[str, dict[str, np.ndarray]]


REFERENCE_DESCRIPTIONS = {
    "thermal_only": "Low nonthermal-density thermal/free-free reference.",
    "no_razin_density_proxy": (
        "Low thermal-density proxy that weakens Razin suppression and free-free opacity together."
    ),
    "optically_thin_proxy": (
        "Thin-slab proxy rescaled back to the original LOS depth to diagnose self-absorption."
    ),
    "background_only": "Zero external-background reference used by the current prototype.",
}

DIAGNOSTIC_DESCRIPTIONS = {
    "absorption_impact": "Optically thin proxy minus full spectrum.",
    "razin_impact": "Low-density no-Razin proxy minus full spectrum.",
}


def arcsec_length_cm() -> float:
    """Return the physical length of one arcsec at 1 AU."""
    return ASTRONOMICAL_UNIT_CM * ARCSEC_TO_RADIAN


def pixel_solid_angle_sr(pixel_size_arcsec: float) -> float:
    """Return one pixel solid angle in steradian."""
    return pixel_size_arcsec * pixel_size_arcsec * ARCSEC_TO_RADIAN * ARCSEC_TO_RADIAN


def voxel_volume_cm3(pixel_size_arcsec: float, los_depth_arcsec: float) -> float:
    """Return one synthetic pixel volume in cm^3."""
    return (pixel_size_arcsec * arcsec_length_cm()) ** 2 * (
        los_depth_arcsec * arcsec_length_cm()
    )


def flux_sfu_to_specific_intensity(flux_sfu: np.ndarray, pixel_size_arcsec: float) -> np.ndarray:
    """Convert flux density per pixel in sfu to specific intensity."""
    return np.asarray(flux_sfu, dtype=np.float64) * SFU_CGS / pixel_solid_angle_sr(pixel_size_arcsec)


def specific_intensity_to_brightness_temperature(
    specific_intensity: np.ndarray, frequency_hz: np.ndarray
) -> np.ndarray:
    """Convert specific intensity to brightness temperature."""
    c = 2.99792458e10
    k_b = 1.380649e-16
    intensity = np.asarray(specific_intensity, dtype=np.float64)
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    reshape = (frequency.size,) + (1,) * (intensity.ndim - 1)
    expanded_frequency = frequency.reshape(reshape)
    return c * c * intensity / (2.0 * k_b * expanded_frequency * expanded_frequency)


def power_law_bin_probabilities(
    energy_bin_edges_mev: np.ndarray,
    power_law_index: float,
) -> np.ndarray:
    """Return normalized bin probabilities for `dN/dE ~ E^{-delta}`."""
    edges = np.asarray(energy_bin_edges_mev, dtype=np.float64)
    delta = float(power_law_index)
    if np.isclose(delta, 1.0):
        weights = np.log(edges[1:] / edges[:-1])
    else:
        exponent = 1.0 - delta
        weights = (edges[1:] ** exponent - edges[:-1] ** exponent) / exponent
        weights = np.abs(weights)
    total = np.sum(weights)
    if total <= 0.0 or not np.isfinite(total):
        raise RuntimeError("invalid power-law bin probabilities")
    return weights / total


def simulate_reconstruction(
    true_nonthermal_density_cm3: np.ndarray,
    true_power_law_index: np.ndarray,
    case: ValidationCase,
    energy_bin_edges_mev: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, bool]:
    """Build reconstructed `n_nth` and `delta` maps from synthetic macro-particle statistics."""
    volume = voxel_volume_cm3(case.pixel_size_arcsec, case.los_depth_arcsec)
    expected_weight = true_nonthermal_density_cm3 * volume / case.macro_particle_electron_count
    deposited_weight = rng.poisson(np.clip(expected_weight, 0.0, None)).astype(np.float64)
    reconstructed_nonthermal_density = deposited_weight * case.macro_particle_electron_count / volume

    histogram_shape = true_nonthermal_density_cm3.shape + (energy_bin_edges_mev.size - 1,)
    histograms = np.zeros(histogram_shape, dtype=np.float64)
    for iy in range(true_nonthermal_density_cm3.shape[0]):
        for ix in range(true_nonthermal_density_cm3.shape[1]):
            count = int(deposited_weight[iy, ix])
            if count <= 0:
                continue
            probabilities = power_law_bin_probabilities(
                energy_bin_edges_mev, true_power_law_index[iy, ix]
            )
            histograms[iy, ix, :] = rng.multinomial(count, probabilities)

    global_delta, global_r_squared, global_accepted = fit_power_law_index_with_quality(
        histograms.sum(axis=(0, 1)),
        1.0e6 * energy_bin_edges_mev,
        case.power_law_index_default,
        minimum_populated_bins=case.minimum_power_law_fit_bins,
        index_bounds=case.power_law_index_bounds,
        minimum_r_squared=case.minimum_power_law_fit_r_squared,
    )

    reconstructed_delta = np.full(true_power_law_index.shape, global_delta, dtype=np.float64)
    fit_r_squared = np.zeros(true_power_law_index.shape, dtype=np.float64)
    fit_accepted = np.zeros(true_power_law_index.shape, dtype=bool)
    for iy in range(true_power_law_index.shape[0]):
        for ix in range(true_power_law_index.shape[1]):
            delta, quality, accepted = fit_power_law_index_with_quality(
                histograms[iy, ix, :],
                1.0e6 * energy_bin_edges_mev,
                global_delta,
                minimum_populated_bins=case.minimum_power_law_fit_bins,
                index_bounds=case.power_law_index_bounds,
                minimum_r_squared=case.minimum_power_law_fit_r_squared,
            )
            reconstructed_delta[iy, ix] = delta
            fit_r_squared[iy, ix] = quality
            fit_accepted[iy, ix] = accepted

    return (
        deposited_weight,
        reconstructed_nonthermal_density,
        reconstructed_delta,
        fit_r_squared,
        fit_accepted,
        global_delta,
        global_r_squared,
        global_accepted,
    )


def build_cube_payload(
    intensity_cube: np.ndarray,
    circular_cube: np.ndarray,
    pixel_size_arcsec: float,
    frequency_hz: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return one standard Stokes-I/V cube payload."""
    intensity_specific = flux_sfu_to_specific_intensity(intensity_cube, pixel_size_arcsec)
    circular_specific = flux_sfu_to_specific_intensity(circular_cube, pixel_size_arcsec)
    return {
        "I_flux_sfu": np.asarray(intensity_cube, dtype=np.float64),
        "I_specific": intensity_specific,
        "I_brightness_temperature": specific_intensity_to_brightness_temperature(
            intensity_specific, frequency_hz
        ),
        "V_flux_sfu": np.asarray(circular_cube, dtype=np.float64),
        "V_specific": circular_specific,
        "V_brightness_temperature": specific_intensity_to_brightness_temperature(
            circular_specific, frequency_hz
        ),
        "integrated_spectrum_sfu": np.asarray(intensity_cube, dtype=np.float64).sum(axis=(1, 2)),
        "integrated_circular_spectrum_sfu": np.asarray(circular_cube, dtype=np.float64).sum(
            axis=(1, 2)
        ),
    }


def evaluate_cube(
    backend: MicrowaveSpectrumBackend,
    frequency_ghz: np.ndarray,
    nonthermal_density_cm3: np.ndarray,
    magnetic_field_gauss: np.ndarray,
    power_law_index: np.ndarray,
    source_mask: np.ndarray,
    case: ValidationCase,
) -> EvaluatedCubeBundle:
    """Evaluate a full image cube and integrated spectrum from local source maps."""
    image_shape = nonthermal_density_cm3.shape
    intensity_cube = np.zeros((frequency_ghz.size, image_shape[0], image_shape[1]), dtype=np.float64)
    circular_cube = np.zeros_like(intensity_cube)
    reference_cubes = {
        name: {
            "I_flux_sfu": np.zeros_like(intensity_cube),
            "V_flux_sfu": np.zeros_like(intensity_cube),
        }
        for name in REFERENCE_DESCRIPTIONS
    }
    diagnostic_cubes = {
        name: {
            "I_flux_sfu": np.zeros_like(intensity_cube),
            "V_flux_sfu": np.zeros_like(intensity_cube),
        }
        for name in DIAGNOSTIC_DESCRIPTIONS
    }
    cache: dict[tuple[float, float, float], dict[str, object]] = {}

    for iy in range(image_shape[0]):
        for ix in range(image_shape[1]):
            if not source_mask[iy, ix]:
                continue
            key = (
                round(float(nonthermal_density_cm3[iy, ix]), 3),
                round(float(magnetic_field_gauss[iy, ix]), 3),
                round(float(power_law_index[iy, ix]), 3),
            )
            if key not in cache:
                bundle = evaluate_reference_bundle(
                    backend,
                    MicrowaveSourceParameters(
                        nonthermal_density_1e7_cm3=float(nonthermal_density_cm3[iy, ix]) / 1.0e7,
                        magnetic_field_100g=float(magnetic_field_gauss[iy, ix]) / 100.0,
                        viewing_angle_deg=case.viewing_angle_deg,
                        thermal_density_1e9_cm3=case.thermal_density_cm3 / 1.0e9,
                        power_law_index=float(power_law_index[iy, ix]),
                        maximum_energy_mev=case.maximum_energy_mev,
                        temperature_mk=case.temperature_mk,
                        minimum_energy_mev=case.minimum_energy_mev,
                        pixel_area_arcsec2=case.pixel_size_arcsec * case.pixel_size_arcsec,
                        los_depth_arcsec=case.los_depth_arcsec,
                    ),
                    frequency_ghz,
                )
                cache[key] = {
                    "full": bundle.full,
                    "references": bundle.references,
                    "diagnostics": bundle.diagnostics,
                }
            cached = cache[key]
            full_spectrum = cached["full"]
            intensity_cube[:, iy, ix] = full_spectrum.total_sfu
            circular_cube[:, iy, ix] = full_spectrum.circular_sfu
            for name, spectrum in cached["references"].items():
                reference_cubes[name]["I_flux_sfu"][:, iy, ix] = spectrum.total_sfu
                reference_cubes[name]["V_flux_sfu"][:, iy, ix] = spectrum.circular_sfu
            for name, spectrum in cached["diagnostics"].items():
                diagnostic_cubes[name]["I_flux_sfu"][:, iy, ix] = spectrum.total_sfu
                diagnostic_cubes[name]["V_flux_sfu"][:, iy, ix] = spectrum.circular_sfu

    return EvaluatedCubeBundle(
        intensity_cube=intensity_cube,
        circular_cube=circular_cube,
        intensity_spectrum=intensity_cube.sum(axis=(1, 2)),
        circular_spectrum=circular_cube.sum(axis=(1, 2)),
        references={
            name: {
                "I_flux_sfu": value["I_flux_sfu"],
                "V_flux_sfu": value["V_flux_sfu"],
                "integrated_spectrum_sfu": value["I_flux_sfu"].sum(axis=(1, 2)),
                "integrated_circular_spectrum_sfu": value["V_flux_sfu"].sum(axis=(1, 2)),
            }
            for name, value in reference_cubes.items()
        },
        diagnostics={
            name: {
                "I_flux_sfu": value["I_flux_sfu"],
                "V_flux_sfu": value["V_flux_sfu"],
                "integrated_spectrum_sfu": value["I_flux_sfu"].sum(axis=(1, 2)),
                "integrated_circular_spectrum_sfu": value["V_flux_sfu"].sum(axis=(1, 2)),
            }
            for name, value in diagnostic_cubes.items()
        },
    )


def make_uniform_case() -> tuple[ValidationCase, dict[str, np.ndarray]]:
    """Return the uniform-source validation case and true source maps."""
    case = ValidationCase(
        name="case1_uniform_source",
        image_nx=40,
        image_ny=40,
        pixel_size_arcsec=1.0,
        los_depth_arcsec=8.0,
        viewing_angle_deg=72.0,
        thermal_density_cm3=5.0e9,
        temperature_mk=9.0,
        minimum_energy_mev=0.02,
        maximum_energy_mev=5.0,
        macro_particle_electron_count=1.0e27,
        morphology_frequencies_ghz=(0.1, 0.3, 1.0, 3.0, 6.0, 10.0),
        spectrum_frequencies_ghz=(
            0.03,
            0.05,
            0.08,
            0.1,
            0.15,
            0.2,
            0.3,
            0.5,
            0.8,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            8.0,
            10.0,
        ),
        power_law_index_default=4.2,
        power_law_index_bounds=(2.5, 6.5),
        minimum_power_law_fit_bins=4,
        minimum_power_law_fit_r_squared=0.7,
    )

    y, x = np.indices((case.image_ny, case.image_nx), dtype=np.float64)
    x_norm = (x - 0.5 * (case.image_nx - 1)) / (0.32 * case.image_nx)
    y_norm = (y - 0.5 * (case.image_ny - 1)) / (0.24 * case.image_ny)
    source_mask = x_norm * x_norm + y_norm * y_norm <= 1.0

    true_nonthermal_density_cm3 = np.where(source_mask, 8.0e7, 0.0)
    true_magnetic_field_gauss = np.where(source_mask, 180.0, 0.0)
    true_power_law_index = np.where(source_mask, 4.2, case.power_law_index_default)

    return case, {
        "source_mask": source_mask,
        "true_nonthermal_density_cm3": true_nonthermal_density_cm3,
        "true_magnetic_field_gauss": true_magnetic_field_gauss,
        "true_power_law_index": true_power_law_index,
    }


def make_loop_top_case() -> tuple[ValidationCase, dict[str, np.ndarray]]:
    """Return the analytic loop-top validation case and true source maps."""
    case = ValidationCase(
        name="case2_analytic_loop_top",
        image_nx=48,
        image_ny=48,
        pixel_size_arcsec=1.0,
        los_depth_arcsec=10.0,
        viewing_angle_deg=78.0,
        thermal_density_cm3=4.0e9,
        temperature_mk=10.0,
        minimum_energy_mev=0.02,
        maximum_energy_mev=6.0,
        macro_particle_electron_count=8.0e26,
        morphology_frequencies_ghz=(0.1, 0.3, 1.0, 3.0, 6.0, 10.0),
        spectrum_frequencies_ghz=(
            0.03,
            0.05,
            0.08,
            0.1,
            0.15,
            0.2,
            0.3,
            0.5,
            0.8,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            8.0,
            10.0,
        ),
        power_law_index_default=4.5,
        power_law_index_bounds=(2.5, 6.5),
        minimum_power_law_fit_bins=4,
        minimum_power_law_fit_r_squared=0.55,
    )

    y, x = np.indices((case.image_ny, case.image_nx), dtype=np.float64)
    x_norm = 2.0 * x / (case.image_nx - 1) - 1.0
    y_norm = 2.0 * y / (case.image_ny - 1) - 1.0

    top = np.exp(-((x_norm / 0.42) ** 2 + ((y_norm + 0.10) / 0.28) ** 2))
    left_foot = np.exp(-(((x_norm + 0.38) / 0.18) ** 2 + ((y_norm - 0.42) / 0.14) ** 2))
    right_foot = np.exp(-(((x_norm - 0.38) / 0.18) ** 2 + ((y_norm - 0.42) / 0.14) ** 2))
    arch = np.exp(-((np.sqrt(x_norm * x_norm + (y_norm + 0.12) ** 2) - 0.55) / 0.18) ** 2)
    source_envelope = np.maximum(top, 0.55 * arch)
    source_mask = source_envelope > 0.12

    magnetic_field = 70.0 + 190.0 * (left_foot + right_foot) + 60.0 * top
    magnetic_field = np.where(source_mask, magnetic_field, 0.0)

    nonthermal_density = (1.2e7 + 1.0e8 * top + 2.0e7 * arch) * source_mask
    delta = 4.9 - 0.9 * top + 0.25 * np.abs(x_norm)
    delta = np.where(source_mask, delta, case.power_law_index_default)

    return case, {
        "source_mask": source_mask,
        "true_nonthermal_density_cm3": nonthermal_density,
        "true_magnetic_field_gauss": magnetic_field,
        "true_power_law_index": delta,
    }


def save_morphology_figures(
    output_dir: Path,
    morphology_frequencies_ghz: tuple[float, ...],
    spectrum_frequencies_ghz: np.ndarray,
    theory_cube: np.ndarray,
    reconstructed_cube: np.ndarray,
) -> None:
    """Save side-by-side morphology figures for selected frequencies."""
    for frequency in morphology_frequencies_ghz:
        index = int(np.argmin(np.abs(spectrum_frequencies_ghz - frequency)))
        theory = np.asarray(theory_cube[index], dtype=np.float64)
        reconstructed = np.asarray(reconstructed_cube[index], dtype=np.float64)
        positive = np.concatenate([theory[theory > 0.0], reconstructed[reconstructed > 0.0]])
        if positive.size == 0:
            vmin, vmax = 1.0, 10.0
        else:
            vmin = float(np.min(positive))
            vmax = float(np.max(positive))

        figure, axes = plt.subplots(1, 2, figsize=(9, 4))
        for axis, image, title in zip(
            axes,
            (theory, reconstructed),
            ("Theory", "Reconstructed"),
        ):
            artist = axis.imshow(
                np.maximum(image, vmin),
                origin="lower",
                cmap="inferno",
                norm=LogNorm(vmin=max(vmin, 1.0e-8), vmax=max(vmax, vmin * 1.01)),
            )
            axis.set_title(f"{title} @ {spectrum_frequencies_ghz[index]:.2f} GHz")
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(artist, ax=axis, shrink=0.82)
        figure.tight_layout()
        figure.savefig(output_dir / f"morphology_{spectrum_frequencies_ghz[index]:.2f}GHz.png", dpi=150)
        plt.close(figure)


def save_spectrum_comparison(
    output_dir: Path,
    frequency_ghz: np.ndarray,
    theory_spectrum: np.ndarray,
    reconstructed_spectrum: np.ndarray,
) -> None:
    """Save integrated spectrum and ratio comparison."""
    ratio = np.ones_like(theory_spectrum)
    valid = np.abs(theory_spectrum) > 0.0
    ratio[valid] = reconstructed_spectrum[valid] / theory_spectrum[valid]

    figure, axes = plt.subplots(2, 1, figsize=(6.5, 7.0), sharex=True)
    axes[0].loglog(frequency_ghz, theory_spectrum, label="Theory", linewidth=2.0)
    axes[0].loglog(frequency_ghz, reconstructed_spectrum, label="Reconstructed", linewidth=2.0)
    axes[0].set_ylabel("Integrated Flux [sfu]")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(frequency_ghz, ratio, linewidth=2.0)
    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel("Frequency [GHz]")
    axes[1].set_ylabel("Recon / Theory")
    axes[1].grid(True, which="both", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_dir / "spectrum_comparison.png", dpi=150)
    plt.close(figure)


def relative_spectrum_error(theory: np.ndarray, reconstructed: np.ndarray) -> float:
    """Return RMS relative spectrum error over bins with finite theory."""
    valid = np.abs(theory) > 0.0
    if not np.any(valid):
        return 0.0
    residual = (reconstructed[valid] - theory[valid]) / theory[valid]
    return float(np.sqrt(np.mean(residual * residual)))


def masked_map_metrics(
    theory: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Return absolute and relative error metrics for one parameter map."""
    valid = np.asarray(mask, dtype=bool) & np.isfinite(theory) & np.isfinite(reconstructed)
    if not np.any(valid):
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "relative_mae": 0.0,
            "relative_rmse": 0.0,
            "peak_theory": 0.0,
            "peak_reconstructed": 0.0,
            "peak_relative_error": 0.0,
        }

    truth = np.asarray(theory[valid], dtype=np.float64)
    recon = np.asarray(reconstructed[valid], dtype=np.float64)
    diff = recon - truth

    positive = np.abs(truth) > 0.0
    relative_diff = np.zeros_like(diff)
    relative_diff[positive] = diff[positive] / truth[positive]

    peak_theory = float(np.max(truth))
    peak_reconstructed = float(np.max(recon))
    peak_relative_error = (
        abs(peak_reconstructed - peak_theory) / peak_theory if peak_theory > 0.0 else 0.0
    )
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "relative_mae": float(np.mean(np.abs(relative_diff[positive]))) if np.any(positive) else 0.0,
        "relative_rmse": float(np.sqrt(np.mean(relative_diff[positive] ** 2)))
        if np.any(positive)
        else 0.0,
        "peak_theory": peak_theory,
        "peak_reconstructed": peak_reconstructed,
        "peak_relative_error": float(peak_relative_error),
    }


def image_metrics(
    theory: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Return one image-comparison metric set for a single frequency plane."""
    valid = np.asarray(mask, dtype=bool) & np.isfinite(theory) & np.isfinite(reconstructed)
    if not np.any(valid):
        return {
            "relative_rmse": 0.0,
            "relative_mae": 0.0,
            "centroid_offset_pixels": 0.0,
            "peak_value_relative_error": 0.0,
            "plane_flux_relative_error": 0.0,
        }

    truth = np.asarray(theory[valid], dtype=np.float64)
    recon = np.asarray(reconstructed[valid], dtype=np.float64)
    diff = recon - truth
    positive = np.abs(truth) > 0.0
    relative_diff = np.zeros_like(diff)
    relative_diff[positive] = diff[positive] / truth[positive]

    y_index, x_index = np.indices(theory.shape, dtype=np.float64)
    theory_weights = np.where(valid, np.maximum(theory, 0.0), 0.0)
    recon_weights = np.where(valid, np.maximum(reconstructed, 0.0), 0.0)

    theory_weight_sum = float(np.sum(theory_weights))
    recon_weight_sum = float(np.sum(recon_weights))
    if theory_weight_sum > 0.0 and recon_weight_sum > 0.0:
        theory_centroid_x = float(np.sum(x_index * theory_weights) / theory_weight_sum)
        theory_centroid_y = float(np.sum(y_index * theory_weights) / theory_weight_sum)
        recon_centroid_x = float(np.sum(x_index * recon_weights) / recon_weight_sum)
        recon_centroid_y = float(np.sum(y_index * recon_weights) / recon_weight_sum)
        centroid_offset = float(
            np.hypot(recon_centroid_x - theory_centroid_x, recon_centroid_y - theory_centroid_y)
        )
    else:
        centroid_offset = 0.0

    peak_theory = float(np.max(truth))
    peak_reconstructed = float(np.max(recon))
    peak_value_relative_error = (
        abs(peak_reconstructed - peak_theory) / peak_theory if peak_theory > 0.0 else 0.0
    )
    plane_flux_relative_error = (
        abs(recon_weight_sum - theory_weight_sum) / theory_weight_sum if theory_weight_sum > 0.0 else 0.0
    )
    return {
        "relative_rmse": float(np.sqrt(np.mean(relative_diff[positive] ** 2)))
        if np.any(positive)
        else 0.0,
        "relative_mae": float(np.mean(np.abs(relative_diff[positive]))) if np.any(positive) else 0.0,
        "centroid_offset_pixels": centroid_offset,
        "peak_value_relative_error": float(peak_value_relative_error),
        "plane_flux_relative_error": float(plane_flux_relative_error),
    }


def band_spectrum_metrics(
    frequency_ghz: np.ndarray,
    theory_spectrum: np.ndarray,
    reconstructed_spectrum: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Return spectrum errors grouped by broad observing bands."""
    bands = {
        "low_0.03_0.3GHz": (0.03, 0.3, False),
        "mid_0.3_3GHz": (0.3, 3.0, False),
        "high_3_10GHz": (3.0, 10.0, True),
    }
    results: dict[str, dict[str, float]] = {}
    for name, (lower, upper, include_upper) in bands.items():
        if include_upper:
            mask = (frequency_ghz >= lower) & (frequency_ghz <= upper)
        else:
            mask = (frequency_ghz >= lower) & (frequency_ghz < upper)
        if not np.any(mask):
            results[name] = {
                "rms_relative_error": 0.0,
                "integrated_flux_theory_sfu": 0.0,
                "integrated_flux_reconstructed_sfu": 0.0,
                "integrated_flux_relative_error": 0.0,
            }
            continue

        theory_band = theory_spectrum[mask]
        recon_band = reconstructed_spectrum[mask]
        integrated_theory = float(np.sum(theory_band))
        integrated_reconstructed = float(np.sum(recon_band))
        integrated_relative_error = (
            abs(integrated_reconstructed - integrated_theory) / integrated_theory
            if integrated_theory > 0.0
            else 0.0
        )
        results[name] = {
            "rms_relative_error": relative_spectrum_error(theory_band, recon_band),
            "integrated_flux_theory_sfu": integrated_theory,
            "integrated_flux_reconstructed_sfu": integrated_reconstructed,
            "integrated_flux_relative_error": float(integrated_relative_error),
        }
    return results


def run_validation_case(
    output_root: Path,
    case: ValidationCase,
    source_maps: dict[str, np.ndarray],
    backend: MicrowaveSpectrumBackend,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Run one synthetic validation case and save products."""
    output_dir = output_root / case.name
    output_dir.mkdir(parents=True, exist_ok=True)

    energy_bin_edges_mev = np.logspace(
        np.log10(case.minimum_energy_mev),
        np.log10(case.maximum_energy_mev),
        14,
    )
    (
        deposited_weight,
        reconstructed_nonthermal_density,
        reconstructed_delta,
        fit_r_squared,
        fit_accepted,
        global_delta,
        global_r_squared,
        global_accepted,
    ) = simulate_reconstruction(
        source_maps["true_nonthermal_density_cm3"],
        source_maps["true_power_law_index"],
        case,
        energy_bin_edges_mev,
        rng,
    )

    frequency_ghz = np.asarray(case.spectrum_frequencies_ghz, dtype=np.float64)
    frequency_hz = 1.0e9 * frequency_ghz

    theory_bundle = evaluate_cube(
        backend,
        frequency_ghz,
        source_maps["true_nonthermal_density_cm3"],
        source_maps["true_magnetic_field_gauss"],
        source_maps["true_power_law_index"],
        source_maps["source_mask"],
        case,
    )
    reconstructed_bundle = evaluate_cube(
        backend,
        frequency_ghz,
        reconstructed_nonthermal_density,
        source_maps["true_magnetic_field_gauss"],
        reconstructed_delta,
        source_maps["source_mask"],
        case,
    )
    theory_cube = theory_bundle.intensity_cube
    theory_spectrum = theory_bundle.intensity_spectrum
    reconstructed_cube = reconstructed_bundle.intensity_cube
    reconstructed_spectrum = reconstructed_bundle.intensity_spectrum

    theory_specific = flux_sfu_to_specific_intensity(theory_cube, case.pixel_size_arcsec)
    reconstructed_specific = flux_sfu_to_specific_intensity(
        reconstructed_cube, case.pixel_size_arcsec
    )
    theory_tb = specific_intensity_to_brightness_temperature(theory_specific, frequency_hz)
    reconstructed_tb = specific_intensity_to_brightness_temperature(
        reconstructed_specific, frequency_hz
    )

    save_morphology_figures(
        output_dir,
        case.morphology_frequencies_ghz,
        frequency_ghz,
        theory_cube,
        reconstructed_cube,
    )
    save_spectrum_comparison(output_dir, frequency_ghz, theory_spectrum, reconstructed_spectrum)

    parameter_metrics = {
        "nonthermal_density_cm3": masked_map_metrics(
            source_maps["true_nonthermal_density_cm3"],
            reconstructed_nonthermal_density,
            source_maps["source_mask"],
        ),
        "power_law_index": masked_map_metrics(
            source_maps["true_power_law_index"],
            reconstructed_delta,
            source_maps["source_mask"],
        ),
    }
    morphology_metrics = {
        f"{frequency:.2f}GHz": image_metrics(
            theory_cube[int(np.argmin(np.abs(frequency_ghz - frequency)))],
            reconstructed_cube[int(np.argmin(np.abs(frequency_ghz - frequency)))],
            source_maps["source_mask"],
        )
        for frequency in case.morphology_frequencies_ghz
    }
    spectrum_band_metrics = band_spectrum_metrics(
        frequency_ghz, theory_spectrum, reconstructed_spectrum
    )

    summary = {
        "case_name": case.name,
        "global_power_law_index": float(global_delta),
        "global_power_law_r_squared": float(global_r_squared),
        "global_power_law_fit_accepted": int(global_accepted),
        "accepted_local_fits": int(np.count_nonzero(fit_accepted)),
        "macro_particle_electron_count": float(case.macro_particle_electron_count),
        "integrated_spectrum_rms_relative_error": relative_spectrum_error(
            theory_spectrum, reconstructed_spectrum
        ),
        "nonthermal_density_peak_theory_cm3": float(np.max(source_maps["true_nonthermal_density_cm3"])),
        "nonthermal_density_peak_reconstructed_cm3": float(np.max(reconstructed_nonthermal_density)),
        "parameter_metrics": parameter_metrics,
        "morphology_metrics": morphology_metrics,
        "spectrum_band_metrics": spectrum_band_metrics,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    write_emission_product(
        output_dir / "validation_product.h5",
        EmissionHdf5Metadata(
            frame_id=0,
            transport_model="synthetic",
            field_source_kind="analytic",
            background_source_kind="analytic",
        ),
        {
            "freq_hz": frequency_hz,
            "truth": {
                "source_mask": source_maps["source_mask"].astype(np.int32),
                "nonthermal_density_cm3": source_maps["true_nonthermal_density_cm3"],
                "magnetic_field_gauss": source_maps["true_magnetic_field_gauss"],
                "power_law_index": source_maps["true_power_law_index"],
                **build_cube_payload(
                    theory_bundle.intensity_cube,
                    theory_bundle.circular_cube,
                    case.pixel_size_arcsec,
                    frequency_hz,
                ),
            },
            "reconstruction": {
                "deposited_weight": deposited_weight,
                "nonthermal_density_cm3": reconstructed_nonthermal_density,
                "power_law_index": reconstructed_delta,
                "power_law_fit_r_squared": fit_r_squared,
                "power_law_fit_accepted": fit_accepted.astype(np.int32),
                **build_cube_payload(
                    reconstructed_bundle.intensity_cube,
                    reconstructed_bundle.circular_cube,
                    case.pixel_size_arcsec,
                    frequency_hz,
                ),
            },
            "references": {
                "truth": {
                    name: {
                        **build_cube_payload(
                            theory_bundle.references[name]["I_flux_sfu"],
                            theory_bundle.references[name]["V_flux_sfu"],
                            case.pixel_size_arcsec,
                            frequency_hz,
                        ),
                        "description": REFERENCE_DESCRIPTIONS[name],
                    }
                    for name in theory_bundle.references
                },
                "reconstruction": {
                    name: {
                        **build_cube_payload(
                            reconstructed_bundle.references[name]["I_flux_sfu"],
                            reconstructed_bundle.references[name]["V_flux_sfu"],
                            case.pixel_size_arcsec,
                            frequency_hz,
                        ),
                        "description": REFERENCE_DESCRIPTIONS[name],
                    }
                    for name in reconstructed_bundle.references
                },
            },
            "diagnostics": {
                "truth": {
                    name: {
                        **build_cube_payload(
                            theory_bundle.diagnostics[name]["I_flux_sfu"],
                            theory_bundle.diagnostics[name]["V_flux_sfu"],
                            case.pixel_size_arcsec,
                            frequency_hz,
                        ),
                        "description": DIAGNOSTIC_DESCRIPTIONS[name],
                    }
                    for name in theory_bundle.diagnostics
                },
                "reconstruction": {
                    name: {
                        **build_cube_payload(
                            reconstructed_bundle.diagnostics[name]["I_flux_sfu"],
                            reconstructed_bundle.diagnostics[name]["V_flux_sfu"],
                            case.pixel_size_arcsec,
                            frequency_hz,
                        ),
                        "description": DIAGNOSTIC_DESCRIPTIONS[name],
                    }
                    for name in reconstructed_bundle.diagnostics
                },
            },
            "component_info": {
                "model_note": (
                    "Self-absorption and Razin effects are stored as reference or differential curves, "
                    "not as additive conserved intensity components."
                ),
                **REFERENCE_DESCRIPTIONS,
                **DIAGNOSTIC_DESCRIPTIONS,
            },
            "metrics": {
                "parameter_metrics": parameter_metrics,
                "morphology_metrics": morphology_metrics,
                "spectrum_band_metrics": spectrum_band_metrics,
            },
            "summary": {
                key: np.asarray([value], dtype=np.float64)
                if isinstance(value, float)
                else np.asarray([value], dtype=np.int32)
                for key, value in summary.items()
                if key not in {"case_name", "parameter_metrics", "morphology_metrics", "spectrum_band_metrics"}
            },
        },
    )
    return summary


def write_root_readme(output_root: Path, summaries: list[dict[str, object]]) -> None:
    """Write a small markdown overview of the validation outputs."""
    lines = [
        "# Emission Validation Outputs",
        "",
        "This folder contains two synthetic validation cases:",
        "",
        "- `case1_uniform_source/`: uniform-source baseline",
        "- `case2_analytic_loop_top/`: analytic loop-top morphology case",
        "",
        "Each case includes:",
        "",
        "- `morphology_*.png`: theory vs reconstructed maps at selected frequencies",
        "- `spectrum_comparison.png`: integrated spectrum comparison",
        "- `validation_product.h5`: theory and reconstruction cubes plus parameter maps and component-aware references",
        "- `summary.json`: compact numerical summary with parameter, morphology, and spectral metrics",
        "",
        "Metric groups:",
        "",
        "- `parameter_metrics`: MAE/RMSE and relative errors for nonthermal density and power-law index",
        "- `morphology_metrics`: per-frequency image relative error, brightness centroid offset, peak-value error, and plane-integrated flux error",
        "- `spectrum_band_metrics`: low/mid/high band RMS spectral error and integrated-flux error",
        "",
        "Component-aware reference groups inside `validation_product.h5`:",
        "",
        "- `references/truth` and `references/reconstruction`: thermal-only, low-density no-Razin proxy, optically thin proxy, and background-only curves",
        "- `diagnostics/truth` and `diagnostics/reconstruction`: absorption impact and Razin impact curves",
        "",
        "Case summaries:",
        "",
    ]
    for summary in summaries:
        lines.append(
            f"- `{summary['case_name']}`: RMS spectrum error = "
            f"{summary['integrated_spectrum_rms_relative_error']:.4e}, "
            f"accepted local power-law fits = {summary['accepted_local_fits']}"
        )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    """Parse the validation-case CLI."""
    parser = argparse.ArgumentParser(
        description="Generate uniform-source and analytic loop-top validation outputs."
    )
    parser.add_argument(
        "--output-root",
        default="emissionValidation",
        help="Root output directory for generated validation cases.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for synthetic reconstruction noise.",
    )
    return parser.parse_args()


def main() -> int:
    """Run both synthetic validation cases."""
    arguments = parse_arguments()
    output_root = Path(arguments.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    backend = MicrowaveSpectrumBackend()
    rng = np.random.default_rng(arguments.seed)

    summaries = []
    for factory in (make_uniform_case, make_loop_top_case):
        case, source_maps = factory()
        summary = run_validation_case(output_root, case, source_maps, backend, rng)
        summaries.append(summary)
        print(
            f"{case.name}: RMS spectrum error = "
            f"{summary['integrated_spectrum_rms_relative_error']:.6e}, "
            f"accepted local fits = {summary['accepted_local_fits']}"
        )

    write_root_readme(output_root, summaries)
    print(f"Validation outputs saved to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
