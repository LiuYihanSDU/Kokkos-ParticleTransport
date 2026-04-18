#!/usr/bin/env python3
"""Simple runnable particle-emission prototype over the current example data."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .compact_field_io import coarse_average_2d, coarse_pixel_centers, read_compact_field_snapshot
from .emission_hdf5 import (
    EmissionHdf5Metadata,
    convolve_image_with_gaussian_beam,
    find_frequency_index,
    gaussian_beam_weights,
    integrate_roi_spectrum,
    write_emission_product,
)
from .emission_references import evaluate_reference_bundle
from .microwave_backend import MicrowaveSourceParameters, MicrowaveSpectrumBackend
from .particle_io import (
    deposit_particle_energy_histograms,
    deposit_particles_to_image_grid,
    fit_power_law_index_with_quality,
    fit_power_law_index_map_with_quality,
    read_particle_snapshot,
)


SFU_CGS = 1.0e-19
ARCSEC_TO_RADIAN = np.pi / (180.0 * 3600.0)
ASTRONOMICAL_UNIT_CM = 1.495978707e13


@dataclass
class SimpleEmissionConfig:
    """Configuration for the first runnable emission prototype."""

    field_path: str = "kokkos_cpu_format_example_frame180_262k/compact_field/field00180.bin"
    particle_path: str = "kokkos_cpu_format_example_frame180_262k/kokkos_cpu/particles_00181.bin"
    output_hdf5: str = "particleEmission/output/simple_emission.h5"
    output_quicklook: str = "particleEmission/output/simple_emission.png"
    transport_model: str = "parker"
    image_nx: int = 32
    image_ny: int = 32
    frequencies_ghz: tuple[float, ...] = (
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
    )
    quicklook_frequency_ghz: float = 3.0
    beam_fwhm_pixels: float = 1.5
    pixel_size_arcsec: float = 1.0
    minimum_particle_energy_ev: float = 0.0
    macro_particle_electron_count: float = 1.0e30
    target_peak_nonthermal_density_cm3: float = 1.0e8
    thermal_density_cm3: float = 5.0e9
    temperature_mk: float = 8.0
    los_depth_arcsec: float = 8.0
    viewing_angle_deg: float = 75.0
    power_law_index_default: float = 4.5
    power_law_index_min: float = 2.0
    power_law_index_max: float = 8.0
    energy_histogram_bins: int = 12
    minimum_power_law_fit_bins: int = 4
    minimum_power_law_fit_r_squared: float = 0.5
    minimum_energy_mev: float = 0.02
    maximum_energy_mev: float = 5.0
    magnetic_field_floor_gauss: float = 80.0
    magnetic_field_peak_gauss: float = 220.0
    magnetic_field_bins: int = 8
    nonthermal_density_bins: int = 8
    power_law_index_bins: int = 8


def _load_config(arguments: argparse.Namespace) -> SimpleEmissionConfig:
    """Load one JSON config file and overlay CLI values."""
    config = SimpleEmissionConfig()
    if arguments.config is not None:
        with Path(arguments.config).open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        config_dict = asdict(config)
        config_dict.update(loaded)
        config = SimpleEmissionConfig(**config_dict)

    for field_name in asdict(config):
        value = getattr(arguments, field_name, None)
        if value is not None:
            setattr(config, field_name, value)
    return config


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the simple prototype runner."""
    parser = argparse.ArgumentParser(
        description="Run a simple particle-emission prototype on the example data."
    )
    parser.add_argument("--config", type=str, help="Optional JSON config file.")
    parser.add_argument("--field-path", dest="field_path", type=str)
    parser.add_argument("--particle-path", dest="particle_path", type=str)
    parser.add_argument("--output-hdf5", dest="output_hdf5", type=str)
    parser.add_argument("--output-quicklook", dest="output_quicklook", type=str)
    parser.add_argument("--transport-model", dest="transport_model", type=str)
    parser.add_argument("--image-nx", dest="image_nx", type=int)
    parser.add_argument("--image-ny", dest="image_ny", type=int)
    parser.add_argument("--quicklook-frequency-ghz", dest="quicklook_frequency_ghz", type=float)
    parser.add_argument("--beam-fwhm-pixels", dest="beam_fwhm_pixels", type=float)
    parser.add_argument("--pixel-size-arcsec", dest="pixel_size_arcsec", type=float)
    parser.add_argument("--minimum-particle-energy-ev", dest="minimum_particle_energy_ev", type=float)
    parser.add_argument("--nonthermal-density-peak-cm3", dest="nonthermal_density_peak_cm3", type=float)
    parser.add_argument("--thermal-density-cm3", dest="thermal_density_cm3", type=float)
    parser.add_argument("--temperature-mk", dest="temperature_mk", type=float)
    parser.add_argument("--los-depth-arcsec", dest="los_depth_arcsec", type=float)
    parser.add_argument("--viewing-angle-deg", dest="viewing_angle_deg", type=float)
    parser.add_argument("--power-law-index", dest="power_law_index", type=float)
    parser.add_argument("--minimum-energy-mev", dest="minimum_energy_mev", type=float)
    parser.add_argument("--maximum-energy-mev", dest="maximum_energy_mev", type=float)
    parser.add_argument("--magnetic-field-floor-gauss", dest="magnetic_field_floor_gauss", type=float)
    parser.add_argument("--magnetic-field-peak-gauss", dest="magnetic_field_peak_gauss", type=float)
    parser.add_argument("--magnetic-field-bins", dest="magnetic_field_bins", type=int)
    parser.add_argument("--nonthermal-density-bins", dest="nonthermal_density_bins", type=int)
    return parser.parse_args()


def flux_sfu_to_specific_intensity(
    flux_sfu: np.ndarray,
    pixel_area_arcsec2: float,
) -> np.ndarray:
    """Convert pixel flux density in sfu to specific intensity."""
    solid_angle_sr = pixel_area_arcsec2 * ARCSEC_TO_RADIAN * ARCSEC_TO_RADIAN
    return np.asarray(flux_sfu, dtype=np.float64) * SFU_CGS / solid_angle_sr


def arcsec_length_cm() -> float:
    """Return the physical length corresponding to one arcsec at 1 AU."""
    return ASTRONOMICAL_UNIT_CM * ARCSEC_TO_RADIAN


def coarse_voxel_volume_cm3(
    pixel_size_arcsec: float,
    los_depth_arcsec: float,
) -> float:
    """Return one coarse image voxel volume in cm^3."""
    length_cm = arcsec_length_cm()
    return (
        pixel_size_arcsec
        * pixel_size_arcsec
        * los_depth_arcsec
        * length_cm
        * length_cm
        * length_cm
    )


def specific_intensity_to_brightness_temperature(
    specific_intensity: np.ndarray,
    frequency_hz: np.ndarray,
) -> np.ndarray:
    """Convert specific intensity to brightness temperature."""
    light_speed_cgs = 2.99792458e10
    boltzmann_cgs = 1.380649e-16
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    intensity = np.asarray(specific_intensity, dtype=np.float64)
    reshape = (frequency.size,) + (1,) * (intensity.ndim - 1)
    frequency_expanded = frequency.reshape(reshape)
    return (
        light_speed_cgs
        * light_speed_cgs
        * intensity
        / (2.0 * boltzmann_cgs * frequency_expanded * frequency_expanded)
    )


def weight_map_to_nonthermal_density_cm3(
    weight_map: np.ndarray,
    macro_particle_electron_count: float,
    pixel_size_arcsec: float,
    los_depth_arcsec: float,
) -> np.ndarray:
    """Convert deposited particle weights into nonthermal density."""
    voxel_volume = coarse_voxel_volume_cm3(pixel_size_arcsec, los_depth_arcsec)
    return (
        np.asarray(weight_map, dtype=np.float64)
        * float(macro_particle_electron_count)
        / voxel_volume
    )


def estimate_macro_particle_electron_count(
    weight_map: np.ndarray,
    pixel_size_arcsec: float,
    los_depth_arcsec: float,
    target_peak_density_cm3: float,
) -> float:
    """Estimate a macro-particle coefficient from a target peak density."""
    maximum_weight = float(np.max(weight_map))
    if maximum_weight <= 0.0:
        raise RuntimeError("cannot estimate macro-particle coefficient from an empty weight map")
    return (
        target_peak_density_cm3
        * coarse_voxel_volume_cm3(pixel_size_arcsec, los_depth_arcsec)
        / maximum_weight
    )


def _scale_magnetic_field_map(
    bmag_map: np.ndarray,
    floor_gauss: float,
    peak_gauss: float,
) -> np.ndarray:
    """Scale field morphology to a user-facing coronal Gauss range."""
    normalized = (bmag_map - np.min(bmag_map)) / max(np.ptp(bmag_map), 1.0e-12)
    return floor_gauss + normalized * (peak_gauss - floor_gauss)


def _quantize(values: np.ndarray, count: int) -> np.ndarray:
    """Quantize one normalized [0, 1] map into integer bins."""
    clipped = np.clip(values, 0.0, 1.0)
    return np.rint(clipped * (count - 1)).astype(np.int32)


def _build_stokes_cube_payload(
    intensity_flux_cube: np.ndarray,
    circular_flux_cube: np.ndarray,
    pixel_area_arcsec2: float,
    frequency_hz: np.ndarray,
    *,
    left_flux_cube: np.ndarray | None = None,
    right_flux_cube: np.ndarray | None = None,
    linear_flux_cube: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Convert Stokes-like flux cubes to a HDF5 payload block."""
    intensity_flux = np.asarray(intensity_flux_cube, dtype=np.float64)
    circular_flux = np.asarray(circular_flux_cube, dtype=np.float64)
    intensity_specific = flux_sfu_to_specific_intensity(intensity_flux, pixel_area_arcsec2)
    circular_specific = flux_sfu_to_specific_intensity(circular_flux, pixel_area_arcsec2)
    intensity_tb = specific_intensity_to_brightness_temperature(intensity_specific, frequency_hz)
    circular_tb = specific_intensity_to_brightness_temperature(circular_specific, frequency_hz)

    payload = {
        "I_flux_sfu": intensity_flux,
        "I_specific": intensity_specific,
        "I_brightness_temperature": intensity_tb,
        "V_flux_sfu": circular_flux,
        "V_specific": circular_specific,
        "V_brightness_temperature": circular_tb,
    }
    if left_flux_cube is not None:
        payload["left_flux_sfu"] = np.asarray(left_flux_cube, dtype=np.float64)
    if right_flux_cube is not None:
        payload["right_flux_sfu"] = np.asarray(right_flux_cube, dtype=np.float64)
    if linear_flux_cube is not None:
        payload["linear_flux_sfu"] = np.asarray(linear_flux_cube, dtype=np.float64)
    return payload


def _save_quicklook(
    output_path: Path,
    frequency_ghz: np.ndarray,
    total_flux_cube: np.ndarray,
    circular_flux_cube: np.ndarray,
    convolved_total_flux_cube: np.ndarray,
    quicklook_frequency_ghz: float,
) -> None:
    """Save a quicklook figure with one image pair and one integrated spectrum pair."""
    frequency_index = find_frequency_index(frequency_ghz, quicklook_frequency_ghz)
    native_image = np.asarray(total_flux_cube[frequency_index], dtype=np.float64)
    beam_image = np.asarray(convolved_total_flux_cube[frequency_index], dtype=np.float64)
    total_spectrum = total_flux_cube.sum(axis=(1, 2))
    circular_spectrum = circular_flux_cube.sum(axis=(1, 2))
    circular_degree = np.zeros_like(total_spectrum)
    valid = np.abs(total_spectrum) > 0.0
    circular_degree[valid] = circular_spectrum[valid] / total_spectrum[valid]

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))

    image0 = axes[0, 0].imshow(native_image, origin="lower", cmap="inferno")
    axes[0, 0].set_title(f"Native I @ {frequency_ghz[frequency_index]:.1f} GHz")
    figure.colorbar(image0, ax=axes[0, 0], shrink=0.85)

    image1 = axes[0, 1].imshow(beam_image, origin="lower", cmap="inferno")
    axes[0, 1].set_title(f"Beam I @ {frequency_ghz[frequency_index]:.1f} GHz")
    figure.colorbar(image1, ax=axes[0, 1], shrink=0.85)

    axes[1, 0].loglog(frequency_ghz, total_spectrum)
    axes[1, 0].set_xlabel("Frequency [GHz]")
    axes[1, 0].set_ylabel("Integrated Flux [sfu]")
    axes[1, 0].set_title("Full-Image Spectrum")
    axes[1, 0].grid(True, which="both", alpha=0.3)

    axes[1, 1].semilogx(frequency_ghz, circular_degree)
    axes[1, 1].set_xlabel("Frequency [GHz]")
    axes[1, 1].set_ylabel("V / I")
    axes[1, 1].set_title("Integrated Circular Polarization")
    axes[1, 1].grid(True, which="both", alpha=0.3)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_simple_emission(config: SimpleEmissionConfig) -> tuple[Path, Path]:
    """Run the simple particle-emission prototype and save its outputs."""
    field = read_compact_field_snapshot(config.field_path)
    particles = read_particle_snapshot(config.particle_path)
    backend = MicrowaveSpectrumBackend()

    weight_map = deposit_particles_to_image_grid(
        particles,
        (float(field.x_edges[0]), float(field.x_edges[-1])),
        (float(field.y_edges[0]), float(field.y_edges[-1])),
        config.image_nx,
        config.image_ny,
        minimum_energy_ev=config.minimum_particle_energy_ev,
    )
    bmag_map_native = field.magnetic_field_magnitude
    bmag_map = coarse_average_2d(
        bmag_map_native,
        field.x_edges,
        field.y_edges,
        config.image_nx,
        config.image_ny,
    )

    suggested_macro_particle_count = estimate_macro_particle_electron_count(
        weight_map,
        config.pixel_size_arcsec,
        config.los_depth_arcsec,
        config.target_peak_nonthermal_density_cm3,
    )
    nonthermal_density_map = weight_map_to_nonthermal_density_cm3(
        weight_map,
        config.macro_particle_electron_count,
        config.pixel_size_arcsec,
        config.los_depth_arcsec,
    )
    active_energy = particles.kinetic_energy_ev[
        (particles.status == 0) & (particles.kinetic_energy_ev > 0.0)
    ]
    if active_energy.size == 0:
        raise RuntimeError("no active particles with positive kinetic energy were found")
    energy_bin_edges_ev = np.logspace(
        np.log10(float(active_energy.min())),
        np.log10(float(active_energy.max())),
        config.energy_histogram_bins + 1,
    )
    energy_histograms = deposit_particle_energy_histograms(
        particles,
        (float(field.x_edges[0]), float(field.x_edges[-1])),
        (float(field.y_edges[0]), float(field.y_edges[-1])),
        config.image_nx,
        config.image_ny,
        energy_bin_edges_ev,
        minimum_energy_ev=config.minimum_particle_energy_ev,
    )
    global_power_law_index, global_power_law_r_squared, global_power_law_accepted = fit_power_law_index_with_quality(
        energy_histograms.sum(axis=(0, 1)),
        energy_bin_edges_ev,
        config.power_law_index_default,
        minimum_populated_bins=config.minimum_power_law_fit_bins,
        index_bounds=(config.power_law_index_min, config.power_law_index_max),
        minimum_r_squared=config.minimum_power_law_fit_r_squared,
    )
    power_law_index_map, power_law_r_squared_map, power_law_fit_accepted_map = fit_power_law_index_map_with_quality(
        energy_histograms,
        energy_bin_edges_ev,
        global_power_law_index,
        minimum_populated_bins=config.minimum_power_law_fit_bins,
        index_bounds=(config.power_law_index_min, config.power_law_index_max),
        minimum_r_squared=config.minimum_power_law_fit_r_squared,
    )
    magnetic_field_map = _scale_magnetic_field_map(
        bmag_map, config.magnetic_field_floor_gauss, config.magnetic_field_peak_gauss
    )

    occupied = weight_map > 0.0
    if not np.any(occupied):
        raise RuntimeError("no active particles were deposited into the image grid")

    frequency_ghz = np.asarray(config.frequencies_ghz, dtype=np.float64)
    frequency_hz = 1.0e9 * frequency_ghz
    pixel_area_arcsec2 = config.pixel_size_arcsec * config.pixel_size_arcsec

    total_flux_cube = np.zeros((frequency_ghz.size, config.image_ny, config.image_nx), dtype=np.float64)
    left_flux_cube = np.zeros_like(total_flux_cube)
    right_flux_cube = np.zeros_like(total_flux_cube)
    circular_flux_cube = np.zeros_like(total_flux_cube)

    nnth_norm = np.zeros_like(nonthermal_density_map)
    nnth_norm[occupied] = nonthermal_density_map[occupied] / np.max(nonthermal_density_map[occupied])
    b_norm = (magnetic_field_map - np.min(magnetic_field_map)) / max(np.ptp(magnetic_field_map), 1.0e-12)
    delta_norm = (
        np.clip(power_law_index_map, config.power_law_index_min, config.power_law_index_max)
        - config.power_law_index_min
    ) / max(config.power_law_index_max - config.power_law_index_min, 1.0e-12)

    nnth_bins = _quantize(nnth_norm, config.nonthermal_density_bins)
    b_bins = _quantize(b_norm, config.magnetic_field_bins)
    delta_bins = _quantize(delta_norm, config.power_law_index_bins)

    reference_flux_cubes = {
        "thermal_only": np.zeros_like(total_flux_cube),
        "no_razin_density_proxy": np.zeros_like(total_flux_cube),
        "optically_thin_proxy": np.zeros_like(total_flux_cube),
        "background_only": np.zeros_like(total_flux_cube),
    }
    reference_circular_flux_cubes = {
        key: np.zeros_like(total_flux_cube) for key in reference_flux_cubes
    }
    diagnostic_flux_cubes = {
        "absorption_impact": np.zeros_like(total_flux_cube),
        "razin_impact": np.zeros_like(total_flux_cube),
    }
    diagnostic_circular_flux_cubes = {
        key: np.zeros_like(total_flux_cube) for key in diagnostic_flux_cubes
    }

    convolved_total_flux_cube = np.zeros_like(total_flux_cube)
    convolved_circular_flux_cube = np.zeros_like(circular_flux_cube)
    spectral_cache: dict[tuple[int, int, int], dict[str, object]] = {}
    for iy in range(config.image_ny):
        for ix in range(config.image_nx):
            if not occupied[iy, ix]:
                continue

            cache_key = (int(nnth_bins[iy, ix]), int(b_bins[iy, ix]), int(delta_bins[iy, ix]))
            if cache_key not in spectral_cache:
                parameters = MicrowaveSourceParameters(
                    nonthermal_density_1e7_cm3=nonthermal_density_map[iy, ix] / 1.0e7,
                    magnetic_field_100g=magnetic_field_map[iy, ix] / 100.0,
                    viewing_angle_deg=config.viewing_angle_deg,
                    thermal_density_1e9_cm3=config.thermal_density_cm3 / 1.0e9,
                    power_law_index=power_law_index_map[iy, ix],
                    maximum_energy_mev=config.maximum_energy_mev,
                    temperature_mk=config.temperature_mk,
                    minimum_energy_mev=config.minimum_energy_mev,
                    pixel_area_arcsec2=pixel_area_arcsec2,
                    los_depth_arcsec=config.los_depth_arcsec,
                )
                bundle = evaluate_reference_bundle(backend, parameters, frequency_ghz)
                spectral_cache[cache_key] = {
                    "full": bundle.full,
                    "references": bundle.references,
                    "diagnostics": bundle.diagnostics,
                }

            cache_entry = spectral_cache[cache_key]
            full_spectrum = cache_entry["full"]
            left_flux_cube[:, iy, ix] = full_spectrum.left_sfu
            right_flux_cube[:, iy, ix] = full_spectrum.right_sfu
            total_flux_cube[:, iy, ix] = full_spectrum.total_sfu
            circular_flux_cube[:, iy, ix] = full_spectrum.circular_sfu

            for name, spectrum in cache_entry["references"].items():
                reference_flux_cubes[name][:, iy, ix] = spectrum.total_sfu
                reference_circular_flux_cubes[name][:, iy, ix] = spectrum.circular_sfu

            for name, spectrum in cache_entry["diagnostics"].items():
                diagnostic_flux_cubes[name][:, iy, ix] = spectrum.total_sfu
                diagnostic_circular_flux_cubes[name][:, iy, ix] = spectrum.circular_sfu

    full_image_payload = _build_stokes_cube_payload(
        total_flux_cube,
        circular_flux_cube,
        pixel_area_arcsec2,
        frequency_hz,
        left_flux_cube=left_flux_cube,
        right_flux_cube=right_flux_cube,
        linear_flux_cube=np.full_like(total_flux_cube, np.nan),
    )
    total_specific_cube = full_image_payload["I_specific"]
    circular_specific_cube = full_image_payload["V_specific"]
    total_tb_cube = full_image_payload["I_brightness_temperature"]
    circular_tb_cube = full_image_payload["V_brightness_temperature"]

    for index in range(frequency_ghz.size):
        convolved_total_flux_cube[index] = convolve_image_with_gaussian_beam(
            total_flux_cube[index], config.beam_fwhm_pixels
        )
        convolved_circular_flux_cube[index] = convolve_image_with_gaussian_beam(
            circular_flux_cube[index], config.beam_fwhm_pixels
        )

    beam_image_payload = _build_stokes_cube_payload(
        convolved_total_flux_cube,
        convolved_circular_flux_cube,
        pixel_area_arcsec2,
        frequency_hz,
    )
    convolved_total_specific_cube = beam_image_payload["I_specific"]
    convolved_total_tb_cube = beam_image_payload["I_brightness_temperature"]
    convolved_circular_specific_cube = beam_image_payload["V_specific"]
    convolved_circular_tb_cube = beam_image_payload["V_brightness_temperature"]

    reference_payload = {
        name: {
            **_build_stokes_cube_payload(
                reference_flux_cubes[name],
                reference_circular_flux_cubes[name],
                pixel_area_arcsec2,
                frequency_hz,
            ),
            "description": {
                "thermal_only": "Low nonthermal-density thermal/free-free reference.",
                "no_razin_density_proxy": (
                    "Low thermal-density proxy that weakens Razin suppression and free-free opacity together."
                ),
                "optically_thin_proxy": (
                    "Thin-slab proxy rescaled back to the original LOS depth to diagnose self-absorption."
                ),
                "background_only": "Zero external-background reference used by the current prototype.",
            }[name],
        }
        for name in reference_flux_cubes
    }
    diagnostic_payload = {
        name: {
            **_build_stokes_cube_payload(
                diagnostic_flux_cubes[name],
                diagnostic_circular_flux_cubes[name],
                pixel_area_arcsec2,
                frequency_hz,
            ),
            "description": {
                "absorption_impact": "Optically thin proxy minus full spectrum.",
                "razin_impact": "Low-density no-Razin proxy minus full spectrum.",
            }[name],
        }
        for name in diagnostic_flux_cubes
    }

    x_centers, y_centers = coarse_pixel_centers(
        field.x_edges, field.y_edges, config.image_nx, config.image_ny
    )
    beam_weights = gaussian_beam_weights(
        (config.image_ny, config.image_nx),
        ((config.image_nx - 1.0) / 2.0, (config.image_ny - 1.0) / 2.0),
        config.beam_fwhm_pixels,
    )

    payload = {
        "freq_hz": frequency_hz,
        "grid": {
            "x_center_code": x_centers,
            "y_center_code": y_centers,
            "pixel_size_arcsec": np.array([config.pixel_size_arcsec], dtype=np.float64),
        },
        "cell": {
            "deposited_weight": weight_map,
            "nonthermal_density_cm3": nonthermal_density_map,
            "magnetic_field_gauss": magnetic_field_map,
            "power_law_index": power_law_index_map,
            "power_law_fit_r_squared": power_law_r_squared_map,
            "power_law_fit_accepted": power_law_fit_accepted_map.astype(np.int32),
        },
        "image": {
            "full": full_image_payload,
            "beam": beam_image_payload,
        },
        "references": reference_payload,
        "diagnostics": diagnostic_payload,
        "component_info": {
            "model_note": (
                "Self-absorption and Razin effects are stored as reference or differential curves, "
                "not as additive conserved intensity components."
            ),
            "thermal_only": reference_payload["thermal_only"]["description"],
            "no_razin_density_proxy": reference_payload["no_razin_density_proxy"]["description"],
            "optically_thin_proxy": reference_payload["optically_thin_proxy"]["description"],
            "background_only": reference_payload["background_only"]["description"],
            "absorption_impact": diagnostic_payload["absorption_impact"]["description"],
            "razin_impact": diagnostic_payload["razin_impact"]["description"],
        },
        "roi": {
            "full_image": {
                "I_flux_sfu": total_flux_cube.sum(axis=(1, 2)),
                "V_flux_sfu": circular_flux_cube.sum(axis=(1, 2)),
            },
            "center_beam": {
                "weights": beam_weights,
                "I_flux_sfu": integrate_roi_spectrum(total_flux_cube, beam_weights),
                "I_specific": integrate_roi_spectrum(total_specific_cube, beam_weights),
                "I_brightness_temperature": integrate_roi_spectrum(total_tb_cube, beam_weights),
                "V_flux_sfu": integrate_roi_spectrum(circular_flux_cube, beam_weights),
                "V_specific": integrate_roi_spectrum(circular_specific_cube, beam_weights),
                "V_brightness_temperature": integrate_roi_spectrum(circular_tb_cube, beam_weights),
            },
        },
        "config": {key: value for key, value in asdict(config).items()},
        "calibration": {
            "voxel_volume_cm3": np.array(
                [coarse_voxel_volume_cm3(config.pixel_size_arcsec, config.los_depth_arcsec)],
                dtype=np.float64,
            ),
            "suggested_macro_particle_electron_count": np.array(
                [suggested_macro_particle_count], dtype=np.float64
            ),
            "global_power_law_index": np.array([global_power_law_index], dtype=np.float64),
            "global_power_law_fit_r_squared": np.array(
                [global_power_law_r_squared], dtype=np.float64
            ),
            "global_power_law_fit_accepted": np.array(
                [int(global_power_law_accepted)], dtype=np.int32
            ),
            "energy_bin_edges_ev": energy_bin_edges_ev,
        },
    }

    output_hdf5 = Path(config.output_hdf5)
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    write_emission_product(
        output_hdf5,
        EmissionHdf5Metadata(
            frame_id=180,
            transport_model=config.transport_model,
            field_source_kind="file",
            background_source_kind="constant",
        ),
        payload,
    )

    quicklook_path = Path(config.output_quicklook)
    _save_quicklook(
        quicklook_path,
        frequency_ghz,
        total_flux_cube,
        circular_flux_cube,
        convolved_total_flux_cube,
        config.quicklook_frequency_ghz,
    )
    print(
        "Macro-particle calibration: "
        f"using {config.macro_particle_electron_count:.6e} electrons per unit particle weight; "
        f"suggested value for target peak density "
        f"{config.target_peak_nonthermal_density_cm3:.6e} cm^-3 is "
        f"{suggested_macro_particle_count:.6e}"
    )
    print(
        "Derived nonthermal density range: "
        f"{float(np.min(nonthermal_density_map)):.6e} .. "
        f"{float(np.max(nonthermal_density_map)):.6e} cm^-3"
    )
    print(
        "Power-law index fit: "
        f"global delta = {global_power_law_index:.6f}, "
        f"global R^2 = {global_power_law_r_squared:.6f}, "
        f"accepted cells = {int(np.count_nonzero(power_law_fit_accepted_map))}/{power_law_fit_accepted_map.size}, "
        f"local range = "
        f"{float(np.min(power_law_index_map)):.6f} .. "
        f"{float(np.max(power_law_index_map)):.6f}"
    )
    return output_hdf5, quicklook_path


def main() -> int:
    """Run the CLI entry point."""
    arguments = _parse_arguments()
    config = _load_config(arguments)
    output_hdf5, quicklook_path = run_simple_emission(config)
    print(f"Emission HDF5 saved to {output_hdf5}")
    print(f"Quicklook figure saved to {quicklook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
