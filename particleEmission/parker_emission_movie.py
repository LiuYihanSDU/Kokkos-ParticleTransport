#!/usr/bin/env python3
"""Render Parker transport snapshots into emission movie frames."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm

from .background_io import coarse_background_maps, read_emission_background_maps
from .compact_field_io import coarse_average_2d, read_compact_field_snapshot
from .emission_hdf5 import convolve_image_with_gaussian_beam, find_frequency_index
from .microwave_backend import MicrowaveSourceParameters, MicrowaveSpectrumBackend
from .particle_io import (
    ELECTRON_VOLT_ERG,
    PARTICLE_BINARY_ENDIAN_MARKER,
    PARTICLE_BINARY_MAGIC,
    PARTICLE_STATUS_ACTIVE,
    fit_power_law_index_with_quality,
    fit_power_law_index_map_with_quality,
)
from .simple_emission_runner import (
    _quantize,
    _scale_magnetic_field_map,
    estimate_macro_particle_electron_count,
    weight_map_to_nonthermal_density_cm3,
)


@dataclass
class FastParticleSnapshot:
    """Particle fields needed by the Parker movie renderer."""

    particle_count: int
    particle_capacity: int
    species: int
    rest_mass: float
    charge: float
    speed_of_light: float
    energy_scale_erg: float
    energy_source: str
    position: np.ndarray
    momentum_magnitude: np.ndarray
    weight: np.ndarray
    status: np.ndarray
    kinetic_energy_ev: np.ndarray


@dataclass(frozen=True)
class FrameInputs:
    """Resolved input paths for one movie frame."""

    frame: int
    field_path: Path
    particle_path: Path
    background_path: Path | None


DEFAULT_FREQUENCIES_GHZ = (
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


def resolve_energy_source(
    requested_source: str,
    species: int,
    rest_mass: float,
    speed_of_light: float,
    energy_scale_erg: float,
) -> str:
    """Select the energy coordinate used for particle threshold panels."""
    if requested_source != "auto":
        return requested_source
    if (
        species == 3
        and rest_mass == 1.0
        and speed_of_light == 1.0
        and energy_scale_erg == 1.0
    ):
        return "transport-p0-kinetic-kev"
    return "snapshot-kinetic"


def particle_energy_ev_from_snapshot(
    momentum_magnitude: np.ndarray,
    rest_mass: float,
    speed_of_light: float,
    energy_scale_erg: float,
    energy_source: str,
    transport_p0: float = 0.1,
    transport_p0_energy_kev: float = 1.0,
) -> np.ndarray:
    """Return the eV coordinate used by the renderer for thresholding and fits."""
    if energy_source in ("transport-p0-kinetic-kev", "transport-p0-kev"):
        if transport_p0 <= 0.0 or transport_p0_energy_kev <= 0.0:
            raise RuntimeError("transport p0 and p0 energy must be positive")
        reference_kinetic_energy = np.sqrt(1.0 + transport_p0 * transport_p0) - 1.0
        if reference_kinetic_energy <= 0.0:
            raise RuntimeError("transport p0 gives a non-positive kinetic energy")
        kinetic_energy = np.sqrt(1.0 + momentum_magnitude * momentum_magnitude) - 1.0
        return kinetic_energy / reference_kinetic_energy * transport_p0_energy_kev * 1.0e3
    if energy_source == "momentum-kev":
        return momentum_magnitude * 1.0e3
    if energy_source != "snapshot-kinetic":
        raise RuntimeError(f"unsupported particle energy source: {energy_source}")
    normalized_momentum = momentum_magnitude / (rest_mass * speed_of_light)
    gamma = np.sqrt(1.0 + normalized_momentum * normalized_momentum)
    kinetic_energy = (gamma - 1.0) * rest_mass * speed_of_light * speed_of_light
    return kinetic_energy * energy_scale_erg / ELECTRON_VOLT_ERG


def parse_frequency_list(text: str) -> tuple[float, ...]:
    """Parse a comma-separated positive frequency list in GHz."""
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values or any((not math.isfinite(value) or value <= 0.0) for value in values):
        raise argparse.ArgumentTypeError("frequency list must contain positive values")
    return tuple(sorted(set(values)))


def read_fast_particle_snapshot(
    path: str | Path,
    energy_source: str = "auto",
    transport_p0: float = 0.1,
    transport_p0_energy_kev: float = 1.0,
) -> FastParticleSnapshot:
    """Read a version-4 scalar-momentum particle snapshot with NumPy."""
    file_path = Path(path)
    with file_path.open("rb") as handle:
        magic = handle.read(8)
        if magic != PARTICLE_BINARY_MAGIC:
            raise RuntimeError(f"{file_path} is not a v3/v4 KPT particle snapshot")
        version_bytes = handle.read(4)
        marker_bytes = handle.read(4)
        marker = struct.unpack("<I", marker_bytes)[0]
        if marker != PARTICLE_BINARY_ENDIAN_MARKER:
            raise RuntimeError(f"{file_path} has unsupported endian marker")
        version = struct.unpack("<I", version_bytes)[0]
        if version != 4:
            raise RuntimeError(
                f"{file_path} uses particle snapshot version {version}; "
                "the movie renderer requires version 4"
            )
        space_dim = struct.unpack("<i", handle.read(4))[0]
        if space_dim < 1 or space_dim > 3:
            raise RuntimeError(f"{file_path} has invalid particle space_dim={space_dim}")
        species = struct.unpack("<i", handle.read(4))[0]
        particle_count = struct.unpack("<Q", handle.read(8))[0]
        particle_capacity = struct.unpack("<Q", handle.read(8))[0]
        rest_mass = struct.unpack("<d", handle.read(8))[0]
        charge = struct.unpack("<d", handle.read(8))[0]
        speed_of_light = struct.unpack("<d", handle.read(8))[0]
        energy_scale_erg = struct.unpack("<d", handle.read(8))[0]
        handle.read(8)  # energy_split_ratio
        handle.read(8)  # minimum_child_weight

        record_dtype = np.dtype(
            [
                ("particle_id", "<u8"),
                ("status", "<i4"),
                ("split_level", "<i4"),
                ("sort_key", "<i4"),
                ("position", "<f8", (space_dim,)),
                ("previous_position", "<f8", (space_dim,)),
                ("previous_step_position", "<f8", (space_dim,)),
                ("momentum_magnitude", "<f8"),
                ("mu", "<f8"),
                ("weight", "<f8"),
                ("initial_kinetic_energy", "<f8"),
            ],
            align=False,
        )
        records = np.fromfile(handle, dtype=record_dtype, count=particle_count)
        if records.size != particle_count:
            raise RuntimeError(f"{file_path} ended before all particle records were read")

    resolved_energy_source = resolve_energy_source(
        energy_source,
        species,
        rest_mass,
        speed_of_light,
        energy_scale_erg,
    )
    kinetic_energy_ev = particle_energy_ev_from_snapshot(
        records["momentum_magnitude"],
        rest_mass,
        speed_of_light,
        energy_scale_erg,
        resolved_energy_source,
        transport_p0,
        transport_p0_energy_kev,
    )

    return FastParticleSnapshot(
        particle_count=int(particle_count),
        particle_capacity=int(particle_capacity),
        species=int(species),
        rest_mass=float(rest_mass),
        charge=float(charge),
        speed_of_light=float(speed_of_light),
        energy_scale_erg=float(energy_scale_erg),
        energy_source=resolved_energy_source,
        position=records["position"],
        momentum_magnitude=records["momentum_magnitude"],
        weight=records["weight"],
        status=records["status"],
        kinetic_energy_ev=kinetic_energy_ev,
    )


def frame_file(directory: Path, prefix: str, frame: int, suffix: str) -> Path:
    """Return one zero-padded frame path."""
    return directory / f"{prefix}{frame:05d}{suffix}"


def discover_frame_inputs(args: argparse.Namespace) -> list[FrameInputs]:
    """Resolve and validate field, particle, and optional background paths."""
    field_dir = Path(args.field_dir)
    particle_dir = Path(args.particle_dir)
    background_dir = Path(args.background_dir) if args.background_dir else None
    frames: list[FrameInputs] = []
    missing: list[str] = []
    for frame in range(args.start_frame, args.end_frame + 1, args.frame_step):
        field_path = frame_file(field_dir, "field", frame, ".bin")
        particle_path = frame_file(particle_dir, "particles_", frame, ".bin")
        background_path = None
        if background_dir is not None:
            candidate = frame_file(background_dir, "emission_background_", frame, ".h5")
            if candidate.exists():
                background_path = candidate
            elif args.require_background:
                missing.append(str(candidate))
        if not field_path.exists():
            missing.append(str(field_path))
        if not particle_path.exists():
            missing.append(str(particle_path))
        frames.append(FrameInputs(frame, field_path, particle_path, background_path))

    if missing:
        preview = "\n".join(missing[:20])
        extra = "" if len(missing) <= 20 else f"\n... {len(missing) - 20} more"
        raise RuntimeError(f"missing required movie inputs:\n{preview}{extra}")
    return frames


def active_mask(snapshot: FastParticleSnapshot) -> np.ndarray:
    """Return active-particle mask."""
    return snapshot.status == PARTICLE_STATUS_ACTIVE


def deposit_weight_map(
    snapshot: FastParticleSnapshot,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    image_nx: int,
    image_ny: int,
    mask: np.ndarray,
) -> np.ndarray:
    """Deposit selected particle weights onto a 2D image grid."""
    selected = active_mask(snapshot) & mask
    histogram, _, _ = np.histogram2d(
        snapshot.position[selected, 1],
        snapshot.position[selected, 0],
        bins=(image_ny, image_nx),
        range=((y_range[0], y_range[1]), (x_range[0], x_range[1])),
        weights=snapshot.weight[selected],
    )
    return histogram.astype(np.float64)


def deposit_energy_histograms(
    snapshot: FastParticleSnapshot,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    image_nx: int,
    image_ny: int,
    energy_edges_ev: np.ndarray,
) -> np.ndarray:
    """Deposit weighted energy histograms onto the image grid."""
    mask = active_mask(snapshot)
    x = snapshot.position[mask, 0]
    y = snapshot.position[mask, 1]
    energy = snapshot.kinetic_energy_ev[mask]
    weights = snapshot.weight[mask]
    x_index = np.floor((x - x_range[0]) / (x_range[1] - x_range[0]) * image_nx).astype(np.int64)
    y_index = np.floor((y - y_range[0]) / (y_range[1] - y_range[0]) * image_ny).astype(np.int64)
    energy_index = np.searchsorted(energy_edges_ev, energy, side="right") - 1
    valid = (
        (x_index >= 0)
        & (x_index < image_nx)
        & (y_index >= 0)
        & (y_index < image_ny)
        & (energy_index >= 0)
        & (energy_index < energy_edges_ev.size - 1)
    )
    histogram = np.zeros((image_ny, image_nx, energy_edges_ev.size - 1), dtype=np.float64)
    flat_index = (
        y_index[valid] * (image_nx * (energy_edges_ev.size - 1))
        + x_index[valid] * (energy_edges_ev.size - 1)
        + energy_index[valid]
    )
    histogram.flat[: histogram.size] = np.bincount(
        flat_index,
        weights=weights[valid],
        minlength=histogram.size,
    )[: histogram.size]
    return histogram


def compute_jz_code_units(field) -> np.ndarray:
    """Compute Jz proxy dBy/dx - dBx/dy in compact-field code units."""
    x_centers = field.x_centers
    y_centers = field.y_centers
    dby_dx = np.gradient(field.by, x_centers, axis=1, edge_order=2)
    dbx_dy = np.gradient(field.bx, y_centers, axis=0, edge_order=2)
    return dby_dx - dbx_dy


def positive_log_norm(values: np.ndarray) -> LogNorm | None:
    """Return a robust positive LogNorm for one image, or None for empty images."""
    positive = np.asarray(values)[np.asarray(values) > 0.0]
    if positive.size == 0:
        return None
    vmax = float(np.nanmax(positive))
    vmin = max(float(np.nanpercentile(positive, 1.0)), vmax * 1.0e-5)
    if not math.isfinite(vmin) or vmin <= 0.0 or vmin >= vmax:
        vmin = max(vmax * 1.0e-5, 1.0e-300)
    return LogNorm(vmin=vmin, vmax=vmax)


def symmetric_norm(values: np.ndarray) -> SymLogNorm:
    """Return a symmetric norm for signed field-like images."""
    finite = np.asarray(values)[np.isfinite(values)]
    vmax = float(np.nanpercentile(np.abs(finite), 99.5)) if finite.size else 1.0
    vmax = max(vmax, 1.0e-12)
    return SymLogNorm(linthresh=vmax * 1.0e-3, vmin=-vmax, vmax=vmax)


def frequency_index(frequency_ghz: np.ndarray, target_ghz: float) -> int:
    """Return the nearest index for a GHz target."""
    return find_frequency_index(frequency_ghz, target_ghz)


def build_energy_edges(snapshot: FastParticleSnapshot, bin_count: int) -> np.ndarray:
    """Build logarithmic energy bin edges from active particles."""
    energy = snapshot.kinetic_energy_ev[active_mask(snapshot)]
    positive = energy[np.isfinite(energy) & (energy > 0.0)]
    if positive.size == 0:
        raise RuntimeError("no active particles with positive kinetic energy")
    emin = float(np.min(positive))
    emax = float(np.max(positive))
    if not emax > emin:
        emax = emin * 1.001
    return np.logspace(np.log10(emin), np.log10(emax), bin_count + 1)


def synthesize_emission(
    args: argparse.Namespace,
    backend: MicrowaveSpectrumBackend,
    field,
    snapshot: FastParticleSnapshot,
    weight_map: np.ndarray,
    thermal_density_map: np.ndarray,
    temperature_mk_map: np.ndarray,
    frequency_ghz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Synthesize beam-convolved intensity cubes and a domain spectrum."""
    energy_edges = build_energy_edges(snapshot, args.energy_histogram_bins)
    energy_histograms = deposit_energy_histograms(
        snapshot,
        (float(field.x_edges[0]), float(field.x_edges[-1])),
        (float(field.y_edges[0]), float(field.y_edges[-1])),
        args.image_nx,
        args.image_ny,
        energy_edges,
    )
    global_delta, _global_r2, _global_ok = fit_power_law_index_with_quality(
        energy_histograms.sum(axis=(0, 1)),
        energy_edges,
        args.power_law_index_default,
        minimum_populated_bins=args.minimum_power_law_fit_bins,
        index_bounds=(args.power_law_index_min, args.power_law_index_max),
        minimum_r_squared=args.minimum_power_law_fit_r_squared,
    )
    power_law_index_map, _r2_map, _accepted_map = fit_power_law_index_map_with_quality(
        energy_histograms,
        energy_edges,
        global_delta,
        minimum_populated_bins=args.minimum_power_law_fit_bins,
        index_bounds=(args.power_law_index_min, args.power_law_index_max),
        minimum_r_squared=args.minimum_power_law_fit_r_squared,
    )

    nonthermal_density_map = weight_map_to_nonthermal_density_cm3(
        weight_map,
        args.macro_particle_electron_count,
        args.pixel_size_arcsec,
        args.los_depth_arcsec,
    )
    suggested_macro_count = estimate_macro_particle_electron_count(
        weight_map,
        args.pixel_size_arcsec,
        args.los_depth_arcsec,
        args.target_peak_nonthermal_density_cm3,
    )
    bmag_map = coarse_average_2d(
        field.magnetic_field_magnitude,
        field.x_edges,
        field.y_edges,
        args.image_nx,
        args.image_ny,
    )
    magnetic_field_map = _scale_magnetic_field_map(
        bmag_map, args.magnetic_field_floor_gauss, args.magnetic_field_peak_gauss
    )

    occupied = weight_map > 0.0
    flux_cube = np.zeros((frequency_ghz.size, args.image_ny, args.image_nx), dtype=np.float64)
    if not np.any(occupied):
        return flux_cube, flux_cube.copy(), np.zeros(frequency_ghz.size), suggested_macro_count

    nnth_norm = np.zeros_like(nonthermal_density_map)
    nnth_norm[occupied] = nonthermal_density_map[occupied] / np.max(nonthermal_density_map[occupied])
    b_norm = (magnetic_field_map - np.min(magnetic_field_map)) / max(np.ptp(magnetic_field_map), 1.0e-12)
    delta_norm = (
        np.clip(power_law_index_map, args.power_law_index_min, args.power_law_index_max)
        - args.power_law_index_min
    ) / max(args.power_law_index_max - args.power_law_index_min, 1.0e-12)
    nth_norm = (
        thermal_density_map - np.min(thermal_density_map)
    ) / max(np.ptp(thermal_density_map), 1.0e-12)
    temp_norm = (
        temperature_mk_map - np.min(temperature_mk_map)
    ) / max(np.ptp(temperature_mk_map), 1.0e-12)

    nnth_bins = _quantize(nnth_norm, args.nonthermal_density_bins)
    b_bins = _quantize(b_norm, args.magnetic_field_bins)
    delta_bins = _quantize(delta_norm, args.power_law_index_bins)
    nth_bins = _quantize(nth_norm, args.thermal_density_bins)
    temp_bins = _quantize(temp_norm, args.temperature_bins)

    spectral_cache: dict[tuple[int, int, int, int, int], np.ndarray] = {}
    for iy in range(args.image_ny):
        for ix in range(args.image_nx):
            if not occupied[iy, ix]:
                continue
            cache_key = (
                int(nnth_bins[iy, ix]),
                int(b_bins[iy, ix]),
                int(delta_bins[iy, ix]),
                int(nth_bins[iy, ix]),
                int(temp_bins[iy, ix]),
            )
            if cache_key not in spectral_cache:
                parameters = MicrowaveSourceParameters(
                    nonthermal_density_1e7_cm3=nonthermal_density_map[iy, ix] / 1.0e7,
                    magnetic_field_100g=magnetic_field_map[iy, ix] / 100.0,
                    viewing_angle_deg=args.viewing_angle_deg,
                    thermal_density_1e9_cm3=thermal_density_map[iy, ix] / 1.0e9,
                    power_law_index=power_law_index_map[iy, ix],
                    maximum_energy_mev=args.maximum_energy_mev,
                    temperature_mk=temperature_mk_map[iy, ix],
                    minimum_energy_mev=args.minimum_energy_mev,
                    pixel_area_arcsec2=args.pixel_size_arcsec * args.pixel_size_arcsec,
                    los_depth_arcsec=args.los_depth_arcsec,
                )
                spectral_cache[cache_key] = backend.evaluate(parameters, frequency_ghz).total_sfu
            flux_cube[:, iy, ix] = spectral_cache[cache_key]

    convolved_cube = np.zeros_like(flux_cube)
    for index in range(frequency_ghz.size):
        convolved_cube[index] = convolve_image_with_gaussian_beam(
            flux_cube[index], args.beam_fwhm_pixels
        )
    return flux_cube, convolved_cube, flux_cube.sum(axis=(1, 2)), suggested_macro_count


def render_frame(
    args: argparse.Namespace,
    backend: MicrowaveSpectrumBackend,
    frame_input: FrameInputs,
    sequence_index: int,
    frequency_ghz: np.ndarray,
    output_frames_dir: Path,
) -> dict[str, float | int | str]:
    """Render one movie panel frame and return summary metadata."""
    field = read_compact_field_snapshot(frame_input.field_path)
    snapshot = read_fast_particle_snapshot(
        frame_input.particle_path,
        energy_source=args.energy_source,
        transport_p0=args.transport_p0,
        transport_p0_energy_kev=args.transport_p0_energy_kev,
    )
    threshold_ev = args.energy_threshold_kev * 1.0e3
    energy_label = (
        "Transport kinetic energy"
        if snapshot.energy_source in ("transport-p0-kinetic-kev", "transport-p0-kev")
        else "Transport p"
        if snapshot.energy_source == "momentum-kev"
        else "Kinetic energy"
    )
    x_range = (float(field.x_edges[0]), float(field.x_edges[-1]))
    y_range = (float(field.y_edges[0]), float(field.y_edges[-1]))

    below_map = deposit_weight_map(
        snapshot,
        x_range,
        y_range,
        args.image_nx,
        args.image_ny,
        snapshot.kinetic_energy_ev < threshold_ev,
    )
    above_map = deposit_weight_map(
        snapshot,
        x_range,
        y_range,
        args.image_nx,
        args.image_ny,
        snapshot.kinetic_energy_ev >= threshold_ev,
    )
    weight_map = below_map + above_map

    bz_map = coarse_average_2d(field.bz, field.x_edges, field.y_edges, args.image_nx, args.image_ny)
    jz_map = coarse_average_2d(
        compute_jz_code_units(field),
        field.x_edges,
        field.y_edges,
        args.image_nx,
        args.image_ny,
    )

    if frame_input.background_path is not None:
        background = read_emission_background_maps(frame_input.background_path)
        thermal_density_map, temperature_mk_map = coarse_background_maps(
            background, args.image_nx, args.image_ny
        )
        background_label = "MHD map"
    else:
        thermal_density_map = np.full(
            (args.image_ny, args.image_nx), args.thermal_density_cm3, dtype=np.float64
        )
        temperature_mk_map = np.full(
            (args.image_ny, args.image_nx), args.temperature_mk, dtype=np.float64
        )
        background_label = "constant"

    _native_cube, convolved_cube, spectrum, suggested_macro_count = synthesize_emission(
        args,
        backend,
        field,
        snapshot,
        weight_map,
        thermal_density_map,
        temperature_mk_map,
        frequency_ghz,
    )

    figure, axes = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    extent = (x_range[0], x_range[1], y_range[0], y_range[1])

    im = axes[0, 0].imshow(bz_map, origin="lower", extent=extent, cmap="RdBu_r", norm=symmetric_norm(bz_map))
    axes[0, 0].set_title(f"Frame {frame_input.frame}: Bz")
    figure.colorbar(im, ax=axes[0, 0], shrink=0.82)

    im = axes[0, 1].imshow(jz_map, origin="lower", extent=extent, cmap="PuOr_r", norm=symmetric_norm(jz_map))
    axes[0, 1].set_title("Jz = dBy/dx - dBx/dy")
    figure.colorbar(im, ax=axes[0, 1], shrink=0.82)

    below_norm = positive_log_norm(below_map)
    im = axes[0, 2].imshow(
        below_map,
        origin="lower",
        extent=extent,
        cmap="viridis",
        norm=below_norm,
    )
    axes[0, 2].set_title(f"{energy_label} < {args.energy_threshold_kev:g} keV")
    figure.colorbar(im, ax=axes[0, 2], shrink=0.82)

    above_norm = positive_log_norm(above_map)
    im = axes[0, 3].imshow(
        above_map,
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=above_norm,
    )
    axes[0, 3].set_title(f"{energy_label} >= {args.energy_threshold_kev:g} keV")
    figure.colorbar(im, ax=axes[0, 3], shrink=0.82)

    for panel_index, target_ghz in enumerate(args.image_frequencies_ghz):
        index = frequency_index(frequency_ghz, target_ghz)
        image = convolved_cube[index]
        norm = positive_log_norm(image)
        im = axes[1, panel_index].imshow(
            image,
            origin="lower",
            extent=extent,
            cmap="inferno",
            norm=norm,
        )
        axes[1, panel_index].set_title(f"{target_ghz:g} GHz, beam")
        figure.colorbar(im, ax=axes[1, panel_index], shrink=0.82)

    spectrum_axis = axes[1, 3]
    spectrum_axis.loglog(frequency_ghz, spectrum, color="black", marker="o", ms=3)
    for target_ghz in args.image_frequencies_ghz:
        spectrum_axis.axvline(target_ghz, color="tab:red", alpha=0.35, lw=1)
    spectrum_axis.set_xlabel("Frequency [GHz]")
    spectrum_axis.set_ylabel("Full-domain flux [sfu]")
    spectrum_axis.set_title("Full-domain spectrum")
    spectrum_axis.grid(True, which="both", alpha=0.25)

    for axis in axes.flat[:7]:
        axis.set_xlabel("x [code]")
        axis.set_ylabel("y [code]")

    figure.suptitle(
        f"Parker emission synthesis, frame {frame_input.frame}, "
        f"{args.image_nx}x{args.image_ny}, beam FWHM={args.beam_fwhm_pixels:g} px, "
        f"thermal background={background_label}, energy source={snapshot.energy_source}",
        fontsize=13,
    )
    output_path = output_frames_dir / f"frame_{sequence_index:05d}.png"
    figure.savefig(output_path, dpi=args.dpi)
    plt.close(figure)

    return {
        "sequence_index": sequence_index,
        "frame": frame_input.frame,
        "particle_count": snapshot.particle_count,
        "energy_source": snapshot.energy_source,
        "active_weight": float(np.sum(weight_map)),
        "below_threshold_weight": float(np.sum(below_map)),
        "above_threshold_weight": float(np.sum(above_map)),
        "suggested_macro_particle_electron_count": float(suggested_macro_count),
        "spectrum_peak_sfu": float(np.max(spectrum)),
        "output_png": str(output_path),
    }


def write_frame_index(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """Write per-frame movie metadata."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_ffmpeg(args: argparse.Namespace, frames_dir: Path, output_movie: Path) -> bool:
    """Encode the PNG sequence into a ProRes MOV with ffmpeg."""
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = ""
    if not ffmpeg:
        message = "ffmpeg was not found in PATH; PNG frames were generated only."
        if args.require_movie:
            raise RuntimeError(message)
        print(message)
        return False
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-c:v",
        "prores_ks",
        "-profile:v",
        "3",
        "-pix_fmt",
        "yuv422p10le",
        str(output_movie),
    ]
    subprocess.run(command, check=True)
    return True


def parse_arguments() -> argparse.Namespace:
    """Parse CLI options for the movie renderer."""
    parser = argparse.ArgumentParser(
        description="Render Parker transport particle snapshots into emission movie frames."
    )
    parser.add_argument("--field-dir", required=True)
    parser.add_argument("--particle-dir", required=True)
    parser.add_argument("--background-dir", default="")
    parser.add_argument("--output-dir", default="benchmark_runs/parker_emission_movie")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int, default=200)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--image-nx", type=int, default=128)
    parser.add_argument("--image-ny", type=int, default=128)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--beam-fwhm-pixels", type=float, default=3.0)
    parser.add_argument("--energy-threshold-kev", type=float, default=1.5)
    parser.add_argument(
        "--energy-source",
        choices=(
            "auto",
            "snapshot-kinetic",
            "momentum-kev",
            "transport-p0-kinetic-kev",
            "transport-p0-kev",
        ),
        default="auto",
        help=(
            "energy coordinate for threshold panels; auto maps placeholder "
            "reconnection snapshots through the relativistic transport p0 kinetic energy"
        ),
    )
    parser.add_argument("--transport-p0", type=float, default=0.1)
    parser.add_argument("--transport-p0-energy-kev", type=float, default=1.0)
    parser.add_argument(
        "--frequencies-ghz",
        type=parse_frequency_list,
        default=DEFAULT_FREQUENCIES_GHZ,
    )
    parser.add_argument(
        "--image-frequencies-ghz",
        type=parse_frequency_list,
        default=(1.0, 3.0, 5.0),
    )
    parser.add_argument("--macro-particle-electron-count", type=float, default=1.0e30)
    parser.add_argument("--target-peak-nonthermal-density-cm3", type=float, default=1.0e8)
    parser.add_argument("--thermal-density-cm3", type=float, default=5.0e9)
    parser.add_argument("--temperature-mk", type=float, default=8.0)
    parser.add_argument("--los-depth-arcsec", type=float, default=8.0)
    parser.add_argument("--pixel-size-arcsec", type=float, default=1.0)
    parser.add_argument("--viewing-angle-deg", type=float, default=75.0)
    parser.add_argument("--power-law-index-default", type=float, default=4.5)
    parser.add_argument("--power-law-index-min", type=float, default=2.0)
    parser.add_argument("--power-law-index-max", type=float, default=8.0)
    parser.add_argument("--minimum-power-law-fit-bins", type=int, default=4)
    parser.add_argument("--minimum-power-law-fit-r-squared", type=float, default=0.5)
    parser.add_argument("--energy-histogram-bins", type=int, default=12)
    parser.add_argument("--minimum-energy-mev", type=float, default=0.02)
    parser.add_argument("--maximum-energy-mev", type=float, default=5.0)
    parser.add_argument("--magnetic-field-floor-gauss", type=float, default=80.0)
    parser.add_argument("--magnetic-field-peak-gauss", type=float, default=220.0)
    parser.add_argument("--magnetic-field-bins", type=int, default=8)
    parser.add_argument("--nonthermal-density-bins", type=int, default=8)
    parser.add_argument("--power-law-index-bins", type=int, default=8)
    parser.add_argument("--thermal-density-bins", type=int, default=4)
    parser.add_argument("--temperature-bins", type=int, default=4)
    parser.add_argument("--require-background", action="store_true")
    parser.add_argument("--no-movie", action="store_true")
    parser.add_argument("--require-movie", action="store_true")
    parser.add_argument("--ffmpeg", default="")
    parser.add_argument("--movie-name", default="parker_emission_movie.mov")
    args = parser.parse_args()

    if args.start_frame > args.end_frame:
        raise SystemExit("--start-frame must be <= --end-frame")
    if args.frame_step <= 0:
        raise SystemExit("--frame-step must be positive")
    if len(args.image_frequencies_ghz) != 3:
        raise SystemExit("--image-frequencies-ghz must contain exactly three values")
    combined = tuple(sorted(set((*args.frequencies_ghz, *args.image_frequencies_ghz))))
    args.frequencies_ghz = combined
    return args


def main() -> int:
    """Run the movie renderer."""
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_inputs = discover_frame_inputs(args)
    frequency_ghz = np.asarray(args.frequencies_ghz, dtype=np.float64)
    backend = MicrowaveSpectrumBackend()

    rows: list[dict[str, float | int | str]] = []
    for sequence_index, frame_input in enumerate(frame_inputs):
        print(f"Rendering frame {frame_input.frame} ({sequence_index + 1}/{len(frame_inputs)})")
        rows.append(
            render_frame(
                args,
                backend,
                frame_input,
                sequence_index,
                frequency_ghz,
                frames_dir,
            )
        )
    write_frame_index(output_dir / "frame_index.csv", rows)
    if not args.no_movie:
        run_ffmpeg(args, frames_dir, output_dir / args.movie_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
