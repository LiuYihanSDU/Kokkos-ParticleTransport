#!/usr/bin/env python3
"""Particle snapshot readers and deposition helpers for particle emission."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PARTICLE_BINARY_MAGIC = b"KPTPRT\0\0"
LEGACY_PARTICLE_BINARY_MAGIC_V2 = b"KPTPRT1\0"
PARTICLE_BINARY_VERSION = 4
PARTICLE_BINARY_MIN_SUPPORTED_VERSION = 2
PARTICLE_BINARY_ENDIAN_MARKER = 0x01020304
ELECTRON_VOLT_ERG = 1.602176634e-12
PARTICLE_STATUS_ACTIVE = 0


@dataclass
class ParticleSnapshot:
    """Minimal particle snapshot fields needed by the emission prototype."""

    version: int
    space_dim: int
    species: int
    particle_count: int
    particle_capacity: int
    rest_mass: float
    charge: float
    speed_of_light: float
    energy_scale_erg: float
    position: np.ndarray
    momentum_magnitude: np.ndarray
    mu: np.ndarray
    weight: np.ndarray
    status: np.ndarray
    kinetic_energy_ev: np.ndarray


class BinaryReader:
    """Small binary reader with explicit endian handling."""

    def __init__(self, path: Path) -> None:
        self.file = path.open("rb")
        self.endian = "<"

    def close(self) -> None:
        self.file.close()

    def read_exact(self, size: int) -> bytes:
        data = self.file.read(size)
        if len(data) != size:
            raise RuntimeError("unexpected end of particle binary snapshot")
        return data

    def read_scalar(self, fmt: str):
        return struct.unpack(self.endian + fmt, self.read_exact(struct.calcsize(fmt)))[0]


def detect_endian(version_bytes: bytes, marker_bytes: bytes) -> tuple[str, int]:
    """Return file endian and decoded binary version."""
    marker_little = struct.unpack("<I", marker_bytes)[0]
    if marker_little == PARTICLE_BINARY_ENDIAN_MARKER:
        return "<", struct.unpack("<I", version_bytes)[0]
    if marker_little == 0x04030201:
        return ">", struct.unpack(">I", version_bytes)[0]
    raise RuntimeError("unsupported particle binary endian marker")


def read_particle_snapshot(path: str | Path) -> ParticleSnapshot:
    """Read the current repository particle snapshot format."""
    reader = BinaryReader(Path(path))
    try:
        magic = reader.read_exact(8)
        if magic not in (PARTICLE_BINARY_MAGIC, LEGACY_PARTICLE_BINARY_MAGIC_V2):
            raise RuntimeError("unrecognized particle binary magic")

        version_bytes = reader.read_exact(4)
        marker_bytes = reader.read_exact(4)
        reader.endian, version = detect_endian(version_bytes, marker_bytes)
        if not (PARTICLE_BINARY_MIN_SUPPORTED_VERSION <= version <= PARTICLE_BINARY_VERSION):
            raise RuntimeError(f"unsupported particle binary version: {version}")

        space_dim = reader.read_scalar("i")
        species = reader.read_scalar("i")
        particle_count = reader.read_scalar("Q")
        particle_capacity = particle_count
        if version >= 3:
            particle_capacity = reader.read_scalar("Q")
        rest_mass = reader.read_scalar("d")
        charge = reader.read_scalar("d")
        speed_of_light = reader.read_scalar("d")
        energy_scale_erg = reader.read_scalar("d")
        if version >= 3:
            reader.read_scalar("d")  # energy_split_ratio
            reader.read_scalar("d")  # minimum_child_weight

        position = np.zeros((particle_count, 3), dtype=np.float64)
        momentum_magnitude = np.empty(particle_count, dtype=np.float64)
        mu = np.empty(particle_count, dtype=np.float64)
        weight = np.empty(particle_count, dtype=np.float64)
        status = np.empty(particle_count, dtype=np.int32)

        for index in range(particle_count):
            reader.read_scalar("Q")  # particle_id
            status[index] = reader.read_scalar("i")
            reader.read_scalar("i")  # split_level
            reader.read_scalar("i")  # sort_key
            for dim in range(space_dim):
                position[index, dim] = reader.read_scalar("d")
            for _ in range(space_dim):
                reader.read_scalar("d")  # previous_position
            if version >= 3:
                for _ in range(space_dim):
                    reader.read_scalar("d")  # previous_step_position
            if version >= 4:
                momentum_magnitude[index] = reader.read_scalar("d")
            else:
                momentum = np.array([reader.read_scalar("d") for _ in range(space_dim)])
                momentum_magnitude[index] = np.linalg.norm(momentum)
            mu[index] = reader.read_scalar("d")
            weight[index] = reader.read_scalar("d")
            reader.read_scalar("d")  # initial_kinetic_energy

        normalized_momentum = momentum_magnitude / (rest_mass * speed_of_light)
        gamma = np.sqrt(1.0 + normalized_momentum * normalized_momentum)
        kinetic_energy = (gamma - 1.0) * rest_mass * speed_of_light * speed_of_light
        kinetic_energy_ev = kinetic_energy * energy_scale_erg / ELECTRON_VOLT_ERG

        return ParticleSnapshot(
            version=version,
            space_dim=space_dim,
            species=species,
            particle_count=particle_count,
            particle_capacity=particle_capacity,
            rest_mass=rest_mass,
            charge=charge,
            speed_of_light=speed_of_light,
            energy_scale_erg=energy_scale_erg,
            position=position,
            momentum_magnitude=momentum_magnitude,
            mu=mu,
            weight=weight,
            status=status,
            kinetic_energy_ev=kinetic_energy_ev,
        )
    finally:
        reader.close()


def active_particle_mask(snapshot: ParticleSnapshot, minimum_energy_ev: float = 0.0) -> np.ndarray:
    """Return a boolean mask for active particles above a minimum kinetic energy."""
    if not math.isfinite(minimum_energy_ev) or minimum_energy_ev < 0.0:
        raise ValueError("minimum_energy_ev must be finite and non-negative")
    return (snapshot.status == PARTICLE_STATUS_ACTIVE) & (
        snapshot.kinetic_energy_ev >= minimum_energy_ev
    )


def deposit_particles_to_image_grid(
    snapshot: ParticleSnapshot,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    image_nx: int,
    image_ny: int,
    minimum_energy_ev: float = 0.0,
) -> np.ndarray:
    """Deposit active particle weights onto one coarse image grid."""
    mask = active_particle_mask(snapshot, minimum_energy_ev)
    x = snapshot.position[mask, 0]
    y = snapshot.position[mask, 1]
    weights = snapshot.weight[mask]
    histogram, _, _ = np.histogram2d(
        y,
        x,
        bins=(image_ny, image_nx),
        range=((y_range[0], y_range[1]), (x_range[0], x_range[1])),
        weights=weights,
    )
    return histogram.astype(np.float64)


def deposit_particle_energy_histograms(
    snapshot: ParticleSnapshot,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    image_nx: int,
    image_ny: int,
    energy_bin_edges_ev: np.ndarray,
    minimum_energy_ev: float = 0.0,
) -> np.ndarray:
    """Deposit weighted particle energy histograms onto one coarse image grid."""
    edges = np.asarray(energy_bin_edges_ev, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("energy_bin_edges_ev must be a 1D array with at least two entries")
    if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
        raise ValueError("energy_bin_edges_ev must be finite and strictly increasing")

    mask = active_particle_mask(snapshot, minimum_energy_ev)
    x = snapshot.position[mask, 0]
    y = snapshot.position[mask, 1]
    energy = snapshot.kinetic_energy_ev[mask]
    weights = snapshot.weight[mask]

    x_index = np.floor((x - x_range[0]) / (x_range[1] - x_range[0]) * image_nx).astype(np.int64)
    y_index = np.floor((y - y_range[0]) / (y_range[1] - y_range[0]) * image_ny).astype(np.int64)
    energy_index = np.searchsorted(edges, energy, side="right") - 1

    valid = (
        (x_index >= 0)
        & (x_index < image_nx)
        & (y_index >= 0)
        & (y_index < image_ny)
        & (energy_index >= 0)
        & (energy_index < edges.size - 1)
    )

    histogram = np.zeros((image_ny, image_nx, edges.size - 1), dtype=np.float64)
    flat_index = (
        y_index[valid] * (image_nx * (edges.size - 1))
        + x_index[valid] * (edges.size - 1)
        + energy_index[valid]
    )
    histogram.flat[: histogram.size] = np.bincount(
        flat_index,
        weights=weights[valid],
        minlength=histogram.size,
    )[: histogram.size]
    return histogram


def fit_power_law_index_with_quality(
    histogram: np.ndarray,
    energy_bin_edges_ev: np.ndarray,
    fallback_index: float,
    minimum_populated_bins: int = 4,
    index_bounds: tuple[float, float] = (2.0, 8.0),
    minimum_r_squared: float = 0.5,
) -> tuple[float, float, bool]:
    """Fit `dN/dE ~ E^{-delta}` from one weighted energy histogram."""
    counts = np.asarray(histogram, dtype=np.float64)
    edges = np.asarray(energy_bin_edges_ev, dtype=np.float64)
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    differential = np.zeros_like(counts)
    positive = counts > 0.0
    differential[positive] = counts[positive] / widths[positive]

    fit_mask = positive & np.isfinite(differential) & (differential > 0.0)
    if int(np.count_nonzero(fit_mask)) < minimum_populated_bins:
        return float(np.clip(fallback_index, index_bounds[0], index_bounds[1])), 0.0, False

    x = np.log10(centers[fit_mask])
    y = np.log10(differential[fit_mask])
    slope, _intercept = np.polyfit(x, y, deg=1)
    y_fit = slope * x + _intercept
    residual_sum = float(np.sum((y - y_fit) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 0.0
    delta = -float(slope)
    if (
        not np.isfinite(delta)
        or not np.isfinite(r_squared)
        or r_squared < minimum_r_squared
        or delta < index_bounds[0]
        or delta > index_bounds[1]
    ):
        return float(np.clip(fallback_index, index_bounds[0], index_bounds[1])), r_squared, False
    return float(delta), r_squared, True


def fit_power_law_index(
    histogram: np.ndarray,
    energy_bin_edges_ev: np.ndarray,
    fallback_index: float,
    minimum_populated_bins: int = 4,
    index_bounds: tuple[float, float] = (2.0, 8.0),
    minimum_r_squared: float = 0.5,
) -> float:
    """Fit `dN/dE ~ E^{-delta}` from one weighted energy histogram."""
    return fit_power_law_index_with_quality(
        histogram,
        energy_bin_edges_ev,
        fallback_index,
        minimum_populated_bins=minimum_populated_bins,
        index_bounds=index_bounds,
        minimum_r_squared=minimum_r_squared,
    )[0]


def fit_power_law_index_map_with_quality(
    histograms: np.ndarray,
    energy_bin_edges_ev: np.ndarray,
    fallback_index: float,
    minimum_populated_bins: int = 4,
    index_bounds: tuple[float, float] = (2.0, 8.0),
    minimum_r_squared: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one power-law index per image cell and return quality information."""
    histogram_cube = np.asarray(histograms, dtype=np.float64)
    if histogram_cube.ndim != 3:
        raise ValueError("histograms must have shape (ny, nx, nbins)")

    power_law_index = np.full(histogram_cube.shape[:2], float(fallback_index), dtype=np.float64)
    r_squared = np.zeros(histogram_cube.shape[:2], dtype=np.float64)
    accepted = np.zeros(histogram_cube.shape[:2], dtype=bool)
    for iy in range(histogram_cube.shape[0]):
        for ix in range(histogram_cube.shape[1]):
            delta, quality, ok = fit_power_law_index_with_quality(
                histogram_cube[iy, ix, :],
                energy_bin_edges_ev,
                fallback_index,
                minimum_populated_bins=minimum_populated_bins,
                index_bounds=index_bounds,
                minimum_r_squared=minimum_r_squared,
            )
            power_law_index[iy, ix] = delta
            r_squared[iy, ix] = quality
            accepted[iy, ix] = ok
    return power_law_index, r_squared, accepted


def fit_power_law_index_map(
    histograms: np.ndarray,
    energy_bin_edges_ev: np.ndarray,
    fallback_index: float,
    minimum_populated_bins: int = 4,
    index_bounds: tuple[float, float] = (2.0, 8.0),
    minimum_r_squared: float = 0.5,
) -> np.ndarray:
    """Fit one power-law index per image cell from local energy histograms."""
    return fit_power_law_index_map_with_quality(
        histograms,
        energy_bin_edges_ev,
        fallback_index,
        minimum_populated_bins=minimum_populated_bins,
        index_bounds=index_bounds,
        minimum_r_squared=minimum_r_squared,
    )[0]
