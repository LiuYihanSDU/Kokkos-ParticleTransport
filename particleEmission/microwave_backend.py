#!/usr/bin/env python3
"""Thin ctypes wrapper around the local microwave backend."""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MicrowaveSourceParameters:
    """One homogeneous microwave source used by the backend."""

    nonthermal_density_1e7_cm3: float
    magnetic_field_100g: float
    viewing_angle_deg: float
    thermal_density_1e9_cm3: float
    power_law_index: float
    maximum_energy_mev: float
    temperature_mk: float
    minimum_energy_mev: float = 0.02
    pixel_area_arcsec2: float = 4.0
    los_depth_arcsec: float = 8.0


@dataclass
class MicrowaveSpectrum:
    """Backend microwave spectrum output for one source."""

    frequency_ghz: np.ndarray
    left_sfu: np.ndarray
    right_sfu: np.ndarray
    total_sfu: np.ndarray
    circular_sfu: np.ndarray
    circular_polarization_degree: np.ndarray


def backend_library_path() -> Path:
    """Return the local shared-library path used by the prototype."""
    return Path(__file__).resolve().parent / "backend" / "fit_Spectrum_Kl.so"


def backend_source_directory() -> Path:
    """Return the original Fortran source directory shipped with pygsfit_cp."""
    return Path(__file__).resolve().parent.parent / "pygsfit_cp-main" / "pygsfit_cp" / "fortran_src"


def build_backend(force: bool = False) -> Path:
    """Build the local microwave backend shared library when needed."""
    library = backend_library_path()
    if library.exists() and not force:
        return library

    source_dir = backend_source_directory()
    library.parent.mkdir(parents=True, exist_ok=True)
    sources = [
        source_dir / "fit_Spectrum_Kl.for",
        source_dir / "Calc_GS_Spec_hom.for",
        source_dir / "angular.for",
        source_dir / "spidrsub.for",
        source_dir / "FitFun.for",
        source_dir / "ResolvedSpectrum.for",
        source_dir / "klein.f",
    ]
    compile_command = [
        "gfortran",
        "-shared",
        "-fPIC",
        "-ffixed-line-length-none",
        "-O2",
        "-o",
        str(library),
        *[str(path) for path in sources],
    ]
    result = subprocess.run(
        compile_command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "microwave backend build failed:\n"
            f"{result.stdout}\n{result.stderr}".rstrip()
        )
    return library


class MicrowaveSpectrumBackend:
    """Minimal wrapper that evaluates spectra at fixed source parameters."""

    def __init__(self) -> None:
        self.library = ctypes.CDLL(str(build_backend()))
        self.function = self.library.get_mw_fit_
        self.function.restype = ctypes.c_double
        self._cache: dict[tuple[float, ...], MicrowaveSpectrum] = {}

    def evaluate(
        self,
        parameters: MicrowaveSourceParameters,
        frequency_ghz: np.ndarray,
    ) -> MicrowaveSpectrum:
        """Evaluate one fixed-parameter spectrum through the backend."""
        frequency = np.asarray(frequency_ghz, dtype=np.float64)
        key = (
            round(parameters.nonthermal_density_1e7_cm3, 6),
            round(parameters.magnetic_field_100g, 6),
            round(parameters.viewing_angle_deg, 6),
            round(parameters.thermal_density_1e9_cm3, 6),
            round(parameters.power_law_index, 6),
            round(parameters.maximum_energy_mev, 6),
            round(parameters.temperature_mk, 6),
            round(parameters.minimum_energy_mev, 6),
            round(parameters.pixel_area_arcsec2, 6),
            round(parameters.los_depth_arcsec, 6),
            *tuple(np.round(frequency, 6)),
        )
        if key in self._cache:
            cached = self._cache[key]
            return MicrowaveSpectrum(
                cached.frequency_ghz.copy(),
                cached.left_sfu.copy(),
                cached.right_sfu.copy(),
                cached.total_sfu.copy(),
                cached.circular_sfu.copy(),
                cached.circular_polarization_degree.copy(),
            )

        n_freq = int(frequency.size)
        ninput = np.array([7, 0, 1, n_freq, 1, 1], dtype=np.int32)
        rinput = np.array(
            [
                0.17,
                1.0e-6,
                1.0,
                parameters.pixel_area_arcsec2,
                parameters.los_depth_arcsec,
                parameters.minimum_energy_mev,
            ],
            dtype=np.float64,
        )

        values = [
            parameters.nonthermal_density_1e7_cm3,
            parameters.magnetic_field_100g,
            parameters.viewing_angle_deg,
            parameters.thermal_density_1e9_cm3,
            parameters.power_law_index,
            parameters.maximum_energy_mev,
            parameters.temperature_mk,
        ]
        parameter_ranges = np.zeros((15, 3), dtype=np.float64, order="F")
        for index, value in enumerate(values):
            delta = max(abs(value) * 1.0e-6, 1.0e-6)
            parameter_ranges[index, 0] = value
            parameter_ranges[index, 1] = value - delta
            parameter_ranges[index, 2] = value + delta

        dummy_flux = np.linspace(200.0, 10.0, n_freq, dtype=np.float64)
        spec_in = np.zeros((1, n_freq, 4), dtype=np.float64, order="F")
        spec_in[0, :, 0] = dummy_flux
        spec_in[0, :, 2] = 1.0

        aparms = np.zeros((1, 8), dtype=np.float64, order="F")
        eparms = np.zeros((1, 8), dtype=np.float64, order="F")
        spec_out = np.zeros((1, n_freq, 2), dtype=np.float64, order="F")

        arrays = [ninput, rinput, parameter_ranges, frequency, spec_in, aparms, eparms, spec_out]
        pointers = [array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for array in arrays]
        argv = (ctypes.POINTER(ctypes.c_double) * 8)(*pointers)
        self.function(ctypes.c_longlong(8), argv)

        left = np.asarray(spec_out[0, :, 0], dtype=np.float64)
        right = np.asarray(spec_out[0, :, 1], dtype=np.float64)
        total = left + right
        circular = right - left
        circular_degree = np.zeros_like(total)
        valid = np.abs(total) > 0.0
        circular_degree[valid] = circular[valid] / total[valid]

        spectrum = MicrowaveSpectrum(
            frequency_ghz=frequency.copy(),
            left_sfu=left,
            right_sfu=right,
            total_sfu=total,
            circular_sfu=circular,
            circular_polarization_degree=circular_degree,
        )
        self._cache[key] = spectrum
        return MicrowaveSpectrum(
            spectrum.frequency_ghz.copy(),
            spectrum.left_sfu.copy(),
            spectrum.right_sfu.copy(),
            spectrum.total_sfu.copy(),
            spectrum.circular_sfu.copy(),
            spectrum.circular_polarization_degree.copy(),
        )
