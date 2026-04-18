#!/usr/bin/env python3
"""Reference-spectrum helpers for component-aware emission analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from particleEmission.microwave_backend import (
    MicrowaveSourceParameters,
    MicrowaveSpectrum,
    MicrowaveSpectrumBackend,
)


@dataclass(frozen=True)
class MicrowaveReferenceConfig:
    """Configuration for diagnostic reference-spectrum evaluation."""

    thin_depth_fraction: float = 1.0e-5
    minimum_thin_depth_arcsec: float = 8.0e-5
    low_density_thermal_1e9_cm3: float = 1.0e-6
    thermal_only_nonthermal_1e7_cm3: float = 1.0e-9


@dataclass(frozen=True)
class MicrowaveReferenceBundle:
    """Reference and differential spectra derived from one full source state."""

    full: MicrowaveSpectrum
    references: dict[str, MicrowaveSpectrum]
    diagnostics: dict[str, MicrowaveSpectrum]


def zero_spectrum(frequency_ghz: np.ndarray) -> MicrowaveSpectrum:
    """Return one zero-valued spectrum."""
    frequency = np.asarray(frequency_ghz, dtype=np.float64)
    zeros = np.zeros_like(frequency)
    return MicrowaveSpectrum(
        frequency_ghz=frequency.copy(),
        left_sfu=zeros.copy(),
        right_sfu=zeros.copy(),
        total_sfu=zeros.copy(),
        circular_sfu=zeros.copy(),
        circular_polarization_degree=zeros.copy(),
    )


def scale_spectrum(spectrum: MicrowaveSpectrum, factor: float) -> MicrowaveSpectrum:
    """Scale one spectrum by a scalar factor."""
    scale = float(factor)
    return MicrowaveSpectrum(
        frequency_ghz=spectrum.frequency_ghz.copy(),
        left_sfu=scale * np.asarray(spectrum.left_sfu, dtype=np.float64),
        right_sfu=scale * np.asarray(spectrum.right_sfu, dtype=np.float64),
        total_sfu=scale * np.asarray(spectrum.total_sfu, dtype=np.float64),
        circular_sfu=scale * np.asarray(spectrum.circular_sfu, dtype=np.float64),
        circular_polarization_degree=np.asarray(
            spectrum.circular_polarization_degree, dtype=np.float64
        ).copy(),
    )


def subtract_spectra(minuend: MicrowaveSpectrum, subtrahend: MicrowaveSpectrum) -> MicrowaveSpectrum:
    """Return the signed difference between two spectra."""
    left = np.asarray(minuend.left_sfu, dtype=np.float64) - np.asarray(
        subtrahend.left_sfu, dtype=np.float64
    )
    right = np.asarray(minuend.right_sfu, dtype=np.float64) - np.asarray(
        subtrahend.right_sfu, dtype=np.float64
    )
    total = np.asarray(minuend.total_sfu, dtype=np.float64) - np.asarray(
        subtrahend.total_sfu, dtype=np.float64
    )
    circular = np.asarray(minuend.circular_sfu, dtype=np.float64) - np.asarray(
        subtrahend.circular_sfu, dtype=np.float64
    )
    circular_degree = np.full_like(total, np.nan, dtype=np.float64)
    valid = np.isfinite(total) & (np.abs(total) > 0.0)
    circular_degree[valid] = circular[valid] / total[valid]
    return MicrowaveSpectrum(
        frequency_ghz=np.asarray(minuend.frequency_ghz, dtype=np.float64).copy(),
        left_sfu=left,
        right_sfu=right,
        total_sfu=total,
        circular_sfu=circular,
        circular_polarization_degree=circular_degree,
    )


def evaluate_reference_bundle(
    backend: MicrowaveSpectrumBackend,
    parameters: MicrowaveSourceParameters,
    frequency_ghz: np.ndarray,
    *,
    config: MicrowaveReferenceConfig | None = None,
) -> MicrowaveReferenceBundle:
    """Evaluate one full spectrum plus diagnostic reference spectra."""
    reference_config = config if config is not None else MicrowaveReferenceConfig()
    full = backend.evaluate(parameters, frequency_ghz)

    thermal_only_parameters = replace(
        parameters,
        nonthermal_density_1e7_cm3=reference_config.thermal_only_nonthermal_1e7_cm3,
    )
    thermal_only = backend.evaluate(thermal_only_parameters, frequency_ghz)

    low_density_parameters = replace(
        parameters,
        thermal_density_1e9_cm3=min(
            parameters.thermal_density_1e9_cm3, reference_config.low_density_thermal_1e9_cm3
        ),
    )
    no_razin_density_proxy = backend.evaluate(low_density_parameters, frequency_ghz)

    thin_depth_arcsec = max(
        parameters.los_depth_arcsec * reference_config.thin_depth_fraction,
        reference_config.minimum_thin_depth_arcsec,
    )
    thin_parameters = replace(parameters, los_depth_arcsec=thin_depth_arcsec)
    optically_thin_proxy = scale_spectrum(
        backend.evaluate(thin_parameters, frequency_ghz),
        parameters.los_depth_arcsec / thin_depth_arcsec,
    )

    background_only = zero_spectrum(frequency_ghz)
    references = {
        "thermal_only": thermal_only,
        "no_razin_density_proxy": no_razin_density_proxy,
        "optically_thin_proxy": optically_thin_proxy,
        "background_only": background_only,
    }
    diagnostics = {
        "absorption_impact": subtract_spectra(optically_thin_proxy, full),
        "razin_impact": subtract_spectra(no_razin_density_proxy, full),
    }
    return MicrowaveReferenceBundle(full=full, references=references, diagnostics=diagnostics)
