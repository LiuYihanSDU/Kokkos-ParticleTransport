#!/usr/bin/env python3
"""Plot every-N-frame reconnection energy spectra for the three benchmark solvers."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter


@dataclass(frozen=True)
class Spectrum:
    """One frame of particle spectrum data."""

    p_left: np.ndarray
    p_right: np.ndarray
    weight: np.ndarray

    @property
    def p_center(self) -> np.ndarray:
        return np.sqrt(self.p_left * self.p_right)

    @property
    def dlog10p(self) -> np.ndarray:
        return np.log10(self.p_right) - np.log10(self.p_left)

    @property
    def dndlog10p(self) -> np.ndarray:
        return self.weight / self.dlog10p


@dataclass(frozen=True)
class CaseConfig:
    """Input directory and reader type for one benchmark case."""

    name: str
    directory: Path
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read reconnection benchmark outputs every N frames and plot three "
            "spectrum panels using the plasma colormap."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("benchmark_runs/reconnection_200"),
        help="Benchmark root containing fortran, kokkos_gpu, and kokkos_cpu directories.",
    )
    parser.add_argument("--frame-start", type=int, default=10)
    parser.add_argument("--frame-end", type=int, default=200)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument(
        "--frame-interval-seconds",
        type=float,
        default=1.0,
        help="Physical seconds per MHD output frame for the colorbar labels.",
    )
    parser.add_argument(
        "--p0",
        type=float,
        default=0.1,
        help="Transport reference momentum corresponding to --p0-energy-kev.",
    )
    parser.add_argument(
        "--p0-energy-kev",
        type=float,
        default=1.0,
        help="Kinetic energy in keV assigned to p0 for the transport momentum scale.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path. Defaults to ROOT/reconnection_spectra_panels.png.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Output combined spectrum CSV. Defaults to ROOT/reconnection_spectra.csv.",
    )
    parser.add_argument(
        "--energy-x-min",
        type=float,
        default=None,
        help="Optional lower kinetic-energy axis limit in keV.",
    )
    parser.add_argument(
        "--energy-x-max",
        type=float,
        default=None,
        help="Optional upper kinetic-energy axis limit in keV.",
    )
    return parser.parse_args()


def frame_sequence(start: int, end: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("frame step must be positive")
    if end < start:
        raise ValueError("frame end must be greater than or equal to frame start")
    return list(range(start, end + 1, step))


def read_kokkos_spectrum(directory: Path, frame: int) -> Spectrum:
    path = directory / f"momentum_histogram_{frame:05d}.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    p_left: list[float] = []
    p_right: list[float] = []
    weight: list[float] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            p_left.append(float(row["p_left"]))
            p_right.append(float(row["p_right"]))
            weight.append(float(row["weight"]))

    return Spectrum(
        p_left=np.asarray(p_left, dtype=float),
        p_right=np.asarray(p_right, dtype=float),
        weight=np.asarray(weight, dtype=float),
    )


def find_dataset(file_handle: h5py.File, names: Iterable[str]) -> np.ndarray:
    for name in names:
        if name in file_handle:
            return np.asarray(file_handle[name], dtype=float)
    raise KeyError(f"none of these datasets exist: {', '.join(names)}")


def project_fortran_global_distribution(
    fglobal: np.ndarray, momentum_bin_count: int
) -> np.ndarray:
    axes = list(range(fglobal.ndim))
    for axis, axis_size in enumerate(fglobal.shape):
        if axis_size == momentum_bin_count:
            sum_axes = tuple(index for index in axes if index != axis)
            return np.asarray(fglobal.sum(axis=sum_axes), dtype=float)
    raise ValueError(
        f"could not identify momentum axis in fglobal shape {fglobal.shape}; "
        f"expected one axis with length {momentum_bin_count}"
    )


def read_fortran_spectrum(directory: Path, frame: int) -> Spectrum:
    path = directory / f"fdists_{frame:04d}.h5"
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as file_handle:
        p_edges = find_dataset(file_handle, ["pbins_edges_global", "pbins_edges"])
        fglobal = find_dataset(file_handle, ["fglobal"])

    if p_edges.ndim != 1 or p_edges.size < 2:
        raise ValueError(f"invalid momentum edges in {path}")
    momentum_bin_count = p_edges.size - 1
    weight = project_fortran_global_distribution(fglobal, momentum_bin_count)
    return Spectrum(
        p_left=p_edges[:-1],
        p_right=p_edges[1:],
        weight=weight,
    )


def read_case_spectrum(case: CaseConfig, frame: int) -> Spectrum:
    if case.kind == "fortran":
        return read_fortran_spectrum(case.directory, frame)
    if case.kind == "kokkos":
        return read_kokkos_spectrum(case.directory, frame)
    raise ValueError(f"unknown case kind: {case.kind}")


def write_combined_csv(
    output_path: Path,
    cases: list[CaseConfig],
    frames: list[int],
    spectra: dict[tuple[str, int], Spectrum],
    p0: float,
    p0_energy_kev: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case",
                "frame",
                "p_left",
                "p_right",
                "p_center",
                "energy_left_kev",
                "energy_right_kev",
                "energy_center_kev",
                "weight",
                "dndlog10p",
                "dndlog10e",
            ]
        )
        for case in cases:
            for frame in frames:
                spectrum = spectra.get((case.name, frame))
                if spectrum is None:
                    continue
                energy_left = kinetic_energy_kev_from_p(
                    spectrum.p_left, p0, p0_energy_kev
                )
                energy_right = kinetic_energy_kev_from_p(
                    spectrum.p_right, p0, p0_energy_kev
                )
                energy_center = np.sqrt(energy_left * energy_right)
                dlog10e = np.log10(energy_right) - np.log10(energy_left)
                dndlog10e = spectrum.weight / dlog10e
                for (
                    p_left,
                    p_right,
                    p_center,
                    e_left,
                    e_right,
                    e_center,
                    weight,
                    dndlog10p,
                    dnde,
                ) in zip(
                    spectrum.p_left,
                    spectrum.p_right,
                    spectrum.p_center,
                    energy_left,
                    energy_right,
                    energy_center,
                    spectrum.weight,
                    spectrum.dndlog10p,
                    dndlog10e,
                    strict=True,
                ):
                    writer.writerow(
                        [
                            case.name,
                            frame,
                            f"{p_left:.16e}",
                            f"{p_right:.16e}",
                            f"{p_center:.16e}",
                            f"{e_left:.16e}",
                            f"{e_right:.16e}",
                            f"{e_center:.16e}",
                            f"{weight:.16e}",
                            f"{dndlog10p:.16e}",
                            f"{dnde:.16e}",
                        ]
                    )


def compact_tick_label(value: float) -> str:
    """Return a compact decimal label for a positive log-axis tick."""
    if value >= 1.0:
        return f"{value:g}"
    return f"{value:.3g}"


def display_case_name(case_name: str) -> str:
    """Return the panel label for a canonical benchmark case name."""
    labels = {
        "Fortran": "LiXiaocan GPAT",
        "Kokkos CPU": "Kokkos-CPU",
        "Kokkos GPU": "Kokkos-GPU",
    }
    return labels.get(case_name, case_name)


def kinetic_energy_kev_from_p(
    momentum: np.ndarray, p0: float, p0_energy_kev: float
) -> np.ndarray:
    """Map dimensionless transport momentum to kinetic energy in keV."""
    if p0 <= 0.0 or p0_energy_kev <= 0.0:
        raise ValueError("p0 and p0-energy-kev must be positive")
    reference_energy = np.sqrt(1.0 + p0 * p0) - 1.0
    if reference_energy <= 0.0:
        raise ValueError("invalid p0 energy normalization")
    return (
        (np.sqrt(1.0 + momentum * momentum) - 1.0)
        / reference_energy
        * p0_energy_kev
    )


def energy_spectrum_arrays(
    spectrum: Spectrum, p0: float, p0_energy_kev: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return energy-bin left/right/center and dN/dlog10(E_keV)."""
    energy_left = kinetic_energy_kev_from_p(spectrum.p_left, p0, p0_energy_kev)
    energy_right = kinetic_energy_kev_from_p(spectrum.p_right, p0, p0_energy_kev)
    energy_center = np.sqrt(energy_left * energy_right)
    dlog10e = np.log10(energy_right) - np.log10(energy_left)
    return energy_left, energy_right, energy_center, spectrum.weight / dlog10e


def energy_axis_ticks(
    spectra: dict[tuple[str, int], Spectrum],
    p0: float,
    p0_energy_kev: float,
    energy_x_min: float | None = None,
    energy_x_max: float | None = None,
) -> tuple[float, float, list[float], list[str]]:
    """Return dense 1-2-5 energy ticks covering all spectra."""
    if (energy_x_min is None) != (energy_x_max is None):
        raise ValueError("energy-x-min and energy-x-max must be specified together")

    if energy_x_min is None or energy_x_max is None:
        left_values: list[np.ndarray] = []
        right_values: list[np.ndarray] = []
        for spectrum in spectra.values():
            energy_left, energy_right, _, _ = energy_spectrum_arrays(
                spectrum, p0, p0_energy_kev
            )
            left_values.append(energy_left[np.isfinite(energy_left)])
            right_values.append(energy_right[np.isfinite(energy_right)])
        if not left_values or not right_values:
            x_min = 1.0e-2
            x_max = 1.0e1
        else:
            x_min = min(
                float(values[values > 0.0].min())
                for values in left_values
                if np.any(values > 0.0)
            )
            x_max = max(
                float(values[values > 0.0].max())
                for values in right_values
                if np.any(values > 0.0)
            )
    else:
        if energy_x_min <= 0.0 or energy_x_max <= 0.0:
            raise ValueError("energy x-axis limits must be positive")
        if energy_x_max <= energy_x_min:
            raise ValueError("energy-x-max must be greater than energy-x-min")
        x_min = energy_x_min
        x_max = energy_x_max

    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("energy x-axis limits must be finite")

    lower_decade = int(np.floor(np.log10(x_min)))
    upper_decade = int(np.ceil(np.log10(x_max)))
    tick_values: list[float] = []
    for exponent in range(lower_decade, upper_decade + 1):
        decade = 10.0**exponent
        for mantissa in (1.0, 2.0, 5.0):
            value = mantissa * decade
            if x_min <= value <= x_max:
                tick_values.append(value)
    if x_min not in tick_values:
        tick_values.insert(0, x_min)
    if x_max not in tick_values:
        tick_values.append(x_max)
    tick_labels = [compact_tick_label(value) for value in tick_values]
    return x_min, x_max, tick_values, tick_labels


def plot_spectra(
    output_path: Path,
    cases: list[CaseConfig],
    frames: list[int],
    spectra: dict[tuple[str, int], Spectrum],
    frame_interval_seconds: float,
    p0: float,
    p0_energy_kev: float,
    energy_x_min: float | None,
    energy_x_max: float | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)
    cmap = plt.get_cmap("plasma")
    frame_times = np.asarray(frames, dtype=float) * frame_interval_seconds
    norm = Normalize(vmin=float(frame_times.min()), vmax=float(frame_times.max()))
    x_min, x_max, x_ticks, x_tick_labels = energy_axis_ticks(
        spectra, p0, p0_energy_kev, energy_x_min, energy_x_max
    )

    for axis, case in zip(axes, cases, strict=True):
        plotted = 0
        for frame in frames:
            spectrum = spectra.get((case.name, frame))
            if spectrum is None:
                continue
            _, _, energy_center, dndlog10e = energy_spectrum_arrays(
                spectrum, p0, p0_energy_kev
            )
            valid = np.isfinite(dndlog10e) & (dndlog10e > 0.0)
            if not np.any(valid):
                continue
            axis.loglog(
                energy_center[valid],
                dndlog10e[valid],
                color=cmap(norm(frame * frame_interval_seconds)),
                linewidth=1.2,
                alpha=0.95,
            )
            plotted += 1
        axis.set_title(display_case_name(case.name))
        axis.set_xlabel("Kinetic energy (keV)")
        axis.set_xlim(x_min, x_max)
        axis.xaxis.set_major_locator(FixedLocator(x_ticks))
        axis.xaxis.set_major_formatter(FixedFormatter(x_tick_labels))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.tick_params(axis="x", which="major", labelsize=8, labelrotation=30)
        axis.grid(True, which="both", linewidth=0.4, alpha=0.35)
        if plotted == 0:
            axis.text(
                0.5,
                0.5,
                "No spectra found",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    axes[0].set_ylabel(r"Particle count $dN/d\log_{10}(E_{\rm keV})$")
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = figure.colorbar(
        scalar_mappable,
        ax=axes,
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("Physical time (s)")
    figure.suptitle("Reconnection Particle Energy Spectra")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.root
    output = args.output or root / "reconnection_spectra_panels.png"
    csv_output = args.csv_output or root / "reconnection_spectra.csv"
    frames = frame_sequence(args.frame_start, args.frame_end, args.frame_step)
    cases = [
        CaseConfig("Fortran", root / "fortran", "fortran"),
        CaseConfig("Kokkos CPU", root / "kokkos_cpu", "kokkos"),
        CaseConfig("Kokkos GPU", root / "kokkos_gpu", "kokkos"),
    ]

    spectra: dict[tuple[str, int], Spectrum] = {}
    for case in cases:
        for frame in frames:
            try:
                spectra[(case.name, frame)] = read_case_spectrum(case, frame)
            except FileNotFoundError as error:
                print(f"Missing spectrum file, skipped: {error}")

    write_combined_csv(csv_output, cases, frames, spectra, args.p0, args.p0_energy_kev)
    plot_spectra(
        output,
        cases,
        frames,
        spectra,
        args.frame_interval_seconds,
        args.p0,
        args.p0_energy_kev,
        args.energy_x_min,
        args.energy_x_max,
    )
    print(f"Wrote {output}")
    print(f"Wrote {csv_output}")


if __name__ == "__main__":
    main()
