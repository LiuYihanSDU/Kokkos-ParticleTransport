#!/usr/bin/env python3
"""Plot every-10-frame reconnection spectra for Fortran, Kokkos CUDA, and Kokkos CPU."""

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
                "weight",
                "dndlog10p",
            ]
        )
        for case in cases:
            for frame in frames:
                spectrum = spectra.get((case.name, frame))
                if spectrum is None:
                    continue
                for p_left, p_right, p_center, weight, dndlog10p in zip(
                    spectrum.p_left,
                    spectrum.p_right,
                    spectrum.p_center,
                    spectrum.weight,
                    spectrum.dndlog10p,
                    strict=True,
                ):
                    writer.writerow(
                        [
                            case.name,
                            frame,
                            f"{p_left:.16e}",
                            f"{p_right:.16e}",
                            f"{p_center:.16e}",
                            f"{weight:.16e}",
                            f"{dndlog10p:.16e}",
                        ]
                    )


def compact_tick_label(value: float) -> str:
    """Return a compact decimal label for a positive log-axis tick."""
    if value >= 1.0:
        return f"{value:g}"
    return f"{value:.3g}"


def momentum_axis_ticks(
    spectra: dict[tuple[str, int], Spectrum],
) -> tuple[float, float, list[float], list[str]]:
    """Return dense 1-2-5 momentum ticks covering all spectra."""
    p_left_values: list[np.ndarray] = []
    p_right_values: list[np.ndarray] = []
    for spectrum in spectra.values():
        p_left_values.append(spectrum.p_left[np.isfinite(spectrum.p_left)])
        p_right_values.append(spectrum.p_right[np.isfinite(spectrum.p_right)])
    if not p_left_values or not p_right_values:
        return 1.0e-2, 1.0e1, [1.0e-2, 1.0e-1, 1.0, 1.0e1], [
            "0.01",
            "0.1",
            "1",
            "10",
        ]

    p_min = min(
        float(values[values > 0.0].min())
        for values in p_left_values
        if np.any(values > 0.0)
    )
    p_max = max(
        float(values[values > 0.0].max())
        for values in p_right_values
        if np.any(values > 0.0)
    )
    lower_decade = int(np.floor(np.log10(p_min)))
    upper_decade = int(np.ceil(np.log10(p_max)))
    tick_values: list[float] = []
    for exponent in range(lower_decade, upper_decade + 1):
        decade = 10.0**exponent
        for mantissa in (1.0, 2.0, 5.0):
            value = mantissa * decade
            if p_min <= value <= p_max:
                tick_values.append(value)
    if p_max not in tick_values:
        tick_values.append(p_max)
    tick_labels = [compact_tick_label(value) for value in tick_values]
    return p_min, p_max, tick_values, tick_labels


def plot_spectra(
    output_path: Path,
    cases: list[CaseConfig],
    frames: list[int],
    spectra: dict[tuple[str, int], Spectrum],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)
    cmap = plt.get_cmap("plasma")
    norm = Normalize(vmin=min(frames), vmax=max(frames))
    x_min, x_max, x_ticks, x_tick_labels = momentum_axis_ticks(spectra)

    for axis, case in zip(axes, cases, strict=True):
        plotted = 0
        for frame in frames:
            spectrum = spectra.get((case.name, frame))
            if spectrum is None:
                continue
            valid = np.isfinite(spectrum.dndlog10p) & (spectrum.dndlog10p > 0.0)
            if not np.any(valid):
                continue
            axis.loglog(
                spectrum.p_center[valid],
                spectrum.dndlog10p[valid],
                color=cmap(norm(frame)),
                linewidth=1.2,
                alpha=0.95,
            )
            plotted += 1
        axis.set_title(case.name)
        axis.set_xlabel("p")
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

    axes[0].set_ylabel(r"$dN/d\log_{10}p$")
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = figure.colorbar(
        scalar_mappable,
        ax=axes,
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("Frame")
    figure.suptitle("Reconnection Particle Spectra")
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
        CaseConfig("Kokkos CUDA", root / "kokkos_gpu", "kokkos"),
        CaseConfig("Kokkos CPU", root / "kokkos_cpu", "kokkos"),
    ]

    spectra: dict[tuple[str, int], Spectrum] = {}
    for case in cases:
        for frame in frames:
            try:
                spectra[(case.name, frame)] = read_case_spectrum(case, frame)
            except FileNotFoundError as error:
                print(f"Missing spectrum file, skipped: {error}")

    write_combined_csv(csv_output, cases, frames, spectra)
    plot_spectra(output, cases, frames, spectra)
    print(f"Wrote {output}")
    print(f"Wrote {csv_output}")


if __name__ == "__main__":
    main()
