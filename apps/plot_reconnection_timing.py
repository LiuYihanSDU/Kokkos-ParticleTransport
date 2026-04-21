#!/usr/bin/env python3
"""Plot reconnection benchmark timing for Fortran, Kokkos CPU, and Kokkos GPU."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator


@dataclass(frozen=True)
class TimingSeries:
    """Timing samples for one benchmark case."""

    name: str
    color: str
    frame: np.ndarray
    particle_count: np.ndarray
    frame_seconds: np.ndarray
    elapsed_seconds: np.ndarray


@dataclass(frozen=True)
class CaseConfig:
    """Input file locations and display metadata for one benchmark case."""

    name: str
    color: str
    kind: str
    directory: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read reconnection benchmark timing logs and plot frame time versus "
            "particle count plus accumulated runtime versus frame number."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("benchmark_runs/reconnection_200"),
        help="Benchmark root containing fortran, kokkos_cpu, and kokkos_gpu directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path. Defaults to ROOT/reconnection_timing_panels.png.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Output combined timing CSV. Defaults to ROOT/reconnection_timing.csv.",
    )
    parser.add_argument(
        "--frame-interval-seconds",
        type=float,
        default=1.0,
        help="Physical seconds per MHD output frame for the accumulated-runtime x-axis.",
    )
    return parser.parse_args()


def parse_fortran_particle_counts(path: Path) -> dict[int, float]:
    """Read frame-indexed active particle counts from Fortran quick.dat."""
    if not path.exists():
        raise FileNotFoundError(path)

    counts: dict[int, float] = {}
    with path.open() as stream:
        header = next(stream, None)
        if header is None or "iframe" not in header:
            raise ValueError(f"invalid Fortran quick.dat header in {path}")
        for line_number, line in enumerate(stream, start=2):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(f"invalid quick.dat row {line_number} in {path}")
            frame = int(fields[0])
            counts[frame] = float(fields[1])
    return counts


def parse_fortran_step_times(path: Path) -> dict[int, float]:
    """Read frame-indexed step times from the Fortran run log."""
    if not path.exists():
        raise FileNotFoundError(path)

    pattern = re.compile(r"Step\s+(\d+)\s+takes\s+([0-9.+\-Ee]+)\s+seconds")
    step_seconds: dict[int, float] = {}
    with path.open(errors="replace") as stream:
        for line in stream:
            match = pattern.search(line)
            if match is None:
                continue
            frame = int(match.group(1))
            step_seconds[frame] = float(match.group(2))
    if not step_seconds:
        raise ValueError(f"no Fortran step timing records found in {path}")
    return step_seconds


def read_fortran_timing(config: CaseConfig) -> TimingSeries:
    counts = parse_fortran_particle_counts(config.directory / "quick.dat")
    step_seconds = parse_fortran_step_times(config.directory / "run.log")

    frame_values: list[int] = []
    count_values: list[float] = []
    frame_time_values: list[float] = []
    elapsed_values: list[float] = []
    elapsed = 0.0
    for frame in sorted(step_seconds):
        if frame not in counts:
            raise ValueError(
                f"Fortran frame {frame} has a step time but no quick.dat particle count"
            )
        elapsed += step_seconds[frame]
        frame_values.append(frame)
        count_values.append(counts[frame])
        frame_time_values.append(step_seconds[frame])
        elapsed_values.append(elapsed)

    return TimingSeries(
        name=config.name,
        color=config.color,
        frame=np.asarray(frame_values, dtype=int),
        particle_count=np.asarray(count_values, dtype=float),
        frame_seconds=np.asarray(frame_time_values, dtype=float),
        elapsed_seconds=np.asarray(elapsed_values, dtype=float),
    )


def read_kokkos_timing(config: CaseConfig) -> TimingSeries:
    path = config.directory / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    frame_values: list[int] = []
    count_values: list[float] = []
    frame_time_values: list[float] = []
    elapsed_values: list[float] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "frame",
            "active",
            "total_frame_seconds",
            "total_elapsed_seconds",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            frame_values.append(int(row["frame"]))
            count_values.append(float(row["active"]))
            frame_time_values.append(float(row["total_frame_seconds"]))
            elapsed_values.append(float(row["total_elapsed_seconds"]))

    if not frame_values:
        raise ValueError(f"no Kokkos timing rows found in {path}")
    return TimingSeries(
        name=config.name,
        color=config.color,
        frame=np.asarray(frame_values, dtype=int),
        particle_count=np.asarray(count_values, dtype=float),
        frame_seconds=np.asarray(frame_time_values, dtype=float),
        elapsed_seconds=np.asarray(elapsed_values, dtype=float),
    )


def read_timing_series(config: CaseConfig) -> TimingSeries:
    if config.kind == "fortran":
        return read_fortran_timing(config)
    if config.kind == "kokkos":
        return read_kokkos_timing(config)
    raise ValueError(f"unknown case kind: {config.kind}")


def write_combined_csv(
    output_path: Path, series: list[TimingSeries], frame_interval_seconds: float
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case",
                "frame",
                "physical_time_seconds",
                "particle_count",
                "frame_seconds",
                "elapsed_seconds",
            ]
        )
        for item in series:
            for frame, count, frame_seconds, elapsed_seconds in zip(
                item.frame,
                item.particle_count,
                item.frame_seconds,
                item.elapsed_seconds,
                strict=True,
            ):
                writer.writerow(
                    [
                        item.name,
                        int(frame),
                        f"{float(frame) * frame_interval_seconds:.16e}",
                        f"{count:.16e}",
                        f"{frame_seconds:.16e}",
                        f"{elapsed_seconds:.16e}",
                    ]
                )


def compact_count_label(value: float, _: int) -> str:
    """Return a compact particle-count tick label."""
    absolute = abs(value)
    if absolute >= 1.0e6:
        return f"{value / 1.0e6:g}M"
    if absolute >= 1.0e3:
        return f"{value / 1.0e3:g}k"
    return f"{value:g}"


def display_case_name(case_name: str) -> str:
    """Return the plot label for a canonical benchmark case name."""
    labels = {
        "Fortran": "LiXiaocan GPAT",
        "Kokkos CPU": "Kokkos-CPU",
        "Kokkos GPU": "Kokkos-GPU",
    }
    return labels.get(case_name, case_name)


def plot_timing(
    output_path: Path, series: list[TimingSeries], frame_interval_seconds: float
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for item in series:
        axes[0].plot(
            item.particle_count,
            item.frame_seconds,
            color=item.color,
            linewidth=1.7,
            label=display_case_name(item.name),
        )
        axes[1].plot(
            item.frame * frame_interval_seconds,
            item.elapsed_seconds / 3600.0,
            color=item.color,
            linewidth=1.7,
            label=display_case_name(item.name),
        )

    axes[0].set_title("Frame Time Scaling")
    axes[0].set_xlabel("Particle count")
    axes[0].set_ylabel("Frame time (s)")
    axes[0].xaxis.set_major_formatter(FuncFormatter(compact_count_label))
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=7))

    axes[1].set_title("Accumulated Runtime")
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("Total runtime (h)")
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=8))
    axes[1].yaxis.set_major_locator(MultipleLocator(1.0))

    axes[0].grid(True, linewidth=0.45, alpha=0.35)
    axes[1].grid(True, axis="y", which="major", linewidth=0.55, alpha=0.45)
    for axis in axes:
        axis.legend(frameon=False)

    figure.suptitle("Reconnection Benchmark Timing")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.root
    output = args.output or root / "reconnection_timing_panels.png"
    csv_output = args.csv_output or root / "reconnection_timing.csv"
    cases = [
        CaseConfig("Fortran", "tab:blue", "fortran", root / "fortran"),
        CaseConfig("Kokkos CPU", "tab:green", "kokkos", root / "kokkos_cpu"),
        CaseConfig("Kokkos GPU", "tab:orange", "kokkos", root / "kokkos_gpu"),
    ]

    series = [read_timing_series(case) for case in cases]
    write_combined_csv(csv_output, series, args.frame_interval_seconds)
    plot_timing(output, series, args.frame_interval_seconds)
    print(f"Wrote {output}")
    print(f"Wrote {csv_output}")


if __name__ == "__main__":
    main()
