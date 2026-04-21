#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

absolute_path() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s\n' "${REPO_ROOT}/${path}"
    fi
}

POSTPROCESS_VENV="${KPT_LOCAL_POSTPROCESS_VENV:-/tmp/kpt_postprocess_venv}"
if [[ -z "${KPT_POSTPROCESS_PYTHON:-}" && "${KPT_LOCAL_PREPARE_POSTPROCESS_PYTHON:-1}" == "1" ]]; then
    if [[ ! -x "${POSTPROCESS_VENV}/bin/python" ]]; then
        python3 -m venv "${POSTPROCESS_VENV}"
    fi
    if ! "${POSTPROCESS_VENV}/bin/python" - <<'PY' >/dev/null 2>&1
import h5py
import matplotlib
import numpy
PY
    then
        "${POSTPROCESS_VENV}/bin/python" -m pip install --upgrade pip
        "${POSTPROCESS_VENV}/bin/python" -m pip install numpy h5py matplotlib
    fi
    export KPT_POSTPROCESS_PYTHON="${POSTPROCESS_VENV}/bin/python"
fi

export KPT_SEQUENCE_LOG_ROOT="${KPT_SEQUENCE_LOG_ROOT:-${REPO_ROOT}/benchmark_runs/local_parker_focused_speed_accuracy}"
export KPT_PARKER_BENCHMARK_ROOT="${KPT_PARKER_BENCHMARK_ROOT:-${REPO_ROOT}/benchmark_runs/local_parker_speed_accuracy}"
export KPT_FOCUSED_BENCHMARK_ROOT="${KPT_FOCUSED_BENCHMARK_ROOT:-${REPO_ROOT}/benchmark_runs/local_focused_speed_accuracy}"

export KPT_FORTRAN_BUILD_MODE="${KPT_FORTRAN_BUILD_MODE:-avx512}"
export KPT_CPU_CORES="${KPT_CPU_CORES:-16}"
export KPT_BUILD_JOBS="${KPT_BUILD_JOBS:-${KPT_CPU_CORES}}"
export KPT_MPI_SIZE="${KPT_MPI_SIZE:-${KPT_CPU_CORES}}"
export KPT_KOKKOS_CPU_THREADS="${KPT_KOKKOS_CPU_THREADS:-${KPT_CPU_CORES}}"
export KPT_KOKKOS_GPU_HOST_THREADS="${KPT_KOKKOS_GPU_HOST_THREADS:-1}"
export KPT_CPU_CXX_FLAGS="${KPT_CPU_CXX_FLAGS:--O3 -march=native}"

export KPT_START_FRAME="${KPT_START_FRAME:-0}"
export KPT_END_FRAME="${KPT_END_FRAME:-200}"
export KPT_HISTOGRAM_INTERVAL="${KPT_HISTOGRAM_INTERVAL:-10}"
export KPT_PARTICLES_PER_RANK="${KPT_PARTICLES_PER_RANK:-1600}"
export KPT_CXX_PARTICLE_CAPACITY="${KPT_CXX_PARTICLE_CAPACITY:-16000000}"
export KPT_FORTRAN_PARTICLE_CAPACITY="${KPT_FORTRAN_PARTICLE_CAPACITY:-1000000}"

# Keep speed comparisons focused on solver time unless explicitly requested.
export KPT_PARTICLE_SNAPSHOT_INTERVAL="${KPT_PARTICLE_SNAPSHOT_INTERVAL:-0}"

"${SCRIPT_DIR}/run_reconnection_parker_focused_comparisons.sh" "$@"

python3 - \
    "$(absolute_path "${KPT_SEQUENCE_LOG_ROOT}")" \
    "$(absolute_path "${KPT_PARKER_BENCHMARK_ROOT}")" \
    "$(absolute_path "${KPT_FOCUSED_BENCHMARK_ROOT}")" <<'PY'
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


def read_timing(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    latest: dict[str, dict[str, float]] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            case = row["case"]
            latest[case] = {
                "frame": float(row["frame"]),
                "particle_count": float(row["particle_count"]),
                "frame_seconds": float(row["frame_seconds"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
            }
    return latest


def read_spectra(path: Path) -> dict[tuple[str, int], list[dict[str, float]]]:
    spectra: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    if not path.exists():
        return spectra
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            key = (row["case"], int(row["frame"]))
            spectra[key].append(
                {
                    "p_center": float(row["p_center"]),
                    "weight": float(row["weight"]),
                    "dndlog10p": float(row["dndlog10p"]),
                }
            )
    return spectra


def compare_series(reference: list[dict[str, float]],
                   candidate: list[dict[str, float]],
                   column: str) -> tuple[float, float]:
    if not reference or not candidate:
        return math.nan, math.nan
    count = min(len(reference), len(candidate))
    ref_values = [reference[i][column] for i in range(count)]
    cand_values = [candidate[i][column] for i in range(count)]
    denom = sum(abs(value) for value in ref_values)
    abs_diff = [abs(cand_values[i] - ref_values[i]) for i in range(count)]
    relative_l1 = sum(abs_diff) / denom if denom > 0.0 else math.nan
    positive_ref = [abs(value) for value in ref_values if abs(value) > 0.0]
    floor = max(positive_ref) * 1.0e-12 if positive_ref else 1.0
    max_relative = max(
        abs_diff[i] / max(abs(ref_values[i]), floor) for i in range(count)
    )
    return relative_l1, max_relative


def write_summary(sequence_root: Path, model_roots: list[tuple[str, Path]]) -> None:
    sequence_root.mkdir(parents=True, exist_ok=True)
    summary_csv = sequence_root / "local_speed_accuracy_summary.csv"
    summary_txt = sequence_root / "local_speed_accuracy_summary.txt"
    rows: list[list[object]] = []
    text_lines: list[str] = []

    for model, root in model_roots:
        text_lines.append(f"[{model}]")
        text_lines.append(f"root={root}")
        timing = read_timing(root / "reconnection_timing.csv")
        for case in ("Fortran", "Kokkos GPU", "Kokkos CPU"):
            item = timing.get(case)
            if not item:
                text_lines.append(f"timing {case}: missing")
                continue
            fortran_elapsed = timing.get("Fortran", {}).get("elapsed_seconds", math.nan)
            speedup = (
                fortran_elapsed / item["elapsed_seconds"]
                if item["elapsed_seconds"] > 0.0 and math.isfinite(fortran_elapsed)
                else math.nan
            )
            rows.append([
                model,
                "timing",
                case,
                int(item["frame"]),
                item["particle_count"],
                item["frame_seconds"],
                item["elapsed_seconds"],
                speedup,
                "",
                "",
                "",
                "",
            ])
            text_lines.append(
                f"timing {case}: frame={int(item['frame'])} "
                f"elapsed={item['elapsed_seconds']:.6g}s "
                f"last_frame={item['frame_seconds']:.6g}s "
                f"speedup_vs_fortran={speedup:.6g}"
            )

        spectra = read_spectra(root / "reconnection_spectra.csv")
        frames = sorted({frame for case, frame in spectra if case == "Fortran"})
        if not frames:
            text_lines.append("spectra: missing reconnection_spectra.csv or Fortran rows")
        for frame in frames:
            reference = spectra.get(("Fortran", frame), [])
            for case in ("Kokkos GPU", "Kokkos CPU"):
                candidate = spectra.get((case, frame), [])
                weight_l1, weight_max = compare_series(reference, candidate, "weight")
                dnd_l1, dnd_max = compare_series(reference, candidate, "dndlog10p")
                rows.append([
                    model,
                    "spectrum",
                    case,
                    frame,
                    "",
                    "",
                    "",
                    "",
                    weight_l1,
                    weight_max,
                    dnd_l1,
                    dnd_max,
                ])
            if frame == frames[-1]:
                text_lines.append(f"spectra final_frame={frame}")
        text_lines.append("")

    with summary_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "model",
            "metric_type",
            "case",
            "frame",
            "particle_count",
            "frame_seconds",
            "elapsed_seconds",
            "speedup_vs_fortran",
            "relative_l1_weight",
            "max_relative_weight",
            "relative_l1_dndlog10p",
            "max_relative_dndlog10p",
        ])
        writer.writerows(rows)
    summary_txt.write_text("\n".join(text_lines) + "\n")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_txt}")


sequence_root = Path(sys.argv[1])
parker_root = Path(sys.argv[2])
focused_root = Path(sys.argv[3])
write_summary(sequence_root, [("parker", parker_root), ("focused", focused_root)])
PY

printf 'Local Parker/focused speed-accuracy test complete. Summary: %s\n' \
    "$(absolute_path "${KPT_SEQUENCE_LOG_ROOT}")"
