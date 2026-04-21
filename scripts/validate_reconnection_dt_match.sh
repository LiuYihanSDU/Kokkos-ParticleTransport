#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ROOT="${1:-${REPO_ROOT}/benchmark_runs/kokkos_cpu_format_example_frame180_262k}"
MHD_CONFIG="${2:-/home/liuyh/data/Athena++/athena_reconnection_test/bin_data/mhd_config.dat}"
SUMMARY_CSV="${3:-${RUN_ROOT}/kokkos_cpu/summary.csv}"

python3 - "${RUN_ROOT}" "${MHD_CONFIG}" "${SUMMARY_CSV}" <<'PY'
from __future__ import annotations

import csv
import math
import os
import re
import struct
import sys
from pathlib import Path


def read_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def read_mhd_config(path: Path) -> dict[str, float]:
    data = path.read_bytes()
    if len(data) < 13 * 8:
        raise RuntimeError(f"{path} is too short to contain mhd_config dt_out")
    values = struct.unpack("<13d", data[:13 * 8])
    names = (
        "dx", "dy", "dz",
        "xmin", "ymin", "zmin",
        "xmax", "ymax", "zmax",
        "lx", "ly", "lz",
        "dt_out",
    )
    return dict(zip(names, values))


def reconnection_time_unit_seconds() -> tuple[float, str]:
    override = os.environ.get("KPT_TIME_UNIT_SECONDS")
    if override:
        value = float(override)
        if value <= 0.0:
            raise RuntimeError("KPT_TIME_UNIT_SECONDS must be positive")
        return value, "KPT_TIME_UNIT_SECONDS"

    length_m = float(os.environ.get("KPT_RECONNECTION_L0_M", "5.0e6"))
    magnetic_field_g = float(os.environ.get("KPT_RECONNECTION_B0_G", "50.0"))
    number_density_cm3 = float(os.environ.get("KPT_RECONNECTION_N0_CM3", "1.0e10"))
    proton_mass_g = 1.6726219e-24
    alfven_speed_cm_s = (
        magnetic_field_g /
        math.sqrt(4.0 * math.pi * number_density_cm3 * proton_mass_g)
    )
    alfven_speed_m_s = alfven_speed_cm_s / 100.0
    if length_m <= 0.0 or alfven_speed_m_s <= 0.0:
        raise RuntimeError("invalid reconnection Alfven time normalization")
    source = (
        "L0/vA from KPT_RECONNECTION_L0_M,KPT_RECONNECTION_B0_G,"
        "KPT_RECONNECTION_N0_CM3 defaults"
    )
    return length_m / alfven_speed_m_s, source


def frame_from_particle_name(path: Path) -> int | None:
    match = re.fullmatch(r"particles_(\d{5})\.bin", path.name)
    return int(match.group(1)) if match else None


run_root = Path(sys.argv[1])
mhd_config = Path(sys.argv[2])
summary_csv = Path(sys.argv[3])
manifest = read_manifest(run_root / "manifest.txt")

if not run_root.exists():
    raise RuntimeError(f"run root does not exist: {run_root}")
if not summary_csv.exists():
    raise RuntimeError(f"summary CSV does not exist: {summary_csv}")

mhd_values = read_mhd_config(mhd_config)
dt_out = mhd_values["dt_out"]
if dt_out <= 0.0:
    raise RuntimeError(f"invalid dt_out in {mhd_config}: {dt_out}")
time_unit_seconds, time_unit_source = reconnection_time_unit_seconds()

rows: list[dict[str, str]] = []
with summary_csv.open(newline="") as stream:
    reader = csv.DictReader(stream)
    required = {"frame", "time"}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise RuntimeError(f"{summary_csv} is missing columns: {sorted(missing)}")
    rows = list(reader)

if not rows:
    raise RuntimeError(f"{summary_csv} has no data rows")

max_abs_error = 0.0
for row in rows:
    frame = int(row["frame"])
    actual_time = float(row["time"])
    expected_time = frame * dt_out
    tolerance = max(1.0e-12, abs(expected_time) * 1.0e-12)
    error = abs(actual_time - expected_time)
    max_abs_error = max(max_abs_error, error)
    if error > tolerance:
        raise RuntimeError(
            f"time mismatch at frame {frame}: summary time={actual_time:.17g}, "
            f"expected frame*dt_out={expected_time:.17g}, dt_out={dt_out:.17g}"
        )

start_frame = int(manifest["start_frame"]) if "start_frame" in manifest else None
end_frame = int(manifest["end_frame"]) if "end_frame" in manifest else int(rows[-1]["frame"])

field_dir = run_root / "compact_field"
if start_frame is not None:
    lower_field = field_dir / f"field{start_frame:05d}.bin"
    if not lower_field.exists():
        raise RuntimeError(f"missing lower compact field frame: {lower_field}")
upper_field = field_dir / f"field{end_frame:05d}.bin"
if not upper_field.exists():
    raise RuntimeError(f"missing upper compact field frame: {upper_field}")

particle_dir = run_root / "kokkos_cpu"
particle_frames = sorted(
    frame for path in particle_dir.glob("particles_*.bin")
    if (frame := frame_from_particle_name(path)) is not None
)
if end_frame not in particle_frames:
    raise RuntimeError(
        f"missing final particle snapshot particles_{end_frame:05d}.bin in {particle_dir}"
    )

print(f"OK: {summary_csv}")
print(f"mhd_config={mhd_config}")
print(f"dt_out_code_time={dt_out:.17g}")
print(f"time_unit_seconds={time_unit_seconds:.17g}")
print(f"time_unit_source={time_unit_source}")
print(f"frame_interval_seconds={dt_out * time_unit_seconds:.17g}")
print(f"summary_rows={len(rows)}")
print(f"last_summary_frame={rows[-1]['frame']}")
last_code_time = float(rows[-1]["time"])
print(f"last_summary_code_time={last_code_time:.17g}")
print(f"last_summary_seconds={last_code_time * time_unit_seconds:.17g}")
print(f"max_abs_time_error={max_abs_error:.3e}")
print(f"compact_field_upper={upper_field}")
print(f"particle_snapshot_final={particle_dir / f'particles_{end_frame:05d}.bin'}")
PY
