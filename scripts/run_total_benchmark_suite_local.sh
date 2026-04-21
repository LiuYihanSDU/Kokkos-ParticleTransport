#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_NAME="$(basename "$0")"
TIMESTAMP="${KPT_SUITE_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

absolute_path() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s\n' "${REPO_ROOT}/${path}"
    fi
}

usage() {
    cat <<'EOF'
Usage:
  scripts/run_total_benchmark_suite_local.sh
  scripts/run_total_benchmark_suite.sh [--no-email]

Runs the full benchmark suite locally without any built-in notification wrapper:
  1. Kokkos-particleTransport three-way reconnection benchmark
     (LiXiaocan GPAT, Kokkos-CPU, Kokkos-GPU).
  2. AMRVAC native NLFFF vs AthenaK CT-NLFFF benchmark from benchmark0419.md.

Compatibility:
  --no-email is accepted as a no-op for older launch commands.

Common overrides:
  KPT_SUITE_ROOT
  KPT_TRANSPORT_MODEL
  KPT_SPLIT_RATIO
  KPT_NLFFF_RUN_NAME
  NP
  NITER
  MF_DIAG_INTERVAL
  OUTPUT_DCYCLE
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --no-email)
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

SUITE_ROOT="$(absolute_path "${KPT_SUITE_ROOT:-benchmark_runs/combined_benchmark_${TIMESTAMP}}")"
SUITE_LOG="${KPT_SUITE_LOG:-${SUITE_ROOT}/combined_benchmark_suite.log}"

mkdir -p "${SUITE_ROOT}"
MANIFEST="${SUITE_ROOT}/combined_benchmark_manifest.txt"
PARTICLE_ROOT="$(absolute_path "${KPT_SUITE_PARTICLE_ROOT:-${SUITE_ROOT}/kpt_reconnection_three_way}")"
AMRVAC_REPO="$(absolute_path "${KPT_AMRVAC_REPO:-/home/liuyh/CLionProjects/amrvac_nlfff_sphere}")"
NLFFF_BENCHMARK_ROOT="$(absolute_path "${KPT_NLFFF_BENCHMARK_ROOT:-${SUITE_ROOT}}")"
NLFFF_RUN_NAME="${KPT_NLFFF_RUN_NAME:-nlfff_compare_0419_${TIMESTAMP}}"
FORTRAN_MHD_DIR="${KPT_FORTRAN_MHD_DIR:-/home/liuyh/data/Athena++/athena_reconnection_test/bin_data/}"
MHD_CONFIG="${KPT_MHD_CONFIG:-${FORTRAN_MHD_DIR%/}/mhd_config.dat}"

compute_frame_interval_seconds() {
    if [[ -n "${KPT_RECONNECTION_FRAME_SECONDS:-}" ]]; then
        printf '%s\n' "${KPT_RECONNECTION_FRAME_SECONDS}"
        return
    fi
    python3 - "${MHD_CONFIG}" <<'PY'
from __future__ import annotations

import math
import os
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = path.read_bytes()
if len(data) < 13 * 8:
    raise SystemExit(f"{path} is too short to contain dt_out")
dt_out = struct.unpack("<13d", data[:13 * 8])[12]
if dt_out <= 0.0:
    raise SystemExit(f"invalid dt_out={dt_out}")

override = os.environ.get("KPT_TIME_UNIT_SECONDS")
if override:
    time_unit_seconds = float(override)
else:
    length_m = float(os.environ.get("KPT_RECONNECTION_L0_M", "5.0e6"))
    magnetic_field_g = float(os.environ.get("KPT_RECONNECTION_B0_G", "50.0"))
    number_density_cm3 = float(os.environ.get("KPT_RECONNECTION_N0_CM3", "1.0e10"))
    proton_mass_g = 1.6726219e-24
    alfven_speed_cm_s = (
        magnetic_field_g
        / math.sqrt(4.0 * math.pi * number_density_cm3 * proton_mass_g)
    )
    time_unit_seconds = length_m / (alfven_speed_cm_s / 100.0)

print(f"{dt_out * time_unit_seconds:.17g}")
PY
}

FRAME_INTERVAL_SECONDS="$(compute_frame_interval_seconds)"

write_manifest() {
    local status="$1"
    cat > "${MANIFEST}" <<EOF
status=${status}
suite_root=${SUITE_ROOT}
timestamp=${TIMESTAMP}
suite_entry_script=${SCRIPT_NAME}
suite_log=${SUITE_LOG}
particle_root=${PARTICLE_ROOT}
amrvac_repo=${AMRVAC_REPO}
nlfff_benchmark_root=${NLFFF_BENCHMARK_ROOT}
nlfff_run_name=${NLFFF_RUN_NAME}
nlfff_run_dir=${NLFFF_BENCHMARK_ROOT}/${NLFFF_RUN_NAME}
fortran_mhd_dir=${FORTRAN_MHD_DIR}
mhd_config=${MHD_CONFIG}
frame_interval_seconds=${FRAME_INTERVAL_SECONDS}
particle_transport_model=${KPT_TRANSPORT_MODEL:-parker}
particle_split_ratio=${KPT_SPLIT_RATIO:-1.2}
particle_pmin_split_over_p0=${KPT_PMIN_SPLIT_OVER_P0:-2.0}
particle_histogram_interval=${KPT_HISTOGRAM_INTERVAL:-10}
updated_at=$(date -Iseconds)
EOF
}

on_exit() {
    local status=$?
    if [[ "${status}" -eq 0 ]]; then
        write_manifest 0
    else
        write_manifest "${status}"
    fi
    exit "${status}"
}
trap on_exit EXIT

write_manifest 999

exec > >(tee -a "${SUITE_LOG}") 2>&1

printf '=== Combined benchmark suite started at %s ===\n' "$(date -Iseconds)"
printf 'Suite root: %s\n' "${SUITE_ROOT}"
printf 'Particle benchmark root: %s\n' "${PARTICLE_ROOT}"
printf 'NLFFF run directory: %s/%s\n' "${NLFFF_BENCHMARK_ROOT}" "${NLFFF_RUN_NAME}"
printf 'Reconnection frame interval: %s seconds\n' "${FRAME_INTERVAL_SECONDS}"

printf '=== Stage 1: KPT three-way reconnection benchmark ===\n'
postprocess_env=()
if [[ -n "${KPT_POSTPROCESS_PYTHON:-}" ]]; then
    postprocess_env=("KPT_POSTPROCESS_PYTHON=${KPT_POSTPROCESS_PYTHON}")
elif [[ -x /tmp/kpt_postprocess_venv/bin/python ]]; then
    postprocess_env=("KPT_POSTPROCESS_PYTHON=/tmp/kpt_postprocess_venv/bin/python")
fi
env \
    KPT_BENCHMARK_ROOT="${PARTICLE_ROOT}" \
    KPT_TRANSPORT_MODEL="${KPT_TRANSPORT_MODEL:-parker}" \
    KPT_FORTRAN_BUILD_MODE="${KPT_FORTRAN_BUILD_MODE:-avx512}" \
    KPT_CPU_CORES="${KPT_CPU_CORES:-16}" \
    KPT_BUILD_JOBS="${KPT_BUILD_JOBS:-${KPT_CPU_CORES:-16}}" \
    KPT_MPI_SIZE="${KPT_MPI_SIZE:-${KPT_CPU_CORES:-16}}" \
    KPT_KOKKOS_CPU_THREADS="${KPT_KOKKOS_CPU_THREADS:-${KPT_CPU_CORES:-16}}" \
    KPT_KOKKOS_GPU_HOST_THREADS="${KPT_KOKKOS_GPU_HOST_THREADS:-1}" \
    KPT_CPU_CXX_FLAGS="${KPT_CPU_CXX_FLAGS:--O3 -march=native}" \
    KPT_START_FRAME="${KPT_START_FRAME:-0}" \
    KPT_END_FRAME="${KPT_END_FRAME:-200}" \
    KPT_HISTOGRAM_INTERVAL="${KPT_HISTOGRAM_INTERVAL:-10}" \
    KPT_PARTICLE_SNAPSHOT_INTERVAL="${KPT_PARTICLE_SNAPSHOT_INTERVAL:-0}" \
    KPT_PARTICLES_PER_RANK="${KPT_PARTICLES_PER_RANK:-1600}" \
    KPT_CXX_PARTICLE_CAPACITY="${KPT_CXX_PARTICLE_CAPACITY:-16000000}" \
    KPT_FORTRAN_PARTICLE_CAPACITY="${KPT_FORTRAN_PARTICLE_CAPACITY:-1000000}" \
    KPT_SPLIT_RATIO="${KPT_SPLIT_RATIO:-1.2}" \
    KPT_PMIN_SPLIT_OVER_P0="${KPT_PMIN_SPLIT_OVER_P0:-2.0}" \
    KPT_RECONNECTION_FRAME_SECONDS="${FRAME_INTERVAL_SECONDS}" \
    "${postprocess_env[@]}" \
    "${SCRIPT_DIR}/run_reconnection_three_cases.sh"

printf '=== Stage 2: AMRVAC native NLFFF vs AthenaK CT-NLFFF benchmark ===\n'
if [[ ! -x "${AMRVAC_REPO}/scripts/run_nlfff_compare_case.sh" ]]; then
    printf 'Missing AMRVAC benchmark entry point: %s\n' \
        "${AMRVAC_REPO}/scripts/run_nlfff_compare_case.sh" >&2
    exit 2
fi

env \
    BENCHMARK_ROOT="${NLFFF_BENCHMARK_ROOT}" \
    RUN_NAME="${NLFFF_RUN_NAME}" \
    NP="${NP:-16}" \
    NITER="${NITER:-100000}" \
    MF_DIAG_INTERVAL="${MF_DIAG_INTERVAL:-10000}" \
    OUTPUT_DCYCLE="${OUTPUT_DCYCLE:-10000}" \
    REQUIRE_AVX512="${REQUIRE_AVX512:-1}" \
    CLEAN_AMRVAC_BUILD="${CLEAN_AMRVAC_BUILD:-1}" \
    CLEAN_ATHENA_BUILD="${CLEAN_ATHENA_BUILD:-1}" \
    "${AMRVAC_REPO}/scripts/run_nlfff_compare_case.sh"

printf '=== Combined benchmark suite completed at %s ===\n' "$(date -Iseconds)"
printf 'Manifest: %s\n' "${MANIFEST}"
