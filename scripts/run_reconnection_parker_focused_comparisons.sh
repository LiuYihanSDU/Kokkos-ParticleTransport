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

SEQUENCE_LOG_ROOT="$(absolute_path "${KPT_SEQUENCE_LOG_ROOT:-${REPO_ROOT}/benchmark_runs/reconnection_parker_focused_sequence}")"
PARKER_BENCHMARK_ROOT="$(absolute_path "${KPT_PARKER_BENCHMARK_ROOT:-${REPO_ROOT}/benchmark_runs/reconnection_optimized_200}")"
FOCUSED_BENCHMARK_ROOT="$(absolute_path "${KPT_FOCUSED_BENCHMARK_ROOT:-${REPO_ROOT}/benchmark_runs/reconnection_focused_200}")"
PARKER_FORTRAN_CONF_NAME="${KPT_PARKER_FORTRAN_CONF_NAME:-conf_reconnection_kpt_parker_benchmark.dat}"
FOCUSED_FORTRAN_CONF_NAME="${KPT_FOCUSED_FORTRAN_CONF_NAME:-conf_reconnection_kpt_focused_benchmark.dat}"

mkdir -p "${SEQUENCE_LOG_ROOT}"

cat > "${SEQUENCE_LOG_ROOT}/manifest.txt" <<EOF
sequence_log_root=${SEQUENCE_LOG_ROOT}
parker_benchmark_root=${PARKER_BENCHMARK_ROOT}
focused_benchmark_root=${FOCUSED_BENCHMARK_ROOT}
parker_fortran_conf_name=${PARKER_FORTRAN_CONF_NAME}
focused_fortran_conf_name=${FOCUSED_FORTRAN_CONF_NAME}
start_time=$(date -Iseconds)
EOF

run_case() {
    local case_name="$1"
    local transport_model="$2"
    local benchmark_root="$3"
    local fortran_conf_name="$4"
    local wrapper="$5"
    shift 5

    local log_file="${SEQUENCE_LOG_ROOT}/${case_name}.log"
    {
        printf '=== Starting %s comparison at %s ===\n' "${case_name}" "$(date -Iseconds)"
        printf 'benchmark_root=%s\n' "${benchmark_root}"
        printf 'transport_model=%s\n' "${transport_model}"
        printf 'fortran_conf_name=%s\n' "${fortran_conf_name}"
    } | tee "${log_file}"

    env \
        KPT_TRANSPORT_MODEL="${transport_model}" \
        KPT_BENCHMARK_ROOT="${benchmark_root}" \
        KPT_FORTRAN_CONF_NAME="${fortran_conf_name}" \
        "${wrapper}" "$@" 2>&1 | tee -a "${log_file}"

    printf '=== Finished %s comparison at %s ===\n' \
        "${case_name}" "$(date -Iseconds)" | tee -a "${log_file}"
}

run_case \
    parker \
    parker \
    "${PARKER_BENCHMARK_ROOT}" \
    "${PARKER_FORTRAN_CONF_NAME}" \
    "${SCRIPT_DIR}/run_reconnection_optimized_comparison.sh" \
    "$@"

run_case \
    focused \
    focused \
    "${FOCUSED_BENCHMARK_ROOT}" \
    "${FOCUSED_FORTRAN_CONF_NAME}" \
    "${SCRIPT_DIR}/run_reconnection_focused_comparison.sh" \
    "$@"

cat >> "${SEQUENCE_LOG_ROOT}/manifest.txt" <<EOF
end_time=$(date -Iseconds)
status=completed
EOF

printf 'Both reconnection comparisons completed. Logs: %s\n' "${SEQUENCE_LOG_ROOT}"
