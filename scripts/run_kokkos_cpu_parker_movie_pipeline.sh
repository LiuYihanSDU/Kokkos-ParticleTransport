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

run_logged() {
    local log_file="$1"
    shift
    mkdir -p "$(dirname "${log_file}")"
    {
        printf '>>>'
        printf ' %q' "$@"
        printf '\n'
    } | tee "${log_file}"
    "$@" 2>&1 | tee -a "${log_file}"
}

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$(absolute_path "${KPT_RUN_ROOT:-benchmark_runs/kokkos_cpu_parker_movie_${timestamp}}")"
CPU_BUILD_DIR="$(absolute_path "${KPT_CPU_BUILD_DIR:-cmake-build-benchmark-cpu}")"
CPU_KOKKOS_DIR="${KPT_CPU_KOKKOS_DIR:-/usr/local/kokkos_cpu/lib/cmake/Kokkos}"
CPU_CXX_FLAGS="${KPT_CPU_CXX_FLAGS:--O3 -march=native}"
CPU_CORES="${KPT_CPU_CORES:-32}"
BUILD_JOBS="${KPT_BUILD_JOBS:-${CPU_CORES}}"
KOKKOS_CPU_THREADS="${KPT_KOKKOS_CPU_THREADS:-${CPU_CORES}}"
PYTHON="${KPT_MOVIE_PYTHON:-/tmp/kpt_postprocess_venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

RESTART_PARTICLE_SNAPSHOT="${KPT_RESTART_PARTICLE_SNAPSHOT:-${KPT_KOKKOS_RESTART_PARTICLE_SNAPSHOT:-}}"
if [[ -n "${RESTART_PARTICLE_SNAPSHOT}" ]]; then
    RESTART_PARTICLE_SNAPSHOT="$(absolute_path "${RESTART_PARTICLE_SNAPSHOT}")"
fi
START_FRAME="${KPT_START_FRAME:-0}"
if [[ -n "${RESTART_PARTICLE_SNAPSHOT}" && -z "${KPT_START_FRAME:-}" ]]; then
    restart_name="$(basename "${RESTART_PARTICLE_SNAPSHOT}")"
    if [[ "${restart_name}" =~ ^particles_([0-9]+)\.bin$ ]]; then
        START_FRAME="$((10#${BASH_REMATCH[1]}))"
    else
        echo "Cannot infer KPT_START_FRAME from ${restart_name}; set KPT_START_FRAME explicitly." >&2
        exit 2
    fi
fi
END_FRAME="${KPT_END_FRAME:-200}"
MOVIE_START_FRAME="${KPT_MOVIE_START_FRAME:-$((START_FRAME + 1))}"
MOVIE_END_FRAME="${KPT_MOVIE_END_FRAME:-${END_FRAME}}"
HISTOGRAM_INTERVAL="${KPT_HISTOGRAM_INTERVAL:-10}"
PARTICLE_SNAPSHOT_INTERVAL="${KPT_PARTICLE_SNAPSHOT_INTERVAL:-1}"
PARTICLES_PER_RANK="${KPT_PARTICLES_PER_RANK:-1600}"
RANK_SCALE="${KPT_RANK_SCALE:-${CPU_CORES}}"
PARTICLE_CAPACITY="${KPT_CXX_PARTICLE_CAPACITY:-16000000}"
PARTICLE_V0="${KPT_PARTICLE_V0:-17.20195}"
DUU0="${KPT_DUU0:-5578.445}"

if [[ -n "${KPT_FIELD_DIR:-}" ]]; then
    FIELD_DIR="$(absolute_path "${KPT_FIELD_DIR}")"
    PREPARE_COMPACT_FIELD="${KPT_PREPARE_COMPACT_FIELD:-0}"
else
    FIELD_DIR="${RUN_ROOT}/compact_field"
    PREPARE_COMPACT_FIELD="${KPT_PREPARE_COMPACT_FIELD:-1}"
fi
FORTRAN_MHD_DIR="${KPT_FORTRAN_MHD_DIR:-/home/liuyh/data/Athena++/athena_reconnection_test/bin_data/}"
ATHENA_INPUT_GLOB="${KPT_ATHENA_INPUT_GLOB:-/home/liuyh/data/Athena++/athena_reconnection_test/reconnection.prim.*.athdf}"
PREPARE_BACKGROUND="${KPT_PREPARE_BACKGROUND:-1}"
BACKGROUND_DIR="$(absolute_path "${KPT_EMISSION_BACKGROUND_DIR:-${RUN_ROOT}/emission_background}")"
BACKGROUND_TARGET_NX="${KPT_BACKGROUND_TARGET_NX:-1024}"
BACKGROUND_TARGET_NY="${KPT_BACKGROUND_TARGET_NY:-1024}"
REFERENCE_DENSITY_CM3="${KPT_REFERENCE_DENSITY_CM3:-5.0e9}"
REFERENCE_TEMPERATURE_MK="${KPT_REFERENCE_TEMPERATURE_MK:-8.0}"
BACKGROUND_NORMALIZATION="${KPT_BACKGROUND_NORMALIZATION:-mean}"

KOKKOS_OUTPUT_DIR="$(absolute_path "${KPT_KOKKOS_OUTPUT_DIR:-${RUN_ROOT}/kokkos_cpu}")"
MOVIE_OUTPUT_DIR="$(absolute_path "${KPT_MOVIE_OUTPUT_DIR:-${RUN_ROOT}/movie}")"
IMAGE_NX="${KPT_MOVIE_IMAGE_NX:-128}"
IMAGE_NY="${KPT_MOVIE_IMAGE_NY:-128}"
BEAM_FWHM_PIXELS="${KPT_MOVIE_BEAM_FWHM_PIXELS:-3.0}"
FPS="${KPT_MOVIE_FPS:-12}"
RENDER_MOVIE="${KPT_RENDER_MOVIE:-1}"
RERUN_SOLVER="${KPT_RERUN_SOLVER:-1}"
ENSURE_IMAGEIO_FFMPEG="${KPT_ENSURE_IMAGEIO_FFMPEG:-1}"

mkdir -p "${RUN_ROOT}" "${KOKKOS_OUTPUT_DIR}" "${MOVIE_OUTPUT_DIR}"

cat > "${RUN_ROOT}/pipeline_manifest.txt" <<EOF
run_root=${RUN_ROOT}
cpu_build_dir=${CPU_BUILD_DIR}
cpu_kokkos_dir=${CPU_KOKKOS_DIR}
cpu_cxx_flags=${CPU_CXX_FLAGS}
cpu_cores=${CPU_CORES}
kokkos_cpu_threads=${KOKKOS_CPU_THREADS}
python=${PYTHON}
start_frame=${START_FRAME}
end_frame=${END_FRAME}
movie_start_frame=${MOVIE_START_FRAME}
movie_end_frame=${MOVIE_END_FRAME}
field_dir=${FIELD_DIR}
prepare_compact_field=${PREPARE_COMPACT_FIELD}
fortran_mhd_dir=${FORTRAN_MHD_DIR}
prepare_background=${PREPARE_BACKGROUND}
background_dir=${BACKGROUND_DIR}
athena_input_glob=${ATHENA_INPUT_GLOB}
kokkos_output_dir=${KOKKOS_OUTPUT_DIR}
movie_output_dir=${MOVIE_OUTPUT_DIR}
particle_snapshot_interval=${PARTICLE_SNAPSHOT_INTERVAL}
restart_particle_snapshot=${RESTART_PARTICLE_SNAPSHOT}
particles_per_rank=${PARTICLES_PER_RANK}
rank_scale=${RANK_SCALE}
particle_capacity=${PARTICLE_CAPACITY}
EOF

echo "Run root: ${RUN_ROOT}"
echo "This pipeline writes particle snapshots every ${PARTICLE_SNAPSHOT_INTERVAL} frame(s)."
echo "For the default 200-frame Parker run, expect roughly 50+ GB of particle snapshots."

if [[ "${ENSURE_IMAGEIO_FFMPEG}" == "1" ]]; then
    if ! "${PYTHON}" - <<'PY' >/dev/null 2>&1
import imageio_ffmpeg
PY
    then
        run_logged "${RUN_ROOT}/install_imageio_ffmpeg.log" \
            "${PYTHON}" -m pip install imageio-ffmpeg
    fi
fi

run_logged "${RUN_ROOT}/configure_cpu.log" \
    cmake -S "${REPO_ROOT}" -B "${CPU_BUILD_DIR}" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="${CPU_CXX_FLAGS}" \
    -DKokkos_DIR="${CPU_KOKKOS_DIR}"
run_logged "${RUN_ROOT}/build_cpu.log" \
    cmake --build "${CPU_BUILD_DIR}" -j "${BUILD_JOBS}"

if [[ "${PREPARE_COMPACT_FIELD}" == "1" ]]; then
    mkdir -p "${FIELD_DIR}"
    run_logged "${RUN_ROOT}/prepare_compact_field.log" \
        "${CPU_BUILD_DIR}/fortran_mhd_to_compact_field" \
        --input-dir "${FORTRAN_MHD_DIR}" \
        --output-dir "${FIELD_DIR}" \
        --start-frame "${START_FRAME}" \
        --end-frame "${END_FRAME}"
fi

if [[ "${PREPARE_BACKGROUND}" == "1" ]]; then
    mkdir -p "${BACKGROUND_DIR}"
    run_logged "${RUN_ROOT}/prepare_emission_background.log" \
        "${PYTHON}" "${REPO_ROOT}/apps/athena2bin.py" \
        --input-glob "${ATHENA_INPUT_GLOB}" \
        --start-frame "${MOVIE_START_FRAME}" \
        --end-frame "${MOVIE_END_FRAME}" \
        --target-nx "${BACKGROUND_TARGET_NX}" \
        --target-ny "${BACKGROUND_TARGET_NY}" \
        --skip-compact-output \
        --emission-background-dir "${BACKGROUND_DIR}" \
        --reference-density-cm3 "${REFERENCE_DENSITY_CM3}" \
        --reference-temperature-mk "${REFERENCE_TEMPERATURE_MK}" \
        --background-normalization "${BACKGROUND_NORMALIZATION}"
fi

walltime_args=()
if [[ -n "${KPT_KOKKOS_WALLTIME_HOURS:-${KPT_WALLTIME_HOURS:-}}" ]]; then
    walltime_hours="${KPT_KOKKOS_WALLTIME_HOURS:-${KPT_WALLTIME_HOURS:-}}"
    walltime_reserve="${KPT_KOKKOS_WALLTIME_RESERVE_MINUTES:-${KPT_WALLTIME_RESERVE_MINUTES:-30}}"
    walltime_args=(--walltime-hours "${walltime_hours}" --walltime-reserve-minutes "${walltime_reserve}")
fi
restart_args=()
if [[ -n "${RESTART_PARTICLE_SNAPSHOT}" ]]; then
    restart_args=(--restart-particle-snapshot "${RESTART_PARTICLE_SNAPSHOT}")
fi

if [[ "${RERUN_SOLVER}" == "1" ]]; then
    run_logged "${KOKKOS_OUTPUT_DIR}/console.log" \
        env \
        OMP_NUM_THREADS="${KOKKOS_CPU_THREADS}" \
        KOKKOS_NUM_THREADS="${KOKKOS_CPU_THREADS}" \
        OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
        OMP_PLACES="${OMP_PLACES:-threads}" \
        "${CPU_BUILD_DIR}/kokkos_particle_transport_app" \
        --profile fortran-global \
        --transport parker \
        "${restart_args[@]}" \
        --field-dir "${FIELD_DIR}" \
        --output-dir "${KOKKOS_OUTPUT_DIR}" \
        --start-frame "${START_FRAME}" \
        --end-frame "${END_FRAME}" \
        --particles-per-frame "${PARTICLES_PER_RANK}" \
        --rank-scale "${RANK_SCALE}" \
        --capacity "${PARTICLE_CAPACITY}" \
        --diagnostic-interval 1 \
        --histogram-interval "${HISTOGRAM_INTERVAL}" \
        --particle-snapshot-interval "${PARTICLE_SNAPSHOT_INTERVAL}" \
        --particle-v0 "${PARTICLE_V0}" \
        --duu0 "${DUU0}" \
        "${walltime_args[@]}"
fi

if [[ "${RENDER_MOVIE}" == "1" ]]; then
    movie_args=(
        --field-dir "${FIELD_DIR}"
        --particle-dir "${KOKKOS_OUTPUT_DIR}"
        --output-dir "${MOVIE_OUTPUT_DIR}"
        --start-frame "${MOVIE_START_FRAME}"
        --end-frame "${MOVIE_END_FRAME}"
        --image-nx "${IMAGE_NX}"
        --image-ny "${IMAGE_NY}"
        --beam-fwhm-pixels "${BEAM_FWHM_PIXELS}"
        --fps "${FPS}"
    )
    if [[ -d "${BACKGROUND_DIR}" ]]; then
        movie_args+=(--background-dir "${BACKGROUND_DIR}")
    fi
    run_logged "${RUN_ROOT}/render_movie.log" \
        "${PYTHON}" -m particleEmission.parker_emission_movie "${movie_args[@]}"
fi

echo "Pipeline complete."
echo "Run root: ${RUN_ROOT}"
echo "Kokkos output: ${KOKKOS_OUTPUT_DIR}"
echo "Movie output: ${MOVIE_OUTPUT_DIR}"
