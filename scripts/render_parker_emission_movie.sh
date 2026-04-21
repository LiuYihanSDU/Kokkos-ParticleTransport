#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${KPT_MOVIE_PYTHON:-/tmp/kpt_postprocess_venv/bin/python}"
RUN_ROOT="${KPT_PARKER_MOVIE_RUN_ROOT:-${ROOT_DIR}/benchmark_runs/local_parker_speed_accuracy}"
FIELD_DIR="${KPT_FIELD_DIR:-${RUN_ROOT}/compact_field}"
PARTICLE_DIR="${KPT_PARTICLE_DIR:-${RUN_ROOT}/kokkos_cpu}"
BACKGROUND_DIR="${KPT_EMISSION_BACKGROUND_DIR:-}"
OUTPUT_DIR="${KPT_PARKER_MOVIE_OUTPUT:-${ROOT_DIR}/benchmark_runs/parker_emission_movie}"
START_FRAME="${KPT_MOVIE_START_FRAME:-1}"
END_FRAME="${KPT_MOVIE_END_FRAME:-200}"
FRAME_STEP="${KPT_MOVIE_FRAME_STEP:-1}"
IMAGE_NX="${KPT_MOVIE_IMAGE_NX:-128}"
IMAGE_NY="${KPT_MOVIE_IMAGE_NY:-128}"
BEAM_FWHM_PIXELS="${KPT_MOVIE_BEAM_FWHM_PIXELS:-3.0}"
FPS="${KPT_MOVIE_FPS:-12}"

args=(
    --field-dir "${FIELD_DIR}"
    --particle-dir "${PARTICLE_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --start-frame "${START_FRAME}"
    --end-frame "${END_FRAME}"
    --frame-step "${FRAME_STEP}"
    --image-nx "${IMAGE_NX}"
    --image-ny "${IMAGE_NY}"
    --beam-fwhm-pixels "${BEAM_FWHM_PIXELS}"
    --fps "${FPS}"
)

if [[ -n "${BACKGROUND_DIR}" ]]; then
    args+=(--background-dir "${BACKGROUND_DIR}")
fi

if [[ "${KPT_MOVIE_NO_MOVIE:-0}" == "1" ]]; then
    args+=(--no-movie)
fi

if [[ -n "${KPT_FFMPEG:-}" ]]; then
    args+=(--ffmpeg "${KPT_FFMPEG}")
fi

cd "${ROOT_DIR}"
"${PYTHON}" -m particleEmission.parker_emission_movie "${args[@]}"
