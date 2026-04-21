#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${KPT_EMISSION_PYTHON:-}" ]]; then
    PYTHON="${KPT_EMISSION_PYTHON}"
elif [[ -n "${KPT_POSTPROCESS_PYTHON:-}" ]]; then
    PYTHON="${KPT_POSTPROCESS_PYTHON}"
elif [[ -x "/tmp/kpt_postprocess_venv/bin/python" ]]; then
    PYTHON="/tmp/kpt_postprocess_venv/bin/python"
else
    PYTHON="python3"
fi

if [[ -n "${KPT_PARKER_RUN_ROOT:-}" ]]; then
    RUN_ROOT="${KPT_PARKER_RUN_ROOT}"
elif [[ -d "${ROOT_DIR}/kokkos_cpu_format_example_frame180_262k" ]]; then
    RUN_ROOT="${ROOT_DIR}/kokkos_cpu_format_example_frame180_262k"
else
    RUN_ROOT="${ROOT_DIR}/benchmark_runs/kokkos_cpu_format_example_frame180_262k"
fi

FIELD_DIR="${KPT_FIELD_DIR:-${RUN_ROOT}/compact_field}"
PARTICLE_DIR="${KPT_PARTICLE_DIR:-${RUN_ROOT}/kokkos_cpu}"
OUTPUT_ROOT="${KPT_PARKER_EMISSION_ROOT:-${ROOT_DIR}/benchmark_runs/parker_emission_synthesis}"
INPUT_LINK_DIR="${OUTPUT_ROOT}/inputs"
CONFIG_PATH="${OUTPUT_ROOT}/parker_emission_config.json"
MANIFEST_PATH="${OUTPUT_ROOT}/manifest.txt"
LOG_PATH="${OUTPUT_ROOT}/emission.log"

mkdir -p "${OUTPUT_ROOT}" "${INPUT_LINK_DIR}"

if [[ -n "${KPT_PARKER_EMISSION_FRAME:-}" ]]; then
    FRAME="${KPT_PARKER_EMISSION_FRAME}"
    PARTICLE_PATH="${KPT_PARTICLE_PATH:-${PARTICLE_DIR}/particles_$(printf '%05d' "${FRAME}").bin}"
else
    PARTICLE_PATH="${KPT_PARTICLE_PATH:-$(find "${PARTICLE_DIR}" -maxdepth 1 -name 'particles_*.bin' | sort | tail -n 1)}"
    if [[ -z "${PARTICLE_PATH}" ]]; then
        echo "No particle snapshot found under ${PARTICLE_DIR}" >&2
        exit 1
    fi
    FRAME="$(basename "${PARTICLE_PATH}" | sed -E 's/^particles_([0-9]+)\.bin$/\1/')"
    FRAME="$((10#${FRAME}))"
fi

FIELD_FRAME="${KPT_FIELD_FRAME:-${FRAME}}"
FIELD_PATH="${KPT_FIELD_PATH:-${FIELD_DIR}/field$(printf '%05d' "${FIELD_FRAME}").bin}"
if [[ ! -f "${FIELD_PATH}" && "${FIELD_FRAME}" -gt 0 ]]; then
    FIELD_FRAME="$((FIELD_FRAME - 1))"
    FIELD_PATH="${FIELD_DIR}/field$(printf '%05d' "${FIELD_FRAME}").bin"
fi

if [[ ! -f "${FIELD_PATH}" ]]; then
    echo "Missing compact field snapshot: ${FIELD_PATH}" >&2
    exit 1
fi
if [[ ! -f "${PARTICLE_PATH}" ]]; then
    echo "Missing particle snapshot: ${PARTICLE_PATH}" >&2
    exit 1
fi

FIELD_LINK="${INPUT_LINK_DIR}/$(basename "${FIELD_PATH}")"
PARTICLE_LINK="${INPUT_LINK_DIR}/$(basename "${PARTICLE_PATH}")"
ln -sfn "${FIELD_PATH}" "${FIELD_LINK}"
ln -sfn "${PARTICLE_PATH}" "${PARTICLE_LINK}"

BACKGROUND_PATH="${KPT_EMISSION_BACKGROUND_PATH:-}"
BACKGROUND_LINK=""
if [[ -n "${BACKGROUND_PATH}" ]]; then
    if [[ ! -f "${BACKGROUND_PATH}" ]]; then
        echo "Missing emission background map: ${BACKGROUND_PATH}" >&2
        exit 1
    fi
    BACKGROUND_LINK="${INPUT_LINK_DIR}/$(basename "${BACKGROUND_PATH}")"
    ln -sfn "${BACKGROUND_PATH}" "${BACKGROUND_LINK}"
fi

cat > "${MANIFEST_PATH}" <<EOF
run_root=${RUN_ROOT}
field_dir=${FIELD_DIR}
particle_dir=${PARTICLE_DIR}
field_frame=${FIELD_FRAME}
particle_frame=${FRAME}
field_path=${FIELD_PATH}
particle_path=${PARTICLE_PATH}
background_path=${BACKGROUND_PATH}
python=${PYTHON}
output_root=${OUTPUT_ROOT}
config_path=${CONFIG_PATH}
EOF

"${PYTHON}" - "${CONFIG_PATH}" "${FIELD_LINK}" "${PARTICLE_LINK}" "${BACKGROUND_LINK}" "${FRAME}" <<'PY'
import json
import os
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
field_path = Path(sys.argv[2])
particle_path = Path(sys.argv[3])
background_path = sys.argv[4]
frame = int(sys.argv[5])
output_root = config_path.parent

def env_float(name, default):
    return float(os.environ.get(name, default))

def env_int(name, default):
    return int(os.environ.get(name, default))

config = {
    "field_path": str(field_path),
    "particle_path": str(particle_path),
    "output_hdf5": str(output_root / f"parker_emission_{frame:05d}.h5"),
    "output_quicklook": str(output_root / f"parker_emission_{frame:05d}.png"),
    "transport_model": "parker",
    "background_path": str(background_path) if background_path else "",
    "frame_id": frame,
    "image_nx": env_int("KPT_EMISSION_IMAGE_NX", 32),
    "image_ny": env_int("KPT_EMISSION_IMAGE_NY", 32),
    "quicklook_frequency_ghz": env_float("KPT_EMISSION_QUICKLOOK_GHZ", 3.0),
    "beam_fwhm_pixels": env_float("KPT_EMISSION_BEAM_FWHM_PIXELS", 1.5),
    "pixel_size_arcsec": env_float("KPT_EMISSION_PIXEL_SIZE_ARCSEC", 1.0),
    "minimum_particle_energy_ev": env_float("KPT_EMISSION_MIN_PARTICLE_EV", 0.0),
    "macro_particle_electron_count": env_float(
        "KPT_EMISSION_MACRO_ELECTRON_COUNT", 1.0e30
    ),
    "target_peak_nonthermal_density_cm3": env_float(
        "KPT_EMISSION_TARGET_PEAK_NNTH_CM3", 1.0e8
    ),
    "thermal_density_cm3": env_float("KPT_EMISSION_THERMAL_DENSITY_CM3", 5.0e9),
    "temperature_mk": env_float("KPT_EMISSION_TEMPERATURE_MK", 8.0),
    "los_depth_arcsec": env_float("KPT_EMISSION_LOS_DEPTH_ARCSEC", 8.0),
    "viewing_angle_deg": env_float("KPT_EMISSION_VIEWING_ANGLE_DEG", 75.0),
    "power_law_index_default": env_float("KPT_EMISSION_POWER_LAW_INDEX", 4.5),
    "magnetic_field_floor_gauss": env_float("KPT_EMISSION_B_FLOOR_GAUSS", 80.0),
    "magnetic_field_peak_gauss": env_float("KPT_EMISSION_B_PEAK_GAUSS", 220.0),
    "magnetic_field_bins": env_int("KPT_EMISSION_B_BINS", 8),
    "nonthermal_density_bins": env_int("KPT_EMISSION_NNTH_BINS", 8),
    "power_law_index_bins": env_int("KPT_EMISSION_DELTA_BINS", 8),
}
config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

(
    cd "${ROOT_DIR}"
    "${PYTHON}" -m particleEmission.simple_emission_runner --config "${CONFIG_PATH}"
) 2>&1 | tee "${LOG_PATH}"

echo "Parker emission output root: ${OUTPUT_ROOT}"
