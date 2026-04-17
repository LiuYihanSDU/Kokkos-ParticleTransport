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

TRANSPORT_MODEL="${KPT_TRANSPORT_MODEL:-parker}"
case "${TRANSPORT_MODEL}" in
    parker|Parker|PARKER)
        TRANSPORT_MODEL="parker"
        FORTRAN_FOCUSED_TRANSPORT=".false."
        ;;
    focused|focus|Focused|FOCUSED|FOCUS)
        TRANSPORT_MODEL="focused"
        FORTRAN_FOCUSED_TRANSPORT=".true."
        ;;
    *)
        echo "Unsupported KPT_TRANSPORT_MODEL=${TRANSPORT_MODEL}; use parker or focused." >&2
        exit 2
        ;;
esac
DEFAULT_RUN_ROOT="${REPO_ROOT}/benchmark_runs/reconnection_200"
if [[ "${TRANSPORT_MODEL}" == "focused" ]]; then
    DEFAULT_RUN_ROOT="${REPO_ROOT}/benchmark_runs/reconnection_focused_200"
fi
RUN_ROOT="$(absolute_path "${KPT_BENCHMARK_ROOT:-${DEFAULT_RUN_ROOT}}")"
FORTRAN_DIR="${RUN_ROOT}/fortran"
KOKKOS_GPU_DIR="${RUN_ROOT}/kokkos_gpu"
KOKKOS_CPU_DIR="${RUN_ROOT}/kokkos_cpu"

if [[ -n "${KPT_FIELD_DIR:-}" ]]; then
    FIELD_DIR="$(absolute_path "${KPT_FIELD_DIR}")"
    PREPARE_COMPACT_FIELD="${KPT_PREPARE_COMPACT_FIELD:-0}"
else
    FIELD_DIR="${RUN_ROOT}/compact_field"
    PREPARE_COMPACT_FIELD="${KPT_PREPARE_COMPACT_FIELD:-1}"
fi
FORTRAN_EXAMPLE_DIR="${REPO_ROOT}/lixiaocanexample/stochastic-parker/examples/reconnection_2d"
FORTRAN_EXEC="${STOCHASTIC_EXEC:-${REPO_ROOT}/lixiaocanexample/stochastic-parker/bin/stochastic-mhd.exec}"
FORTRAN_MHD_DIR="${KPT_FORTRAN_MHD_DIR:-/home/liuyh/data/Athena++/athena_reconnection_test/bin_data/}"
FORTRAN_CONF_NAME="${KPT_FORTRAN_CONF_NAME:-conf_reconnection_kpt_${TRANSPORT_MODEL}_benchmark.dat}"
FORTRAN_CONF_PATH="${FORTRAN_EXAMPLE_DIR}/config/${FORTRAN_CONF_NAME}"

CPU_BUILD_DIR="$(absolute_path "${KPT_CPU_BUILD_DIR:-${REPO_ROOT}/cmake-build-benchmark-cpu}")"
GPU_BUILD_DIR="$(absolute_path "${KPT_GPU_BUILD_DIR:-${REPO_ROOT}/cmake-build-benchmark-cuda}")"
CPU_KOKKOS_DIR="${KPT_CPU_KOKKOS_DIR:-/usr/local/kokkos_cpu/lib/cmake/Kokkos}"
GPU_KOKKOS_DIR="${KPT_GPU_KOKKOS_DIR:-/usr/local/kokkos/lib/cmake/Kokkos}"
GPU_CXX_COMPILER="${KPT_GPU_CXX_COMPILER:-/usr/local/kokkos/bin/nvcc_wrapper}"
FORTRAN_BUILD_MODE="${KPT_FORTRAN_BUILD_MODE:-existing}"
FORTRAN_SOURCE_ROOT="$(absolute_path "${KPT_FORTRAN_SOURCE_ROOT:-${REPO_ROOT}/lixiaocanexample/stochastic-parker}")"
FORTRAN_AVX512_WORK_ROOT="$(absolute_path "${KPT_FORTRAN_AVX512_WORK_ROOT:-${REPO_ROOT}/benchmark_runs/fortran_avx512_build}")"
FORTRAN_AVX512_SOURCE_DIR="${FORTRAN_AVX512_WORK_ROOT}/source"
FORTRAN_AVX512_BUILD_DIR="${FORTRAN_AVX512_SOURCE_DIR}/build-avx512"
FORTRAN_AVX512_EXEC="${FORTRAN_AVX512_SOURCE_DIR}/bin/stochastic-mhd.exec"
FORTRAN_COMPILER="${KPT_FORTRAN_COMPILER:-/usr/bin/mpif90}"
MT_STREAM_DIR="${MT_STREAM:-/home/liuyh/codes/mt_stream_f90-1.11}"
HDF5_ROOT_DIR="${KPT_HDF5_ROOT:-/usr/lib/x86_64-linux-gnu/hdf5/openmpi}"
HDF5_FORTRAN_COMPILER="${KPT_HDF5_FORTRAN_COMPILER:-/usr/bin/h5pfc.openmpi}"
CPU_CORE_COUNT="${KPT_CPU_CORES:-16}"
BUILD_JOBS="${KPT_BUILD_JOBS:-${CPU_CORE_COUNT}}"
CPU_CXX_FLAGS="${KPT_CPU_CXX_FLAGS:--march=native}"

START_FRAME="${KPT_START_FRAME:-0}"
END_FRAME="${KPT_END_FRAME:-200}"
HISTOGRAM_INTERVAL="${KPT_HISTOGRAM_INTERVAL:-10}"
MPI_SIZE="${KPT_MPI_SIZE:-${CPU_CORE_COUNT}}"
PARTICLES_PER_RANK="${KPT_PARTICLES_PER_RANK:-1600}"
FORTRAN_PARTICLE_CAPACITY="${KPT_FORTRAN_PARTICLE_CAPACITY:-1000000}"
CXX_PARTICLE_CAPACITY="${KPT_CXX_PARTICLE_CAPACITY:-16000000}"
FORTRAN_OMP_THREADS="${KPT_FORTRAN_OMP_THREADS:-1}"
KOKKOS_GPU_HOST_THREADS="${KPT_KOKKOS_GPU_HOST_THREADS:-1}"
KOKKOS_CPU_THREADS="${KPT_KOKKOS_CPU_THREADS:-${CPU_CORE_COUNT}}"
PARTICLE_V0="${KPT_PARTICLE_V0:-17.20195}"
DUU0="${KPT_DUU0:-5578.445}"
POSTPROCESS_PYTHON="${KPT_POSTPROCESS_PYTHON:-}"
SKIP_POSTPROCESS="${KPT_SKIP_POSTPROCESS:-0}"

TIME_CMD="${KPT_TIME_CMD:-/usr/bin/time}"
MPI_RUN="${KPT_MPI_RUN:-mpirun}"
FORTRAN_AVX512_FLAGS="${KPT_FORTRAN_AVX512_FLAGS:--O3 -g -fimplicit-none -Wline-truncation -fwhole-file -std=gnu -fall-intrinsics -fPIC -cpp -fallow-argument-mismatch -march=native -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vnni}"

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

run_logged_in_dir() {
    local log_file="$1"
    local work_dir="$2"
    shift 2
    mkdir -p "$(dirname "${log_file}")"
    {
        printf '>>> cd %q &&' "${work_dir}"
        printf ' %q' "$@"
        printf '\n'
    } | tee "${log_file}"
    (cd "${work_dir}" && "$@") 2>&1 | tee -a "${log_file}"
}

prepare_case_dir() {
    local case_dir="$1"
    rm -rf "${case_dir}"
    mkdir -p "${case_dir}/restart"
}

require_file() {
    local path="$1"
    local message="$2"
    if [[ ! -f "${path}" ]]; then
        echo "${message}: ${path}" >&2
        exit 1
    fi
}

require_dir() {
    local path="$1"
    local message="$2"
    if [[ ! -d "${path}" ]]; then
        echo "${message}: ${path}" >&2
        exit 1
    fi
}

prepare_fortran_source_copy() {
    require_dir "${FORTRAN_SOURCE_ROOT}" "Missing Fortran source tree"
    mkdir -p "${FORTRAN_AVX512_WORK_ROOT}"
    rm -rf "${FORTRAN_AVX512_SOURCE_DIR}"
    mkdir -p "${FORTRAN_AVX512_SOURCE_DIR}"
    tar -C "${FORTRAN_SOURCE_ROOT}" \
        --exclude='./.git' \
        --exclude='./build' \
        --exclude='./bin' \
        --exclude='./lib' \
        --exclude='./run' \
        --exclude='./data' \
        -cf - . | tar -C "${FORTRAN_AVX512_SOURCE_DIR}" -xf -
    cat > "${FORTRAN_AVX512_SOURCE_DIR}/cmake/submodules.cmake" <<'EOF'
set(GIT_SUBMODULES_DIRECTORY src/third_party)
EOF
}

build_fortran_avx512() {
    require_dir "${MT_STREAM_DIR}" "Missing MT_STREAM directory"
    require_dir "${HDF5_ROOT_DIR}" "Missing parallel HDF5 root directory"
    require_file "${HDF5_FORTRAN_COMPILER}" "Missing HDF5 Fortran compiler wrapper"
    run_logged "${RUN_ROOT}/configure_fortran_avx512.log" \
        env MT_STREAM="${MT_STREAM_DIR}" \
        cmake -S "${FORTRAN_AVX512_SOURCE_DIR}" -B "${FORTRAN_AVX512_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_Fortran_COMPILER="${FORTRAN_COMPILER}" \
        -DCMAKE_Fortran_FLAGS="${FORTRAN_AVX512_FLAGS}" \
        -DCMAKE_PREFIX_PATH="${HDF5_ROOT_DIR}" \
        -DHDF5_ROOT="${HDF5_ROOT_DIR}" \
        -DHDF5_PREFER_PARALLEL=TRUE \
        -DHDF5_Fortran_COMPILER_EXECUTABLE="${HDF5_FORTRAN_COMPILER}" \
        -DUSE_OPENMP=ON
    run_logged "${RUN_ROOT}/build_fortran_avx512.log" \
        env MT_STREAM="${MT_STREAM_DIR}" \
        cmake --build "${FORTRAN_AVX512_BUILD_DIR}" -j "${BUILD_JOBS}"
    require_file "${FORTRAN_AVX512_EXEC}" \
        "Missing AVX512 Fortran executable after build"
    cp "${FORTRAN_AVX512_SOURCE_DIR}/bin/CMakeFiles/stochastic-mhd.exec.dir/flags.make" \
        "${RUN_ROOT}/fortran_avx512_flags.make"
    if command -v objdump >/dev/null 2>&1; then
        objdump -d "${FORTRAN_AVX512_EXEC}" \
            | grep -E '\b(zmm|ymm)[0-9]|\bv(add|mul|fmadd|movap|movup|broadcast|gather|scatter)' \
            | head -80 > "${RUN_ROOT}/fortran_avx512_vector_instructions.txt" || true
    fi
}

prepare_fortran_executable() {
    case "${FORTRAN_BUILD_MODE}" in
        existing)
            require_file "${FORTRAN_EXEC}" "Missing Fortran stochastic executable"
            ;;
        avx512)
            prepare_fortran_source_copy
            build_fortran_avx512
            FORTRAN_EXEC="${FORTRAN_AVX512_EXEC}"
            ;;
        *)
            echo "Unsupported KPT_FORTRAN_BUILD_MODE=${FORTRAN_BUILD_MODE}; use existing or avx512." >&2
            exit 2
            ;;
    esac
}

set_fortran_config_value() {
    local key="$1"
    local value="$2"
    sed -i -E "s|^(${key}[[:space:]]*=[[:space:]]*).*|\\1${value}|" "${FORTRAN_CONF_PATH}"
}

build_kokkos_apps() {
    mkdir -p "${RUN_ROOT}"
    run_logged "${RUN_ROOT}/configure_cpu.log" \
        cmake -S "${REPO_ROOT}" -B "${CPU_BUILD_DIR}" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="${CPU_CXX_FLAGS}" \
        -DKokkos_DIR="${CPU_KOKKOS_DIR}"
    run_logged "${RUN_ROOT}/build_cpu.log" \
        cmake --build "${CPU_BUILD_DIR}" -j "${BUILD_JOBS}"

    run_logged "${RUN_ROOT}/configure_gpu.log" \
        cmake -S "${REPO_ROOT}" -B "${GPU_BUILD_DIR}" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="${GPU_CXX_COMPILER}" \
        -DKokkos_DIR="${GPU_KOKKOS_DIR}"
    run_logged "${RUN_ROOT}/build_gpu.log" \
        cmake --build "${GPU_BUILD_DIR}" -j "${BUILD_JOBS}"
}

prepare_compact_fields() {
    if [[ "${PREPARE_COMPACT_FIELD}" != "1" ]]; then
        return
    fi
    rm -rf "${FIELD_DIR}"
    mkdir -p "${FIELD_DIR}"
    run_logged "${RUN_ROOT}/prepare_compact_field.log" \
        "${CPU_BUILD_DIR}/fortran_mhd_to_compact_field" \
        --input-dir "${FORTRAN_MHD_DIR}" \
        --output-dir "${FIELD_DIR}" \
        --start-frame "${START_FRAME}" \
        --end-frame "${END_FRAME}"
}

write_manifest() {
    cat > "${RUN_ROOT}/manifest.txt" <<EOF
run_root=${RUN_ROOT}
transport_model=${TRANSPORT_MODEL}
field_dir=${FIELD_DIR}
prepare_compact_field=${PREPARE_COMPACT_FIELD}
fortran_dir=${FORTRAN_DIR}
kokkos_gpu_dir=${KOKKOS_GPU_DIR}
kokkos_cpu_dir=${KOKKOS_CPU_DIR}
start_frame=${START_FRAME}
end_frame=${END_FRAME}
histogram_interval=${HISTOGRAM_INTERVAL}
cpu_core_count=${CPU_CORE_COUNT}
build_jobs=${BUILD_JOBS}
cpu_cxx_flags=${CPU_CXX_FLAGS}
fortran_build_mode=${FORTRAN_BUILD_MODE}
fortran_source_root=${FORTRAN_SOURCE_ROOT}
fortran_avx512_source_dir=${FORTRAN_AVX512_SOURCE_DIR}
fortran_avx512_build_dir=${FORTRAN_AVX512_BUILD_DIR}
fortran_avx512_flags=${FORTRAN_AVX512_FLAGS}
mt_stream=${MT_STREAM_DIR}
hdf5_root=${HDF5_ROOT_DIR}
hdf5_fortran_compiler=${HDF5_FORTRAN_COMPILER}
mpi_size=${MPI_SIZE}
particles_per_rank=${PARTICLES_PER_RANK}
fortran_particle_capacity=${FORTRAN_PARTICLE_CAPACITY}
cxx_particle_capacity=${CXX_PARTICLE_CAPACITY}
fortran_omp_threads=${FORTRAN_OMP_THREADS}
kokkos_gpu_host_threads=${KOKKOS_GPU_HOST_THREADS}
kokkos_cpu_threads=${KOKKOS_CPU_THREADS}
particle_v0=${PARTICLE_V0}
duu0=${DUU0}
fortran_exec=${FORTRAN_EXEC}
fortran_mhd_dir=${FORTRAN_MHD_DIR}
cpu_build_dir=${CPU_BUILD_DIR}
gpu_build_dir=${GPU_BUILD_DIR}
postprocess_python=${POSTPROCESS_PYTHON}
EOF
}

prepare_fortran_config() {
    mkdir -p "${FORTRAN_EXAMPLE_DIR}/config"
    cp "${FORTRAN_EXAMPLE_DIR}/conf_reconnection.dat" "${FORTRAN_CONF_PATH}"
    set_fortran_config_value momentum_dependency 1
    set_fortran_config_value gamma_turb 1.6666667
    set_fortran_config_value mag_dependency 1
    set_fortran_config_value kpara0 0.00743592
    set_fortran_config_value kret 0.01
}

run_fortran_case() {
    prepare_case_dir "${FORTRAN_DIR}"
    prepare_fortran_config

    local output_dir="${FORTRAN_DIR}/"
    local args=(
        -qh 12.0 -rf .false.
        -ft "${FORTRAN_FOCUSED_TRANSPORT}" -nl .false. -kk 6.770161725403334
        -pv "${PARTICLE_V0}" -sm 1
        -dm "${FORTRAN_MHD_DIR}" -mc mhd_config.dat -np "${PARTICLES_PER_RANK}"
        -ti 1 -ts "${START_FRAME}" -te "${END_FRAME}" -tm "${END_FRAME}"
        -st 0
        -df 1 -pi 6.2
        -sf 1 -sr 2.0 -ps 2.0
        -tf .false. -ptf tags_selected_01.h5
        -ni 100
        -dd "${output_dir}" -cf "${FORTRAN_CONF_NAME}"
        -ld .true. -ded .true. -de .false.
        -nm "${FORTRAN_PARTICLE_CAPACITY}" -in .true. -ij .false.
        -sn .true. -tti "${END_FRAME}" -ip .false.
        -jz 500 -nn 20000 -ib .false.
        -db2 0.03 -nb 2000
        -iv .false. -dv 10.0 -nv 2000
        -ir .false. -rm 2.0 -nr 2000
        -iaj .false. -ajm 0.2 -naj 20000
        -xs -0.25 -xe 0.25
        -ys 0.00 -ye 1.00 -zs -0.25 -ze 0.25
        -dw 0 -ds 0 -t0 7.53877e-5
        -ws 1 -db 0 -co 0
        -nd 2 -dp1 850964.408 -dp2 13575468.975
        -ch -1 -sc 0 -ug 1
        -cd 0 -pd 0
        -i3 0 -as 0
        -sf1 ts_plane_top -sn1 +y
        -s2e .false. -ii .false.
        -sf2 ts_plane_top -sn2 +y
        -vdt .false. -du "${DUU0}"
    )

    run_logged_in_dir "${FORTRAN_DIR}/run.log" "${FORTRAN_EXAMPLE_DIR}" \
        env OMP_NUM_THREADS="${FORTRAN_OMP_THREADS}" \
        "${TIME_CMD}" -p "${MPI_RUN}" -n "${MPI_SIZE}" "${FORTRAN_EXEC}" "${args[@]}"
}

run_kokkos_case() {
    local app="$1"
    local case_dir="$2"
    local host_threads="$3"
    local log_file="${case_dir}/console.log"

    prepare_case_dir "${case_dir}"
    run_logged "${log_file}" \
        env OMP_NUM_THREADS="${host_threads}" KOKKOS_NUM_THREADS="${host_threads}" \
        OMP_PROC_BIND=spread OMP_PLACES=threads \
        "${TIME_CMD}" -p "${app}" \
        --profile fortran-global \
        --transport "${TRANSPORT_MODEL}" \
        --field-dir "${FIELD_DIR}" \
        --output-dir "${case_dir}" \
        --start-frame "${START_FRAME}" \
        --end-frame "${END_FRAME}" \
        --particles-per-frame "${PARTICLES_PER_RANK}" \
        --rank-scale "${MPI_SIZE}" \
        --capacity "${CXX_PARTICLE_CAPACITY}" \
        --particle-v0 "${PARTICLE_V0}" \
        --duu0 "${DUU0}" \
        --diagnostic-interval 1 \
        --histogram-interval "${HISTOGRAM_INTERVAL}"
}

select_postprocess_python() {
    if [[ -n "${POSTPROCESS_PYTHON}" ]]; then
        printf '%s\n' "${POSTPROCESS_PYTHON}"
        return
    fi
    if [[ -x /tmp/kpt_postprocess_venv/bin/python ]]; then
        printf '%s\n' /tmp/kpt_postprocess_venv/bin/python
        return
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return
    fi
    echo "Missing Python interpreter for post-processing." >&2
    exit 1
}

run_postprocess_plots() {
    if [[ "${SKIP_POSTPROCESS}" == "1" ]]; then
        echo "=== Skipping post-processing plots because KPT_SKIP_POSTPROCESS=1 ==="
        return
    fi

    local python_bin
    python_bin="$(select_postprocess_python)"
    local spectrum_start="${HISTOGRAM_INTERVAL}"
    local spectrum_step="${HISTOGRAM_INTERVAL}"
    if [[ "${HISTOGRAM_INTERVAL}" -le 0 ]]; then
        spectrum_start="${END_FRAME}"
        spectrum_step="${END_FRAME}"
    fi
    run_logged "${RUN_ROOT}/postprocess_spectra.log" \
        env MPLBACKEND=Agg "${python_bin}" \
        "${REPO_ROOT}/apps/plot_reconnection_benchmark_spectra.py" \
        --root "${RUN_ROOT}" \
        --frame-start "${spectrum_start}" \
        --frame-end "${END_FRAME}" \
        --frame-step "${spectrum_step}" \
        --output "${RUN_ROOT}/reconnection_spectra_panels.png" \
        --csv-output "${RUN_ROOT}/reconnection_spectra.csv"
    run_logged "${RUN_ROOT}/postprocess_timing.log" \
        env MPLBACKEND=Agg "${python_bin}" \
        "${REPO_ROOT}/apps/plot_reconnection_timing.py" \
        --root "${RUN_ROOT}" \
        --output "${RUN_ROOT}/reconnection_timing_panels.png" \
        --csv-output "${RUN_ROOT}/reconnection_timing.csv"
}

main() {
    require_dir "${FORTRAN_MHD_DIR}" "Missing Fortran MHD binary directory"

    mkdir -p "${RUN_ROOT}"
    write_manifest
    prepare_fortran_executable
    write_manifest
    build_kokkos_apps
    prepare_compact_fields
    require_dir "${FIELD_DIR}" "Missing compact Athena field directory"
    require_file "${FIELD_DIR}/field$(printf '%05d' "${START_FRAME}").bin" \
        "Missing first compact Athena field frame"
    require_file "${FIELD_DIR}/field$(printf '%05d' "${END_FRAME}").bin" \
        "Missing final compact Athena field frame"

    echo "=== Running Fortran reference case ==="
    run_fortran_case

    echo "=== Running Kokkos CUDA case ==="
    run_kokkos_case "${GPU_BUILD_DIR}/kokkos_particle_transport_app" "${KOKKOS_GPU_DIR}" \
        "${KOKKOS_GPU_HOST_THREADS}"

    echo "=== Running Kokkos CPU case ==="
    run_kokkos_case "${CPU_BUILD_DIR}/kokkos_particle_transport_app" "${KOKKOS_CPU_DIR}" \
        "${KOKKOS_CPU_THREADS}"

    echo "=== Writing spectra and timing comparison plots ==="
    run_postprocess_plots

    echo "=== All reconnection benchmark cases completed ==="
    echo "Results: ${RUN_ROOT}"
}

main "$@"
