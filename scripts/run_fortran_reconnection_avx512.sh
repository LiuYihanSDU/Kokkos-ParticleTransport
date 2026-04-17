#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/run_fortran_reconnection_avx512.sh -AVX512

Build a separate AVX512-enabled Fortran stochastic-mhd executable and run only
the reconnection_2d Fortran benchmark case. The original Fortran executable is
not overwritten.

Useful overrides:
  KPT_BENCHMARK_ROOT              Output root for this run
  KPT_CPU_CORES                   Default build jobs and MPI ranks, default 16
  KPT_MPI_SIZE                    MPI ranks, default KPT_CPU_CORES
  KPT_PARTICLES_PER_RANK          Fortran -np value, default 1600
  KPT_END_FRAME                   Final MHD frame, default 200
  KPT_FORTRAN_COMPILER            Fortran MPI compiler, default /usr/bin/mpif90
  KPT_FORTRAN_AVX512_FLAGS        Full Fortran optimization and ISA flags
  KPT_HDF5_ROOT                   Parallel HDF5 root, default OpenMPI HDF5
  KPT_HDF5_FORTRAN_COMPILER       HDF5 Fortran wrapper, default h5pfc.openmpi
  KPT_FORTRAN_AVX512_WORK_ROOT    Build-copy root
  MT_STREAM                       mt_stream_f90 installation
EOF
}

if [[ "${1:-}" != "-AVX512" && "${1:-}" != "--avx512" ]]; then
    usage >&2
    exit 2
fi
shift

if [[ "$#" -gt 0 ]]; then
    echo "Unexpected arguments: $*" >&2
    usage >&2
    exit 2
fi

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

set_fortran_config_value() {
    local key="$1"
    local value="$2"
    sed -i -E "s|^(${key}[[:space:]]*=[[:space:]]*).*|\\1${value}|" "${FORTRAN_CONF_PATH}"
}

RUN_ROOT="$(absolute_path "${KPT_BENCHMARK_ROOT:-${REPO_ROOT}/benchmark_runs/reconnection_200_fortran_avx512}")"
FORTRAN_DIR="${RUN_ROOT}/fortran"

FORTRAN_SOURCE_ROOT="$(absolute_path "${KPT_FORTRAN_SOURCE_ROOT:-${REPO_ROOT}/lixiaocanexample/stochastic-parker}")"
FORTRAN_EXAMPLE_DIR="${REPO_ROOT}/lixiaocanexample/stochastic-parker/examples/reconnection_2d"
FORTRAN_CONF_NAME="${KPT_FORTRAN_CONF_NAME:-conf_reconnection_kpt_fortran_avx512.dat}"
FORTRAN_CONF_PATH="${FORTRAN_EXAMPLE_DIR}/config/${FORTRAN_CONF_NAME}"
FORTRAN_MHD_DIR="${KPT_FORTRAN_MHD_DIR:-/home/liuyh/data/Athena++/athena_reconnection_test/bin_data/}"

AVX_WORK_ROOT="$(absolute_path "${KPT_FORTRAN_AVX512_WORK_ROOT:-${REPO_ROOT}/benchmark_runs/fortran_avx512_build}")"
AVX_SOURCE_DIR="${AVX_WORK_ROOT}/source"
AVX_BUILD_DIR="${AVX_SOURCE_DIR}/build-avx512"
AVX_EXEC="${AVX_SOURCE_DIR}/bin/stochastic-mhd.exec"
FORTRAN_COMPILER="${KPT_FORTRAN_COMPILER:-/usr/bin/mpif90}"
MT_STREAM_DIR="${MT_STREAM:-/home/liuyh/codes/mt_stream_f90-1.11}"
HDF5_ROOT_DIR="${KPT_HDF5_ROOT:-/usr/lib/x86_64-linux-gnu/hdf5/openmpi}"
HDF5_FORTRAN_COMPILER="${KPT_HDF5_FORTRAN_COMPILER:-/usr/bin/h5pfc.openmpi}"

CPU_CORE_COUNT="${KPT_CPU_CORES:-16}"
BUILD_JOBS="${KPT_BUILD_JOBS:-${CPU_CORE_COUNT}}"
START_FRAME="${KPT_START_FRAME:-0}"
END_FRAME="${KPT_END_FRAME:-200}"
MPI_SIZE="${KPT_MPI_SIZE:-${CPU_CORE_COUNT}}"
PARTICLES_PER_RANK="${KPT_PARTICLES_PER_RANK:-1600}"
FORTRAN_PARTICLE_CAPACITY="${KPT_FORTRAN_PARTICLE_CAPACITY:-1000000}"
FORTRAN_OMP_THREADS="${KPT_FORTRAN_OMP_THREADS:-1}"
TIME_CMD="${KPT_TIME_CMD:-/usr/bin/time}"
MPI_RUN="${KPT_MPI_RUN:-mpirun}"

FORTRAN_AVX512_FLAGS="${KPT_FORTRAN_AVX512_FLAGS:--O3 -g -fimplicit-none -Wline-truncation -fwhole-file -std=gnu -fall-intrinsics -fPIC -cpp -fallow-argument-mismatch -march=native -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vnni}"

prepare_fortran_source_copy() {
    require_dir "${FORTRAN_SOURCE_ROOT}" "Missing Fortran source tree"
    mkdir -p "${AVX_WORK_ROOT}"
    rm -rf "${AVX_SOURCE_DIR}"
    mkdir -p "${AVX_SOURCE_DIR}"
    tar -C "${FORTRAN_SOURCE_ROOT}" \
        --exclude='./.git' \
        --exclude='./build' \
        --exclude='./bin' \
        --exclude='./lib' \
        --exclude='./run' \
        --exclude='./data' \
        -cf - . | tar -C "${AVX_SOURCE_DIR}" -xf -
    cat > "${AVX_SOURCE_DIR}/cmake/submodules.cmake" <<'EOF'
set(GIT_SUBMODULES_DIRECTORY src/third_party)
EOF
}

build_fortran_avx512() {
    require_dir "${MT_STREAM_DIR}" "Missing MT_STREAM directory"
    require_dir "${HDF5_ROOT_DIR}" "Missing parallel HDF5 root directory"
    require_file "${HDF5_FORTRAN_COMPILER}" "Missing HDF5 Fortran compiler wrapper"
    run_logged "${RUN_ROOT}/configure_fortran_avx512.log" \
        env MT_STREAM="${MT_STREAM_DIR}" \
        cmake -S "${AVX_SOURCE_DIR}" -B "${AVX_BUILD_DIR}" \
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
        cmake --build "${AVX_BUILD_DIR}" -j "${BUILD_JOBS}"
    require_file "${AVX_EXEC}" "Missing AVX512 Fortran executable after build"
    cp "${AVX_SOURCE_DIR}/bin/CMakeFiles/stochastic-mhd.exec.dir/flags.make" \
        "${RUN_ROOT}/fortran_avx512_flags.make"
    if command -v objdump >/dev/null 2>&1; then
        objdump -d "${AVX_EXEC}" \
            | grep -E '\b(zmm|ymm)[0-9]|\bv(add|mul|fmadd|movap|movup|broadcast|gather|scatter)' \
            | head -80 > "${RUN_ROOT}/fortran_avx512_vector_instructions.txt" || true
    fi
}

write_manifest() {
    cat > "${RUN_ROOT}/manifest.txt" <<EOF
run_root=${RUN_ROOT}
fortran_dir=${FORTRAN_DIR}
fortran_source_root=${FORTRAN_SOURCE_ROOT}
fortran_avx512_source_dir=${AVX_SOURCE_DIR}
fortran_avx512_build_dir=${AVX_BUILD_DIR}
fortran_avx512_exec=${AVX_EXEC}
fortran_avx512_flags=${FORTRAN_AVX512_FLAGS}
mt_stream=${MT_STREAM_DIR}
hdf5_root=${HDF5_ROOT_DIR}
hdf5_fortran_compiler=${HDF5_FORTRAN_COMPILER}
fortran_mhd_dir=${FORTRAN_MHD_DIR}
start_frame=${START_FRAME}
end_frame=${END_FRAME}
cpu_core_count=${CPU_CORE_COUNT}
build_jobs=${BUILD_JOBS}
mpi_size=${MPI_SIZE}
particles_per_rank=${PARTICLES_PER_RANK}
fortran_particle_capacity=${FORTRAN_PARTICLE_CAPACITY}
fortran_omp_threads=${FORTRAN_OMP_THREADS}
EOF
}

prepare_case_dir() {
    local case_dir="$1"
    rm -rf "${case_dir}"
    mkdir -p "${case_dir}/restart"
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
        -ft .false. -nl .false. -kk 6.770161725403334
        -pv 17.20195 -sm 1
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
        -vdt .false. -du 5578.445
    )

    run_logged_in_dir "${FORTRAN_DIR}/run.log" "${FORTRAN_EXAMPLE_DIR}" \
        env OMP_NUM_THREADS="${FORTRAN_OMP_THREADS}" \
        "${TIME_CMD}" -p "${MPI_RUN}" -n "${MPI_SIZE}" "${AVX_EXEC}" "${args[@]}"
}

main() {
    require_dir "${FORTRAN_MHD_DIR}" "Missing Fortran MHD binary directory"
    mkdir -p "${RUN_ROOT}"
    write_manifest
    prepare_fortran_source_copy
    build_fortran_avx512
    write_manifest
    echo "=== Running Fortran AVX512 reconnection case ==="
    run_fortran_case
    echo "=== Fortran AVX512 reconnection case completed ==="
    echo "Results: ${RUN_ROOT}"
}

main "$@"
