cmake_minimum_required(VERSION 3.24)

project(KokkosParticleTransport LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Kokkos_ROOT "$ENV{Kokkos_ROOT}" CACHE PATH "Root path of the Kokkos installation")
set(Kokkos_DIR "$ENV{Kokkos_DIR}" CACHE PATH "Path to the directory containing KokkosConfig.cmake")
set(OpenMP_ROOT "$ENV{OpenMP_ROOT}" CACHE PATH "Root path of the OpenMP installation")

if(APPLE)
  if((NOT DEFINED Kokkos_ROOT OR Kokkos_ROOT STREQUAL "") AND EXISTS "/opt/homebrew/opt/kokkos")
    set(Kokkos_ROOT "/opt/homebrew/opt/kokkos" CACHE PATH "Root path of the Kokkos installation" FORCE)
  endif()

  if((NOT DEFINED OpenMP_ROOT OR OpenMP_ROOT STREQUAL "") AND EXISTS "/opt/homebrew/opt/libomp")
    set(OpenMP_ROOT "/opt/homebrew/opt/libomp" CACHE PATH "Root path of the OpenMP installation" FORCE)
  endif()
endif()

if(Kokkos_ROOT)
  list(PREPEND CMAKE_PREFIX_PATH "${Kokkos_ROOT}")
endif()

if(OpenMP_ROOT)
  list(PREPEND CMAKE_PREFIX_PATH "${OpenMP_ROOT}")
endif()

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND OpenMP_ROOT)
  set(OpenMP_CXX_FLAGS "-fopenmp=libomp" CACHE STRING "OpenMP flags for Clang on macOS" FORCE)
  set(OpenMP_CXX_INCLUDE_DIR "${OpenMP_ROOT}/include" CACHE PATH "OpenMP include directory" FORCE)
  set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "OpenMP CXX library names" FORCE)
  set(OpenMP_omp_LIBRARY "${OpenMP_ROOT}/lib/libomp.dylib" CACHE FILEPATH "OpenMP runtime library" FORCE)
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Kokkos CONFIG QUIET)

if(NOT Kokkos_FOUND)
  message(FATAL_ERROR
    "Kokkos was not found. Set Kokkos_ROOT to the install prefix "
    "(for example /opt/homebrew/opt/kokkos) or set Kokkos_DIR to the directory "
    "containing KokkosConfig.cmake "
    "(for example /opt/homebrew/opt/kokkos/lib/cmake/Kokkos).")
endif()

add_library(kokkos_particle_transport INTERFACE)
add_library(KokkosParticleTransport::headers ALIAS kokkos_particle_transport)

target_compile_features(kokkos_particle_transport INTERFACE cxx_std_20)

target_include_directories(kokkos_particle_transport
  INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(kokkos_particle_transport
  INTERFACE
    Kokkos::kokkos
)

set(KPT_WITH_KOKKOS_CUDA OFF)
if(TARGET Kokkos::CUDA)
  set(KPT_WITH_KOKKOS_CUDA ON)
endif()

add_executable(kokkos_particle_transport_app
  src/main.cpp
)

if(NOT KPT_WITH_KOKKOS_CUDA)
  target_sources(kokkos_particle_transport_app
    PRIVATE
      src/header_smoke.cpp
  )
endif()

target_link_libraries(kokkos_particle_transport_app
  PRIVATE
    KokkosParticleTransport::headers
)

add_executable(fortran_mhd_to_compact_field
  apps/fortran_mhd_to_compact_field.cpp
)

target_compile_features(fortran_mhd_to_compact_field
  PRIVATE
    cxx_std_20
)

set(KPT_DEFAULT_PARTICLE_FIELD_DIR "${PROJECT_SOURCE_DIR}/cmake-build-debug/field_fortran_check")
if(NOT EXISTS "${KPT_DEFAULT_PARTICLE_FIELD_DIR}/field00002.bin")
  set(KPT_DEFAULT_PARTICLE_FIELD_DIR "${PROJECT_SOURCE_DIR}/cmake-build-debug/field")
endif()

set(KPT_RECONNECTION_PARTICLE_FIELD_DIR "${KPT_DEFAULT_PARTICLE_FIELD_DIR}"
  CACHE PATH "Field directory for the 64000-particles-per-frame reconnection run target")
set(KPT_RECONNECTION_FULL_FIELD_DIR "${PROJECT_SOURCE_DIR}/cmake-build-debug/field"
  CACHE PATH "Field directory for the full Fortran-global reconnection run target")
set(KPT_RECONNECTION_OUTPUT_ROOT "${PROJECT_BINARY_DIR}/runs"
  CACHE PATH "Output root used by reconnection run targets")

add_custom_target(run_reconnection_smoke
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${KPT_RECONNECTION_OUTPUT_ROOT}/smoke"
  COMMAND $<TARGET_FILE:kokkos_particle_transport_app>
          --profile smoke
          --field-dir "${KPT_RECONNECTION_PARTICLE_FIELD_DIR}"
          --output-dir "${KPT_RECONNECTION_OUTPUT_ROOT}/smoke"
          --overwrite-output
  DEPENDS kokkos_particle_transport_app
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  USES_TERMINAL
  COMMENT "Run a one-frame reconnection smoke test"
)

add_custom_target(run_reconnection_particle_64000
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${KPT_RECONNECTION_OUTPUT_ROOT}/particle_64000"
  COMMAND $<TARGET_FILE:kokkos_particle_transport_app>
          --profile particle-64000
          --field-dir "${KPT_RECONNECTION_PARTICLE_FIELD_DIR}"
          --output-dir "${KPT_RECONNECTION_OUTPUT_ROOT}/particle_64000"
          --overwrite-output
  DEPENDS kokkos_particle_transport_app
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  USES_TERMINAL
  COMMENT "Run the default 64000-particles-per-frame reconnection calibration test"
)

add_custom_target(run_reconnection_fortran_global
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${KPT_RECONNECTION_OUTPUT_ROOT}/fortran_global"
  COMMAND $<TARGET_FILE:kokkos_particle_transport_app>
          --profile fortran-global
          --field-dir "${KPT_RECONNECTION_FULL_FIELD_DIR}"
          --output-dir "${KPT_RECONNECTION_OUTPUT_ROOT}/fortran_global"
          --overwrite-output
  DEPENDS kokkos_particle_transport_app
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  USES_TERMINAL
  COMMENT "Run the 200-frame Fortran-global reconnection calibration"
)

add_custom_target(run_reconnection_three_cases_200
  COMMAND bash "${PROJECT_SOURCE_DIR}/scripts/run_reconnection_three_cases.sh"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  USES_TERMINAL
  COMMENT "Run Fortran, Kokkos CUDA, and Kokkos CPU reconnection cases sequentially"
)

add_custom_target(run_reconnection_optimized_comparison_200
  COMMAND bash "${PROJECT_SOURCE_DIR}/scripts/run_reconnection_optimized_comparison.sh"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  USES_TERMINAL
  COMMENT "Run optimized Parker Fortran AVX512, Kokkos CUDA, and Kokkos CPU comparison with plots"
)

add_custom_target(run_reconnection_focused_comparison_200
  COMMAND bash "${PROJECT_SOURCE_DIR}/scripts/run_reconnection_focused_comparison.sh"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  USES_TERMINAL
  COMMENT "Run focused-transport Fortran AVX512, Kokkos CUDA, and Kokkos CPU comparison with plots"
)

message(STATUS "Using Kokkos from: ${Kokkos_DIR}")
message(STATUS "Kokkos CUDA backend target available: ${KPT_WITH_KOKKOS_CUDA}")
