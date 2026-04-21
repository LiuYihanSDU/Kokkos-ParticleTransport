# Code Map

This document tracks the current repository structure, public-facing types, and the
implementation status of major interfaces. Update this file whenever code changes alter
the shape, semantics, or responsibility boundaries of the project.

## Repository Status

The repository now contains an active Kokkos reconnection calibration driver for Parker
and focused transport, compact Athena++/Fortran field converters, benchmark wrappers,
particle snapshot diagnostics, and a Python microwave emission-synthesis prototype.
Several generic solver headers still retain scaffold-level interfaces, and
`QLTModel.hpp` remains a placeholder. This document reflects the code as it exists now,
not the intended final design.

## High-Level Module View

- `readme.md`: Chinese user-facing project README covering build/run workflow,
  background-field conversion, walltime controls, benchmark wrappers, and emission movie
  synthesis.
- `README.en.md`: English companion README that mirrors the Chinese workflow overview for
  collaborators who do not read Chinese.
- `CMakeLists.txt`: project build entry point; locates an installed Kokkos package,
  exports compile commands, and defines both the interface target and calibration
  executable plus local reconnection run targets.
- `CMakePresets.json`: Linux CPU/CUDA CMake presets for command-line and CLion builds on
  this workstation.
- `CMakeUserPresets.json`: user-local CLion/CMake preset configuration for the Homebrew
  LLVM toolchain and Kokkos installation on this machine.
- `src/header_smoke.cpp`: header aggregation translation unit used to force full-header
  parsing and provide IDE/build smoke coverage for local interfaces.
- `src/main.cpp`: Li Xiaocan `reconnection_2d` calibration and timing driver built on
  the repository Athena field reader, interpolation utilities, Kokkos device aliases, and
  random pool wrapper, with runtime Parker and focused-transport branches.
- `apps/athena2bin.py`: Python converter from 2D Athena++ `.athdf` snapshots to the
  compact background-field binary format consumed by `AthenaFieldReader.hpp`, with an
  optional emission-background HDF5 output for density/temperature maps.
- `apps/field_binary_to_xdmf_hdf5.py`: Python converter from the repository
  field binary snapshot format to HDF5 plus XDMF.
- `apps/fortran_mhd_to_compact_field.cpp`: C++ converter from Li Xiaocan Fortran
  `mhd_data_NNNN` binary frames to the compact Kokkos background-field binary format.
- `apps/plot_reconnection_benchmark_spectra.py`: Python post-processing script that
  reads the three-case reconnection benchmark outputs and plots every-10-frame spectra
  as kinetic-energy spectra in LiXiaocan GPAT, Kokkos-CPU, and Kokkos-GPU panels.
- `apps/plot_reconnection_timing.py`: Python post-processing script that reads the
  three-case reconnection timing outputs and plots frame time versus particle count plus
  accumulated runtime versus physical frame time.
- `apps/particle_binary_to_xdmf_hdf5.py`: Python converter from the repository
  particle binary snapshot format to ParaView-readable HDF5 plus XDMF Polyvertex data.
- `particleEmission/EmissionTypes.hpp`: emission workflow common enums and value
  types for transport-model metadata, source provenance, observer setup, local
  field/background samples, reduced particle moments, and source-cell closure.
- `particleEmission/EmissionProviders.hpp`: host-side emission provider interfaces
  plus first concrete implementations for analytic field access, constant/functional
  background access, and reduced particle providers that enforce the Parker `mu`
  handling rule.
- `particleEmission/__init__.py`: Python package marker for the particle-emission
  prototype modules.
- `particleEmission/compact_field_io.py`: compact 2D field reader and coarse-grid
  averaging helpers for the current sample reconnection field format.
- `particleEmission/background_io.py`: emission background-map reader for Athena/MHD
  converted HDF5 files containing thermal-density and temperature maps.
- `particleEmission/particle_io.py`: minimal particle snapshot reader and weighted
  coarse-grid deposition utilities for active-particle morphology maps.
- `particleEmission/microwave_backend.py`: local ctypes wrapper around a rebuilt
  microwave backend shared library, with fixed-parameter spectrum evaluation and
  automatic on-demand compilation through `gfortran`.
- `particleEmission/emission_references.py`: reference-spectrum utilities that build
  component-aware diagnostic curves from the microwave backend, including thermal-only,
  low-density no-Razin proxy, optically thin proxy, zero-background reference, and
  signed absorption/Razin impact spectra.
- `particleEmission/validation_cases.py`: synthetic validation driver that builds a
  uniform-source baseline and an analytic loop-top case, reconstructs nonthermal
  density and power-law index from synthetic macro-particle statistics, and writes
  theory-vs-reconstruction morphology figures plus spectrum comparisons under a
  project-root validation output folder, together with parameter-map errors,
  per-frequency image metrics, broad-band spectral error summaries, and component-aware
  reference/diagnostic cubes for truth and reconstruction products.
- `particleEmission/emission_viewer_core.py`: generic emission-product discovery and
  flattening helpers for HDF5 browsing, including automatic cube/map/spectrum
  classification, derived polarization-degree products, component/reference discovery,
  ROI integration, beam scaling, and export support.
- `particleEmission/emission_viewer_app.py`: Streamlit-based local emission viewer app
  that opens the simple prototype outputs and synthetic validation products, provides
  frequency browsing, ROI- and beam-based spectrum extraction, image/map comparison,
  metric inspection, component/reference spectral overlays, and export buttons for PNG
  and CSV products.
- `particleEmission/run_emission_viewer.py`: small launcher that starts the Streamlit
  emission viewer from the repository-managed virtual environment with project-root
  default search paths and optional default-product selection.
- `emissionValidation/`: generated project-root validation output folder containing
  the runnable synthetic benchmark cases `case1_uniform_source` and
  `case2_analytic_loop_top` with morphology panels, spectrum comparisons, HDF5
  products, component-aware reference/diagnostic groups, and summary metadata including
  parameter, image, and banded-spectrum validation metrics.
- `kokkos_cpu_format_example_frame180_262k/`: tracked Parker CPU example containing two
  compact field frames, one version-4 particle snapshot, summary/log outputs, and a
  manifest for default emission-synthesis smoke runs.
- `pygsfit_cp-main/`: bundled external gyrosynchrotron fitting/synthesis package whose
  Fortran sources and prebuilt platform libraries are used as reference material and as
  the local microwave backend source tree.
- `particleEmission/emission_hdf5.py`: Python utilities for emission HDF5 product
  writing/reading, Rayleigh-Jeans specific-intensity and brightness-temperature
  conversion, frequency slicing, ROI integration, and Gaussian beam convolution.
- `particleEmission/simple_emission_runner.py`: first runnable end-to-end prototype
  that reads the example field and particle files, deposits particles to a coarse
  image grid, optionally reads an MHD/Athena-derived background map, calibrates a
  macro-particle-to-electron conversion factor, synthesizes per-pixel microwave spectra
  plus component-aware reference/diagnostic cubes, writes an HDF5 product, and saves a
  quicklook figure.
- `particleEmission/parker_emission_movie.py`: Parker movie renderer that combines
  compact fields, version-4 particle snapshots, optional Athena/MHD background maps,
  Bz/Jz diagnostics, energy-split particle maps, beam-convolved 1/3/5 GHz images, and
  full-domain spectra into a PNG sequence and optional ProRes MOV.
- `particleEmission/example_coronal_config.json`: example active-region coronal
  parameter set for the simple emission prototype CLI.
- `particleEmission/requirements.txt`: Python dependency list currently used to
  manage the local particle-emission virtual environment for HDF5 I/O, array
  operations, plotting, optional bundled ffmpeg access, and the local Streamlit viewer
  app.
- `particleEmission/workflow_plan.md`: planning document for the future particle
  emission workflow, covering local source reconstruction, gyrosynchrotron forward
  synthesis, binary emission products, ROI/beam analysis, staged Kokkos adoption,
  reserved background-parameter input interfaces, Parker `mu` handling, and support
  for both file-backed and analytic field providers.
- `scripts/run_reconnection_three_cases.sh`: top-level benchmark runner that builds the
  CPU/CUDA Kokkos executables and then runs the Fortran, Kokkos CUDA, and Kokkos CPU
  reconnection cases sequentially with separate output directories, log files, and
  post-processing plots.
- `scripts/run_total_benchmark_suite_local.sh`: extracted local benchmark-suite driver
  that runs the Kokkos-particleTransport three-way benchmark first, then the AMRVAC
  native NLFFF versus AthenaK CT-NLFFF benchmark, and records the suite log plus
  manifest under one benchmark root.
- `scripts/run_total_benchmark_suite.sh`: compatibility entry point that forwards to
  `scripts/run_total_benchmark_suite_local.sh` so existing launch commands still work
  without the old built-in email wrapper.
- `scripts/run_reconnection_optimized_comparison.sh`: Parker comparison wrapper that
  defaults to AVX512 Fortran, optimized Kokkos CPU, Kokkos CUDA, and automatic spectra
  plus timing plots.
- `scripts/run_reconnection_focused_comparison.sh`: focused-transport comparison wrapper
  with the same reconnection input defaults, isotropic pitch-angle injection, and
  automatic spectra plus timing plots.
- `scripts/run_reconnection_parker_focused_comparisons.sh`: sequential wrapper that runs
  the Parker optimized comparison first and the focused-transport comparison second,
  with independent benchmark roots and sequence-level logs.
- `scripts/run_local_parker_focused_speed_accuracy_test.sh`: local benchmark wrapper
  that runs Parker then focused three-way comparisons, prepares a post-processing Python
  environment when needed, and writes timing plus spectrum-difference summaries.
- `scripts/run_with_email_notification.sh`: GPT-independent command wrapper that records
  stdout/stderr to a log file and sends a completion email through local sendmail or
  Python standard-library SMTP.
- `scripts/validate_reconnection_dt_match.sh`: host-side validation helper that checks
  Kokkos summary times against the Fortran MHD `mhd_config.dat` `dt_out` value and
  verifies that the expected compact-field and particle snapshot frames exist, while
  also reporting the default reconnection Alfvén-time conversion to seconds.
- `scripts/run_parker_emission_synthesis.sh`: Parker emission bridge that discovers a
  Kokkos particle snapshot plus matching compact field, writes an emission config and
  manifest under a new output root, then invokes the simple emission synthesis runner.
- `scripts/render_parker_emission_movie.sh`: shell wrapper for the Parker emission movie
  renderer, defaulting to `/tmp/kpt_postprocess_venv/bin/python` and local Parker
  benchmark paths.
- `scripts/run_kokkos_cpu_parker_movie_pipeline.sh`: one-command CPU Parker rerun plus
  movie pipeline that writes per-frame particle snapshots, optional Athena/MHD emission
  background maps, 2x4 diagnostic/emission PNG frames, and a ProRes MOV.
- `scripts/run_fortran_reconnection_avx512.sh`: Fortran-only AVX512 comparison runner
  that builds a separate stochastic-mhd executable from an isolated source copy and runs
  the reconnection_2d Fortran benchmark without overwriting the baseline executable.
- `include/SolverBase.hpp`: common solver state for transport solvers.
- `include/ParkerSolver.hpp`: Parker transport solver derived from `SolverBase`.
- `include/ParkerDebuger.hpp`: host-side debug helper for `ParkerSolver`.
- `include/Grid.hpp`: device-backed common grid metadata and coordinate placement enums.
- `include/OrthogonalCoordinate.hpp`: orthogonal coordinate-system geometry and Lame
  coefficient helpers.
- `include/StoredCoordinateGrid.hpp`: explicit face-coordinate storage grid scaffold that
  derives cell-centered coordinates from cell edges.
- `include/AnalyticCoordinateGrid.hpp`: coefficient-backed analytic coordinate mapping
  profiles plus an analytic grid wrapper over `MappedCoordinateGrid`.
- `include/MappedCoordinateGrid.hpp`: generic device-callable coordinate mapping grid
  scaffold used by analytic and custom coordinate maps.
- `include/GridValidation.hpp`: host-side validation helpers for grid metadata,
  analytic coordinate coefficients, and stored coordinate views.
- `include/Field.hpp`: generic field abstraction plus scalar/vector aliases.
- `include/FieldInterpolator.hpp`: field time interpolation, finite-difference time
  derivatives, and device-callable particle-position field sampling.
- `include/FieldOperator.hpp`: generic Kokkos field operators for gradient,
  divergence, curl, magnitude, scaling, dot product, and direction fields.
- `include/FieldOutputTypes.hpp`: host-side serializable field output snapshot type.
- `include/FieldOutputCommon.hpp`: host-side field-to-binary-snapshot builder used by
  binary output and offline conversion.
- `include/FieldBinaryIO.hpp`: custom field binary snapshot serialization helpers.
- `include/FieldBinaryWriter.hpp`: field binary output entry point.
- `include/AthenaFieldReader.hpp`: host-side reader that loads `apps/athena2bin.py`
  compact 2D Athena++ background-field binaries into a uniform
  `AnalyticCoordinateGrid` plus magnetic-field and velocity `VectorField` objects for
  solver construction.
- `include/ParticleSystem.hpp`: single-species relativistic particle container,
  particle properties, lifecycle status, grid-boundary handling, sorting, splitting,
  and binary snapshot output.
- `include/ParticleSystemDebugger.hpp`: debug-only host-side particle diagnostics and
  statistics printing.
- `include/ParticleGenerator.hpp`: particle injection regions, rejection-sampling
  predicates, field/grid candidate sets, and device-side particle injection kernels.
- `include/RandomManager.hpp`: device-side random pool wrapper and stochastic samplers.
- `include/FocusSolver.hpp`: focused transport solver derived from `SolverBase`.
- `include/FocusDebuger.hpp`: host-side debug helper for `FocusTransportSolver`.
- `include/FocusCoefficient.hpp`: focused transport coefficient helpers and
  orthogonal-coordinate focused-transport precompute operators.
- `include/ParkerCoefficient.hpp`: Parker transport coefficient helpers including
  device-side `curl(B / B^2)` and `div(kappa_ij)` precomputation.
- `include/LegencyModel.hpp`: README legacy diffusion-coefficient model with
  gamma-independent field precomputation and particle-gamma correction helpers.
- `include/QLTModel.hpp`: quasi-linear theory model placeholder.
- `include/TurbulenceProperties.hpp`: turbulence-amplitude and correlation-length
  models plus grid-bound field initialization helpers for coefficient calculations.
- `include/PhysicalConstant.hpp`: CGS physical constants, symbolic metadata, and
  heliospheric unit-system helpers.
- `include/UnitConverter.hpp`: dimensional scalar types, unit systems, and
  nondimensionalization helpers.
- `include/KokkosDevice.hpp`: repository device aliases and device trait helpers.

## Dependency Sketch

```text
CMakeLists.txt
  -> find_package(Kokkos CONFIG)
  -> interface target kokkos_particle_transport
     -> include/
     -> Kokkos::kokkos
  -> executable target kokkos_particle_transport_app
     -> src/main.cpp
     -> KokkosParticleTransport::headers
     -> src/header_smoke.cpp only when Kokkos::CUDA is not available
  -> custom run targets:
     -> run_reconnection_smoke
     -> run_reconnection_particle_64000
     -> run_reconnection_fortran_global
     -> run_reconnection_three_cases_200
     -> run_reconnection_optimized_comparison_200
     -> run_reconnection_focused_comparison_200
     -> shell-only Parker/focused sequence wrapper in scripts/

readme.md
  -> Chinese workflow entry point for local users
  -> README.en.md as the English companion document
  -> docs/code_map.md for maintainer-facing interface details

README.en.md
  -> English workflow entry point mirroring readme.md

src/main.cpp
  -> AthenaFieldReader for compact Athena frame loading
  -> Field and FieldInterpolator for device-side coefficient storage and sampling
  -> RandomManager for Kokkos::Random_XorShift64_Pool injection and SDE increments
  -> KokkosDevice for execution-space and memory-space selection
  -> --transport parker for the Fortran-style Parker calibration pusher
  -> --transport focused for the 2D focused-transport calibration pusher with
     isotropic pitch-angle cosine injection

apps/field_binary_to_xdmf_hdf5.py
  -> Python 3
  -> h5py and numpy at runtime for HDF5 writing
  -> repository field binary snapshot format

apps/particle_binary_to_xdmf_hdf5.py
  -> Python 3
  -> h5py and numpy at runtime for HDF5 writing
  -> repository particle binary snapshot format versions 2, 3, and 4

apps/athena2bin.py
  -> Python 3
  -> h5py and numpy at runtime; scipy is optional for cubic interpolation
  -> Athena++ .athdf input files
  -> compact Athena background-field binary format:
     int32 nx, int32 ny, x/y face coordinates, Bx/By/Bz, Vx/Vy/Vz
  -> optional emission_background_NNNNN.h5 files:
     x/y face coordinates, rho_code, pressure_code when available,
     temperature_proxy_code, thermal_density_cm3, temperature_mk
  -> --skip-compact-output mode for generating only emission background maps

apps/fortran_mhd_to_compact_field.cpp
  -> C++20 standard library only
  -> Fortran mhd_config.dat and mhd_data_NNNN files
  -> compact Athena background-field binary format:
     int32 nx, int32 ny, x/y face coordinates, Bx/By/Bz, Vx/Vy/Vz

apps/plot_reconnection_benchmark_spectra.py
  -> Python 3
  -> h5py, matplotlib, and numpy at runtime
  -> benchmark_runs/reconnection_200/fortran/fdists_NNNN.h5
  -> benchmark_runs/reconnection_200/kokkos_gpu/momentum_histogram_NNNNN.csv
  -> benchmark_runs/reconnection_200/kokkos_cpu/momentum_histogram_NNNNN.csv
  -> benchmark_runs/reconnection_200/reconnection_spectra_panels.png
  -> benchmark_runs/reconnection_200/reconnection_spectra.csv
  -> momentum-to-energy mapping with p0 and p0-energy-keV controls
  -> optional kinetic-energy x-axis limits through --energy-x-min and --energy-x-max
  -> physical colorbar time through --frame-interval-seconds

apps/plot_reconnection_timing.py
  -> Python 3
  -> matplotlib and numpy at runtime
  -> benchmark_runs/reconnection_200/fortran/run.log
  -> benchmark_runs/reconnection_200/fortran/quick.dat
  -> benchmark_runs/reconnection_200/kokkos_gpu/summary.csv
  -> benchmark_runs/reconnection_200/kokkos_cpu/summary.csv
  -> benchmark_runs/reconnection_200/reconnection_timing_panels.png
  -> benchmark_runs/reconnection_200/reconnection_timing.csv
  -> physical accumulated-runtime x-axis through --frame-interval-seconds

scripts/run_reconnection_three_cases.sh
  -> cmake configure/build for CPU and CUDA benchmark build directories
  -> optional isolated AVX512 Fortran configure/build when KPT_FORTRAN_BUILD_MODE=avx512
  -> apps/fortran_mhd_to_compact_field.cpp executable from the CPU benchmark build
  -> lixiaocanexample/stochastic-parker/bin/stochastic-mhd.exec
  -> apps/plot_reconnection_benchmark_spectra.py after all three cases finish
  -> apps/plot_reconnection_timing.py after all three cases finish
  -> benchmark_runs/reconnection_200/compact_field by default for Kokkos compact field inputs
  -> /home/liuyh/data/Athena++/athena_reconnection_test/bin_data by default for Fortran MHD inputs
  -> benchmark_runs/reconnection_200/{fortran,kokkos_gpu,kokkos_cpu,compact_field}

scripts/run_total_benchmark_suite.sh
  -> scripts/run_total_benchmark_suite_local.sh

scripts/run_total_benchmark_suite_local.sh
  -> scripts/run_reconnection_three_cases.sh for the KPT three-way benchmark
  -> /home/liuyh/CLionProjects/amrvac_nlfff_sphere/scripts/run_nlfff_compare_case.sh
     for the AMRVAC/AthenaK NLFFF benchmark
  -> benchmark_runs/combined_benchmark_YYYYMMDD_HHMMSS by default

scripts/run_reconnection_optimized_comparison.sh
  -> scripts/run_reconnection_three_cases.sh
  -> KPT_TRANSPORT_MODEL=parker
  -> KPT_FORTRAN_BUILD_MODE=avx512
  -> benchmark_runs/reconnection_optimized_200 by default

scripts/run_reconnection_focused_comparison.sh
  -> scripts/run_reconnection_three_cases.sh
  -> KPT_TRANSPORT_MODEL=focused
  -> KPT_FORTRAN_BUILD_MODE=avx512
  -> benchmark_runs/reconnection_focused_200 by default

scripts/run_fortran_reconnection_avx512.sh
  -> copied Fortran source tree under benchmark_runs/fortran_avx512_build/source
  -> cmake configure/build of stochastic-mhd.exec with explicit GCC AVX512 ISA flags
  -> lixiaocanexample/stochastic-parker/examples/reconnection_2d for runtime config
  -> /home/liuyh/data/Athena++/athena_reconnection_test/bin_data by default for Fortran MHD inputs
  -> benchmark_runs/reconnection_200_fortran_avx512/fortran

scripts/run_parker_emission_synthesis.sh
  -> particleEmission.simple_emission_runner
  -> kokkos_cpu_format_example_frame180_262k by default, or KPT_PARKER_RUN_ROOT
  -> optional KPT_EMISSION_BACKGROUND_PATH emission-background HDF5 map
  -> benchmark_runs/parker_emission_synthesis/parker_emission_config.json
  -> benchmark_runs/parker_emission_synthesis/parker_emission_NNNNN.h5
  -> benchmark_runs/parker_emission_synthesis/parker_emission_NNNNN.png

particleEmission/parker_emission_movie.py
  -> Python 3
  -> numpy, h5py, matplotlib, optional imageio-ffmpeg, and local microwave backend at runtime
  -> compact fieldNNNNN.bin inputs
  -> version-4 particles_NNNNN.bin inputs
  -> optional emission_background_NNNNN.h5 inputs
  -> benchmark_runs/parker_emission_movie/frames/frame_NNNNN.png
  -> benchmark_runs/parker_emission_movie/frame_index.csv
  -> optional benchmark_runs/parker_emission_movie/parker_emission_movie.mov

particleEmission/microwave_backend.py
  -> pygsfit_cp-main/pygsfit_cp/fortran_src by default for backend source files
  -> local rebuilt shared library loaded through ctypes

particleEmission/simple_emission_runner.py
  -> kokkos_cpu_format_example_frame180_262k by default through the shell bridge
  -> optional emission_background_NNNNN.h5 maps from apps/athena2bin.py

scripts/render_parker_emission_movie.sh
  -> /tmp/kpt_postprocess_venv/bin/python by default, or KPT_MOVIE_PYTHON
  -> particleEmission.parker_emission_movie
  -> benchmark_runs/local_parker_speed_accuracy/{compact_field,kokkos_cpu} by default

scripts/run_kokkos_cpu_parker_movie_pipeline.sh
  -> cmake configure/build for cmake-build-benchmark-cpu
  -> apps/fortran_mhd_to_compact_field.cpp executable for compact field preparation
  -> apps/athena2bin.py --skip-compact-output for optional background-map preparation
  -> cmake-build-benchmark-cpu/kokkos_particle_transport_app
  -> particleEmission.parker_emission_movie
  -> benchmark_runs/kokkos_cpu_parker_movie_YYYYMMDD_HHMMSS by default

scripts/run_with_email_notification.sh
  -> any shell command passed after --
  -> Python 3 standard library email/smtplib/gzip modules
  -> optional local sendmail command when present
  -> SMTP settings from KPT_NOTIFY_* environment variables
  -> optional KPT_NOTIFY_SMTP_CONNECT_HOST override for DNS-proxied SMTP networks
  -> benchmark_runs/email_notifications by default for wrapper logs

CMakePresets.json
  -> configure preset linux-kokkos-cpu-debug
     -> /usr/local/kokkos_cpu
  -> configure preset linux-kokkos-cpu-release
     -> /usr/local/kokkos_cpu
  -> configure preset linux-kokkos-cuda-release
     -> /usr/local/kokkos
     -> /usr/local/kokkos/bin/nvcc_wrapper

CMakeUserPresets.json
  -> configure preset llvm-kokkos-debug
     -> /opt/homebrew/opt/llvm/bin/clang
     -> /opt/homebrew/opt/llvm/bin/clang++
     -> /opt/homebrew/opt/kokkos
     -> /opt/homebrew/opt/libomp

ParkerSolver
  -> SolverBase
     -> Grid
     -> Field
     -> ParticleSystem
     -> UnitConverter
  -> FieldInterpolator
  -> ParkerCoefficient
  -> RandomManager

FocusTransportSolver
  -> SolverBase
     -> Grid
     -> Field
     -> ParticleSystem including scalar momentum and mu storage
     -> UnitConverter
  -> FieldInterpolator
  -> FocusCoefficient
  -> LegencyModel for pitch-angle scattering and gamma scaling
  -> RandomManager

Grid
  -> KokkosDevice
  -> OrthogonalCoordinate

StoredCoordinateGrid
  -> Grid

AnalyticCoordinateGrid
  -> MappedCoordinateGrid
  -> Kokkos_MathematicalFunctions

MappedCoordinateGrid
  -> Grid
  -> Kokkos_MathematicalFunctions

GridValidation
  -> AnalyticCoordinateGrid
  -> StoredCoordinateGrid
  -> stdexcept/string host diagnostics

FieldBinaryWriter
  -> FieldOutputCommon
  -> FieldBinaryIO

AthenaFieldReader
  -> Field
  -> GridValidation
  -> AnalyticCoordinateGrid
  -> Kokkos host mirrors and deep_copy for host-to-device loading

FieldOutputCommon
  -> MappedCoordinateHostAccess for generic mapped-grid host coordinate providers

FieldInterpolator
  -> Field
  -> concrete coordinate grids through coordinate/grid_index/contains/wrap APIs
  -> KokkosDevice through grid/field execution-space traits

FieldOperator
  -> Field
  -> KokkosDevice through grid/field execution-space traits

ParticleSystem
  -> Grid
  -> KokkosDevice
  -> PhysicalConstant
  -> UnitConverter
  -> Kokkos_Sort

ParticleSystemDebugger
  -> ParticleSystem
  -> Kokkos host mirrors
  -> std::ostream host output

ParkerSolverDebugger
  -> ParkerSolver
  -> UnitConverter for code-to-CGS diagnostic scaling
  -> std::ostream host output

FocusTransportSolverDebugger
  -> FocusTransportSolver
  -> UnitConverter for code-to-CGS diagnostic scaling
  -> std::ostream host output

ParkerCoefficient
  -> Field
  -> FieldOperator for solar-wind divergence setup
  -> LegencyModel for legacy diffusion branch selection and gamma scaling conventions
  -> KokkosDevice through grid/field execution-space traits

FocusCoefficient
  -> Field
  -> FieldOperator for magnitude, direction, gradient, and divergence setup
  -> ParkerCoefficient for the shared orthogonal tensor-divergence backend
  -> LegencyModel for legacy diffusion fields and pitch-angle scattering parameters
  -> KokkosDevice through grid/field execution-space traits

LegencyModel
  -> Field
  -> ParticleSystem for ParticleProperties
  -> TurbulenceProperties
  -> KokkosDevice through grid/field execution-space traits

ParticleGenerator
  -> Field
  -> ParticleSystem
  -> RandomManager

TurbulenceProperties
  -> Field
  -> PhysicalConstant
  -> UnitConverter

RandomManager
  -> KokkosDevice

UnitConverter
  -> KokkosDevice

PhysicalConstant
  -> UnitConverter
```

## File-by-File Interface Map

### readme.md

- Chinese project README.
- Current responsibilities:
  - gives the normal project entry point for local users;
  - summarizes the Kokkos transport targets, Athena++/Fortran background inputs,
    particle diagnostics, emission synthesis, and movie pipeline;
  - documents the local CPU/CUDA build commands, compact field format, Parker/focused
    transport run commands, and frame-boundary walltime controls;
  - records the practical data-volume warning for 200-frame particle snapshots;
  - points maintainers to `docs/code_map.md` for the detailed interface map.
- Current notes:
  - this file is intentionally Chinese because it is the primary README requested for
    the current local workflow;
  - physics formulas remain schematic and should stay aligned with active solver
    behavior when coefficient or SDE conventions change.

### README.en.md

- English companion project README.
- Current responsibilities:
  - mirrors `readme.md` at the workflow level for English-speaking collaborators;
  - documents the same build, input, transport, walltime, benchmark, and emission movie
    entry points without adding a separate technical contract.
- Current notes:
  - keep this file synchronized with `readme.md` when user-facing workflow commands or
    run-size defaults change.

### emissionValidation/

- Tracked synthetic emission-validation product directory.
- Current responsibilities:
  - stores the current validation artifacts for the uniform-source and analytic loop-top
    cases produced by `particleEmission/validation_cases.py`;
  - provides known HDF5 products and quicklook figures for exercising
    `particleEmission/emission_viewer_app.py` without rerunning validation first.
- Current notes:
  - despite the name, this directory is currently tracked reference output rather than a
    disposable `benchmark_runs/` result tree.

### kokkos_cpu_format_example_frame180_262k/

- Tracked Parker CPU example input/output bundle.
- Current responsibilities:
  - provides `compact_field/field00180.bin` and `compact_field/field00181.bin`;
  - provides `kokkos_cpu/particles_00181.bin`, `summary.csv`, `run.log`,
    `momentum_histogram_00181.csv`, and `console.log`;
  - provides `manifest.txt`, `format_summary.txt`, and `prepare_compact_field.log` for
    provenance of the compact-field and particle snapshot example;
  - acts as the default input root for `scripts/run_parker_emission_synthesis.sh` when no
    `KPT_PARKER_RUN_ROOT` override is supplied.
- Current notes:
  - keep this directory small and curated; large rerun products belong under
    `benchmark_runs/`, which is ignored by Git.

### pygsfit_cp-main/

- Bundled external gyrosynchrotron fitting/synthesis package.
- Current responsibilities:
  - provides Fortran source files under `pygsfit_cp/fortran_src/` used by
    `particleEmission/microwave_backend.py` for local backend compilation;
  - carries upstream Python wrappers, demos, documentation, and platform-specific
    library artifacts that help verify backend calling conventions.
- Current notes:
  - treat this as third-party/reference source unless an explicit task asks to update the
    backend package itself;
  - current emission synthesis uses a narrow ctypes wrapper rather than importing the
    full upstream Python application workflow.

### CMakeLists.txt

- Project build entry point.
- Current responsibilities:
  - requires CMake 3.24 and C++20;
  - exports `compile_commands.json` for editor and tool integration;
  - accepts `Kokkos_ROOT` and `Kokkos_DIR` as cache/environment inputs;
  - accepts `OpenMP_ROOT` as a cache/environment input;
  - applies Homebrew default paths for Kokkos and libomp on macOS when they are not
    explicitly provided;
  - preconfigures Clang/libomp OpenMP discovery on macOS before locating Kokkos;
  - uses `find_package(Kokkos CONFIG)` to locate an installed Kokkos package;
  - defines `kokkos_particle_transport` as an `INTERFACE` library target;
  - links the header-only project target against `Kokkos::kokkos`;
  - defines `kokkos_particle_transport_app` as the reconnection calibration executable;
  - defines `fortran_mhd_to_compact_field` as a standard-library converter executable
    used by the benchmark script to prepare Kokkos field inputs from the same Fortran MHD
    frames;
  - detects a CUDA-enabled Kokkos package through the `Kokkos::CUDA` target;
  - omits `src/header_smoke.cpp` from the CUDA app so the runnable GPU calibration path
    is not blocked by CPU-only smoke coverage;
  - adds `run_reconnection_smoke`, `run_reconnection_particle_64000`, and
    `run_reconnection_fortran_global` custom targets for one-click local runs, passing
    `--overwrite-output` explicitly so repeated IDE smoke/calibration targets remain
    rerunnable under the solver's output-directory overwrite guard;
  - adds `run_reconnection_three_cases_200`, which invokes the top-level benchmark shell
    script for the sequential Fortran, Kokkos CUDA, and Kokkos CPU 200-frame run.
- Current notes:
  - IDE configuration must provide a visible Kokkos package path if it is not already on
    `CMAKE_PREFIX_PATH`.
  - The custom targets use `KPT_RECONNECTION_PARTICLE_FIELD_DIR`,
    `KPT_RECONNECTION_FULL_FIELD_DIR`, and `KPT_RECONNECTION_OUTPUT_ROOT` cache
    variables so CLion or command-line users can redirect field/input paths without
    editing source.
  - `run_reconnection_three_cases_200` intentionally delegates to a shell script because
    it needs to configure and build two separate Kokkos installations before running the
    three cases in a fixed order.
  - HDF5 conversion is intentionally outside CMake; the Python converter depends on
    `h5py` and `numpy` at runtime instead of linking the C++ target to HDF5.

### CMakePresets.json

- Linux CMake preset entry point for command-line and CLion loading.
- Current responsibilities:
  - defines `linux-kokkos-cpu-debug` against `/usr/local/kokkos_cpu`;
  - defines `linux-kokkos-cpu-release` against `/usr/local/kokkos_cpu`;
  - defines `linux-kokkos-cuda-release` against `/usr/local/kokkos` and
    `/usr/local/kokkos/bin/nvcc_wrapper`;
  - defines matching build presets for all three configure presets.
- Current notes:
  - the CUDA preset expects the installed Kokkos package to have CUDA enabled with an
    architecture compatible with the local RTX 4060 Ti (`ADA89` in the current install);
  - presets use `Unix Makefiles` so they work from a normal shell even when CLion's
    bundled Ninja is not on `PATH`.

### CMakeUserPresets.json

- User-local CMake preset entry point for IDE loading.
- Current responsibilities:
  - defines the `llvm-kokkos-debug` configure preset;
  - pins the Homebrew LLVM compiler toolchain;
  - provides `Kokkos_ROOT` for `find_package(Kokkos CONFIG)`;
  - provides `OpenMP_ROOT` for `find_package(OpenMP)`;
  - defines a matching build preset for the same configure preset.
- Current notes:
  - this file is intended to avoid manual per-profile editing inside CLion;
  - paths are currently tailored to this machine's Homebrew installation.

### src/header_smoke.cpp

- Header smoke translation unit.
- Current responsibilities:
  - includes all active local headers in one translation unit;
  - includes `AthenaFieldReader.hpp` so the solver background-field reader is covered by
    the normal build smoke path;
  - includes the field binary output headers;
  - gives the build system and IDE a concrete way to validate header parseability;
  - helps surface syntax and dependency regressions early during scaffolding.

### src/main.cpp

- Li Xiaocan `reconnection_2d` calibration and timing executable.
- Current responsibilities:
  - initializes and finalizes Kokkos;
  - owns named run profiles in source (`particle-64000`, `smoke`, `fortran-rank`,
    `fortran-global`, and `full`) so normal local runs do not require long command-line
    argument lists;
  - defaults to the two-frame `particle-64000` profile: 64000 injected particles per
    frame, `rank_scale=1`, `capacity=1000000`, and the Fortran-derived
    `cmake-build-debug/field_fortran_check` field directory when it contains frames 0
    through 2;
  - parses command-line controls as profile overrides for compact Athena field location,
    output directory, transport model, frame range, particle injection rate, rank
    scaling, particle capacity, diagnostics cadence, random seed, MHD output cadence,
    splitting, focused pitch-angle controls, time interpolation, and optional walltime
    stopping controls;
  - exposes split-threshold controls through `--split-ratio` and
    `--pmin-split-over-p0`, matching the Fortran benchmark wrapper's `-sr` and `-ps`
    controls;
  - loads `fieldNNNNN.bin` files through `AthenaFieldReader.hpp` as uniform analytic
    x/y grids with periodic boundaries and two ghost cells;
  - precomputes a legacy 16-component Parker coefficient field per MHD frame containing
    `Vx`, `Vy`, `Bx`, `By`, `Bz`, `|B|`, the Parker velocity-gradient subset,
    magnetic-component gradients, and `|B|` gradients;
  - precomputes a separate 21-component focused coefficient field only for focused runs,
    adding `Vz` plus the x/y velocity-gradient components consumed by the focused
    pusher;
  - samples coefficient fields at particle positions shifted by `+0.5 dx/+0.5 dy` so
    the repository cell-centered grid matches the Fortran example's interpolation
    convention, where the first physical field value is sampled at `xmin/ymin`;
  - injects uniform particles with the Fortran example's delta momentum distribution
    (`p=p0`), random pitch-angle cosine, random injection time inside each MHD interval,
    and unit statistical weight;
  - advances particles through one interval with a device kernel matching either the
    non-focused 2D Parker branch in `particle_module.f90::push_particle_2d` or the
    focused 2D branch in `particle_module.f90::push_particle_2d_ft` for Cartesian,
    uniform-grid, periodic, `dpp_wave=0`, `dpp_shear=0`, `nlgc=false`,
    `include_3rd_dim=false` settings;
  - wraps periodic particle coordinates through a branch-first device helper that only
    falls back to `floor` when a single step crosses more than one full domain width,
    matching the Fortran common path while preserving robust large-jump wrapping;
  - applies the Fortran-style diffusion normalization
    `kappa_parallel = kpara0 * B^(gamma_turb-2) * (p/p0)^(3-gamma_turb)` and
    `kappa_perp = kret * kappa_parallel`;
  - in focused mode, includes along-field streaming, focused drift, focused adiabatic
    momentum change, pitch-angle advection, and pitch-angle diffusion with the same
    reduced `duu0 * (1-mu^2) * (|mu|^(gamma_turb-1) + 0.2)` branch used by the Fortran
    local example;
  - uses `Kokkos::Random_XorShift64_Pool` through `RandomManager` and
    `U[-sqrt(3), sqrt(3)]` stochastic increments, matching the active Fortran pusher
    branch rather than the normal sampler used by the generic `ParkerSolver`;
  - splits high-momentum particles after each interval using the same threshold sequence
    `pmin_split * p0 * split_ratio^split_level`;
  - writes `summary.csv` timing/statistics rows, `run.log` frame-by-frame timing logs
    with total elapsed time, momentum histogram CSV files for comparison with Fortran
    diagnostics, and `particles_NNNNN.bin` binary particle snapshots for direct
    particle-cloud analysis;
  - rejects accidental non-restart overwrites when the requested output directory already
    contains solver-owned artifacts (`summary.csv`, `run.log`, `particles_*.bin`, or
    `momentum_histogram_*.csv`), unless `--overwrite-output` is supplied;
  - can restart from a completed-frame `particles_NNNNN.bin` snapshot through
    `--restart-particle-snapshot`, restoring particle position, scalar momentum,
    pitch-angle cosine, weight, lifecycle status, and split level before continuing at
    the matching frame boundary;
  - can stop cleanly after a completed frame when the configured walltime limit minus
    reserve has been reached, writing final diagnostics for that completed frame before
    returning successfully. If the reserve is greater than the limit, the effective
    stop threshold is clamped to zero elapsed seconds, so the driver stops after the
    first completed frame.
- Local interfaces:
  - `Reconnection2DSettings`
    - Role: host-side run configuration mirroring the Fortran shell-script controls that
      matter for this reduced Parker calibration.
    - Key data members: profile name, field/output paths, optional restart particle
      snapshot path, frame range, particles per frame, rank scale, capacity, seed,
      diagnostic cadence, histogram cadence, particle snapshot cadence, transport model,
      split/time-interpolation switches, overwrite-output switch, walltime
      limit/reserve seconds, MHD `dt_out`, `p0`, `pmin`, `pmax`,
      `gamma_turb`, `kpara0`, `kret`, `dt_min_rel`, `dt_max_rel`, drift parameters,
      charge, `particle_v0`, `duu0`, `mu_max`, and splitting thresholds.
    - Execution-space assumptions: host-only parsing and validation.
    - Call points: built by `parse_settings`, consumed by `run`, and converted to
      `Reconnection2DDeviceSettings`.
  - `apply_run_profile`
    - Role: source-owned profile switch that resets host settings to a known local run
      before applying explicit command-line overrides.
    - Supported profiles: `particle-64000`/`particle-test`, `smoke`, `fortran-rank`,
      `fortran-global`, and `full`.
    - Execution-space assumptions: host-only; may inspect the local filesystem to prefer
      `cmake-build-debug/field_fortran_check` for short tests.
    - Call points: invoked by `parse_settings` after pre-scanning `--profile`.
  - `Reconnection2DDeviceSettings`
    - Role: device-copyable scalar configuration for injection, pusher, and splitting
      kernels.
    - Key data members: physical bounds/widths, uniform `dx/dy`, interpolation coordinate
      shifts, interval duration, diffusion/drift/focused/splitting constants, timestep
      bounds, and boolean switches as integers.
    - Execution-space assumptions: trivially copied into Kokkos lambdas; no STL members.
    - Call points: built by `make_device_settings`, captured by
      `inject_uniform_particles`, `move_particles_one_interval`, and `split_particles`.
  - `wrap_periodic_coordinate`
    - Role: device-side periodic coordinate wrapper for the reconnection pusher.
    - Key behavior: uses compare/add/subtract for the normal one-domain-crossing path and
      applies a `floor`-based fallback only for rare multi-width jumps.
    - Execution-space assumptions: `KOKKOS_INLINE_FUNCTION`; valid in CPU and GPU
      kernels.
    - Call points: used before field interpolation and before storing final positions in
      `move_particles_one_interval`.
  - `square`
    - Role: device-side scalar squaring helper used where the Fortran source uses `**2`.
    - Execution-space assumptions: `KOKKOS_INLINE_FUNCTION`; avoids the general `pow`
      path for exact squares in CPU and GPU kernels.
    - Call points: used in drift-denominator and timestep-limit calculations in
      `move_particles_one_interval`.
  - `clamp_pitch_angle_cosine`
    - Role: device-side focused-transport limiter that clamps `mu` into
      `[-mu_max, mu_max]`, matching the Fortran local focused pusher.
    - Execution-space assumptions: `KOKKOS_INLINE_FUNCTION`; valid in CPU and GPU
      kernels.
    - Call points: used when reading, updating, and storing `mu` in
      `move_particles_one_interval`.
  - `ReconnectionParticleStorage`
    - Role: lightweight device-backed particle arrays for the Fortran-style calibration,
      including per-particle `time` and `step_dt` that the generic `ParticleSystem`
      currently does not store.
    - Key data members: host `count/capacity`, 2D `position`, `momentum`, `time`,
      `step_dt`, `weight`, `mu`, integer `status`, and integer `split_level`.
    - Execution-space assumptions: arrays use the repository `Device` abstraction and
      the device's default layout; host cache updates occur only after append/split calls.
    - Call points: allocated in `run`, filled by `inject_uniform_particles`, advanced by
      `move_particles_one_interval`, split by `split_particles`, and copied to host by
      diagnostics.
  - `infer_particle_snapshot_frame`
    - Role: host-side parser for repository particle snapshot names of the form
      `particles_NNNNN.bin`.
    - Call points: used by `parse_settings` so `--restart-particle-snapshot` can infer
      `start_frame` when no explicit `--start-frame` is supplied.
  - `is_reconnection_output_artifact` / `output_directory_has_reconnection_outputs` /
    `require_fresh_output_directory`
    - Role: host-side overwrite guard for non-restart runs.
    - Protected artifacts: `summary.csv`, `run.log`, `particles_*.bin`, and
      `momentum_histogram_*.csv` in the selected output directory.
    - Call points: `run` invokes the guard before creating or opening output files;
      restart runs append/continue, and explicit `--overwrite-output` fresh runs bypass
      the guard.
  - `load_reconnection_particle_snapshot`
    - Role: reads a version-4 repository particle binary snapshot into
      `ReconnectionParticleStorage` for frame-boundary restart.
    - Restored data: current 2D position, scalar momentum magnitude, pitch-angle cosine
      `mu`, macro-particle weight, lifecycle status, and split level.
    - Restart assumptions: particle binary snapshots do not carry the reconnection
      pusher's per-particle substep time or Kokkos random-pool state, so the loader sets
      every particle time to `start_frame * dt_out`, initializes `step_dt` from
      `dt_min_rel * dt_out`, and reseeds the random pool from the configured seed plus
      a frame-dependent offset. This is a statistical frame-boundary restart, not a
      bitwise replay checkpoint.
    - Call points: invoked by `run` before loading the first coefficient frame when
      `Reconnection2DSettings::restart_particle_snapshot_path` is non-empty.
  - `write_reconnection_particle_snapshot`
    - Role: writes the custom reconnection particle arrays to the repository particle
      binary snapshot format for offline conversion and particle-cloud inspection.
    - Output format: version-4 `KPTPRT\0\0` particle snapshot containing synthetic
      per-storage-index IDs, status, split level, current 2D position, scalar momentum
      magnitude, pitch-angle cosine `mu`, and macro-particle weight.
    - Current notes: the reduced reconnection storage does not keep previous positions,
      so previous-position fields in this compatibility snapshot are filled with the
      current position. Per-particle `time` and `step_dt` remain runtime diagnostics and
      are not part of particle binary format version 4.
    - Call points: invoked by `run_transport_loop` when
      `particle_snapshot_interval > 0`, always including the final completed frame for
      normal end-frame completion or walltime stop.
  - `ReconnectionCoefficientField`
    - Role: alias for the 21-component field consumed by the focused calibration pusher.
    - Execution-space assumptions: uses the loaded Athena grid, same ghosted storage, and
      `Field` random-access support for particle interpolation.
    - Call points: produced by `make_reconnection_coefficients` and
      `load_coefficient_frame`, sampled by `FieldInterpolator::LinearPositionInterpolator`
      inside the particle mover.
  - `ParkerReconnectionCoefficientField`
    - Role: alias for the 16-component field consumed by the Parker calibration pusher,
      preserving the pre-focused interpolation payload and Parker timing path.
    - Execution-space assumptions: uses the loaded Athena grid, same ghosted storage, and
      `Field` random-access support for particle interpolation.
    - Call points: produced by `make_parker_reconnection_coefficients` and
      `load_parker_coefficient_frame`, sampled by
      `FieldInterpolator::LinearPositionInterpolator` inside
      `move_parker_particles_one_interval`.
  - `move_parker_particles_one_interval`
    - Role: device-side legacy Parker interval pusher using the 16-component coefficient
      field and the original non-focused random-number sequence.
    - Execution-space assumptions: Kokkos range kernel over active stored particles.
    - Call points: selected by `run_transport_loop` when `--transport parker` is active.
  - `run_transport_loop`
    - Role: transport-coefficient-type-templated frame loop that keeps result writing,
      particle snapshots, injection, splitting, walltime checks, and diagnostics shared
      while dispatching to Parker or focused movers at compile time.
    - Execution-space assumptions: host-side orchestration; kernel dispatch is delegated
      to the selected mover.
    - Call points: invoked by `run` after loading either Parker or focused coefficient
      frames.
- Current limitations:
  - This driver targets the exact `reconnection_2d` branch used by the provided script,
    not the full Fortran feature matrix. NLGC, momentum diffusion, 3D-in-2D motion,
    open particle boundaries, MPI exchange, HDF5 local distributions, and
    tracked-particle outputs are intentionally outside this calibration entry point.
  - Kokkos walltime stopping is a clean frame-boundary stop. It preserves diagnostics
    and optional particle snapshots for the last completed frame; those snapshots can be
    used for frame-boundary particle-state restart, but the driver still does not save
    RNG state for bitwise-identical continuation.
  - `particle-64000` is a local particle-adaptation and timing profile, not the original
    Fortran script's particle count. Use `--profile fortran-rank` for one Fortran MPI
    rank (`1600` particles/frame), or `--profile fortran-global`/`--profile full` for
    the script-global 16-rank injection count (`1600 * 16` particles/frame).

### apps/athena2bin.py

- Python Athena++ background-field extraction tool.
- Current responsibilities:
  - reads Athena++ `.athdf` files through `h5py`;
  - takes the first two values of `RootGridX1` and `RootGridX2` as the physical x/y
    coordinate bounds;
  - reads Athena++ `DatasetNames`, `NumVariables`, and `VariableNames` metadata to map
    `prim` components by name instead of assuming a fixed raw index order;
  - reconstructs a 2D global mesh from Athena++ mesh blocks using `LogicalLocations`;
  - extracts cell-centered magnetic components `Bcc1`, `Bcc2`, `Bcc3` and primitive
    velocity components `vel1`, `vel2`, `vel3`;
  - optionally extracts primitive density and pressure-like components (`rho`, `press`,
    or common aliases) and writes an emission-background HDF5 file per frame;
  - resamples the reconstructed arrays onto a uniform target grid with SciPy
    `RegularGridInterpolator` when available, otherwise with a NumPy bilinear fallback;
  - writes one compact binary file per input snapshot under `./field/`;
  - exposes a CLI for input glob/explicit files, output frame range, target shape, and
    optional background-map scaling through reference density/temperature values;
  - can run in `--skip-compact-output` mode to write only emission-background HDF5 maps
    without duplicating compact solver field files.
- Current binary payload:
  - `int32 nx`
  - `int32 ny`
  - `double x_edges[nx + 1]`
  - `double y_edges[ny + 1]`
  - `double Bx[nx * ny]`
  - `double By[nx * ny]`
  - `double Bz[nx * ny]`
  - `double Vx[nx * ny]`
  - `double Vy[nx * ny]`
  - `double Vz[nx * ny]`
- Current notes:
  - Optional emission-background files use schema `kpt_emission_background_v1` and store
    `x_edges`, `y_edges`, `rho_code`, optional `pressure_code`,
    `temperature_proxy_code`, `thermal_density_cm3`, and `temperature_mk`. The physical
    density/temperature maps are reference-scaled by mean or max normalization unless
    `--background-normalization none` is requested.
  - The payload has no magic, version, endian marker, units, time stamp, or coordinate
    system metadata. The C++ reader therefore validates only dimensions, payload length,
    monotonic coordinates, finite values, and caller-supplied unit scales.
  - Component arrays are flattened in row-major 2D order with x-index contiguous:
    `source_index = j * nx + i`. This matches `Field` flattening for a 2D grid because
    repository field storage also makes dimension zero contiguous.
  - The script explicitly requires a 2D input (`nz == 1`) before sampling
    `data[..., 0, :, :]`.
  - The output edges represent cell faces, while the field arrays are cell-centered.
  - Source interpolation coordinates are built at cell centers,
    `x_min + (i + 0.5) dx` and `y_min + (j + 0.5) dy`, matching cell-centered Athena++
    values rather than treating source samples as domain endpoints.
  - Cubic interpolation is smooth but can overshoot and does not preserve `div B = 0`
    exactly because vector components are resampled independently. The NumPy fallback
    avoids the SciPy dependency but is only bilinear.
  - Values remain in Athena/code units unless the caller applies scale factors while
    loading or converting.

### particleEmission/background_io.py

- Python emission-background map reader.
- Current responsibilities:
  - reads HDF5 files using the `kpt_emission_background_v1` layout written by
    `apps/athena2bin.py --emission-background-dir`;
  - validates `x_edges`, `y_edges`, `thermal_density_cm3`, and `temperature_mk` shape,
    finiteness, and non-negativity;
  - coarsens thermal-density and temperature maps onto the simple emission image grid
    with the same `coarse_average_2d` helper used for compact magnetic fields.
- Local interfaces:
  - `EmissionBackgroundMaps`
    - Role: in-memory container for thermal/background maps on a 2D source grid.
    - Key data members: `x_edges`, `y_edges`, `thermal_density_cm3`,
      `temperature_mk`, `source_path`, and `source_kind`.
    - Execution-space assumptions: Python/NumPy host-only post-processing.
    - Call points: returned by `read_emission_background_maps` and consumed by
      `coarse_background_maps` inside `particleEmission/simple_emission_runner.py`.
  - `read_emission_background_maps`
    - Role: load one HDF5 background map with clear dependency errors when `h5py` is
      unavailable.
  - `coarse_background_maps`
    - Role: produce image-grid thermal density and temperature arrays for per-pixel
      microwave source closure.

### apps/field_binary_to_xdmf_hdf5.py

- Python field-output conversion tool.
- Current responsibilities:
  - reads repository custom binary field snapshot format versions 1 and 2;
  - reads version-2 binary unit metadata including coordinate/value unit labels and CGS
    scale factors;
  - applies metadata-derived CGS dimensionalization by default, with command-line
    overrides for scale factors, unit labels, and field name;
  - writes ParaView-oriented HDF5 heavy data plus an XDMF sidecar;
  - returns nonzero exit codes for invalid CLI usage or conversion failures.
- Current notes:
  - the converter is not a CMake target;
  - it requires Python 3 and imports `h5py` plus `numpy` at runtime;
  - the converter is the AthenaK-style offline output path: simulations may write a
    lightweight binary snapshot first, then convert to HDF5/XDMF after the run.
  - `--keep-code-units` writes raw code-unit payloads instead of applying CGS scale
    metadata.

### apps/fortran_mhd_to_compact_field.cpp

- C++ Fortran-MHD-to-compact-field conversion executable.
- Current responsibilities:
  - reads `mhd_config.dat` written by the Fortran `mhd_config_module`;
  - reads `mhd_data_NNNN` frames containing single-precision Fortran-ordered arrays with
    variables `vx`, `vy`, `vz`, `rho`, `bx`, `by`, `bz`, and `btot`;
  - extracts the physical 2D domain from the Fortran ghosted storage;
  - writes the compact field format consumed by `AthenaFieldReader.hpp` with component
    order `Bx`, `By`, `Bz`, `Vx`, `Vy`, `Vz`;
  - supports `--input-dir`, `--output-dir`, `--start-frame`, `--end-frame`, and
    `--config` command-line controls.
- Current notes:
  - this converter is intentionally C++ instead of Python so the benchmark setup does not
    depend on NumPy availability in the shell used by CLion/CMake;
  - it currently supports the 2D `nz=1` reconnection data layout used by the bundled
    Fortran example.

### apps/plot_reconnection_benchmark_spectra.py

- Python reconnection benchmark post-processing tool.
- Current responsibilities:
  - reads Fortran global distribution files `fdists_NNNN.h5` from the benchmark
    `fortran` output directory;
  - reads Kokkos momentum histogram CSV files from the benchmark `kokkos_gpu` and
    `kokkos_cpu` output directories;
  - samples frames `10, 20, ..., 200` by default, with command-line controls for start,
    end, and stride;
  - sums the Fortran `fglobal` distribution over non-momentum axes to produce an
    all-particle momentum spectrum;
  - converts all cases to `dN/dlog10(p)` using the stored logarithmic momentum-bin
    edges;
  - writes a combined CSV table and a three-panel Matplotlib figure using the `plasma`
    colormap to encode increasing physical frame time;
  - converts transport momentum to kinetic energy in keV using
    `E_keV=(sqrt(1+p^2)-1)/(sqrt(1+p0^2)-1)*E0_keV`, defaulting to
    `p0=0.1` and `E0_keV=1.0`;
  - fixes the energy x-axis to dense 1-2-5 logarithmic ticks spanning all loaded
    spectra, or to explicit `--energy-x-min`/`--energy-x-max` limits when supplied,
    so each panel has readable tick labels instead of relying on Matplotlib's sparse
    automatic log ticks.
- Current notes:
  - the script expects `h5py`, `matplotlib`, and `numpy` in the Python environment;
  - it is intentionally not wired into CMake because it is an interactive analysis step
    normally run from PyCharm or a plotting environment.

### apps/plot_reconnection_timing.py

- Python reconnection benchmark timing post-processing tool.
- Current responsibilities:
  - reads Fortran per-frame wall times from `fortran/run.log`;
  - reads Fortran frame-indexed active particle counts from `fortran/quick.dat`;
  - reads Kokkos CPU and GPU frame timing, accumulated timing, and active particle counts
    from each case's `summary.csv`;
  - writes a combined timing CSV with columns `case`, `frame`, `particle_count`,
    `physical_time_seconds`, `frame_seconds`, and `elapsed_seconds`;
  - writes a two-panel Matplotlib figure where the first panel plots frame time against
    particle count and the second panel plots accumulated runtime against physical
    simulation time;
  - keeps one fixed color per solver across both timing panels.
- Current notes:
  - the script expects `matplotlib` and `numpy` in the Python environment;
  - Fortran accumulated time is reconstructed as the cumulative sum of printed per-frame
    step times, while Kokkos accumulated time comes from `total_elapsed_seconds`;
  - it is intentionally not wired into CMake because it is an interactive analysis step
    normally run from PyCharm or a plotting environment.

### apps/particle_binary_to_xdmf_hdf5.py

- Python particle-output visualization conversion tool.
- Current responsibilities:
  - reads repository particle binary snapshot versions 2, 3, and 4;
  - accepts both legacy version-2 magic `KPTPRT1\0` and current version-4 magic
    `KPTPRT\0\0`;
  - validates endian marker, `SpaceDim`, particle count, particle capacity, and
    dimensionless particle-property metadata before allocating payload arrays;
  - reconstructs 3-component visualization arrays from `SpaceDim`-component particle
    data by zero-padding missing coordinates;
  - computes relativistic diagnostic arrays from dimensionless momentum magnitude and
    particle properties: `momentum_magnitude`, `gamma`, `speed`, `kinetic_energy`, and
    `kinetic_energy_ev`;
  - derives ParaView-friendly scalar attributes including `initial_kinetic_energy_ev`
    and integer lifecycle masks `is_active`, `is_escaped`, and `is_inactive`;
  - writes HDF5 heavy data plus an XDMF `Polyvertex` sidecar for ParaView particle-cloud
    visualization, with per-particle arrays exposed as XDMF node attributes;
  - exposes visualization-only `--position-scale`, `--position-unit`,
    `--momentum-scale`, and `--momentum-unit` CLI controls.
- Current HDF5 datasets:
  - `position`
  - `previous_position`
  - `previous_step_position`
  - `momentum`
  - `momentum_magnitude`
  - `particle_id`
  - `status`
  - `is_active`
  - `is_escaped`
  - `is_inactive`
  - `split_level`
  - `sort_key`
  - `mu`
  - `weight`
  - `initial_kinetic_energy`
  - `initial_kinetic_energy_ev`
  - `gamma`
  - `speed`
  - `kinetic_energy`
  - `kinetic_energy_ev`
- Current notes:
  - The converter is not a CMake target.
  - It requires Python 3 and imports `h5py` plus `numpy` at runtime.
  - The XDMF file should be opened in ParaView; it references the HDF5 file as heavy
    data.
  - ParaView coloring/filtering should use XDMF node attributes such as
    `kinetic_energy_ev`, `initial_kinetic_energy_ev`, `momentum_magnitude`, `weight`,
    `status`, or `is_active`.
  - HDF5 datasets carry lightweight unit/description attributes where the converter can
    infer them.
  - Position and momentum scaling are output-only visualization controls. The binary
    snapshot currently stores dimensionless particle positions and scalar momentum
    magnitudes, plus the energy scale needed to report kinetic energy in eV.
  - Version-4 snapshots no longer store momentum direction. The HDF5 `momentum` vector
    dataset is retained for legacy ParaView compatibility and is zero-filled for
    version-4 inputs; color/filter by `momentum_magnitude` or energy diagnostics.

### scripts/run_reconnection_three_cases.sh

- Top-level sequential benchmark runner.
- Current responsibilities:
  - selects the transport equation through `KPT_TRANSPORT_MODEL=parker` or
    `KPT_TRANSPORT_MODEL=focused`;
  - can either use an existing Fortran executable or build an isolated AVX512 Fortran
    executable when `KPT_FORTRAN_BUILD_MODE=avx512`;
  - configures and builds a CPU Release Kokkos executable in
    `cmake-build-benchmark-cpu`;
  - configures and builds a CUDA Release Kokkos executable in
    `cmake-build-benchmark-cuda`;
  - generates Kokkos compact field files under the benchmark root from the same Fortran
    MHD data by default, unless `KPT_FIELD_DIR` is provided;
  - creates a benchmark-local Fortran configuration file derived from the example
    `conf_reconnection.dat` and patches the calibration diffusion parameters;
  - prepares both `fortran/restart` and benchmark-root `restart` directories before the
    Fortran reference launch so MT-stream PRNG state dumps can be written reliably in the
    benchmark workspace;
  - passes `-ft .true.` to the Fortran executable and `--transport focused` to the
    Kokkos executable for focused-transport runs;
  - exposes a walltime interface for scheduler-limited runs: Fortran receives
    `-qh ${KPT_FORTRAN_WALLTIME_HOURS}` and Kokkos receives
    `--walltime-hours/--walltime-reserve-minutes` when a Kokkos walltime is configured;
  - exposes particle splitting thresholds through `KPT_SPLIT_RATIO` and
    `KPT_PMIN_SPLIT_OVER_P0`, forwarding them to both the Fortran and Kokkos command
    lines;
  - runs the Fortran reference case, Kokkos CUDA case, and Kokkos CPU case in that order;
  - limits the default CPU budget to `16` cores through `KPT_CPU_CORES`, using that value
    for build parallelism, Fortran MPI rank count, and Kokkos CPU thread count unless
    more specific overrides are provided;
  - configures the CPU benchmark build with `KPT_CPU_CXX_FLAGS=-march=native` by default
    so host kernels can use the local CPU ISA, including AVX512 on machines that expose
    it;
  - writes separate result directories under the benchmark root:
    `fortran`, `kokkos_gpu`, and `kokkos_cpu`;
  - enables Kokkos particle binary snapshots by default at the same cadence as
    `KPT_HISTOGRAM_INTERVAL`, writing `particles_NNNNN.bin` files under each Kokkos
    result directory;
  - forwards `KPT_RESTART_PARTICLE_SNAPSHOT` or `KPT_KOKKOS_RESTART_PARTICLE_SNAPSHOT`
    to the Kokkos executable as `--restart-particle-snapshot`; when `KPT_START_FRAME`
    is unset, the wrapper infers it from `particles_NNNNN.bin`, and Kokkos case
    directories are preserved instead of deleted before launch;
  - writes configure/build logs plus per-case runtime logs so CLion runs can be inspected
    after completion;
  - runs `apps/plot_reconnection_benchmark_spectra.py` and
    `apps/plot_reconnection_timing.py` after all three cases complete, unless
    `KPT_SKIP_POSTPROCESS=1`;
  - sets the default post-processed energy-spectrum x-axis to `0.05`-`20` keV for
    the three-case comparison figure.
- Current notes:
  - defaults match the Fortran-global calibration count: `1600` injected particles per
    MPI rank, `16` ranks, and `200` frames;
  - the Parker default benchmark root is `benchmark_runs/reconnection_200`, while the
    focused default benchmark root is `benchmark_runs/reconnection_focused_200`;
  - the CUDA case defaults to one host thread through `KPT_KOKKOS_GPU_HOST_THREADS=1`,
    while the CPU case defaults to `KPT_KOKKOS_CPU_THREADS=${KPT_CPU_CORES}`;
  - `KPT_CPU_CXX_FLAGS` can be set to a more conservative value such as `-march=znver4`
    or an empty string when portable CPU binaries are needed;
  - `KPT_PREPARE_COMPACT_FIELD=0` disables automatic field conversion when a trusted
    compact field directory is supplied through `KPT_FIELD_DIR`;
  - `KPT_PARTICLE_SNAPSHOT_INTERVAL=0` disables Kokkos particle snapshots, while any
    positive value writes snapshots at that frame cadence and always includes the final
    frame;
  - `KPT_RECONNECTION_FRAME_SECONDS` supplies the physical frame interval used by the
    spectra colorbar and accumulated-runtime x-axis;
  - `KPT_WALLTIME_HOURS` sets a common scheduler-style walltime for Fortran and Kokkos;
    `KPT_FORTRAN_WALLTIME_HOURS` overrides the Fortran `-qh` value and defaults to
    `12.0` hours when no common walltime is set; `KPT_KOKKOS_WALLTIME_HOURS` enables
    Kokkos frame-boundary walltime stopping and overrides the common value for Kokkos;
    `KPT_KOKKOS_WALLTIME_RESERVE_MINUTES` or `KPT_WALLTIME_RESERVE_MINUTES` controls
    the Kokkos reserve, defaulting to `30` minutes when Kokkos walltime is enabled;
  - for walltime-limited production jobs, prefer `KPT_SKIP_POSTPROCESS=1` and run
    plotting after confirming the completed frame ranges, because spectra post-processing
    expects matching output files at its configured frame cadence;
  - environment variables such as `KPT_BENCHMARK_ROOT`, `KPT_FIELD_DIR`,
    `KPT_FORTRAN_MHD_DIR`, `KPT_TRANSPORT_MODEL`, `KPT_FORTRAN_BUILD_MODE`,
    `KPT_END_FRAME`, `KPT_CPU_CORES`, `KPT_MPI_SIZE`, `KPT_KOKKOS_CPU_THREADS`,
    `KPT_PARTICLE_V0`, `KPT_DUU0`, `KPT_CXX_PARTICLE_CAPACITY`,
    `KPT_PARTICLE_SNAPSHOT_INTERVAL`, and walltime variables can override local paths,
    transport model, output cadence, scheduler limits, and run size without editing the
    script.

### scripts/run_reconnection_optimized_comparison.sh

- Parker optimized-comparison wrapper.
- Current responsibilities:
  - sets conservative workstation defaults for a 16-core benchmark run;
  - defaults `KPT_TRANSPORT_MODEL=parker`;
  - defaults `KPT_FORTRAN_BUILD_MODE=avx512` so the Fortran reference is rebuilt with
    explicit AVX512 flags in an isolated source copy;
  - defaults `KPT_CPU_CXX_FLAGS=-O3 -march=native` for the Kokkos CPU executable;
  - delegates the build, sequential three-case execution, and plotting to
    `scripts/run_reconnection_three_cases.sh`;
  - writes results under `benchmark_runs/reconnection_optimized_200` unless
    `KPT_BENCHMARK_ROOT` overrides it.

### scripts/run_reconnection_focused_comparison.sh

- Focused-transport comparison wrapper.
- Current responsibilities:
  - keeps the reconnection input data, particle count, frame count, and plotting cadence
    aligned with the Parker optimized wrapper by default;
  - defaults `KPT_TRANSPORT_MODEL=focused`, causing the Fortran command to use
    `-ft .true.` and the Kokkos command to use `--transport focused`;
  - preserves isotropic pitch-angle-cosine injection through the Fortran and Kokkos
    `mu_max * (2 rand - 1)` initialization path;
  - defaults `KPT_FORTRAN_BUILD_MODE=avx512` and
    `KPT_CPU_CXX_FLAGS=-O3 -march=native`;
  - delegates the build, sequential three-case execution, and plotting to
    `scripts/run_reconnection_three_cases.sh`;
  - writes results under `benchmark_runs/reconnection_focused_200` unless
    `KPT_BENCHMARK_ROOT` overrides it.

### scripts/run_reconnection_parker_focused_comparisons.sh

- Sequential Parker-plus-focused comparison wrapper.
- Current responsibilities:
  - runs `scripts/run_reconnection_optimized_comparison.sh` first with
    `KPT_TRANSPORT_MODEL=parker`;
  - runs `scripts/run_reconnection_focused_comparison.sh` second with
    `KPT_TRANSPORT_MODEL=focused`;
  - forces separate per-case benchmark roots through `KPT_BENCHMARK_ROOT` so inherited
    shell state does not make the two runs overwrite each other;
  - writes a sequence manifest and per-case logs under
    `benchmark_runs/reconnection_parker_focused_sequence` by default.
- Current notes:
  - `KPT_PARKER_BENCHMARK_ROOT`, `KPT_FOCUSED_BENCHMARK_ROOT`, and
    `KPT_SEQUENCE_LOG_ROOT` can override the per-case result locations and the combined
    log directory;
  - command-line arguments passed to this wrapper are forwarded unchanged to both
    underlying comparison wrappers;
  - per-case Fortran configuration names can be overridden with
    `KPT_PARKER_FORTRAN_CONF_NAME` and `KPT_FOCUSED_FORTRAN_CONF_NAME`.

### scripts/run_local_parker_focused_speed_accuracy_test.sh

- Local Parker/focused speed-and-result comparison wrapper.
- Current responsibilities:
  - prepares `/tmp/kpt_postprocess_venv` with `numpy`, `h5py`, and `matplotlib` by
    default when `KPT_POSTPROCESS_PYTHON` is not already set;
  - runs `scripts/run_reconnection_parker_focused_comparisons.sh`, which in turn runs
    Fortran, Kokkos CUDA, and Kokkos CPU for Parker first and focused transport second;
  - writes local benchmark roots under `benchmark_runs/local_parker_speed_accuracy`,
    `benchmark_runs/local_focused_speed_accuracy`, and sequence logs under
    `benchmark_runs/local_parker_focused_speed_accuracy` by default;
  - parses each run's `reconnection_timing.csv` and `reconnection_spectra.csv` after
    post-processing completes;
  - writes `local_speed_accuracy_summary.csv` and
    `local_speed_accuracy_summary.txt` with elapsed-time speedups relative to Fortran
    plus relative spectrum differences between Kokkos and Fortran.
- Current notes:
  - defaults keep `KPT_PARTICLE_SNAPSHOT_INTERVAL=0` so solver speed comparisons are not
    dominated by particle binary I/O. Set `KPT_PARTICLE_SNAPSHOT_INTERVAL=10` or another
    positive cadence when particle snapshots are part of the test target.
  - `KPT_LOCAL_PREPARE_POSTPROCESS_PYTHON=0` disables automatic Python environment
    preparation; `KPT_LOCAL_POSTPROCESS_VENV` can change the venv path.
  - all standard `KPT_*` run-size controls from
    `scripts/run_reconnection_three_cases.sh` still apply.

### scripts/validate_reconnection_dt_match.sh

- Reconnection output time-step validation helper.
- Current responsibilities:
  - reads the Fortran MHD binary `mhd_config.dat` and extracts `dt_out` from the same
    field used by `apps/fortran_mhd_to_compact_field.cpp`;
  - reads a Kokkos `summary.csv` and verifies that every reported physical time equals
    `frame * dt_out` within floating-point tolerance;
  - verifies that the compact lower/upper field frames and final Kokkos particle
    snapshot exist under the selected run root;
  - reports the code-time interval converted to seconds using the default reconnection
    `tau_A=L0/v_A` normalization (`L0=5e6 m`, `B0=50 G`, `n0=1e10 cm^-3`) unless an
    override is supplied.
- Current defaults:
  - run root:
    `benchmark_runs/kokkos_cpu_format_example_frame180_262k`;
  - MHD config:
    `/home/liuyh/data/Athena++/athena_reconnection_test/bin_data/mhd_config.dat`;
  - summary CSV:
    `${run_root}/kokkos_cpu/summary.csv`.
- Current notes:
  - the script uses only Python standard-library modules, so it does not require NumPy
    or h5py;
  - optional arguments are `run_root`, `mhd_config`, and `summary_csv` in that order.
  - `KPT_TIME_UNIT_SECONDS` can override the physical time unit directly. Otherwise,
    `KPT_RECONNECTION_L0_M`, `KPT_RECONNECTION_B0_G`, and
    `KPT_RECONNECTION_N0_CM3` control the reported Alfvén-time conversion.

### scripts/run_with_email_notification.sh

- GPT-independent long-job notification wrapper.
- Current responsibilities:
  - runs any command supplied after `--` without requiring modifications to that command;
  - writes stdout/stderr plus command metadata to a selected log file;
  - preserves the wrapped command's exit code as the wrapper exit code by default;
  - sends an immediate `STARTED` email when the command begins;
  - sends periodic `RUNNING` status emails while the command is still active, defaulting
    to a 12-hour cadence and including the next automatic status time in every status
    email;
  - sends a final completion email with host, command, start/end times, duration, exit
    status, log path, tail lines, and an optional log attachment;
  - uses a local `sendmail` command when available, otherwise sends through SMTP using
    only the Python standard library.
- Current notes:
  - this workstation currently has Python 3 but no detected `sendmail`, `mail`,
    `mailx`, or `msmtp`, so practical use requires `KPT_NOTIFY_SMTP_HOST`,
    `KPT_NOTIFY_TO`, and the corresponding SMTP credential variables;
  - `KPT_NOTIFY_TO` defaults to `liu-yh@outlook.com` for this workstation;
  - when local DNS maps SMTP hosts to a proxy address, `KPT_NOTIFY_SMTP_CONNECT_HOST`
    can point at the real SMTP IP while `KPT_NOTIFY_SMTP_HOST` remains the TLS SNI and
    certificate host;
  - SMTP passwords should be provided through `KPT_NOTIFY_SMTP_PASSWORD_FILE` when
    possible, so credentials are not embedded in shell history or command lines;
  - `--dry-run-email` builds and prints the email payload without connecting to SMTP,
    which is the preferred local syntax test path;
  - `--status-interval-hours` or `KPT_NOTIFY_STATUS_INTERVAL_HOURS` controls the
    periodic status cadence; setting it to `0` disables periodic status emails while
    keeping start and final notifications;
  - `KPT_NOTIFY_SMTP_TIMEOUT_SECONDS` bounds SMTP connection attempts so notification
    failures do not indefinitely stall wrapper bookkeeping;
  - large logs are attached only up to `KPT_NOTIFY_ATTACH_MAX_MB`; the email body always
    includes the configured log tail.

### scripts/run_total_benchmark_suite.sh

- Thin compatibility launcher for the suite driver.
- Current responsibilities:
  - forwards all arguments directly to `scripts/run_total_benchmark_suite_local.sh`;
  - preserves the long-standing entry-point path used by existing terminal history,
    notes, and automation.

### scripts/run_total_benchmark_suite_local.sh

- Top-level benchmark suite driver.
- Current responsibilities:
  - computes the reconnection physical frame interval from the Fortran MHD
    `mhd_config.dat` `dt_out` and the same Alfvén-time normalization used by
    `scripts/validate_reconnection_dt_match.sh`;
  - runs the Kokkos-particleTransport three-way reconnection benchmark under
    `benchmark_runs/combined_benchmark_YYYYMMDD_HHMMSS/kpt_reconnection_three_way`;
  - mirrors the full suite stdout/stderr into `combined_benchmark_suite.log` through a
    process-substitution `tee` so local runs still leave a single suite-level log file;
  - configures that particle benchmark with every-10-frame spectra, `KPT_SPLIT_RATIO=1.2`
    by default, and disabled particle snapshots for the speed/spectrum run;
  - runs `/home/liuyh/CLionProjects/amrvac_nlfff_sphere/scripts/run_nlfff_compare_case.sh`
    with the benchmark contract described by that repository's `benchmark0419.md`;
  - writes `combined_benchmark_manifest.txt` with the suite entry script, suite log,
    particle benchmark root, NLFFF run directory, frame interval, and current/final
    status.
- Current notes:
  - `--no-email` is accepted as a no-op compatibility flag so older launch snippets do
    not fail after the notification wrapper was removed from the suite entry path;
  - `KPT_SUITE_ROOT`, `KPT_SUITE_PARTICLE_ROOT`, `KPT_NLFFF_BENCHMARK_ROOT`, and
    `KPT_NLFFF_RUN_NAME` can redirect the two benchmark output locations;
  - AMRVAC/AthenaK controls such as `NP`, `NITER`, `MF_DIAG_INTERVAL`,
    `OUTPUT_DCYCLE`, `REQUIRE_AVX512`, `CLEAN_AMRVAC_BUILD`, and `CLEAN_ATHENA_BUILD`
    are forwarded to the NLFFF benchmark script.

### scripts/run_parker_emission_synthesis.sh

- Parker emission bridge for the current particle-snapshot workflow.
- Current responsibilities:
  - discovers the latest `particles_NNNNN.bin` under the selected Kokkos Parker output
    directory, unless `KPT_PARKER_EMISSION_FRAME` or `KPT_PARTICLE_PATH` is supplied;
  - selects the matching compact field frame, falling back from `fieldNNNNN.bin` to the
    previous frame when the particle output frame is one interval beyond the loaded
    background frame;
  - creates an output root under `benchmark_runs/parker_emission_synthesis` by default,
    with symlinks to the selected field, particle snapshot, and optional background map;
  - writes `manifest.txt` and `parker_emission_config.json` so the synthesis inputs and
    coronal closure parameters are reproducible;
  - invokes `python -m particleEmission.simple_emission_runner` and writes
    `parker_emission_NNNNN.h5`, `parker_emission_NNNNN.png`, and `emission.log`.
- Current notes:
  - defaults target the repository Parker CPU format example
    `kokkos_cpu_format_example_frame180_262k`;
  - `KPT_PARKER_RUN_ROOT`, `KPT_FIELD_DIR`, `KPT_PARTICLE_DIR`, `KPT_FIELD_PATH`,
    `KPT_PARTICLE_PATH`, and `KPT_PARKER_EMISSION_ROOT` can redirect input/output
    locations;
  - `KPT_EMISSION_BACKGROUND_PATH` can point to an Athena/MHD converted background HDF5
    file from `apps/athena2bin.py --emission-background-dir`; otherwise the synthesis
    uses the constant thermal density and temperature from the generated config;
  - `KPT_EMISSION_*` variables override image size, viewing angle, LOS depth, magnetic
    field scaling, macro-particle electron count, and default thermal/source closure
    parameters.

### particleEmission/parker_emission_movie.py

- Parker diagnostic and emission movie renderer.
- Current responsibilities:
  - validates that each requested frame has a compact `fieldNNNNN.bin` and matching
    version-4 `particles_NNNNN.bin`, plus an optional
    `emission_background_NNNNN.h5`;
  - reads particle snapshots with a NumPy structured dtype for the scalar-momentum
    version-4 record layout, avoiding per-particle Python `struct` loops;
  - computes Bz and a code-unit current proxy `Jz = dBy/dx - dBx/dy` from the compact
    field and coarse-averages them onto the movie image grid;
  - deposits active particle weights below and above the configured kinetic-energy
    threshold, defaulting to `1.5 keV`;
  - supports
    `--energy-source auto|snapshot-kinetic|momentum-kev|transport-p0-kinetic-kev`;
    the default `auto` mode treats placeholder reconnection snapshots with custom
    species, `rest_mass=1`, `c=1`, and `energy_scale_erg=1` as relativistic
    transport-p0 kinetic-energy data using `--transport-p0` and
    `--transport-p0-energy-kev` instead of applying a bogus CGS kinetic-energy
    conversion;
  - reconstructs local nonthermal density and a power-law slope approximation from
    weighted particle energy histograms;
  - synthesizes microwave spectra on the requested frequency grid with the local
    backend, using quantized cache keys that include nonthermal density, magnetic field,
    power-law index, thermal density, and temperature;
  - writes a 2x4 PNG panel per frame: Bz, Jz, sub-threshold particles,
    super-threshold particles, beam-convolved 1/3/5 GHz images, and a full-domain
    spectrum panel;
  - optionally calls `ffmpeg` with `prores_ks`, profile 3, and `yuv422p10le` to write a
    ProRes MOV.
- Current notes:
  - movie encoding first uses `--ffmpeg` or `ffmpeg` from `PATH`, then falls back to the
    optional `imageio-ffmpeg` binary when that package is installed in the Python
    environment;
  - the renderer intentionally fails when particle snapshots are missing rather than
    reusing one particle frame across multiple MHD frames;
  - current reconnection particle snapshots use placeholder particle metadata, so the
    renderer's auto energy mode intentionally maps the stored scalar transport `p`
    through the configured p0 kinetic energy
    (`E_keV = (sqrt(1+p^2)-1)/(sqrt(1+p0^2)-1) * transport_p0_energy_kev`) for those
    files; absolute CGS kinetic energies still require a finalized transport-to-CGS
    normalization.

### scripts/render_parker_emission_movie.sh

- Shell wrapper for `particleEmission.parker_emission_movie`.
- Current responsibilities:
  - selects `/tmp/kpt_postprocess_venv/bin/python` by default;
  - targets `benchmark_runs/local_parker_speed_accuracy/compact_field` and
    `benchmark_runs/local_parker_speed_accuracy/kokkos_cpu` by default;
  - forwards `KPT_MOVIE_*`, `KPT_FIELD_DIR`, `KPT_PARTICLE_DIR`,
    `KPT_EMISSION_BACKGROUND_DIR`, `KPT_FFMPEG`, and output-root overrides to the Python
    renderer.
- Current notes:
  - the existing local Parker speed benchmark did not write particle snapshots, so this
    wrapper needs a rerun with `KPT_PARTICLE_SNAPSHOT_INTERVAL=1` or another suitable
    cadence before it can render a real 200-frame particle/emission movie.

### scripts/run_kokkos_cpu_parker_movie_pipeline.sh

- One-command CPU Parker rerun and movie pipeline.
- Current responsibilities:
  - configures and builds the CPU Kokkos executable in `cmake-build-benchmark-cpu`;
  - prepares compact MHD field inputs with `fortran_mhd_to_compact_field` unless
    `KPT_FIELD_DIR` is supplied;
  - prepares Athena/MHD emission-background maps with
    `apps/athena2bin.py --skip-compact-output` unless `KPT_PREPARE_BACKGROUND=0`;
  - reruns `kokkos_particle_transport_app` with `--transport parker` and
    `--particle-snapshot-interval 1` by default;
  - forwards `KPT_RESTART_PARTICLE_SNAPSHOT` or `KPT_KOKKOS_RESTART_PARTICLE_SNAPSHOT`
    to the solver and infers `START_FRAME` from `particles_NNNNN.bin` when
    `KPT_START_FRAME` is not set;
  - renders the requested movie frames with `particleEmission.parker_emission_movie`;
  - installs `imageio-ffmpeg` into the selected Python environment when needed and when
    `KPT_ENSURE_IMAGEIO_FFMPEG=1`.
- Current defaults:
  - output root: timestamped `benchmark_runs/kokkos_cpu_parker_movie_YYYYMMDD_HHMMSS`;
  - solver frame range: `0..200`;
  - movie frame range: `1..200`;
  - particle injection: `1600 * 16` particles per frame, matching the local Parker
    speed/accuracy benchmark;
  - particle snapshots: every completed frame;
  - movie grid: `128x128`, with beam FWHM `3` pixels and ProRes MOV output.
- Current notes:
  - the default 200-frame run writes roughly `50+ GB` of particle snapshots, so disk
    space should be checked before launching;
  - `KPT_KOKKOS_WALLTIME_HOURS` or `KPT_WALLTIME_HOURS` are forwarded to the solver
    walltime interface for scheduler-style jobs;
  - `KPT_RENDER_MOVIE=0` runs only the CPU solver/data-preparation stages, and
    `KPT_RERUN_SOLVER=0` can render from an already completed snapshot directory.

### scripts/run_fortran_reconnection_avx512.sh

- Fortran-only AVX512 comparison runner for the Li Xiaocan `reconnection_2d` benchmark.
- Current responsibilities:
  - requires an explicit `-AVX512` or `--avx512` command-line marker before doing any
    build or run work;
  - copies the Fortran solver source tree into
    `benchmark_runs/fortran_avx512_build/source` while excluding generated build, bin,
    lib, run, and data directories;
  - replaces the copied `cmake/submodules.cmake` with a minimal local path definition so
    the isolated build uses the already-copied FLAP sources instead of trying to update
    Git submodules from a non-repository copy;
  - configures and builds a separate Fortran executable with
    `KPT_FORTRAN_COMPILER=/usr/bin/mpif90`, `USE_OPENMP=ON`,
    `KPT_FORTRAN_AVX512_FLAGS`, and the OpenMPI HDF5 installation;
  - defaults `KPT_FORTRAN_AVX512_FLAGS` to the Fortran solver's required release flags
    plus `-march=native` and explicit GCC AVX512 feature switches for the local Ryzen 9
    9950X class CPU;
  - defaults `KPT_HDF5_ROOT` to `/usr/lib/x86_64-linux-gnu/hdf5/openmpi` and
    `KPT_HDF5_FORTRAN_COMPILER` to `/usr/bin/h5pfc.openmpi` so the parallel HDF5 MPI
    APIs used by the Fortran diagnostics link correctly;
  - writes the resulting Fortran compile flags to
    `benchmark_runs/reconnection_200_fortran_avx512/fortran_avx512_flags.make`;
  - runs only the Fortran reconnection case with the same calibration settings as
    `scripts/run_reconnection_three_cases.sh`;
  - passes `KPT_FORTRAN_WALLTIME_HOURS` to the Fortran `-qh` quota, defaulting through
    `KPT_WALLTIME_HOURS` to `12.0` hours;
  - writes output under `benchmark_runs/reconnection_200_fortran_avx512/fortran` so the
    baseline three-case run remains intact.
- Current notes:
  - the original `lixiaocanexample/stochastic-parker/bin/stochastic-mhd.exec` executable
    is not overwritten because the AVX build happens from a copied source tree;
  - the default particle count remains `1600` particles per MPI rank and `16` ranks,
    matching the current 25,600-particle-per-frame benchmark unless environment
    overrides change it;
  - `MT_STREAM` defaults to `/home/liuyh/codes/mt_stream_f90-1.11` when the environment
    variable is not already set.

### include/AthenaFieldReader.hpp

- `namespace AthenaFieldIO`
  - Intended role: host-side loader for the compact 2D Athena++ background-field
    binaries written by `apps/athena2bin.py`.
- `inline constexpr int athena_binary_space_dim = 2`
  - Intended role: fixes the on-disk coordinate rank of the current compact Athena
    format.
- `inline constexpr int athena_binary_vector_dim = 3`
  - Intended role: fixes the magnetic-field and velocity component count to 3 so 2.5D
    guide-field data can be carried on a 2D coordinate grid.
- `template <int SpaceDim> struct AthenaBinaryFieldReadOptions`
  - Intended role: runtime loading policy for one compact Athena binary file.
  - Current visible data members:
    - `ghost_array_type ghost_cells`
    - `periodic_array_type periodic`
    - `OrthogonalCoordinateSystem coordinate_system`
    - `FieldGhostFillMode ghost_fill_mode`
    - `bool close_periodic_faces`
    - `double coordinate_scale`
    - `double magnetic_field_scale`
    - `double velocity_scale`
    - `GridValidation::GridValidationOptions<SpaceDim> grid_validation`
  - Current notes:
    - Defaults use two lower and two upper ghost cells per axis and non-periodic
      Cartesian coordinates.
    - Unit scales default to one because the compact Athena payload carries no unit
      metadata.
    - The caller can set periodic flags before loading; loaded field ghost cells are
      then filled through `Field::fill_ghost_cells`.
- `template <typename DeviceType, typename LayoutType> struct AthenaBackgroundField`
  - Intended role: solver-ready bundle returned by the reader.
  - Current visible type aliases:
    - `grid_type = AnalyticCoordinateGrid<2, DeviceType, LayoutType, 3>`
    - `vector_field_type = VectorField<grid_type, 3, LayoutType>`
  - Current visible data members:
    - `grid_type grid`
    - `vector_field_type magnetic_field`
    - `vector_field_type velocity_field`
    - `std::int32_t nx`
    - `std::int32_t ny`
  - Current notes:
    - `magnetic_field` maps to solver `B_vec`.
    - `velocity_field` maps to solver `V_body`/solar-wind velocity input.
    - `SpaceDim = 2, VecDim = 3` allows `Bx`, `By`, and `Bz` to be available while
      particle coordinates remain on the x-y grid.
- `AthenaBackgroundField<...> read_athena_binary_background_field(...)`
  - Intended role: read one compact Athena binary snapshot into device-ready Kokkos
    fields.
  - Current call points:
    - accepts a file path and optional `AthenaBinaryFieldReadOptions<2>`;
    - reads `int32 nx`, `int32 ny`, x/y face coordinates, then contiguous `Bx`, `By`,
      `Bz`, `Vx`, `Vy`, and `Vz` arrays;
    - validates that x/y face coordinates are uniformly spaced and constructs a uniform
      `AnalyticCoordinateGrid` from the physical lower face and grid spacing;
    - copies field payloads through host mirrors and `Kokkos::deep_copy`;
    - fills field ghost cells after loading.
  - Current execution-space assumptions:
    - file IO and validation are host-only;
    - returned grid and fields use the repository `Device` abstraction unless template
      parameters override it;
    - `Field::fill_ghost_cells` launches device kernels and therefore requires Kokkos to
      be initialized before calling the reader.
  - Current validation:
    - rejects non-positive dimensions, non-finite or non-ascending coordinates,
      non-uniform compact-grid face coordinates, non-finite field values, non-positive
      scales, trailing payload, and host endian
      mismatch;
    - delegates grid metadata and analytic-coordinate validation to `GridValidation`.

### include/Field.hpp

- `enum class FieldGhostFillMode`
  - Intended role: selects how field ghost cells are filled.
  - Current values:
    - `PeriodicOrClamped`: periodic dimensions wrap to physical indices, non-periodic
      dimensions copy the nearest physical boundary value.
    - `PeriodicOnly`: periodic dimensions wrap, non-periodic ghost cells are left
      unchanged.
- `template <typename GridType, int ComponentDim, typename LayoutType> struct Field`
  - Intended role: grid-bound scalar/vector field wrapper over a flattened Kokkos view.
  - Current dependencies:
    - `AnalyticCoordinateGrid.hpp`
    - `MappedCoordinateGrid.hpp`
    - `StoredCoordinateGrid.hpp`
    - `KokkosDevice.hpp`
    - host-only `<fstream>` and `<string>` for VTK output.
  - Current visible type aliases:
    - `grid_type = GridType`
    - `execution_space = grid_type::execution_space`
    - `memory_space = grid_type::memory_space`
    - `layout_type = LayoutType`
    - `index_array_type = grid_type::index_array_type`
    - `coordinate_array_type = grid_type::coordinate_array_type`
    - `centering_array_type = Kokkos::Array<GridCentering, grid_type::space_dim>`
    - `view_type = Kokkos::View<double*[ComponentDim], layout_type, memory_space>`
    - `const_random_access_view_type = Kokkos::View<const double*[ComponentDim], ..., RandomAccess>`
    - `size_type = view_type::size_type`
    - `stride_array_type = Kokkos::Array<size_type, grid_type::space_dim>`
  - Current visible members:
    - `grid_type grid`
    - `centering_array_type centerings`
    - `index_array_type lower_index_cache`
    - `index_array_type upper_index_cache`
    - `index_array_type extents_cache`
    - `stride_array_type strides_cache`
    - `size_type value_count_cache`
    - `view_type data`
  - Current visible call points:
    - `Field(const grid_type& input_grid, const char* label, GridCentering centering)`
    - `Field(const grid_type& input_grid, const char* label, const centering_array_type& input_centerings)`
    - `size_type point_count() const`
    - `int extent(int dim) const`
    - `size_type extent(std::size_t dim) const`
    - `GridCentering centering(int dim) const`
    - `int lower_index(int dim, GridDomain domain) const`
    - `int upper_index(int dim, GridDomain domain) const`
    - `int domain_extent(int dim, GridDomain domain) const`
    - `size_type flatten(const index_array_type& indices) const`
    - `index_array_type logical_indices(size_type linear_index) const`
    - `double& operator()(size_type point_index, int component)`
    - `double operator()(size_type point_index, int component) const`
    - `double& operator()(const index_array_type& indices, int component)`
    - `double operator()(const index_array_type& indices, int component) const`
    - `double& operator()(const index_array_type& indices)`
    - `double operator()(const index_array_type& indices) const`
    - `const_random_access_view_type random_access_view() const`
    - `RandomAccessAccessor random_access() const`
    - `void fill_ghost_cells(FieldGhostFillMode mode, bool close_periodic_faces)`
    - `void enforce_periodic_face_closure()`
    - `void write_vtk(const std::string& file_path, const std::string& field_name, GridDomain domain) const`
  - Current nested types:
    - `RandomAccessAccessor`: device-side read-only accessor carrying a
      `RandomAccess` view plus logical-index flattening caches.
  - Current notes:
    - The field is bound to a concrete grid object and allocates over the selected
      ghosted coordinate centering.
    - Storage is flattened with dimension zero contiguous, then field component.
    - Logical-index strides are cached as `size_type` values rather than `int` values to
      avoid overflow in flattened offsets on large grids.
    - Scalar fields use `ComponentDim = 1`; vector fields use `ComponentDim > 1`.
    - `ComponentDim` must be positive.
    - `fill_ghost_cells` is a device kernel. Periodic ghost values are copied from
      wrapped physical indices on the opposite side; non-periodic ghost values are
      clamped or left unchanged depending on `FieldGhostFillMode`.
    - `fill_ghost_cells` enforces periodic face closure by default before ghost filling.
      For face-centered periodic dimensions, the redundant upper physical face
      `i = n` is copied from the lower physical face `i = 0` so future face-centered
      curl/flux operators see a single consistent periodic seam.
    - `enforce_periodic_face_closure` can be called explicitly before differential
      operators when a face-centered field is modified without immediately refilling
      ghost cells.
    - `write_vtk` is a host-side legacy ASCII VTK `STRUCTURED_GRID` exporter for
      `SpaceDim` 1 through 3.
    - VTK export requires a concrete coordinate grid such as `StoredCoordinateGrid`,
      `AnalyticCoordinateGrid`, or host-callable `MappedCoordinateGrid`; metadata-only
      `Grid` does not contain coordinates to export.
    - VTK point coordinates are always written in Cartesian coordinates. Polar
      `(r, phi[, z])` grids are converted through `(r cos(phi), r sin(phi), z)`.
      Spherical `(r, theta[, phi])` grids are converted through the standard spherical
      basis, with 2D spherical output placed in the `phi = 0` meridional plane.
    - Vector fields are interpreted as coordinate-basis components and converted to
      Cartesian vectors for VTK output.
    - VTK output for generic `MappedCoordinateGrid` uses
      `MappedCoordinateHostAccess::create_coordinate_provider_from_grid`, so custom
      mapping types can provide host-copy behavior through `create_host_mirror()` or
      `host_mapping()`.
    - Binary output is implemented as external `FieldOutput::*` free functions rather
      than additional `Field` data members, so field storage and output remain separated.
    - `write_vtk` is retained as a debug-only host-side helper. In release builds it
      throws if called.
- `template <typename GridType, typename LayoutType> using ScalarField = Field<GridType, 1, LayoutType>;`
- `template <typename GridType, int ComponentDim, typename LayoutType> using VectorField = Field<GridType, ComponentDim, LayoutType>;`

### include/FieldInterpolator.hpp

- `namespace FieldInterpolator`
  - Intended role: reusable interpolation helpers over existing `Field`,
    `ScalarField`, and `VectorField` storage.
- `enum class TimeInterpolationBoundsPolicy`
  - Intended role: controls whether linear time interpolation may extrapolate beyond
    the two supplied time stamps.
  - Values:
    - `RequireBracketed`
    - `AllowExtrapolation`
- `enum class InterpolationBoundaryPolicy`
  - Intended role: controls coordinate preprocessing before particle-position
    interpolation.
  - Values:
    - `RequireInside`
    - `WrapPeriodic`
    - `ClampToDomain`
- `template <typename FieldType> struct LinearPositionInterpolator`
  - Intended role: device-callable multilinear interpolator for sampling a field at a
    particle or arbitrary coordinate-grid position.
  - Current visible type aliases:
    - `field_type = FieldType`
    - `grid_type = field_type::grid_type`
    - `index_array_type = field_type::index_array_type`
    - `coordinate_array_type = field_type::coordinate_array_type`
    - `centering_array_type = field_type::centering_array_type`
    - `value_array_type = Kokkos::Array<double, field_type::component_dim>`
    - `field_accessor_type = field_type::RandomAccessAccessor`
  - Current visible members:
    - `grid_type grid`
    - `centering_array_type centerings`
    - `field_accessor_type values`
    - `GridDomain domain`
    - `InterpolationBoundaryPolicy boundary_policy`
  - Current visible call points:
    - `LinearPositionInterpolator(const field_type& input_field, GridDomain input_domain, InterpolationBoundaryPolicy input_boundary_policy)`
    - `bool contains_position(const coordinate_array_type& position) const`
    - `double sample_component(const coordinate_array_type& position, int component) const`
    - `value_array_type sample(const coordinate_array_type& position) const`
    - `value_array_type sample_all_components(const coordinate_array_type& position) const`
    - `value_array_type sample_by_component(const coordinate_array_type& position) const`
    - `double operator()(const coordinate_array_type& position, int component) const`
    - `value_array_type operator()(const coordinate_array_type& position) const`
- Current visible host/setup call points:
  - `void validate_coordinate_grid_for_interpolation<GridType>()`
  - `void validate_compatible_fields(const LeftFieldType& left_field, const RightFieldType& right_field, const char* operation_name)`
  - `void validate_time_pair(double lower_time, double upper_time, const char* operation_name)`
  - `double time_interpolation_weight(double lower_time, double upper_time, double target_time, TimeInterpolationBoundsPolicy bounds_policy)`
  - `void fill_time_interpolated_field(const LowerFieldType& lower_field, const UpperFieldType& upper_field, OutputFieldType& output_field, double lower_time, double upper_time, double target_time, TimeInterpolationBoundsPolicy bounds_policy)`
  - `auto make_time_interpolated_field(const LowerFieldType& lower_field, const UpperFieldType& upper_field, double lower_time, double upper_time, double target_time, const char* label, TimeInterpolationBoundsPolicy bounds_policy)`
  - `void fill_time_derivative_field(const LowerFieldType& lower_field, const UpperFieldType& upper_field, OutputFieldType& output_field, double lower_time, double upper_time)`
  - `auto make_time_derivative_field(const LowerFieldType& lower_field, const UpperFieldType& upper_field, double lower_time, double upper_time, const char* label)`
  - `LinearPositionInterpolator<FieldType> make_linear_position_interpolator(const FieldType& field, GridDomain domain, InterpolationBoundaryPolicy boundary_policy)`
  - `void fill_interpolated_values_at_positions(const FieldType& field, const PositionViewType& positions, OutputViewType& output_values, std::size_t count, GridDomain domain, InterpolationBoundaryPolicy boundary_policy)`
  - `void fill_interpolated_values_at_positions(const FieldType& field, const PositionViewType& positions, OutputViewType& output_values, GridDomain domain, InterpolationBoundaryPolicy boundary_policy)`
- Current notes:
  - Time interpolation and time-derivative construction are deliberately separate call
    paths. The former computes `(1 - w) field0 + w field1`; the latter computes
    `(field1 - field0) / (t1 - t0)`.
  - Time interpolation writes every stored field point, including ghost points, and
    assumes the input fields already contain valid ghost values if ghost output matters.
  - Position interpolation uses multilinear interpolation over the field's own
    per-dimension centering. It obtains continuous logical coordinates through the
    concrete grid's `grid_index()` API, so stored, analytic, and mapped coordinate grids
    share the same sampler path.
  - `LinearPositionInterpolator` stores a `Field::RandomAccessAccessor` and is intended
    to be captured directly by particle-update kernels that perform irregular field
    lookups.
  - `sample(position)` computes the interpolation stencil once and accumulates all
    components for vector fields. `sample_by_component(position)` remains available as
    a compatibility/debug path that calls `sample_component()` repeatedly.
  - The default position-interpolation setup uses `GridDomain::GhostedDomain` and
    `InterpolationBoundaryPolicy::WrapPeriodic`. This allows particles near physical
    boundaries to use existing ghost cells while wrapping periodic coordinates before
    sampling.
  - Non-periodic coordinates outside the selected domain abort under `RequireInside` or
    `WrapPeriodic`. `ClampToDomain` is available for diagnostic or visualization paths
    where endpoint clamping is acceptable.
  - Concrete grids used for position interpolation must provide `coordinate()`,
    `grid_index()`, `contains_coordinate()`, and `wrap_coordinate()`. Metadata-only
    `Grid` is intentionally rejected.

### include/FieldOperator.hpp

- `namespace FieldOperator`
  - Intended role: reusable Kokkos field operators over existing `Field`,
    `ScalarField`, and `VectorField` storage.
- `enum class ZeroMagnitudePolicy`
  - Intended role: controls how direction normalization handles zero-magnitude vectors.
  - Values:
    - `ZeroVector`
    - `Abort`
- Current visible device helpers:
  - `bool indices_inside_domain(const FieldType& field, const FieldType::index_array_type& indices, GridDomain domain)`
  - `double component_or_zero(const FieldType& field, const FieldType::index_array_type& indices, int component)`
  - `int vector_calculus_basis_dim(const GridType& grid)`
  - `double vector_calculus_jacobian(const GridType& grid, const GridType::coordinate_array_type& q)`
  - `double centered_three_point_derivative(...)`
  - `double centered_component_derivative(...)`
  - `double gradient_component(const ScalarFieldType& scalar_field, const ScalarFieldType::index_array_type& indices, int component)`
  - `double divergence_value(const VectorFieldType& vector_field, const VectorFieldType::index_array_type& indices)`
  - `double curl_component(const VectorFieldType& vector_field, const VectorFieldType::index_array_type& indices, int component)`
  - `double magnitude_value(const VectorFieldType& vector_field, const VectorFieldType::index_array_type& indices)`
  - `double dot_product_value(const LeftFieldType& left_field, const RightFieldType& right_field, const LeftFieldType::index_array_type& indices)`
- Current visible host/setup call points:
  - `void fill_gradient(const ScalarFieldType& scalar_field, GradientFieldType& gradient_field, FieldGhostFillMode ghost_fill_mode)`
  - `auto make_gradient_field(const ScalarFieldType& scalar_field, const char* label, FieldGhostFillMode ghost_fill_mode)`
  - `void fill_divergence(const VectorFieldType& vector_field, DivergenceFieldType& divergence_field, FieldGhostFillMode ghost_fill_mode)`
  - `auto make_divergence_field(const VectorFieldType& vector_field, const char* label, FieldGhostFillMode ghost_fill_mode)`
  - `void fill_curl(const VectorFieldType& vector_field, CurlFieldType& curl_field, FieldGhostFillMode ghost_fill_mode)`
  - `auto make_curl_field(const VectorFieldType& vector_field, const char* label, FieldGhostFillMode ghost_fill_mode)`
  - `void fill_magnitude(const VectorFieldType& vector_field, MagnitudeFieldType& magnitude_field)`
  - `auto make_magnitude_field(const VectorFieldType& vector_field, const char* label)`
  - `void fill_scaled(const InputFieldType& input_field, OutputFieldType& output_field, double alpha)`
  - `void scale_in_place(FieldType& field, double alpha)`
  - `auto make_scaled_field(const InputFieldType& input_field, double alpha, const char* label)`
  - `void fill_dot_product(const LeftFieldType& left_field, const RightFieldType& right_field, DotFieldType& dot_field)`
  - `auto make_dot_product_field(const LeftFieldType& left_field, const RightFieldType& right_field, const char* label)`
  - `void fill_direction(const VectorFieldType& vector_field, DirectionFieldType& direction_field, double minimum_magnitude, ZeroMagnitudePolicy zero_policy)`
  - `auto make_direction_field(const VectorFieldType& vector_field, const char* label, double minimum_magnitude, ZeroMagnitudePolicy zero_policy)`
- Current notes:
  - Differential operators compute physical-domain values only, then fill output ghost
    cells through `Field::fill_ghost_cells`.
  - Input fields used by gradient, divergence, or curl must have current ghost cells.
  - Centered derivatives use a three-point nonuniform-grid formula over the field's
    own centering.
  - Gradient, divergence, and curl use orthogonal-coordinate Lame coefficients from the
    grid geometry and assume vector components are physical orthonormal-basis
    components.
  - The vector-calculus Jacobian multiplies up to `min(VecDim, 3)` basis Lame
    coefficients. This supports 2.5D use when `SpaceDim < 3` but `VecDim = 3`.
  - Missing coordinate derivatives are treated as zero, so 2.5D fields can retain a
    third vector component without allocating a third coordinate dimension.
  - Algebraic operators such as magnitude, scaling, dot product, and direction operate
    over all stored points, including ghost points.

### include/FieldOutputTypes.hpp

- `namespace FieldOutput`
  - Intended role: groups host-side field output helpers.
- `struct FieldOutputMetadata`
  - Intended role: small host-side unit metadata bundle for binary snapshots and offline
    HDF5/XDMF conversion.
  - Current visible members:
    - `std::string coordinate_unit`
    - `std::string value_unit`
    - `double coordinate_scale_cgs`
    - `double value_scale_cgs`
- `struct FieldOutputSnapshot`
  - Intended role: serializable, host-side visualization snapshot shared by binary and
    offline HDF5/XDMF conversion paths.
  - Current visible members:
    - `std::string field_name`
    - `std::string coordinate_unit`
    - `std::string value_unit`
    - `std::int32_t space_dim`
    - `std::int32_t component_dim`
    - `std::int32_t output_component_dim`
    - `std::int32_t coordinate_system`
    - `std::int32_t domain`
    - `double coordinate_scale_cgs`
    - `double value_scale_cgs`
    - `std::array<std::int32_t, 3> dimensions`
    - `std::array<std::int32_t, 3> lower_indices`
    - `std::array<std::int32_t, 3> upper_indices`
    - `std::array<std::int32_t, 3> centerings`
    - `std::vector<double> points`
    - `std::vector<double> values`
  - Current visible call points:
    - `std::size_t point_count() const`
    - `bool is_vector() const`
  - Current notes:
    - `points` always stores Cartesian coordinates with three components per point.
    - Scalar snapshots store one value per point; vector snapshots store three Cartesian
      vector components per point.
    - The snapshot is host-only and intended for output/conversion paths, not device
      kernels.

### include/FieldOutputCommon.hpp

- Host-side field snapshot construction helpers for binary output and offline conversion.
- Current dependencies:
  - `AnalyticCoordinateGrid.hpp`
  - `FieldOutputTypes.hpp`
  - `Grid.hpp`
  - `MappedCoordinateGrid.hpp`
  - `StoredCoordinateGrid.hpp`
  - `Kokkos_Core.hpp` for host mirrors and device-to-host copies.
- Current visible call points:
  - `FieldOutputSnapshot make_field_output_snapshot(const FieldType& field, const std::string& field_name, GridDomain domain)`
  - `double analytic_coordinate_on_host(...)`
  - `double analytic_face_coordinate_on_host(...)`
  - `FieldOutputSnapshot build_snapshot_from_coordinate_provider(...)`
  - `void for_each_output_index(...)`
  - `std::array<double, 3> cartesian_point_from_coordinates(...)`
  - `std::array<double, 3> cartesian_vector_from_components(...)`
- Current notes:
  - Direct field snapshot construction must be called while Kokkos is initialized
    because it creates host mirrors and copies device views.
  - This module copies field values and any required coordinate metadata to host mirrors.
  - Stored grids use mirrored stored cell/face coordinate views.
  - Analytic grids mirror coefficient views and evaluate analytic coordinate formulas on
    the host.
  - Generic mapped grids build a host coordinate provider through
    `MappedCoordinateHostAccess::create_coordinate_provider_from_grid`.
  - Custom mapped coordinate mappings that own device-only Views should implement
    `create_host_mirror()` or `host_mapping()` so output paths use mirrored host
    storage.
  - Polar and spherical points are converted to Cartesian coordinates for ParaView-ready
    output.
  - Vector field values are interpreted as coordinate-basis components and converted to
    Cartesian vector components.

### include/FieldBinaryIO.hpp

- Custom field binary snapshot serialization module.
- Current visible constants:
  - `field_binary_magic`
  - `field_binary_version`
  - `field_binary_min_supported_version`
  - `field_binary_endian_marker`
- Current visible call points:
  - `void validate_field_output_snapshot(const FieldOutputSnapshot& snapshot)`
  - `void write_field_binary_snapshot(const FieldOutputSnapshot& snapshot, const std::string& file_path)`
  - `FieldOutputSnapshot read_field_binary_snapshot(const std::string& file_path)`
- Current notes:
  - Binary format version 2 stores fixed metadata, field name, coordinate and value unit
    labels, CGS scale factors, Cartesian point coordinates, and scalar/vector output
    values.
  - Binary format version 1 remains readable and is interpreted as code-unit data with
    `coordinate_scale_cgs = 1` and `value_scale_cgs = 1`.
  - Binary snapshots are intended to carry code-unit payloads from normal runs plus
    enough metadata for offline CGS dimensionalization.
  - The current binary format is a visualization snapshot format, not a full restart
    checkpoint format.
  - The reader validates magic, supported version range, endian marker, dimensions, CGS
    scale metadata, and payload sizes
    before returning a snapshot.

### include/FieldBinaryWriter.hpp

- Field-to-binary writer entry point.
- Current dependencies:
  - `FieldBinaryIO.hpp`
  - `FieldOutputCommon.hpp`
- Current visible call points:
  - `void FieldOutput::write_field_binary(const FieldType& field, const std::string& file_path, const std::string& field_name, GridDomain domain, const FieldOutputMetadata& metadata)`
- Current notes:
  - This is the AthenaK-style lightweight output path.
  - It writes the same host-side snapshot consumed by the offline converter.
  - Optional `FieldOutputMetadata` stores unit labels and CGS scales in the binary file
    without mutating the field payload.
  - It does not require HDF5.

### include/OrthogonalCoordinate.hpp

- `enum class OrthogonalCoordinateSystem`
  - Intended role: identifies the orthogonal curvilinear coordinate system used for grid
    metric operations.
  - Current values:
    - `Cartesian`: Cartesian coordinates with `h_i = 1`.
    - `Polar`: polar coordinates `(r, phi[, z])` with `h = (1, r[, 1])`.
    - `Spherical`: spherical coordinates `(r, theta, phi)` with
      `h = (1, r, r sin(theta))`.
- `template <int SpaceDim, int VecDim> struct OrthogonalGeometry`
  - Intended role: device-callable Lame coefficient and geometry helper for orthogonal
    coordinate systems, including 2.5D cases where the active coordinate dimension and
    vector component dimension differ.
  - Current visible type aliases:
    - `coordinate_array_type = Kokkos::Array<double, SpaceDim>`
    - `metric_array_type = Kokkos::Array<double, VecDim>`
  - Current visible members:
    - `static constexpr int space_dim`
    - `static constexpr int vec_dim`
    - `static constexpr double singularity_tolerance`
    - `OrthogonalCoordinateSystem coordinate_system`
  - Current visible call points:
    - `OrthogonalGeometry(OrthogonalCoordinateSystem input_coordinate_system)`
    - `double h(int component, const coordinate_array_type& q) const`
    - `double h_inv(int component, const coordinate_array_type& q) const`
    - `metric_array_type h(const coordinate_array_type& q) const`
    - `metric_array_type h_inv(const coordinate_array_type& q) const`
    - `double jacobian(const coordinate_array_type& q) const`
    - `double physical_distance(int component, double coordinate_spacing, const coordinate_array_type& q) const`
    - `bool is_singular(const coordinate_array_type& q) const`
  - Current notes:
    - All public geometry methods are `KOKKOS_INLINE_FUNCTION` and can be called inside
      device kernels.
    - The geometry layer assumes an orthogonal coordinate basis and is intended for
      gradient, divergence, curl, interpolation-distance, and finite-volume geometry
      helpers.
    - `jacobian` currently multiplies active coordinate dimensions only; full 2.5D
      volume/area semantics remain deferred until finite-volume operators are added.
    - `h()` and related hot-path helpers no longer abort on every call. Metric
      singularities are rejected once by host-side grid validation during construction.
    - `template <int SpaceDim> using OrthogonalMetric = OrthogonalGeometry<SpaceDim, SpaceDim>`
      remains as a backward-compatible alias for coordinate-space-only geometry.

### include/Grid.hpp

- `enum class GridCoordinateRepresentation`
  - Intended role: identifies the coordinate representation used by a concrete grid
    scaffold.
  - Current values:
    - `StoredCoordinates`
    - `AnalyticCoordinates`
    - `MappedCoordinates`
- `enum class GridCentering`
  - Intended role: identifies whether a coordinate query refers to cell centers or grid
    faces.
  - Current values:
    - `CellCentered`: one coordinate per physical cell in a dimension.
    - `FaceCentered`: one coordinate per physical face or cell edge in a dimension.
- `enum class GridDomain`
  - Intended role: selects whether coordinate/index queries use only the physical cells
    or include ghost cells.
  - Current values:
    - `PhysicalDomain`: physical cell range only.
    - `GhostedDomain`: physical cells plus lower and upper ghost-cell ranges.
- `enum GridBoundarySide`
  - Intended role: names the lower and upper boundary slots used by ghost-cell metadata.
  - Current values:
    - `LowerBoundary = 0`
    - `UpperBoundary = 1`
- `template <int SpaceDim, typename DeviceType, typename LayoutType, int VecDim> struct Grid`
  - Intended role: common grid metadata container stored in the project device memory
    space.
  - Current dependencies:
    - `KokkosDevice.hpp`
    - `OrthogonalCoordinate.hpp`
  - Current visible type aliases:
    - `device_type = DeviceType`
    - `execution_space = DeviceTraits<DeviceType>::execution_space`
    - `memory_space = DeviceTraits<DeviceType>::memory_space`
    - `layout_type = LayoutType`
    - `index_array_type = Kokkos::Array<int, SpaceDim>`
    - `ghost_array_type = Kokkos::Array<Kokkos::Array<int, 2>, SpaceDim>`
    - `periodic_array_type = Kokkos::Array<int, SpaceDim>`
    - `extent_view_type = Kokkos::View<int*, layout_type, memory_space>`
    - `ghost_extent_view_type = Kokkos::View<int*[2], layout_type, memory_space>`
    - `periodic_view_type = Kokkos::View<int*, layout_type, memory_space>`
    - `coordinate_array_type = Kokkos::Array<double, SpaceDim>`
    - `metric_array_type = Kokkos::Array<double, VecDim>`
    - `metric_type = OrthogonalGeometry<SpaceDim, VecDim>`
  - Current visible members:
    - `static constexpr int space_dim`
    - `static constexpr int vec_dim`
    - `static constexpr int default_ghost_cell_layers = 2`
    - `extent_view_type n`
    - `ghost_extent_view_type ghost_cells`
    - `periodic_view_type periodic`
    - `index_array_type n_cache`
    - `ghost_array_type ghost_cells_cache`
    - `periodic_array_type periodic_cache`
    - `metric_type metric`
  - Current visible call points:
    - `Grid(const index_array_type& extents, OrthogonalCoordinateSystem input_coordinate_system)`
    - `Grid(const index_array_type& extents, const ghost_array_type& ghosts, const periodic_array_type& periodic_flags, OrthogonalCoordinateSystem input_coordinate_system)`
    - `Grid(extent_view_type input_n, ghost_extent_view_type input_ghost_cells, periodic_view_type input_periodic, OrthogonalCoordinateSystem input_coordinate_system)`
    - `void refresh_metadata_cache_on_host()`
    - `OrthogonalCoordinateSystem coordinate_system() const`
    - `double h(int component, const coordinate_array_type& q) const`
    - `metric_array_type h(const coordinate_array_type& q) const`
    - `double h_inv(int component, const coordinate_array_type& q) const`
    - `metric_array_type h_inv(const coordinate_array_type& q) const`
    - `double jacobian(const coordinate_array_type& q) const`
    - `double physical_distance(int component, double coordinate_spacing, const coordinate_array_type& q) const`
    - `int extent(int dim) const`
    - `int extent(int dim, GridCentering centering) const`
    - `int ghost_extent(int dim, int side) const`
    - `int total_extent(int dim) const`
    - `int total_extent(int dim, GridCentering centering) const`
    - `bool is_periodic(int dim) const`
    - `int lower_cell_index(int dim, GridDomain domain) const`
    - `int upper_cell_index(int dim, GridDomain domain) const`
    - `int lower_face_index(int dim, GridDomain domain) const`
    - `int upper_face_index(int dim, GridDomain domain) const`
    - `bool contains_cell_index(int dim, int index, GridDomain domain) const`
    - `int wrap_cell_index(int dim, int index) const`
  - Current notes:
    - `n` is now a device View rather than a host-side fixed array.
    - `n` represents the number of physical cells along each dimension.
    - The extent-only constructor creates two lower and two upper ghost cells per
      dimension by default to support fourth-order interpolation stencils and provide a
      buffer for particles that have crossed the physical boundary.
    - `FaceCentered` coordinate counts add one geometric face to the cell count.
    - Physical logical cell indices are `[0, n - 1]`.
    - Ghosted logical cell indices are `[-lower_ghost, n + upper_ghost - 1]`.
    - Ghost metadata stores lower and upper ghost-cell counts per dimension.
    - Periodic metadata is stored as integer flags for device-friendly indexing.
    - Host-side constructors allocate metadata Views and initialize them through host
      mirrors plus `Kokkos::deep_copy`.
    - Metadata Views remain the owned storage, but small fixed-array metadata caches are
      refreshed at construction and used by hot device-side metadata accessors.
    - The grid immutability contract is that metadata Views must not be modified after
      construction unless `refresh_metadata_cache_on_host` and validation are rerun.
    - The coordinate system defaults to `Cartesian` unless explicitly selected.
    - `VecDim` defaults to `SpaceDim`; callers can instantiate a grid with a larger vector
      metric dimension for 2.5D curl-related operations.
    - The base `Grid` exposes metric operations for known coordinate points; concrete
      coordinate grids provide index-based coordinate gathering and spacing helpers.

### include/StoredCoordinateGrid.hpp

- `template <int SpaceDim, typename DeviceType, typename LayoutType, int VecDim> struct StoredCoordinateGrid`
  - Intended role: concrete grid scaffold whose coordinates are explicitly stored in
    device memory.
  - Current dependencies:
    - `Grid.hpp`
    - `Kokkos_MathematicalFunctions.hpp`
  - Current inheritance:
    - derives from `Grid<SpaceDim, DeviceType, LayoutType, VecDim>`
  - Current visible type aliases:
    - `index_array_type = Grid::index_array_type`
    - `coordinate_array_type = Grid::coordinate_array_type`
    - `metric_array_type = Grid::metric_array_type`
    - `centering_array_type = Kokkos::Array<GridCentering, SpaceDim>`
    - `coordinate_view_type = Kokkos::View<double**, layout_type, memory_space>`
    - `bound_array_type = Kokkos::Array<double, SpaceDim>`
  - Current visible members:
    - `coordinate_view_type cell_coordinates`
    - `coordinate_view_type face_coordinates`
    - `bound_array_type physical_lower_bound`
    - `bound_array_type physical_upper_bound`
    - `bound_array_type ghost_lower_bound`
    - `bound_array_type ghost_upper_bound`
    - `bound_array_type periodic_width`
    - `bound_array_type periodic_inv_width`
  - Current visible call points:
    - `StoredCoordinateGrid(const base_type& metadata, coordinate_view_type input_face_coordinates)`
    - `void refresh_bounds_cache_on_host()`
    - `double coordinate(int dim, int index, GridCentering centering) const`
    - `double grid_index(int dim, double x, GridCentering centering) const`
    - `double grid_index(int dim, double x, GridCentering centering, GridDomain domain) const`
    - `coordinate_array_type coordinate_tuple(const index_array_type& indices, GridCentering centering) const`
    - `coordinate_array_type coordinate_tuple(const index_array_type& indices, const centering_array_type& centerings) const`
    - `double h(int component, const index_array_type& indices, const centering_array_type& centerings) const`
    - `metric_array_type h(const index_array_type& indices, const centering_array_type& centerings) const`
    - `double h_inv(int component, const index_array_type& indices, const centering_array_type& centerings) const`
    - `metric_array_type h_inv(const index_array_type& indices, const centering_array_type& centerings) const`
    - `double jacobian(const index_array_type& indices, const centering_array_type& centerings) const`
    - `double coordinate_spacing(int dim, int index, GridCentering centering) const`
    - `double cell_width(int dim, int cell_index) const`
    - `double physical_distance(int component, const index_array_type& indices, const centering_array_type& centerings) const`
    - `double physical_cell_width(int component, const index_array_type& indices, const centering_array_type& metric_centerings) const`
    - `int cell_index(int dim, double x, GridDomain domain) const`
    - `bool contains_coordinate(int dim, double x, GridDomain domain) const`
    - `double wrap_coordinate(int dim, double x) const`
  - Current notes:
    - The coordinate Views are axis-major 2D storage: dimension by coordinate index.
    - Stored coordinate arrays are addressed by logical grid indices offset by the lower
      ghost-cell count.
    - Stored grids now require face-centered cell-edge coordinates as construction input.
    - Cell-centered coordinates are derived during construction as arithmetic midpoints of
      adjacent faces and stored for later cell-centered coordinate queries.
    - Cell-centered-only construction and independent cell-plus-face construction are
      intentionally deleted so boundary caches are always backed by valid face bounds.
    - `grid_index` uses a device-side binary search over the selected stored centering,
      returns a fractional logical coordinate, and linearly extrapolates with the nearest
      interval outside the selected domain.
    - `cell_index` uses a device-side binary search over face coordinates and assumes a
      strictly ascending face-coordinate array in each dimension.
    - `contains_coordinate` checks whether a coordinate lies between the selected
      domain's lower and upper cached face bounds.
    - `wrap_coordinate` maps coordinates back into the physical face range only when the
      dimension is periodic; non-periodic dimensions return the input unchanged.
    - Periodic coordinate wrapping requires ascending physical face coordinates and aborts
      otherwise.
    - Stored grids validate face-coordinate view shape before construction-time cache
      refresh, reject non-ascending physical face bounds, and always refresh coordinate
      bound caches from face coordinates.
    - Metric helpers gather local coordinates from the stored coordinate arrays and then
      delegate Lame coefficient evaluation to the base `Grid` metric.
    - `coordinate_spacing` measures the adjacent coordinate difference for the selected
      centering; callers must choose indices with a valid forward neighbor.
    - `physical_cell_width` converts face-to-face coordinate width into the local
      physical line element using the selected metric-evaluation centerings.

### include/AnalyticCoordinateGrid.hpp

- `enum class AnalyticGridDiscretization`
  - Intended role: identifies the analytic coordinate spacing profile used by each
    dimension.
  - Current values:
    - `UniformSpacing`: constant spacing in physical coordinate space.
    - `LogarithmicSpacing`: constant spacing in log-coordinate space.
    - `LinearWidthSpacing`: cell or point-interval width grows linearly by dimension.
- `template <int SpaceDim, typename DeviceType, typename LayoutType> struct AnalyticCoordinateMapping`
  - Intended role: device-callable mapping object that evaluates analytic coordinates and
    inverse grid indices from compact per-dimension coefficients.
  - Current dependencies:
    - `MappedCoordinateGrid.hpp`
    - `Kokkos_MathematicalFunctions.hpp`
  - Current visible type aliases:
    - `discretization_view_type = Kokkos::View<int*, layout_type, memory_space>`
    - `scalar_view_type = Kokkos::View<double*, layout_type, memory_space>`
    - `discretization_array_type = Kokkos::Array<AnalyticGridDiscretization, SpaceDim>`
    - `scalar_array_type = Kokkos::Array<double, SpaceDim>`
  - Current visible members:
    - `discretization_view_type discretization`
    - `scalar_view_type start`
    - `scalar_view_type step`
    - `scalar_view_type step_growth`
  - Current visible call points:
    - `AnalyticCoordinateMapping(scalar_view_type input_start, scalar_view_type input_step)`
    - `AnalyticCoordinateMapping(discretization_view_type input_discretization, scalar_view_type input_start, scalar_view_type input_step, scalar_view_type input_step_growth)`
    - `AnalyticCoordinateMapping(const discretization_array_type& input_discretization, const scalar_array_type& input_start, const scalar_array_type& input_step, const scalar_array_type& input_step_growth)`
    - `AnalyticCoordinateMapping(const discretization_array_type& input_discretization, const scalar_array_type& input_start, const scalar_array_type& input_step)`
    - `AnalyticGridDiscretization discretization_type(int dim) const`
    - `double coordinate(int dim, double grid_position, GridCentering centering) const`
    - `double coordinate(int dim, int index, GridCentering centering) const`
    - `double grid_index(int dim, double x, GridCentering centering) const`
    - `auto create_host_mirror() const`
  - Current notes:
    - The analytic mapping supports uniform spacing, logarithmic spacing, and linearly
      increasing interval width; these are now mapping profiles rather than separate
      grid implementations.
    - Each dimension can select its own analytic discretization profile.
    - Analytic mappings store coefficients only; coordinates are evaluated through
      explicit formulas instead of expanded coordinate arrays.
    - `start` and `step` are interpreted as lower physical face plus face spacing for
      uniform spacing, lower log-face plus log spacing for logarithmic spacing, and lower
      physical face plus first interval width for linearly increasing interval width.
    - `step_growth` is reserved for `LinearWidthSpacing` and is zero for the default
      uniform constructor.
    - `FaceCentered` coordinates evaluate the analytic face expression directly.
    - `CellCentered` coordinates currently evaluate the physical midpoint between
      neighboring faces.
    - `grid_index` provides the reserved inverse map `x -> i` for analytic coordinate
      systems and returns a continuous grid coordinate.
    - `create_host_mirror` copies coefficient Views to `HostSpace` and returns a
      host-callable analytic mapping for generic `MappedCoordinateGrid` setup and output
      paths.
    - Boundary checks, periodic wrapping, spacing helpers, and metric helpers are
      inherited by `AnalyticCoordinateGrid` from `MappedCoordinateGrid`.
- `template <int SpaceDim, typename DeviceType, typename LayoutType, int VecDim> struct AnalyticCoordinateGrid`
  - Intended role: user-facing analytic grid wrapper that binds `AnalyticCoordinateMapping`
    into the generic `MappedCoordinateGrid` implementation.
  - Current inheritance:
    - derives from `MappedCoordinateGrid<SpaceDim, AnalyticCoordinateMapping<...>, DeviceType, LayoutType, VecDim>`
  - Current visible type aliases:
    - `mapping_type = AnalyticCoordinateMapping<SpaceDim, DeviceType, LayoutType>`
    - `mapped_base_type = MappedCoordinateGrid<SpaceDim, mapping_type, DeviceType, LayoutType, VecDim>`
    - `base_type = Grid<SpaceDim, DeviceType, LayoutType, VecDim>`
    - `index_array_type = mapped_base_type::index_array_type`
    - `coordinate_array_type = mapped_base_type::coordinate_array_type`
    - `centering_array_type = mapped_base_type::centering_array_type`
    - `discretization_view_type = mapping_type::discretization_view_type`
    - `scalar_view_type = mapping_type::scalar_view_type`
    - `discretization_array_type = mapping_type::discretization_array_type`
    - `scalar_array_type = mapping_type::scalar_array_type`
  - Current visible call points:
    - `AnalyticCoordinateGrid(const base_type& metadata, mapping_type input_mapping)`
    - `AnalyticCoordinateGrid(const base_type& metadata, scalar_view_type input_start, scalar_view_type input_step)`
    - `AnalyticCoordinateGrid(const base_type& metadata, discretization_view_type input_discretization, scalar_view_type input_start, scalar_view_type input_step, scalar_view_type input_step_growth)`
    - `AnalyticCoordinateGrid(const base_type& metadata, const discretization_array_type& input_discretization, const scalar_array_type& input_start, const scalar_array_type& input_step, const scalar_array_type& input_step_growth)`
    - `AnalyticCoordinateGrid(const base_type& metadata, const discretization_array_type& input_discretization, const scalar_array_type& input_start, const scalar_array_type& input_step)`
    - `void refresh_bounds_cache_on_host()`
    - `AnalyticGridDiscretization discretization_type(int dim) const`
    - Inherits `coordinate`, `coordinate_tuple`, `grid_index`, `cell_index`,
      `contains_coordinate`, `wrap_coordinate`, `cell_width`, `coordinate_spacing`,
      `h`, `h_inv`, `jacobian`, `physical_distance`, and `physical_cell_width` from
      `MappedCoordinateGrid`.
  - Current notes:
    - `AnalyticCoordinateGrid` is now an analytic specialization of the generic mapped
      grid path; the only analytic-specific state lives in `mapping`.
    - The wrapper preserves convenient constructors so callers do not need to manually
      construct `AnalyticCoordinateMapping` for common cases.
    - Constructors validate coefficient view shapes before refreshing cached bounds.
    - Constructors reject non-ascending physical analytic face bounds during cached-bound
      refresh.
    - The wrapper defers the generic mapped-grid bounds refresh and computes cached bounds
      from host mirrors of analytic coefficient views, avoiding direct host reads of
      device-memory coefficient views.
- `template <int SpaceDim, typename DeviceType, typename LayoutType, int VecDim> using AnalyticUniformGrid = AnalyticCoordinateGrid<...>;`
  - Intended role: compatibility alias for earlier scaffold code that referred to the
    analytic grid as uniform-only.
  - Current notes:
    - New code should prefer `AnalyticCoordinateGrid` when using non-uniform analytic
      discretizations.

### include/MappedCoordinateGrid.hpp

- `namespace MappedCoordinateHostAccess`
  - Intended role: host-side adapter helpers for mapped coordinate setup and output.
  - Current visible call points:
    - `auto create_host_mapping(const MappingType& mapping)`
    - `auto create_coordinate_provider(const MappingType& mapping)`
    - `auto create_coordinate_provider_from_grid(const GridType& grid)`
  - Current notes:
    - `create_host_mapping` first tries `mapping.create_host_mirror()`, then
      `mapping.host_mapping()`, and finally falls back to copying the mapping object.
    - The fallback is valid for plain host-callable functors. Mappings that own
      device-only Views should provide one of the explicit host-copy methods.
    - Field output paths use `create_coordinate_provider_from_grid` for generic mapped
      grids before writing VTK, binary snapshots, or HDF5/XDMF snapshots.
- `template <int SpaceDim, typename MappingType, typename DeviceType, typename LayoutType, int VecDim> struct MappedCoordinateGrid`
  - Intended role: generic structured coordinate grid whose physical coordinates and
    inverse grid coordinates are delegated to a device-callable mapping object.
  - Current dependencies:
    - `Grid.hpp`
    - `Kokkos_MathematicalFunctions.hpp`
  - Current inheritance:
    - derives from `Grid<SpaceDim, DeviceType, LayoutType, VecDim>`
  - Current visible type aliases:
    - `index_array_type = Grid::index_array_type`
    - `coordinate_array_type = Grid::coordinate_array_type`
    - `metric_array_type = Grid::metric_array_type`
    - `centering_array_type = Kokkos::Array<GridCentering, SpaceDim>`
    - `bound_array_type = Kokkos::Array<double, SpaceDim>`
    - `mapping_type = MappingType`
  - Current visible members:
    - `mapping_type mapping`
    - `bound_array_type physical_lower_bound`
    - `bound_array_type physical_upper_bound`
    - `bound_array_type ghost_lower_bound`
    - `bound_array_type ghost_upper_bound`
    - `bound_array_type periodic_width`
    - `bound_array_type periodic_inv_width`
  - Current visible call points:
    - `MappedCoordinateGrid(const base_type& metadata, mapping_type input_mapping)`
    - `MappedCoordinateGrid(const base_type& metadata, mapping_type input_mapping, DeferBoundsCacheRefresh)`
    - `void refresh_bounds_cache_on_host()`
    - `double coordinate(int dim, double grid_position, GridCentering centering) const`
    - `double coordinate(int dim, int index, GridCentering centering) const`
    - `coordinate_array_type coordinate_tuple(const index_array_type& indices, GridCentering centering) const`
    - `coordinate_array_type coordinate_tuple(const index_array_type& indices, const centering_array_type& centerings) const`
    - `double h(int component, const index_array_type& indices, const centering_array_type& centerings) const`
    - `metric_array_type h(const index_array_type& indices, const centering_array_type& centerings) const`
    - `double h_inv(int component, const index_array_type& indices, const centering_array_type& centerings) const`
    - `metric_array_type h_inv(const index_array_type& indices, const centering_array_type& centerings) const`
    - `double jacobian(const index_array_type& indices, const centering_array_type& centerings) const`
    - `double grid_index(int dim, double x, GridCentering centering) const`
    - `int cell_index(int dim, double x, GridDomain domain) const`
    - `bool contains_coordinate(int dim, double x, GridDomain domain) const`
    - `double wrap_coordinate(int dim, double x) const`
    - `double cell_width(int dim, double cell_index) const`
    - `double coordinate_spacing(int dim, double grid_position, GridCentering centering) const`
    - `double coordinate_spacing(int dim, int index, GridCentering centering) const`
    - `double physical_distance(int component, const index_array_type& indices, const centering_array_type& centerings) const`
    - `double physical_cell_width(int component, const index_array_type& indices, const centering_array_type& metric_centerings) const`
  - Current notes:
    - `MappingType` must provide `coordinate` and `grid_index` methods with the same
      signatures for device-side grid operations.
    - Host-side bounds-cache construction uses
      `MappedCoordinateHostAccess::create_host_mapping` rather than directly reading the
      device mapping object.
    - `AnalyticCoordinateGrid` is implemented as this generic grid instantiated with
      `AnalyticCoordinateMapping`.
    - This is also the reserved `grid_i = fun(x)` extension path for custom grids that
      should not be added to `AnalyticGridDiscretization`.
    - No virtual dispatch is used, so the adapter remains compatible with Kokkos device
      kernels when the mapping object itself is device-safe.
    - Boundary checks and periodic wrapping are implemented generically from cached
      physical and ghost face bounds plus the mapping's inverse face map.
    - Constructor-time bounds refresh rejects non-ascending physical face coordinates.
    - Periodic coordinate wrapping still aborts if a corrupted grid reaches wrapping with
      non-positive cached periodic width.
    - Generic mapped grids can support device-backed mapping coefficients when the
      mapping type provides `create_host_mirror()` or `host_mapping()` returning a
      host-callable mapping with mirrored storage.
    - Metric helpers gather local mapped coordinates and delegate Lame coefficient
      evaluation to the base `Grid` metric.

### include/GridValidation.hpp

- `namespace GridValidation`
  - Intended role: host-side validation entry point for grid construction and debug
    checks before launching device kernels.
  - Current dependencies:
    - `AnalyticCoordinateGrid.hpp`
    - `StoredCoordinateGrid.hpp`
    - `Kokkos_Core.hpp`
    - host-only `<stdexcept>` and `<string>` diagnostics
  - Current notes:
    - Validation helpers are host-side setup calls. Core grid constructors only perform
      cheap shape checks needed to avoid unsafe construction-time cache refreshes.
    - New setup code should prefer the `make_validated_*` factory helpers so validation is
      attached to construction without requiring an explicit validation call in `main`.
    - Validation failure throws `std::runtime_error` on the host.
    - The helpers copy metadata and coordinate views to host mirrors, so they are intended
      for setup/debug paths rather than per-step kernels.
- `template <int SpaceDim> struct GridValidationOptions`
  - Intended role: controls how strict host-side validation should be.
  - Current visible members:
    - `int required_ghost_layers`
    - `bool require_positive_extents`
    - `bool require_non_negative_ghost_cells`
    - `bool require_periodic_flags_binary`
    - `bool require_cell_coordinates`
    - `bool require_face_coordinates`
    - `bool require_monotonic_coordinates`
    - `bool require_periodic_ascending`
    - `bool reject_metric_singularities`
  - Current notes:
    - `required_ghost_layers` defaults to `Grid<SpaceDim>::default_ghost_cell_layers`.
    - Cell and face coordinate views are required by default for stored-grid validation;
      stored-grid factories derive cell coordinates from face coordinates before running
      full validation.
- `template <int SpaceDim> struct GridMetadataSnapshot`
  - Intended role: small host-side copy of `Grid` metadata used by validation routines.
  - Current visible members:
    - `Kokkos::Array<int, SpaceDim> extents`
    - `Kokkos::Array<Kokkos::Array<int, 2>, SpaceDim> ghost_cells`
    - `Kokkos::Array<int, SpaceDim> periodic`
    - `OrthogonalCoordinateSystem coordinate_system`
- `GridMetadataSnapshot<GridType::space_dim> metadata_snapshot_on_host(const GridType& grid)`
  - Intended role: copies grid metadata views to host memory for diagnostics or validation.
- `void validate_grid_metadata_on_host(const GridType& grid, const GridValidationOptions<...>& options)`
  - Intended role: checks common metadata before launching kernels.
  - Current checks:
    - metadata View values match the small fixed-array metadata cache;
    - positive extents when enabled;
    - non-negative ghost-cell counts when enabled;
    - lower and upper ghost-cell counts satisfy `required_ghost_layers`;
    - periodic flags are binary when enabled.
- `Grid<...> make_validated_grid_metadata(...)`
  - Intended role: constructs common grid metadata and immediately validates it before
    returning the immutable grid object.
  - Current overloads:
    - extent-only metadata with default ghost cells and periodic flags;
    - explicit extents, ghost cells, periodic flags, and coordinate system.
- `void validate_stored_face_coordinate_input_on_host(const GridType& metadata_grid, const CoordinateViewType& face_coordinates, const GridValidationOptions<...>& options)`
  - Intended role: validates stored face-coordinate input before constructing a
    `StoredCoordinateGrid`, so bad View shapes or descending coordinates are rejected
    before constructor cache refresh reads the View.
- `void validate_stored_coordinate_grid_on_host(const StoredGridType& grid, const GridValidationOptions<...>& options)`
  - Intended role: validates stored coordinate views and coordinate assumptions.
  - Current checks:
    - common metadata checks;
    - metadata View values match the small fixed-array metadata cache;
    - required cell and face coordinate views are present;
    - coordinate view rank extents cover `SpaceDim` and the required ghosted coordinate
      count in every dimension;
    - stored cell and face coordinates are finite;
    - stored coordinates are strictly ascending and have no duplicate adjacent entries
      when monotonic validation is enabled;
    - periodic dimensions have ascending physical face coordinates when periodic wrapping
      validation is enabled;
    - stored coordinate face bounds match the cached physical and ghost bounds;
    - polar and spherical stored coordinates do not touch metric singularities when
      singularity rejection is enabled.
- `StoredCoordinateGrid<...> make_validated_stored_coordinate_grid(...)`
  - Intended role: validates face-coordinate input, constructs stored-coordinate grids,
    derives cell coordinates, and validates the completed grid before returning.
  - Current overloads:
    - metadata plus face-centered cell-edge coordinate views.
  - Current notes:
    - Cell-centered-only stored-grid construction is intentionally unsupported because
      boundary checks and periodic wrapping require physical face bounds.
- `void validate_analytic_mapping_input_on_host(const GridType& metadata_grid, const AnalyticMappingType& mapping, const GridValidationOptions<...>& options)`
  - Intended role: validates analytic mapping coefficients and sampled coordinates before
    constructing an `AnalyticCoordinateGrid`, so invalid coefficient views are rejected
    before constructor cache refresh.
- `void validate_analytic_coordinate_grid_on_host(const AnalyticGridType& grid, const GridValidationOptions<...>& options)`
  - Intended role: validates analytic grid coefficients and sampled ghosted coordinate
    ranges.
  - Current checks:
    - common metadata checks;
    - analytic coefficient views cover `SpaceDim`;
    - analytic discretization values are valid;
    - analytic coefficients are finite and `step` is non-zero;
    - sampled ghosted face-centered and cell-centered coordinates are finite, strictly
      ascending, and non-duplicated when monotonic validation is enabled;
    - periodic dimensions have ascending physical analytic face coordinates when
      periodic wrapping validation is enabled;
    - analytic coordinate face bounds match the cached physical and ghost bounds;
    - polar and spherical sampled coordinates do not touch metric singularities when
      singularity rejection is enabled.
- `AnalyticCoordinateGrid<...> make_validated_analytic_coordinate_grid(...)`
  - Intended role: constructs analytic grids and immediately validates them before
    returning.
  - Current overloads:
    - metadata plus an analytic mapping object;
    - metadata plus uniform coefficient views;
    - metadata plus device-side discretization and coefficient views;
    - metadata plus host-side discretization and coefficient arrays with `step_growth`;
    - metadata plus host-side discretization and coefficient arrays without `step_growth`.

### include/KokkosDevice.hpp

- Repository-wide device abstraction entry point.
- Current visible members:
  - `ExecutionSpace = Kokkos::DefaultExecutionSpace`
  - `Device = ExecutionSpace::device_type`
  - `HostExecutionSpace = Kokkos::DefaultHostExecutionSpace`
  - `HostDevice = HostExecutionSpace::device_type`
  - `template <typename DeviceType> struct DeviceTraits`
- Current notes:
  - centralizes execution-space, memory-space, and default layout discovery for other
    repository types;
  - active physics and container headers now default to this `Device` alias.

### include/UnitConverter.hpp

- `namespace Units`
  - Intended role: host/device-friendly unit conversion and nondimensionalization layer.
- `struct UnitDimension`
  - Intended role: stores integer exponents of base dimensions.
  - Current visible members:
    - `int length_power`
    - `int mass_power`
    - `int time_power`
    - `int charge_power`
    - `int temperature_power`
  - Current visible call points:
    - `bool operator==(const UnitDimension& other) const`
    - `bool operator!=(const UnitDimension& other) const`
- `namespace Units::Dimensions`
  - Intended role: common dimension constants used by transport equations.
  - Current dimensions include:
    - `dimensionless`
    - `length`
    - `mass`
    - `time`
    - `charge`
    - `temperature`
    - `velocity`
    - `acceleration`
    - `angular_frequency`
    - `diffusion_coefficient`
    - `number_density`
    - `momentum`
    - `rigidity`
    - `energy`
    - `magnetic_field`
    - `electric_field`
    - `electric_potential`
- `struct UnitSystem`
  - Intended role: defines CGS base scales per one code unit.
  - Current visible members:
    - `double length_cgs`
    - `double mass_cgs`
    - `double time_cgs`
    - `double charge_cgs`
    - `double temperature_cgs`
  - Current visible call points:
    - `static UnitSystem cgs()`
    - `static UnitSystem cgs_gaussian()`
    - `double scale(const UnitDimension& dimension) const`
    - `void validate() const`
- `struct DimensionlessValue`
  - Intended role: explicit wrapper for already nondimensionalized scalar values passed
    into solvers and device kernels.
  - Current visible members:
    - `double value`
  - Current visible call points:
    - `double raw() const`
    - `explicit operator double() const`
- `template <int L, int M, int T, int Q, int Temp> struct DimensionalScalar`
  - Intended role: compile-time-dimensional scalar stored in CGS units.
  - Current visible members:
    - `double value_cgs`
  - Current aliases:
    - `DimensionlessScalar`
    - `Length`
    - `Mass`
    - `Time`
    - `Charge`
    - `Temperature`
    - `Velocity`
    - `Acceleration`
    - `AngularFrequency`
    - `DiffusionCoefficient`
    - `NumberDensity`
    - `Momentum`
    - `Rigidity`
    - `Energy`
    - `MagneticField`
    - `ElectricField`
    - `ElectricPotential`
- `struct Nondimensionalizer`
  - Intended role: converts CGS dimensional quantities to code-dimensionless values before
    solver or coefficient execution.
  - Current visible members:
    - `UnitSystem units`
  - Current visible call points:
    - `Nondimensionalizer(const UnitSystem& input_units)`
    - `DimensionlessValue to_dimensionless(double value_cgs, const UnitDimension& dimension) const`
    - `double to_cgs(DimensionlessValue value, const UnitDimension& dimension) const`
    - `DimensionlessValue to_dimensionless(DimensionalScalar<...> quantity) const`
    - `DimensionalScalar<...> to_cgs(DimensionlessValue value) const`
    - `DimensionlessValue from_dimensionless_in_system(DimensionlessValue source_value, const UnitSystem& source_units, const UnitDimension& dimension) const`
    - `DimensionlessValue to_dimensionless_in_system(DimensionlessValue value, const UnitSystem& target_units, const UnitDimension& dimension) const`
    - `void nondimensionalize_field_in_place(FieldType& field, const UnitDimension& dimension) const`
    - `void dimensionalize_field_in_place(FieldType& field, const UnitDimension& dimension) const`
    - `void convert_field_from_unit_system_in_place(FieldType& field, const UnitSystem& source_units, const UnitDimension& dimension) const`
    - `void convert_field_to_unit_system_in_place(FieldType& field, const UnitSystem& target_units, const UnitDimension& dimension) const`
- `UnitSystem make_unit_system(Length length, Mass mass, Time time, Charge charge, Temperature temperature)`
  - Intended role: builds a base unit system from typed CGS scales.
- `UnitSystem make_unit_system_from_magnetic_field_scale(Length length, Mass mass, Time time, MagneticField magnetic_field, Temperature temperature)`
  - Intended role: builds a CGS base unit system for external MHD code units whose
    magnetic-field scale is known directly.
  - Current notes:
  - Conversion is intentionally separated from solver execution; users should convert
    physical input values during configuration/setup and pass only `DimensionlessValue`
    or raw dimensionless doubles into kernels.
  - The unit system stores only five base scales, so derived quantities such as magnetic
    field, rigidity, diffusion coefficient, and angular frequency are computed from
    dimension exponents.
  - Field conversion helpers launch Kokkos kernels over all stored field points and
    components. They are intended for setup/output paths around solver execution, not
    inside hot per-particle kernels.
  - `Field` intentionally does not store dimensional-state metadata in its device view.
    Unit state is a setup/output contract handled by `Nondimensionalizer`, source
    `UnitSystem` descriptions, and future host-side output metadata.
  - External simulation outputs, such as Athena-family dimensionless fields, can be
    converted by describing the source code units with `UnitSystem` and using
    `convert_field_from_unit_system_in_place`.

### include/PhysicalConstant.hpp

- `namespace PhysicalConstants`
  - Intended role: CGS physical constants plus symbolic metadata used by configuration and
    diagnostics.
- Current visible constants:
  - `pi_value`
  - `speed_of_light_cgs`
  - `elementary_charge_cgs`
  - `boltzmann_constant_cgs`
  - `planck_constant_cgs`
  - `electron_volt_erg`
  - `proton_mass_cgs`
  - `electron_mass_cgs`
  - `gaussian_vacuum_permeability_factor_cgs`
  - `astronomical_unit_cgs`
  - `solar_radius_cgs`
  - `day_cgs`
  - `julian_year_cgs`
- Current typed constants:
  - `speed_of_light`
  - `elementary_charge`
  - `boltzmann_constant`
  - `planck_constant`
  - `electron_volt`
  - `proton_mass`
  - `electron_mass`
  - `gaussian_vacuum_permeability_factor`
  - `astronomical_unit`
  - `solar_radius`
  - `day`
  - `julian_year`
- `enum class ConstantSymbol`
  - Intended role: stable symbolic identifiers for constants.
  - Current notes:
    - Enumerators use explicit integer values so adding a new constant later does not
      silently renumber existing serialized/configuration-facing identifiers.
- `struct ConstantInfo`
  - Intended role: host-side physical constant metadata for diagnostics and
    configuration.
  - Current visible members:
    - `ConstantSymbol id`
    - `const char* ascii_symbol`
    - `const char* latex_symbol`
    - `const char* name`
    - `Units::UnitDimension dimension`
    - `double value_cgs`
- Current visible call points:
  - `ConstantInfo constant_info(ConstantSymbol symbol)`
  - `Units::UnitSystem heliospheric_unit_system(...)`
- Current notes:
  - The default heliospheric unit system uses length `1 au`, time `1 day`, mass
    `m_p`, charge `e`, and temperature `1 K`.
  - Physical constants remain in CGS; `Units::Nondimensionalizer` performs conversion
    before values are used by solvers or coefficient calculators.

### include/TurbulenceProperties.hpp

- `enum class TurbulencePropertyModelKind`
  - Intended role: selects the turbulence-property model evaluated during setup.
  - Values:
    - `Constant`
    - `HeliosphericRadial`
- `enum class TurbulenceCorrelationLengthModel`
  - Intended role: selects how the effective correlation length `Lc` is initialized.
  - Values:
    - `SlabLength`: use `lslab` as the effective `Lc` placeholder.
    - `Constant`: use a user-supplied constant `Lc`.
- `enum class TurbulencePropertyStorageMode`
  - Intended role: controls whether turbulence properties are exposed as constants or
    grid fields.
  - Values:
    - `Auto`: constants stay scalar-only; spatially varying models allocate fields.
    - `ConstantOnly`: reject non-constant models.
    - `Field`: allocate fields even for constant models.
- `struct TurbulencePropertyState`
  - Intended role: carries turbulence properties and their radial derivatives at one
    location.
  - Current visible members:
    - `double sigma2`
    - `double l2d`
    - `double lslab`
    - `double lc`
    - `double d_sigma2_dr`
    - `double d_l2d_dr`
    - `double d_lslab_dr`
    - `double d_lc_dr`
  - Current notes:
    - Derivatives are with respect to code-unit radial coordinate `r`, not CGS radius.
- `struct TurbulencePropertyModel`
  - Intended role: host-validated, device-callable model for turbulence amplitudes and
    correlation lengths.
  - Current visible members:
    - `TurbulencePropertyModelKind kind`
    - `TurbulenceCorrelationLengthModel lc_model`
    - `double au_in_code_units`
    - constant-model coefficients for `sigma2`, `l2d`, `lslab`, and `lc`
    - heliospheric-model coefficients for `l2d`, radial power, and `lslab/l2d`
  - Current visible call points:
    - `static TurbulencePropertyModel constant_code_units(double sigma2, double l2d, double lslab, double lc)`
    - `static TurbulencePropertyModel constant_au(double sigma2, double l2d_au, double lslab_au, const Units::Nondimensionalizer& nondimensionalizer, double lc_au)`
    - `static TurbulencePropertyModel constant_cgs(double sigma2, Units::Length l2d, Units::Length lslab, const Units::Nondimensionalizer& nondimensionalizer, Units::Length lc)`
    - `static TurbulencePropertyModel heliospheric_radial(const Units::Nondimensionalizer& nondimensionalizer, TurbulenceCorrelationLengthModel lc_model, double constant_lc_au)`
    - `bool is_constant() const`
    - `TurbulencePropertyState evaluate_from_radius(double radius_code) const`
    - `void validate() const`
  - Current heliospheric preset:
    - `sigma2 = 0.1 * (r / 0.1 au)^0.5` for `r < 0.5 au`;
    - `sigma2 = 0.15 * (r / 0.1 au)^0.25` for `0.5 au <= r < 2 au`;
    - `sigma2 = 0.32` for `r >= 2 au`;
    - `l2d = 0.0074 au * (r / 1 au)^1.1`;
    - `lslab = 3.9 * l2d`;
    - `lc = lslab` by default until a coefficient model chooses a more specific
      effective correlation length.
- `double turbulence_radial_distance(const GridType& grid, const GridType::coordinate_array_type& q)`
  - Intended role: extracts radial distance from a coordinate tuple for Cartesian,
    polar, and spherical coordinate systems.
  - Current notes:
    - Cartesian uses Euclidean radius from active coordinate components.
    - Polar and spherical use coordinate component zero as radius.
- `template <typename GridType> struct TurbulenceProperties`
  - Intended role: grid-bound turbulence-property storage used by future Parker/Focused
    coefficient kernels.
  - Current visible type aliases:
    - `grid_type`
    - `scalar_field_type`
    - `index_array_type`
    - `coordinate_array_type`
    - `size_type`
    - `execution_space`
  - Current visible members:
    - `grid_type grid`
    - `TurbulencePropertyModel model`
    - `TurbulencePropertyStorageMode storage_mode`
    - `TurbulencePropertyState constant_state`
    - scalar fields for `sigma2`, `l2d`, `lslab`, `lc`, and radial derivatives.
  - Current visible call points:
    - `TurbulenceProperties(const grid_type& grid, const TurbulencePropertyModel& model, TurbulencePropertyStorageMode requested_storage, const char* label)`
    - `bool stores_fields() const`
    - `Accessor accessor() const`
    - `void initialize_fields()`
  - `TurbulenceProperties::Accessor`
    - Intended role: device-side read-only accessor for coefficient kernels.
    - Current behavior:
      - returns `constant_state` for `ConstantOnly` storage;
      - reads RandomAccess scalar field views for `Field` storage.
  - Current notes:
    - `Auto` storage removes constant turbulence properties from field storage while
      preserving the same accessor interface for coefficient kernels.
    - Spatially varying models initialize all stored field points, including ghosted
      points, from cell-centered coordinates.
    - `initialize_fields()` is public as an implementation helper because CUDA extended
      lambdas require public enclosing member functions; normal setup should rely on the
      constructor rather than calling it directly.
    - The current derivative fields are radial derivatives because the active model is
      radial. General mapped coordinate gradients should be added when a non-radial
      turbulence model is introduced.

### include/ParticleSystem.hpp

- `enum class ParticleSpecies`
  - Intended role: identifies built-in particle species whose CGS parameters can be
    reused during setup.
  - Current values:
    - `Proton`
    - `Electron`
    - `Alpha`
    - `Custom`
- `enum class ParticleStatus`
  - Intended role: stores particle lifecycle state for transport and diagnostics.
  - Current values:
    - `Active`
    - `Escaped`
    - `Inactive`
  - Current notes:
    - This is the authoritative alive/recycle flag. `Active` means the particle is
      currently transported; `Escaped` and `Inactive` are recyclable storage slots.
- Helper status functions:
  - `KOKKOS_INLINE_FUNCTION bool particle_status_is_alive(ParticleStatus status)`
  - `KOKKOS_INLINE_FUNCTION bool particle_status_is_alive(int status)`
  - `KOKKOS_INLINE_FUNCTION bool particle_status_is_recyclable(ParticleStatus status)`
  - `KOKKOS_INLINE_FUNCTION bool particle_status_is_recyclable(int status)`
- `enum class ParticleConnectionCorrectionMode`
  - Intended role: reserved selector for momentum-direction connection correction.
  - Current values:
    - `Disabled`
    - `Curvilinear`
  - Current notes:
    - The current `ParticleSystem` stores only scalar momentum magnitude, so
      `Curvilinear` is rejected by `set_connection_correction_mode()` and by the
      correction hook. Directional momentum transport needs a future directional
      particle state rather than this scalar-momentum container.
- `struct ParticleSpeciesParametersCgs`
  - Intended role: stores built-in particle rest mass and charge in CGS units.
  - Current visible members:
    - `ParticleSpecies species`
    - `double rest_mass_cgs`
    - `double charge_cgs`
  - Current visible call points:
    - `ParticleSpeciesParametersCgs particle_species_parameters_cgs(ParticleSpecies species)`
    - `const char* particle_species_name(ParticleSpecies species)`
- `struct ParticleProperties`
  - Intended role: stores already nondimensionalized single-species particle parameters
    for device kernels.
  - Current visible members:
    - `ParticleSpecies species`
    - `double rest_mass`
    - `double charge`
    - `double speed_of_light`
    - `double energy_scale_erg`
  - Current visible call points:
    - `void validate() const`
    - `double gamma_from_momentum_magnitude(double momentum_magnitude) const`
    - `double speed_from_momentum_magnitude(double momentum_magnitude) const`
    - `double kinetic_energy_from_momentum_magnitude(double momentum_magnitude) const`
    - `double energy_to_electron_volt(double dimensionless_energy) const`
    - `double kinetic_energy_ev_from_momentum_magnitude(double momentum_magnitude) const`
    - `ParticleProperties make_particle_properties_from_cgs(...)`
    - `ParticleProperties make_particle_properties(ParticleSpecies species, const Units::Nondimensionalizer& nondimensionalizer)`
  - Current notes:
    - `ParticleProperties` is dimensionless by construction. Built-in species use CGS
      constants only during host-side setup through `Units::Nondimensionalizer`.
    - Alpha-particle parameters use charge `2e` and rest mass `6.6446573357e-24 g`.
- `template <int SpaceDim> struct ParticleState`
  - Intended role: compact host/device single-particle state used as an interface or
    snapshot object.
  - Current visible members:
    - `std::uint64_t particle_id`
    - `Kokkos::Array<double, SpaceDim> position`
    - `double momentum_magnitude`
    - `double mu`
    - `double weight`
    - `double initial_kinetic_energy`
    - `int split_level`
    - `ParticleStatus status`
  - Current notes:
    - `mu` is stored in the common state for focused transport. Parker transport can
      leave it at zero and ignore it, avoiding separate particle containers at this
      stage.
- `struct ParticleSplitStatistics`
  - Intended role: host-side diagnostic summary for one fixed-ratio energy-growth
    splitting pass.
  - Current visible members:
    - `std::uint64_t active_count`
    - `std::uint64_t split_count`
    - `std::uint64_t skipped_capacity_count`
    - `std::uint64_t skipped_invalid_baseline_count`
    - `std::uint64_t skipped_minimum_weight_count`
- `struct ParticleSplittingPolicy`
  - Intended role: fixed per-`ParticleSystem` controls for energy-growth particle
    splitting.
  - Current visible members:
    - `double energy_split_ratio`
    - `double minimum_child_weight`
  - Current visible call points:
    - `void validate() const`
  - Current notes:
    - The policy is copied into `ParticleSystem` during construction and exposed only
      through `splitting_policy()`. Splitting calls do not accept a runtime ratio, so one
      run cannot accidentally change the meaning of existing `split_level` values.
- `template <int SpaceDim, typename DeviceType, typename LayoutType> struct ParticleSystem`
  - Intended role: device-backed single-species relativistic particle container.
  - Current visible type aliases:
    - `execution_space`
    - `memory_space`
    - `layout_type`
    - `size_type`
    - `particle_id_view_type = Kokkos::View<std::uint64_t*, ...>`
    - `scalar_view_type = Kokkos::View<double*, ...>`
    - `vector_view_type = Kokkos::View<double*[SpaceDim], ...>`
    - `int_view_type = Kokkos::View<int*, ...>`
    - `coordinate_array_type = Kokkos::Array<double, SpaceDim>`
    - `particle_state_type = ParticleState<SpaceDim>`
  - Current visible members:
    - `ParticleProperties properties`
    - `ParticleConnectionCorrectionMode connection_correction_mode`
    - `size_type particle_count_cache`
    - `size_type particle_capacity_cache`
    - `particle_id_view_type particle_id`
    - `vector_view_type position`
    - `vector_view_type previous_position`
    - `vector_view_type previous_step_position`
    - `scalar_view_type momentum`
    - `scalar_view_type mu`
    - `scalar_view_type weight`
    - `scalar_view_type initial_kinetic_energy`
    - `int_view_type status`
    - `int_view_type split_level`
    - `int_view_type sort_key`
  - Current visible call points:
    - `ParticleSystem()`
    - `ParticleSystem(size_type particle_count, const ParticleProperties& input_properties, const char* label)`
    - `ParticleSystem(size_type particle_count, size_type particle_capacity, const ParticleProperties& input_properties, const char* label)`
    - `ParticleSystem(size_type particle_count, size_type particle_capacity, const ParticleProperties& input_properties, const ParticleSplittingPolicy& input_splitting_policy, const char* label)`
    - `size_type particle_count() const`
    - `size_type particle_capacity() const`
    - `ParticleSplittingPolicy splitting_policy() const`
    - `bool empty() const`
    - `ParticleStatus particle_status(size_type index) const`
    - `void set_status(size_type index, ParticleStatus input_status) const`
    - `bool is_alive(size_type index) const`
    - `bool is_recyclable(size_type index) const`
    - `void commit_particle_range(size_type first_index, size_type inserted_count)`
    - `particle_state_type particle(size_type index) const`
    - `void set_particle(size_type index, const particle_state_type& state) const`
    - `double momentum_squared(size_type index) const`
    - `double momentum_magnitude(size_type index) const`
    - `double gamma(size_type index) const`
    - `double speed(size_type index) const`
    - `double kinetic_energy(size_type index) const`
    - `double kinetic_energy_ev(size_type index) const`
    - `double velocity_component(size_type index, int dim) const`
    - `coordinate_array_type velocity(size_type index) const`
    - `void set_connection_correction_mode(ParticleConnectionCorrectionMode mode)`
    - `void capture_previous_positions()`
    - `void reset_split_baselines()`
    - `void apply_grid_boundary_conditions(const GridType& grid, GridDomain domain)`
    - `void assign_sort_keys(KeyFunctor key_functor)`
    - `void sort_by_key(int minimum_key, int maximum_key, bool sort_within_bins)`
    - `size_type compact_active_particles()`
    - `ParticleSplitStatistics split_particles_on_energy_growth()`
    - `void apply_curvilinear_momentum_correction(CorrectionFunctor correction_functor)`
    - `void write_binary_snapshot(const std::string& file_path) const`
    - `static ParticleSystem read_binary_snapshot(const std::string& file_path, size_type minimum_capacity, const char* label)`
    - `void initialize_default_state()`
  - Current notes:
    - The container currently supports one particle species per `ParticleSystem`.
      Multi-species transport should be represented by separate solver runs or a future
      multi-species owner that holds multiple `ParticleSystem` instances.
    - Storage is SoA-style Kokkos views for GPU-friendly access. `ParticleState` exists
      as an interface object, not as the primary storage layout.
    - Position has `SpaceDim` components, while `momentum` is a scalar View storing
      the dimensionless momentum magnitude. Relativistic gamma, speed, and energy are
      computed directly from this stored magnitude and dimensionless
      `ParticleProperties`.
    - `velocity_component()` and `velocity()` are retained as compile-time-visible
      guard interfaces but abort if called, because a scalar momentum magnitude does not
      determine a velocity direction.
    - `mu` is always allocated. Parker transport should leave it zero/unused; focused
      transport can use it directly without changing the storage type.
    - `previous_position` stores the position immediately before the current transport
      update. `previous_step_position` stores one additional previous-step position for
      curvilinear connection-correction functors that need two historical positions.
    - `weight` stores the statistical macro-particle weight. Splitting halves the
      parent weight and assigns the same half-weight to the child, conserving total
      weight.
    - `initial_kinetic_energy` is the per-particle dimensionless baseline energy used
      by fixed-ratio splitting thresholds. Injection utilities reset it from the
      injected momentum magnitude.
    - `split_level` records the highest fixed-ratio energy threshold already handled for
      each particle. Its semantics depend on the construction-time
      `ParticleSplittingPolicy`.
    - `assign_sort_keys` fills integer keys with a user-provided device functor.
      `sort_by_key` uses `Kokkos::BinSort` to permute all particle arrays consistently,
      including previous-step positions, weight, and splitting metadata.
    - `compact_active_particles` sorts active particles first, shrinks
      `particle_count()` to the active count, and leaves escaped/inactive storage beyond
      the new count available for later append-style injection.
    - `apply_curvilinear_momentum_correction` now rejects non-disabled correction modes
      because the scalar-momentum container has no stored momentum direction to rotate
      across a curvilinear basis.
    - `split_particles_on_energy_growth` launches a device-side append pass. Each active
      parent can append at most one child at the current storage tail, the host-side
      count is updated before the function returns, and skipped children are counted
      when capacity is exhausted.
    - `write_binary_snapshot` writes a host-side particle binary snapshot containing ids,
      status, split levels, sort keys, position, previous position,
      previous-step position, momentum magnitude, `mu`, weights, split baselines, capacity,
      splitting policy, and dimensionless particle properties. The particle snapshot
      format currently uses magic `KPTPRT\0\0` and version `4`. Version 4 stores one
      scalar momentum magnitude per particle; the Python converter still reads legacy
      version-2 and version-3 vector-momentum snapshots.
    - `read_binary_snapshot` reconstructs a new `ParticleSystem` from a version-4
      snapshot, validates species/properties/splitting metadata, restores all particle
      arrays, and allows a larger `minimum_capacity` so a restarted run can keep
      injecting or splitting particles after loading. It intentionally supports only the
      current scalar-momentum restart format, not legacy vector-momentum visualization
      snapshots.
    - `apply_grid_boundary_conditions` wraps active particles in periodic dimensions and
      applies the same periodic shift to `previous_position` and
      `previous_step_position` so later curvilinear corrections see a continuous local
      trajectory. Active particles that leave non-periodic grid bounds are marked as
      `Escaped`; recycling is handled by particle injection utilities because injection
      distributions and kinematics are solver-policy decisions.
    - `initialize_default_state()` is public as a constructor-time implementation helper
      because CUDA extended lambdas require public enclosing member functions; callers
      should not use it to reset a live simulation unless they intentionally want to
      overwrite all particle state.

### include/ParticleSystemDebugger.hpp

- `struct ParticleStatistics`
  - Intended role: debug-only host-side diagnostic summary for local experiments.
  - Current visible members:
    - `std::uint64_t total_count`
    - `std::uint64_t active_count`
    - `std::uint64_t escaped_count`
    - `std::uint64_t inactive_count`
    - `double active_weight_sum`
    - `double mean_gamma`
    - `double max_gamma`
    - `double mean_speed`
    - `double max_speed`
    - `double active_mean_kinetic_energy_ev`
    - `double active_min_kinetic_energy_ev`
    - `double active_max_kinetic_energy_ev`
- `template <int SpaceDim, typename DeviceType, typename LayoutType> struct ParticleSystemDebugger`
  - Intended role: friend diagnostic helper for `ParticleSystem`.
  - Current visible type aliases:
    - `particle_system_type = ParticleSystem<SpaceDim, DeviceType, LayoutType>`
    - `size_type = particle_system_type::size_type`
  - Current visible call points:
    - `static ParticleStatistics compute_statistics(const particle_system_type& particles)`
    - `static void print_statistics(const particle_system_type& particles, std::ostream& output)`
  - Current notes:
    - This helper performs host mirrors of scalar momentum magnitude, weight, and
      status data. It is for local debug/pre-experiment diagnostics, not production
      transport loops.
    - `mean_gamma`, `max_gamma`, `mean_speed`, and `max_speed` are computed over active
      particles only. Active mean kinetic energy is weight-weighted and reported in eV.
    - In builds with `NDEBUG`, both public methods throw at runtime to prevent
      accidental release diagnostics.

### include/SolverBase.hpp

- `template <typename GridType, int VecDim, typename DeviceType> struct SolverBase`
  - Intended role: shared state for Parker and focused transport solvers while
    preserving the concrete coordinate-grid type.
  - Current visible type aliases:
    - `grid_type = GridType`
    - `device_type = DeviceType`
    - `execution_space = grid_type::execution_space`
    - `memory_space = grid_type::memory_space`
    - `layout_type = grid_type::layout_type`
    - `vector_field_type = VectorField<grid_type, VecDim, layout_type>`
    - `particle_system_type = ParticleSystem<grid_type::space_dim, device_type, layout_type>`
    - `dimensionless_value_type = Units::DimensionlessValue`
  - Current visible members:
    - `particle_system_type particles;`
    - `grid_type grid;`
    - `vector_field_type B_vec, V_body;`
    - `dimensionless_value_type dT{1.0};`
    - `dimensionless_value_type courant_scale{0.3};`
    - `dimensionless_value_type Omega0{0.0};`
  - Constructors:
    - `SolverBase(grid_type input_grid, vector_field_type input_B_vec, vector_field_type input_V_body)`
    - `SolverBase(particle_system_type input_particles, grid_type input_grid, vector_field_type input_B_vec, vector_field_type input_V_body)`
  - Current visible call points:
    - `void set_dimensionless_time_step(dimensionless_value_type input_time_step)`
    - `void set_dimensionless_courant_scale(dimensionless_value_type input_courant_scale)`
    - `void set_dimensionless_rotation_rate(dimensionless_value_type input_rotation_rate)`
    - `double time_step() const`
    - `double courant_number() const`
    - `double rotation_rate() const`
  - Current notes:
    - Acts as a storage base only.
    - The solver is templated on the concrete coordinate grid type, so stored, analytic,
      and mapped coordinate APIs are not sliced down to metadata-only `Grid`.
    - Scalar solver controls are stored as `Units::DimensionlessValue`; setup code must
      nondimensionalize physical inputs through `Units::Nondimensionalizer` before
      assigning them.
    - Kernel-facing scalar accessors return raw dimensionless doubles.
    - Parker-specific deterministic and stochastic stepping is implemented in
      `ParkerSolver`; the base class remains storage/control scaffolding only.

### include/ParkerSolver.hpp

- `template <int SpaceDim, int VecDim> struct ParkerDeterministicParticleTerms`
  - Intended role: host/device debug snapshot of one particle's Parker equation terms.
  - Current visible members:
    - `std::uint64_t particle_id`
    - `coordinate_array_type position`
    - `double momentum_magnitude`
    - `double gamma`
    - `double speed`
    - `vector_array_type magnetic_field`
    - `vector_array_type magnetic_direction`
    - `double magnetic_field_magnitude`
    - `vector_array_type solar_wind_velocity`
    - `vector_array_type drift_velocity`
    - `vector_array_type diffusion_advection`
    - `vector_array_type total_advection`
    - `coordinate_array_type coordinate_rate`
    - `double solar_wind_divergence`
    - `double momentum_rate`
    - `double kappa_parallel`
    - `double kappa_perpendicular`
    - `coordinate_array_type kappa_effective`
    - `double diffusion_time_step`
    - `double advection_time_step`
    - `double momentum_time_step`
    - `double stable_time_step`
- `template <typename GridType, int VecDim, typename DeviceType> struct ParkerSolverDebugger;`
  - Forward declaration for debug access.
- `template <typename GridType, int VecDim, typename DeviceType> struct ParkerSolver : public SolverBase<...>`
  - Intended role: Parker transport solver over a concrete coordinate grid.
  - Current visible type aliases:
    - `base_type = SolverBase<GridType, VecDim, DeviceType>`
    - `grid_type = base_type::grid_type`
    - `device_type = base_type::device_type`
    - `execution_space = base_type::execution_space`
    - `memory_space = base_type::memory_space`
    - `layout_type = base_type::layout_type`
    - `particle_system_type = base_type::particle_system_type`
    - `vector_field_type = base_type::vector_field_type`
    - `scalar_field_type = ScalarField<grid_type, layout_type>`
    - `random_manager_type = RandomManager<DeviceType>`
    - `stochastic_sampler_type = StochasticSampler<DeviceType>`
    - `coefficient_field_set_type = ParkerCoefficient::TransportCoefficientFieldSet<grid_type, layout_type>`
    - `packed_coefficient_field_type = Field<grid_type, packed_component_count, layout_type>`
    - `debug_terms_type = ParkerDeterministicParticleTerms<GridType::space_dim, VecDim>`
  - Current visible call points:
    - Constructor:
      - `ParkerSolver(grid_type input_grid, vector_field_type input_B_vec, vector_field_type input_V_body)`
      - `ParkerSolver(grid_type input_grid, vector_field_type input_B_vec, vector_field_type input_V_body, coefficient_field_set_type input_coefficients)`
      - `ParkerSolver(particle_system_type input_particles, grid_type input_grid, vector_field_type input_B_vec, vector_field_type input_V_body)`
      - `ParkerSolver(particle_system_type input_particles, grid_type input_grid, vector_field_type input_B_vec, vector_field_type input_V_body, coefficient_field_set_type input_coefficients)`
    - `void set_coefficient_fields(coefficient_field_set_type input_coefficients)`
    - `void set_use_packed_coefficient_basis(bool enabled)`
    - `bool packed_coefficient_basis_enabled() const`
    - `void rebuild_packed_coefficient_basis()`
    - `KOKKOS_INLINE_FUNCTION void initializer()`
    - `KOKKOS_INLINE_FUNCTION void operator()(int i) const`
    - `double compute_adaptive_time_step() const`
    - `double advance_deterministic()`
    - `double advance_stochastic(const random_manager_type& random_manager)`
    - `double advance_sde(const random_manager_type& random_manager)`
  - Current status:
    - Implements deterministic Parker stepping and Euler-Maruyama stochastic Parker SDE
      stepping.
    - Uses precomputed coefficient fields from `ParkerCoefficient::TransportCoefficientFieldSet`.
    - Validates that solver-consumed coefficient fields are configured and share the
      same ghosted storage domain and centering as `V_body`.
    - Uses the packed spatial-basis path by default on all execution backends. It can
      still be disabled explicitly through `set_use_packed_coefficient_basis(false)`.
      The packed field stores `B`, `V_sw`, `curl(B/B^2)`, primary and secondary diffusion
      divergence bases, `kappa_parallel_gamma_one`, `kappa_perpendicular_gamma_one`,
      and `div(V_sw)` in one multi-component `Field` so particle kernels can sample
      those basis quantities with one position interpolation. Particle-specific gamma,
      momentum, charge, and drift prefactors remain evaluated in registers.
    - Kernel-launch implementation helpers are public because CUDA extended lambdas
      require public enclosing member functions. External code should prefer the
      high-level stepping API and use lower-level helpers only for tests or diagnostics.
    - Constructors that receive a configured coefficient set build the packed field
      immediately. `rebuild_packed_coefficient_basis()` refreshes the packed field from
      the currently assigned coefficient set. `set_coefficient_fields()` also rebuilds
      it when the packed path is enabled; callers that mutate coefficient field
      contents in place must rebuild explicitly before using the packed path.
    - Particle updates use
      `dX/dt = V_sw + V_d + div(kappa)` and
      `dp/dt = -(p / 3) div(V_sw)`.
    - Vector coefficients are interpreted as physical orthonormal-basis components.
      Coordinate updates divide each active coordinate component by the corresponding
      Lame coefficient through `grid.h_inv`.
    - `V_d` is assembled from the interpolated `curl(B / B^2)` field using the CGS
      Parker prefactor `p v c / (3 q)` after interpolation.
    - Stochastic spatial diffusion uses an analytic local magnetic-direction
      decomposition. The kernel draws an isotropic Gaussian vector, removes its
      component along `b = B / |B|`, and replaces that component with an independent
      field-aligned Gaussian. This gives covariance
      `2 [kappa_perp I + (kappa_parallel - kappa_perp) b b] dt` without storing or
      factorizing a diffusion matrix.
    - For reduced-dimensional `SpaceDim < VecDim` runs, only the represented coordinate
      components of the sampled physical increment are written back; out-of-plane
      vector components still affect the retained covariance through the normalized
      magnetic direction.
    - `div(kappa)` uses the selected legacy branch: README branch interpolates
      parallel and perpendicular divergence bases and applies `gamma^(1/3)` and
      `gamma^(1/9)`; constant-ratio branch interpolates one basis and applies
      `gamma^(1/3)`.
    - The adaptive local time-step bound includes three enabled limiter classes:
      diffusion resolution
      `(0.5 dx_i)^2 / (2 kappa_eff_i)`, advective CFL
      `dx_i / |V_sw_i + V_d_i + div(kappa)_i|`, and relative momentum change
      `3 / |div(V_sw)|`. Here `dx_i` is a physical cell width and
      `kappa_eff_i = kappa_perp + (kappa_parallel - kappa_perp) b_i^2`. The global
      solver step is the active-particle minimum times `courant_scale`.
    - `advance_deterministic()` stores the used dimensionless step in `dT`, captures
      previous positions, updates active particles, clamps negative updated momentum
      magnitude to zero, and then applies grid boundary conditions.
    - `advance_stochastic()` and `advance_sde()` use the same adaptive step and
      deterministic terms, add the sampled magnetic-direction diffusion increment, and
      convert physical orthonormal increments back to coordinate increments through
      `grid.h_inv`.
    - Particle-update kernels request `evaluate_particle_terms<..., false>()` so the
      already completed adaptive-step reduction is not repeated inside the update pass.
      The stochastic kernel also acquires the Kokkos random-pool state only after field
      interpolation and term assembly, reducing the time each thread holds a generator.
    - Runtime stepping requires a concrete coordinate grid with `cell_index`,
      `physical_cell_width`, `contains_coordinate`, and `wrap_coordinate`; the
      metadata-only alias is not a runtime solver.
    - Marked as friend with `ParkerSolverDebugger`.
- `template <int SpaceDim, int VecDim, typename DeviceType, typename LayoutType> using MetadataParkerSolver`
  - Intended role: metadata-only compatibility alias for early scaffolding or smoke
    tests that do not need concrete coordinate APIs.

### include/ParkerDebuger.hpp

- `template <typename GridType, int VecDim, typename DeviceType> struct ParkerSolverDebugger`
  - Intended role: host-side inspection helper for solver internals.
  - Current visible call points:
    - `static void inspect_solver(const ParkerSolver<...>& solver, std::ostream& output)`
    - `static void print_particle_transport_terms(const ParkerSolver<...>& solver, size_type particle_index, const Units::UnitSystem& unit_system, std::ostream& output)`
    - `static void print_particle_equation_terms(const ParkerSolver<...>& solver, size_type particle_index, const Units::UnitSystem& unit_system, std::ostream& output)`
  - Current notes:
    - Uses `std::ostream`, so it is host-only.
    - Reads grid extents through `solver.grid.extent(d)`.
    - Particle term diagnostics launch a one-element Kokkos kernel to reuse the same
      device interpolation and term assembly path as the solver.
    - Every printed line starts with `[Debug]`.
    - Diagnostics print both code-unit values and dimensional CGS values from a supplied
      `Units::UnitSystem`.
    - The equation-style output includes `x`, `p`, `B`, `|B|`, `b`, `V_sw`, `V_d`,
      `nabla_dot_kappa`, total deterministic `dX/dt`, `div(V_sw)`,
      `-(p / 3) div(V_sw)`, local coefficient values, projected `kappa_eff`, and
      diffusion/advection/momentum local time-step diagnostics.

### include/FocusSolver.hpp

- `template <int SpaceDim, int VecDim> struct FocusTransportParticleTerms`
  - Intended role: host/device debug snapshot of one particle's focused transport
    equation terms.
  - Current visible members include:
    - `particle_id`, `position`, scalar `momentum_magnitude`, and pitch-angle cosine
      `mu`;
    - relativistic `gamma` and `speed`;
    - interpolated `B`, normalized `b`, `|B|`, solar-wind velocity, streaming velocity,
      focused drift velocity, perpendicular diffusion advection, total advection, and
      coordinate-space rate;
    - `div(V_sw)`, `bb:grad(V_sw)`, `b dot dV_sw/dt`, `b dot grad ln|B|`,
      `dp/dt`, focused `dmu/dt`, `D_mumu`, and `dD_mumu/dmu`;
    - scalar diffusion diagnostics and local time-step bounds for spatial diffusion,
      spatial advection, momentum change, pitch-angle drift, pitch-angle diffusion, and
      pitch-angle drift/diffusion balance.
- `template <typename GridType, int VecDim, typename DeviceType> struct FocusTransportSolver`
  - Intended role: focused transport solver over a concrete coordinate grid.
  - Current inheritance:
    - derives from `SolverBase<GridType, VecDim, DeviceType>`.
  - Current visible type aliases:
    - `coefficient_field_set_type = FocusCoefficient::TransportCoefficientFieldSet<grid_type, layout_type>`
    - `random_manager_type = RandomManager<DeviceType>`
    - `stochastic_sampler_type = StochasticSampler<DeviceType>`
    - `packed_coefficient_field_type = Field<grid_type, packed_component_count, layout_type>`
    - `debug_terms_type = FocusTransportParticleTerms<GridType::space_dim, VecDim>`
  - Current visible members:
    - `coefficient_field_set_type coefficients`
    - `packed_coefficient_field_type packed_coefficient_basis`
    - `bool use_packed_coefficient_basis`
    - `double maximum_pitch_angle_cosine`
  - Current visible call points:
    - constructors matching the Parker solver pattern, with optional particle system and
      optional coefficient field set;
    - `void set_coefficient_fields(coefficient_field_set_type input_coefficients)`
    - `void set_use_packed_coefficient_basis(bool enabled)`
    - `bool packed_coefficient_basis_enabled() const`
    - `void rebuild_packed_coefficient_basis()`
    - `void set_maximum_pitch_angle_cosine(double input_maximum_mu)`
    - `double compute_adaptive_time_step() const`
    - `double advance_deterministic()`
    - `double advance_stochastic(const random_manager_type& random_manager)`
    - `double advance_sde(const random_manager_type& random_manager)`
  - Current status:
    - Implements focused-transport deterministic stepping and Euler-Maruyama stochastic
      stepping for `X`, scalar momentum magnitude `p`, and pitch-angle cosine `mu`.
    - Uses precomputed fields from `FocusCoefficient::TransportCoefficientFieldSet`.
    - Uses the packed spatial-basis path by default on all execution backends. It can
      still be disabled explicitly through `set_use_packed_coefficient_basis(false)`.
      The packed field stores `B`, precomputed `b`, `V_sw`, gradient and curvature drift bases,
      perpendicular diffusion advection basis, `|B|`,
      `kappa_parallel_gamma_one`, `kappa_perpendicular_gamma_one`, `div(V_sw)`,
      magnetic focusing, `bb:grad(V_sw)`, `b dot dV_sw/dt`, scattering `sigma2`, and
      scattering correlation length in one multi-component `Field`. Particle-specific
      `p`, `mu`, gamma factors, drift prefactors, and `D_mumu` are still evaluated
      after interpolation.
    - Kernel-launch implementation helpers are public because CUDA extended lambdas
      require public enclosing member functions. External code should prefer the
      high-level stepping API and use lower-level helpers only for tests or diagnostics.
    - Constructors that receive a configured coefficient set build the packed field
      immediately. `rebuild_packed_coefficient_basis()` refreshes the packed field from
      the currently assigned coefficient set. `set_coefficient_fields()` also rebuilds
      it when the packed path is enabled; callers that mutate coefficient field
      contents in place must rebuild explicitly before using the packed path.
    - Spatial advection is assembled as
      `v mu b + V_sw + V_d + div(kappa_perp)`.
    - Focused drift uses the README gradient/curvature form:
      `(p v c / q) * [0.5(1 - mu^2) (b x grad ln|B|)/|B| + mu^2 (b x (b dot grad)b)/|B|]`.
    - Perpendicular spatial diffusion samples a projected Gaussian increment with
      covariance `2 kappa_perp (I - b b) dt`; only represented coordinate components
      are written back through `grid.h_inv`.
    - Pitch-angle scattering uses `LegencyModel::evaluate_pitch_angle_scattering`,
      including the configured `resonance_regularization_h0` denominator offset to
      avoid the `mu = 0` resonant singularity without clamping nonzero `mu` values.
    - Pitch-angle boundary handling uses reflection at
      `[-maximum_pitch_angle_cosine, maximum_pitch_angle_cosine]`; the default maximum
      is exactly `1.0`.
    - The `mu` update is semi-implicit by three Picard fixed-point iterations. The same
      normal deviate is held fixed while each iteration recomputes the focused
      deterministic rate, `D_mumu`, and `dD_mumu/dmu` at the current trial `mu`.
    - The adaptive local time-step bound takes the minimum of spatial diffusion
      resolution, spatial advection CFL, relative momentum change, pitch-angle
      deterministic change, pitch-angle stochastic RMS displacement, and
      pitch-angle drift/diffusion balance. Pitch-angle limiter candidates are relaxed
      by a factor of ten relative to the other local bounds before the active-particle
      minimum is multiplied by `courant_scale`.
    - Runtime stepping requires a concrete coordinate grid with `cell_index`,
      `physical_cell_width`, `contains_coordinate`, and `wrap_coordinate`; metadata-only
      grids remain smoke-test/scaffold types.
- `template <typename GridType, int VecDim, typename DeviceType> using FocusSolver`
  - Intended role: compatibility alias for `FocusTransportSolver`.
- `template <int SpaceDim, int VecDim, typename DeviceType, typename LayoutType> using MetadataFocusTransportSolver`
  - Intended role: metadata-only compatibility alias for early scaffolding or smoke
    tests that do not need runtime stepping.

### include/FocusDebuger.hpp

- `template <typename GridType, int VecDim, typename DeviceType> struct FocusTransportSolverDebugger`
  - Intended role: host-side inspection helper for focused solver internals.
  - Current visible call points:
    - `static void inspect_solver(const FocusTransportSolver<...>& solver, std::ostream& output)`
    - `static void print_particle_transport_terms(const FocusTransportSolver<...>& solver, size_type particle_index, const Units::UnitSystem& unit_system, std::ostream& output)`
    - `static void print_particle_equation_terms(const FocusTransportSolver<...>& solver, size_type particle_index, const Units::UnitSystem& unit_system, std::ostream& output)`
  - Current notes:
    - Uses `std::ostream`, so it is host-only.
    - Particle diagnostics launch a one-element Kokkos kernel through
      `FocusTransportSolver::debug_terms_for_particle`, reusing the same device
      interpolation and term assembly path as the solver.
    - Diagnostics print code-unit and dimensional CGS values for position, momentum,
      `mu`, `B`, `b`, streaming/advection/drift terms, `dp/dt`, pitch-angle scattering
      terms, diffusion coefficients, and all local time-step bounds.
- `template <typename GridType, int VecDim, typename DeviceType> using FocusSolverDebugger`
  - Intended role: compatibility alias for `FocusTransportSolverDebugger`.

### include/FocusCoefficient.hpp

- `namespace FocusCoefficient`
  - Intended role: focused transport coefficient utilities and specialized
    orthogonal-coordinate operators.
- `template <typename GridType, typename LayoutType> struct TransportCoefficientFieldSet`
  - Intended role: solver-facing bundle of precomputed focused transport coefficient
    fields.
  - Current visible members:
    - branch metadata: `perpendicular_model`, `perpendicular_parallel_ratio`,
      `pitch_angle_scattering`;
    - magnetic geometry fields: `magnetic_direction`, `magnetic_field_magnitude`,
      `log_magnetic_field`, `grad_log_magnetic_field`, and `magnetic_curvature`;
    - focused drift basis fields divided by `|B|`:
      `gradient_drift_basis_over_magnetic_field` and
      `curvature_drift_basis_over_magnetic_field`;
    - perpendicular transport fields:
      `perpendicular_diffusion_advection_gamma_one`,
      `kappa_parallel_gamma_one`, and `kappa_perpendicular_gamma_one`;
    - flow contraction fields:
      `solar_wind_divergence`, `magnetic_focusing`, `bb_grad_solar_wind`,
      `solar_wind_time_derivative`, and `b_dot_solar_wind_total_derivative`;
    - pitch-angle scattering fields:
      `scattering_sigma2` and `scattering_correlation_length`.
  - Current visible call points:
    - `TransportCoefficientFieldSet(...)`
    - `bool configured() const`
- Current visible device helpers:
  - `bool field_indices_inside_domain(...)`
  - `double field_component_or_zero(...)`
  - `int vector_calculus_basis_dim(...)`
  - `double physical_vector_gradient_component(...)`
  - `double magnetic_direction_solar_wind_gradient_contraction(...)`
  - `double magnetic_direction_solar_wind_convective_derivative(...)`
  - `double magnetic_direction_solar_wind_total_derivative(...)`
  - `double magnetic_focusing_value(...)`
  - `double magnetic_curvature_component(...)`
  - `double cross_product_component(...)`
  - `double magnetic_field_magnitude_value(...)`
- Current visible host/setup call points:
  - `void fill_log_magnetic_field(...)`
  - `auto make_log_magnetic_field(...)`
  - `void fill_magnetic_curvature(...)`
  - `auto make_magnetic_curvature_field(...)`
  - `void fill_focused_drift_basis_fields(...)`
  - `void fill_focused_transport_contraction_fields(...)`
  - `void fill_scattering_turbulence_fields(...)`
  - `void fill_perpendicular_diffusion_advection_field(...)`
  - `auto make_legacy_transport_coefficient_fields_with_time_derivative(...)`
  - `auto make_legacy_transport_coefficient_fields(...)`
- Current notes:
  - Magnetic and velocity vector components are interpreted as physical
    orthonormal-basis components in the grid's orthogonal coordinate system.
  - `physical_vector_gradient_component` implements the needed Cartesian, polar, and
    spherical connection terms for vector-gradient contractions such as
    `bb:grad(V_sw)` and `b dot (V_sw dot grad) V_sw`.
  - The focused equation uses the full derivative
    `dV_sw/dt = partial_t V_sw + (V_sw dot grad) V_sw`. The conversion from the
    supplied partial-time derivative field to the scalar projection
    `b dot dV_sw/dt` is performed in `FocusCoefficient`. The default one-call setup
    path creates a zero `partial_t V_sw` field.
  - Missing coordinate derivatives in reduced-dimensional 2.5D runs are treated as zero,
    while retained vector-basis connection terms are still evaluated when the metric
    provides the needed coordinate information.
  - The perpendicular diffusion-advection field uses the shared Parker tensor-divergence
    backend with the `PerpendicularProjection` basis, so orthogonal-coordinate metric
    terms stay consistent between Parker and Focused transport.
  - The one-call setup path computes magnetic direction, `ln|B|`, `grad ln|B|`,
    curvature, focused drift bases, legacy diffusion scalars, perpendicular diffusion
    advection, flow contractions including `b dot dV_sw/dt`, and scattering
    turbulence scalars.

### include/ParkerCoefficient.hpp

- `namespace ParkerCoefficient`
  - Intended role: Parker transport coefficient utilities that can be reused by solver
    setup kernels and later particle-update kernels.
- `enum class DiffusionTensorDivergenceBasis`
  - Intended role: selects which gamma-independent tensor basis is used when computing
    `div(kappa_ij)`.
  - Values:
    - `ParallelProjection`
    - `PerpendicularProjection`
    - `ConstantParallelRatio`
- `template <typename GridType> struct DiffusionTensorDivergenceFieldSet`
  - Intended role: branch-aware storage for precomputed `div(kappa_ij)` basis vector
    fields.
  - Current visible type aliases:
    - `grid_type`
    - `vector_field_type = VectorField<grid_type, grid_type::vec_dim>`
  - Current visible members:
    - `LegencyModel::PerpendicularDiffusionModelKind perpendicular_model`
    - `vector_field_type parallel_basis`
    - `vector_field_type perpendicular_basis`
    - `vector_field_type constant_ratio_basis`
  - Current visible call points:
    - `DiffusionTensorDivergenceFieldSet(const grid_type& grid, const char* label, LegencyModel::PerpendicularDiffusionModelKind model_kind)`
    - `bool stores_readme_basis_fields() const`
    - `bool stores_constant_ratio_basis_field() const`
- `template <typename GridType, typename LayoutType> struct TransportCoefficientFieldSet`
  - Intended role: solver-facing bundle of precomputed Parker coefficient fields.
  - Current visible type aliases:
    - `grid_type`
    - `layout_type`
    - `scalar_field_type = ScalarField<grid_type, layout_type>`
    - `vector_field_type = VectorField<grid_type, grid_type::vec_dim, layout_type>`
  - Current visible members:
    - `LegencyModel::PerpendicularDiffusionModelKind perpendicular_model`
    - `double perpendicular_parallel_ratio`
    - `vector_field_type magnetic_drift_curl`
    - `DiffusionTensorDivergenceFieldSet<grid_type> diffusion_tensor_divergence`
    - `scalar_field_type kappa_parallel_gamma_one`
    - `scalar_field_type kappa_perpendicular_gamma_one`
    - `scalar_field_type solar_wind_divergence`
  - Current visible call points:
    - `TransportCoefficientFieldSet(...)`
    - `bool configured() const`
- Current visible device helpers:
  - `bool field_indices_inside_domain(const FieldType& field, const FieldType::index_array_type& indices, GridDomain domain)`
  - `double field_component_or_zero(const FieldType& field, const FieldType::index_array_type& indices, int component)`
  - `int vector_calculus_basis_dim(const GridType& grid)`
  - `double vector_calculus_jacobian(const GridType& grid, const GridType::coordinate_array_type& q)`
  - `double centered_three_point_derivative(...)`
  - `double magnetic_field_magnitude(const MagneticFieldType& magnetic_field, const MagneticFieldType::index_array_type& indices)`
  - `double magnetic_direction_component(const MagneticFieldType& magnetic_field, const MagneticFieldType::index_array_type& indices, int component, double minimum_magnetic_field_magnitude)`
  - `double magnetic_field_over_magnitude_squared_component(const MagneticFieldType& magnetic_field, const MagneticFieldType::index_array_type& indices, int component, double minimum_magnetic_field_squared)`
  - `double metric_weighted_magnetic_field_over_magnitude_squared_component(...)`
  - `double centered_metric_weighted_derivative(...)`
  - `double magnetic_field_over_magnitude_squared_curl_component(...)`
  - `double diffusion_tensor_basis_component(...)`
  - `double metric_weighted_diffusion_tensor_basis_component(...)`
  - `double centered_metric_weighted_diffusion_tensor_basis_derivative(...)`
  - `double centered_lame_derivative(...)`
  - `double diffusion_tensor_divergence_basis_component(...)`
  - `Kokkos::Array<double, VecDim> apply_readme_diffusion_tensor_divergence_gamma(...)`
  - `Kokkos::Array<double, VecDim> apply_constant_ratio_diffusion_tensor_divergence_gamma(...)`
- Current visible host/setup call points:
  - `void validate_magnetic_field_curl_inputs(const MagneticFieldType& magnetic_field, const CurlFieldType& curl_field, double minimum_magnetic_field_squared)`
  - `void fill_magnetic_field_over_magnitude_squared_curl(const MagneticFieldType& magnetic_field, CurlFieldType& curl_field, double minimum_magnetic_field_squared, FieldGhostFillMode ghost_fill_mode)`
  - `auto make_magnetic_field_over_magnitude_squared_curl_field(const MagneticFieldType& magnetic_field, const char* label, double minimum_magnetic_field_squared, FieldGhostFillMode ghost_fill_mode)`
  - `void validate_diffusion_tensor_storage_compatibility(const ReferenceFieldType& reference_field, const CheckedFieldType& checked_field, const char* operation_name)`
  - `void validate_diffusion_tensor_divergence_inputs(const MagneticFieldType& magnetic_field, const OutputFieldType& output_field, double minimum_magnetic_field_magnitude, const char* operation_name)`
  - `void fill_diffusion_tensor_divergence_basis_field(...)`
  - `void fill_readme_diffusion_tensor_divergence_basis_fields(...)`
  - `void fill_constant_ratio_diffusion_tensor_divergence_basis_field(...)`
  - `auto make_readme_diffusion_tensor_divergence_basis_fields(...)`
  - `auto make_constant_ratio_diffusion_tensor_divergence_basis_field(...)`
  - `auto make_diffusion_tensor_divergence_fields(const MagneticFieldType& magnetic_field, const DiffusionFieldSetType& diffusion_fields, const LegencyModel::DiffusionModelParameters& parameters, const char* label, double minimum_magnetic_field_magnitude, FieldGhostFillMode ghost_fill_mode)`
  - `auto make_legacy_transport_coefficient_fields(const MagneticFieldType& magnetic_field, const SolarWindVelocityFieldType& solar_wind_velocity, const TurbulencePropertiesType& turbulence_properties, const ParticleProperties& particle_properties, const char* label, const LegencyModel::DiffusionModelParameters& parameters, double minimum_magnetic_field_squared_for_drift, FieldGhostFillMode ghost_fill_mode)`
- Current notes:
  - The precomputed field represents only `curl(B / B^2)`. The particle-dependent
    Parker drift prefactor `p v c / (3 q)` should be applied later during particle
    updates after interpolation.
  - Magnetic field components are interpreted as physical orthonormal-basis components,
    such as `(B_r, B_theta, B_phi)` in spherical coordinates.
  - The curl implementation uses the general orthogonal-coordinate formula with Lame
    coefficients from `Grid::h`.
  - Missing coordinate dimensions are treated as 2.5D ignored directions, so the
    derivative in a dimension not present in `SpaceDim` is zero while vector components
    up to `VecDim = 3` can still be retained.
  - Derivatives use second-order centered differences over the field centering and
    require valid input magnetic-field ghost cells. The routine fills output ghost cells
    after computing physical-domain values.
  - A positive `minimum_magnetic_field_squared` acts as a regularizing floor. Without a
    positive floor, points with `B^2 <= 0` abort because drift is undefined at magnetic
    nulls.
  - Diffusion tensor divergence precomputation assumes cell-centered magnetic-field and
    diffusion-coefficient fields on the same concrete coordinate grid. Input ghost
    cells must be current before calling the precompute routines.
  - The tensor divergence formula is implemented for physical orthonormal-basis tensor
    components in orthogonal coordinates. It includes Lame-coefficient derivative
    connection terms, so polar/spherical `div(kappa_ij)` is not reduced to a Cartesian
    row-wise vector divergence.
  - For the README perpendicular branch, precomputation produces two vector fields:
    `parallel_basis = div(kappa_parallel_gamma_one b_i b_j)` and
    `perpendicular_basis = div(kappa_perpendicular_gamma_one (delta_ij - b_i b_j))`.
    Particle kernels should interpolate both and combine them with
    `gamma^(1/3)` and `gamma^(1/9)` respectively.
  - For the constant-ratio branch, precomputation produces one vector field:
    `constant_ratio_basis = div(kappa_parallel_gamma_one
    [a delta_ij + (1 - a) b_i b_j])`. Particle kernels should interpolate only this
    vector field and multiply it by `gamma^(1/3)`.
  - `make_diffusion_tensor_divergence_fields` consumes the
    `LegencyModel::DiffusionModelParameters` branch flag to allocate and fill the
    correct field set. This keeps the solver-side interpolation count aligned with the
    selected diffusion model.
  - `make_legacy_transport_coefficient_fields` is the one-call setup path for the
    Parker solver. It computes `curl(B/B^2)`, gamma-independent
    `kappa_parallel/kappa_perpendicular`, branch-aware `div(kappa_ij)` bases, and
    `div(V_sw)`, then returns a `TransportCoefficientFieldSet`.

### include/LegencyModel.hpp

- `namespace LegencyModel`
  - Intended role: legacy diffusion-coefficient helpers based on the README
    `kappa_parallel` and `kappa_perpendicular` formulas.
- `enum class PerpendicularDiffusionModelKind`
  - Intended role: selects how perpendicular diffusion is related to the parallel
    diffusion coefficient.
  - Values:
    - `ReadmeFormula`
    - `ConstantParallelRatio`
- `struct DiffusionModelParameters`
  - Intended role: host-validated numerical prefactors and optional safety floors for
    diffusion coefficient evaluation.
  - Current visible members:
    - `PerpendicularDiffusionModelKind perpendicular_model`
    - `double parallel_prefactor`
    - `double perpendicular_prefactor`
    - `double perpendicular_parallel_ratio`
    - `double minimum_magnetic_field_magnitude`
    - `double minimum_solar_wind_speed`
    - `double minimum_sigma2`
  - Current visible call points:
    - `static DiffusionModelParameters with_constant_perpendicular_parallel_ratio(double ratio)`
    - `static DiffusionModelParameters with_constant_perpendicular_parallel_ratio(double ratio, DiffusionModelParameters parameters)`
    - `void set_constant_perpendicular_parallel_ratio(double ratio)`
    - `void use_readme_perpendicular_formula()`
    - `bool uses_constant_perpendicular_parallel_ratio() const`
    - `void validate() const`
- `struct GyrofrequencyReference`
  - Intended role: optional reference split for local gyrofrequency,
    `Omega_ref = |q| B_ref / (m0 c)`.
  - Current visible members:
    - `double magnetic_field_reference`
    - `double omega_reference`
  - Current visible call points:
    - `static GyrofrequencyReference from_particle_properties(const ParticleProperties& properties, double input_magnetic_field_reference)`
    - `double local_nonrelativistic_omega(double magnetic_field_magnitude) const`
    - `double local_relativistic_omega(double magnetic_field_magnitude, double gamma) const`
- `struct GammaIndependentDiffusionCoefficients`
  - Intended role: local diffusion coefficients and diagnostics before applying a
    particle's relativistic gamma factor.
  - Current visible members:
    - `double kappa_parallel_gamma_one`
    - `double kappa_perpendicular_gamma_one`
    - `double omega_nonrelativistic`
    - `double magnetic_field_magnitude`
    - `double solar_wind_speed`
    - `PerpendicularDiffusionModelKind perpendicular_model`
- `struct DiffusionCoefficients`
  - Intended role: particle-specific parallel/perpendicular diffusion coefficients.
  - Current visible members:
    - `double kappa_parallel`
    - `double kappa_perpendicular`
- `struct PitchAngleScatteringParameters`
  - Intended role: host-validated numerical controls for focused-transport
    pitch-angle scattering.
  - Current visible members:
    - `double spectral_index`
    - `double resonance_regularization_h0`
    - `double minimum_magnetic_field_magnitude`
    - `double minimum_sigma2`
    - `double minimum_correlation_length`
    - `double minimum_speed`
  - Current visible call points:
    - `void validate() const`
- `struct PitchAngleScatteringCoefficients`
  - Intended role: local focused-transport pitch-angle scattering diagnostics.
  - Current visible members:
    - `double D_mumu`
    - `double dD_mumu_dmu`
    - `double omega`
    - `double xi`
    - `double regularized_abs_mu`
- `template <typename GridType> struct GammaIndependentDiffusionFieldSet`
  - Intended role: pair of grid-bound scalar fields storing `gamma = 1` diffusion
    coefficients.
  - Current visible members:
    - `PerpendicularDiffusionModelKind perpendicular_model`
    - `ScalarField<GridType> kappa_parallel_gamma_one`
    - `ScalarField<GridType> kappa_perpendicular_gamma_one`
  - Current visible call points:
    - `GammaIndependentDiffusionFieldSet(const grid_type& grid, const char* label, PerpendicularDiffusionModelKind model_kind)`
- Current visible device helpers:
  - `double absolute_value(double value)`
  - `double vector_magnitude_at(const VectorFieldType& vector_field, const VectorFieldType::index_array_type& indices)`
  - `double nonrelativistic_gyrofrequency(const ParticleProperties& properties, double magnetic_field_magnitude)`
  - `double relativistic_gyrofrequency(const ParticleProperties& properties, double magnetic_field_magnitude, double gamma)`
  - `double parallel_diffusion_gamma_one(double solar_wind_speed, double correlation_length, double omega_nonrelativistic, double sigma2, const DiffusionModelParameters& parameters)`
  - `double perpendicular_diffusion_gamma_one(double solar_wind_speed, double l2d, double sigma2_2d, double kappa_parallel_gamma_one, const DiffusionModelParameters& parameters)`
  - `double perpendicular_diffusion_constant_ratio_gamma_one(double kappa_parallel_gamma_one, double ratio)`
  - `double perpendicular_diffusion_gamma_one(double solar_wind_speed, double l2d, double sigma2_2d, double kappa_parallel_gamma_one, const DiffusionModelParameters& parameters, PerpendicularDiffusionModelKind model_kind)`
  - `GammaIndependentDiffusionCoefficients evaluate_gamma_independent_coefficients(...)`
  - `DiffusionCoefficients apply_particle_gamma(const GammaIndependentDiffusionCoefficients& coefficients, double gamma)`
  - `DiffusionCoefficients apply_particle_gamma(double kappa_parallel_gamma_one, double kappa_perpendicular_gamma_one, double gamma, PerpendicularDiffusionModelKind perpendicular_model)`
  - `DiffusionCoefficients apply_particle_gamma(double kappa_parallel_gamma_one, double kappa_perpendicular_gamma_one, double gamma, const DiffusionModelParameters& parameters)`
  - `double pitch_angle_sign(double mu)`
  - `double regularized_abs_pitch_angle_mu(double mu, const PitchAngleScatteringParameters& parameters)`
  - `double pitch_angle_spectrum_normalization(double spectral_index)`
  - `double pitch_angle_diffusion_coefficient(double mu, double omega, double speed, double sigma2, double correlation_length, const PitchAngleScatteringParameters& parameters)`
  - `double pitch_angle_diffusion_derivative(double mu, double D_mumu, double omega, double speed, double correlation_length, const PitchAngleScatteringParameters& parameters)`
  - `PitchAngleScatteringCoefficients evaluate_pitch_angle_scattering(...)`
  - `double diffusion_tensor_component(double kappa_parallel, double kappa_perpendicular, double magnetic_direction_i, double magnetic_direction_j, int i, int j)`
  - `double diffusion_tensor_component_from_field(...)`
- Current visible host/setup call points:
  - `void fill_gamma_independent_diffusion_fields(const MagneticFieldType& magnetic_field, const SolarWindVelocityFieldType& solar_wind_velocity, const TurbulencePropertiesType& turbulence_properties, const ParticleProperties& particle_properties, ParallelFieldType& kappa_parallel_gamma_one, PerpendicularFieldType& kappa_perpendicular_gamma_one, const DiffusionModelParameters& parameters, FieldGhostFillMode ghost_fill_mode)`
  - `auto make_gamma_independent_diffusion_fields(const MagneticFieldType& magnetic_field, const SolarWindVelocityFieldType& solar_wind_velocity, const TurbulencePropertiesType& turbulence_properties, const ParticleProperties& particle_properties, const char* label, const DiffusionModelParameters& parameters, FieldGhostFillMode ghost_fill_mode)`
- Current notes:
  - All quantities are expected to be already nondimensionalized in repository code
    units. In those units, `|q| |B| / (m c)` is a dimensionless angular frequency
    measured in inverse code time.
  - The model uses the local nonrelativistic gyrofrequency
    `Omega_nonrel = |q| |B| / (m0 c)` for field precomputation. The particle
    gyrofrequency is `Omega = Omega_nonrel / gamma`.
  - Therefore `kappa_parallel(gamma) = kappa_parallel_gamma_one * gamma^(1/3)`.
  - The README perpendicular branch is implemented as
    `kappa_perp = (V_sw / 3) * (3 kappa_parallel / V_sw)^(1/3) *
    (C_perp sigma2_2D L_2D)^(2/3)`. The final `2/3` exponent keeps the formula
    dimensionally closed in CGS units when `sigma2_2D` and `C_perp` are
    dimensionless and `L_2D` is a physical length.
  - Because the README perpendicular formula contains `kappa_parallel^(1/3)`,
    `kappa_perpendicular(gamma) = kappa_perpendicular_gamma_one * gamma^(1/9)`.
  - When `PerpendicularDiffusionModelKind::ConstantParallelRatio` is selected through
    `set_constant_perpendicular_parallel_ratio(a)` or
    `with_constant_perpendicular_parallel_ratio(a)`, field precomputation uses
    `kappa_perpendicular_gamma_one = a * kappa_parallel_gamma_one` instead of the
    README perpendicular formula.
  - In the constant-ratio branch the perpendicular coefficient inherits the same gamma
    scaling as the parallel coefficient:
    `kappa_perpendicular(gamma) = a * kappa_parallel_gamma_one * gamma^(1/3)`.
    Particle kernels that interpolate only the two scalar coefficient fields should call
    the `apply_particle_gamma(..., parameters)` overload so this branch information is
    preserved.
  - The current perpendicular implementation uses `TurbulencePropertyState::sigma2`
    as the `sigma2_2D` term until turbulence storage is split into slab and 2D
    amplitudes.
  - The remaining model uncertainty is provenance and normalization of the empirical
    perpendicular prefactor `0.198`, not pure dimensional closure. The constant-ratio
    branch should remain the conservative production default until that reference
    convention is verified.
  - Field precomputation currently requires cell-centered magnetic-field, solar-wind,
    and output coefficient fields on the same concrete grid. It computes the physical
    domain and then fills output ghost cells.
  - Diffusion tensor helpers implement
    `kappa_ij = kappa_perp delta_ij + (kappa_parallel - kappa_perp) b_i b_j` and
    assume magnetic-field vector components are physical orthonormal-basis components.
  - Pitch-angle scattering implements the README focused-transport formula for
    `D_mumu` and `dD_mumu/dmu`. It evaluates the relativistic gyrofrequency
    `Omega = |q||B|/(gamma m c)`, uses the local particle speed, `sigma2`, and `Lc`,
    and evaluates the resonant denominator with `|mu| + h0`, where `h0` is
    `PitchAngleScatteringParameters::resonance_regularization_h0`.
  - `pitch_angle_diffusion_derivative` uses the compact derivative expression from
    `readme.md`; at exactly `mu = 0` the sign term is treated as zero while the
    coefficient evaluation still uses the configured `|mu| + h0` denominator offset.

### include/RandomManager.hpp

- `template <typename DeviceType> struct RandomManager`
  - Intended role: wrapper around a Kokkos random pool for device kernels.
  - Current visible type aliases:
    - `device_type = DeviceType`
    - `execution_space = DeviceTraits<DeviceType>::execution_space`
    - `memory_space = DeviceTraits<DeviceType>::memory_space`
    - `random_pool_type = Kokkos::Random_XorShift64_Pool<DeviceType>`
    - `generator_type = random_pool_type::generator_type`
  - Current visible members:
    - `random_pool_type pool`
    - `std::uint64_t seed`
  - Current visible call points:
    - `RandomManager(std::uint64_t input_seed)`
    - `KOKKOS_INLINE_FUNCTION generator_type get_state() const`
    - `KOKKOS_INLINE_FUNCTION generator_type get_generator() const`
    - `KOKKOS_INLINE_FUNCTION void free_state(generator_type& generator) const`
    - `KOKKOS_INLINE_FUNCTION void free_generator(generator_type& generator) const`
  - Current notes:
    - Uses the repository `Device` abstraction by default.
    - `get_generator` and `free_generator` remain compatibility aliases around the
      normalized `get_state` and `free_state` names.
    - Kernels should always return generator states to the pool before exiting.
- `template <typename DeviceType> struct StochasticSampler`
  - Intended role: helper distribution samplers built on the generator.
  - Current visible call points:
    - `static double uniform01(generator_type& generator)`
    - `static double uniform(generator_type& generator, double lower, double upper)`
    - `static double normal(generator_type& generator)`
    - `static double normal(generator_type& generator, double mean, double standard_deviation)`
    - `static double uniform_sqrt3(...)`
    - `static double random_sign(generator_type& generator)`

### include/ParticleGenerator.hpp

- Includes:
  - `Field.hpp`
  - `ParticleSystem.hpp`
  - `RandomManager.hpp`
- Intended role:
  - device-side particle injection utilities backed by `RandomManager`.
- `template <int SpaceDim> struct AxisAlignedParticleRegion`
  - Intended role: host-validated axis-aligned coordinate region used by injection
    kernels.
  - Current visible members:
    - `coordinate_array_type lower`
    - `coordinate_array_type upper`
  - Current visible call points:
    - `void validate() const`
    - `KOKKOS_INLINE_FUNCTION bool contains(const coordinate_array_type& point) const`
    - `KOKKOS_INLINE_FUNCTION coordinate_array_type sample(GeneratorType& generator) const`
- `template <int SpaceDim> struct ParticleInjectionKinematics`
  - Intended role: constant momentum magnitude, pitch-angle cosine, and status assigned
    to newly injected particles.
  - Current visible members:
    - `double momentum_magnitude`
    - `double mu`
    - `double weight`
    - `ParticleStatus status`
  - Current notes:
    - Injection and recycling kernels reset `initial_kinetic_energy` from
      `momentum_magnitude`, reset `split_level` to zero, and assign this statistical
      weight.
- `struct ParticleInjectionPlan`
  - Intended role: contiguous target particle range plus generic rejection-sampling
    controls.
  - Current visible members:
    - `std::uint64_t first_particle_index`
    - `std::uint64_t particle_count`
    - `std::uint64_t first_particle_id`
    - `int max_attempts`
  - Current notes:
    - `max_attempts` is used only by generic custom-predicate rejection sampling.
      Field-conditioned injection uses compacted candidate grid points instead.
- `enum class FieldPredicateComparison`
  - Intended role: threshold comparison selector for field-filtered injection.
  - Values:
    - `Greater`, `GreaterEqual`, `Less`, `LessEqual`, `AbsGreater`, `AbsLess`
- `enum class ParticleRecycleSelection`
  - Intended role: selects which non-active lifecycle states are reinitialized by
    recycling kernels.
  - Values:
    - `EscapedOnly`
    - `InactiveOnly`
    - `Recyclable`
- Helper predicate functions:
  - `bool particle_recycle_selection_matches(int status, ParticleRecycleSelection selection)`
  - `bool compare_field_predicate_value(double value, double threshold, FieldPredicateComparison comparison)`
- `struct AlwaysAcceptParticlePredicate`
  - Intended role: device predicate for unconditional injection.
- `template <typename FieldType> struct FieldThresholdParticlePredicate`
  - Intended role: device predicate that accepts candidate positions when an existing
    scalar or component field value passes a configured threshold.
  - Current visible members:
    - `grid_type grid`
    - `centering_array_type centerings`
    - `field_accessor_type values`
    - `FieldPredicateComparison comparison`
    - `GridDomain domain`
    - `double threshold`
    - `int component`
  - Current visible call points:
    - `FieldThresholdParticlePredicate(const FieldType& field, double input_threshold, FieldPredicateComparison input_comparison, int input_component, GridDomain input_domain)`
    - `KOKKOS_INLINE_FUNCTION bool operator()(const coordinate_array_type& point) const`
  - Current notes:
    - The predicate consumes a precomputed `Field` through its random-access accessor.
      It intentionally does not compute derived quantities such as `div V`.
    - Coordinates are mapped to cell-centered or nearest face-centered logical indices
      according to the field centering.
- `template <typename FieldType> struct FieldParticleCandidateSet`
  - Intended role: compact device-side list of field grid points that pass a threshold
    and optional coordinate-region filter.
  - Current visible members:
    - `field_type field`
    - `candidate_view_type point_indices`
    - `cumulative_weight_view_type cumulative_weights`
    - `size_type candidate_count_cache`
    - `double total_weight_cache`
  - Current visible call points:
    - `size_type candidate_count() const`
    - `bool empty() const`
    - `size_type sample_candidate_index(GeneratorType& generator) const`
    - `coordinate_array_type coordinate(size_type candidate_index) const`
  - Current notes:
    - The candidate set stores field point indices, not arbitrary continuous
      coordinates. Injection places particles on the selected field grid coordinate.
    - Sampling is weighted by an approximate physical cell volume. Face-centered field
      points use the neighboring/clamped cell for the volume weight.
- `template <typename GridType> struct GridParticleCandidateSet`
  - Intended role: device-backed physical-cell candidate set for grid-wide injection and
    recycling.
  - Current visible members:
    - `grid_type grid`
    - `index_array_type lower_cell_index_cache`
    - `index_array_type extents_cache`
    - `cumulative_weight_view_type cumulative_weights`
    - `size_type candidate_count_cache`
    - `double total_weight_cache`
  - Current visible call points:
    - `size_type candidate_count() const`
    - `bool empty() const`
    - `index_array_type cell_indices(size_type linear_index) const`
    - `size_type sample_candidate_index(GeneratorType& generator) const`
    - `coordinate_array_type sample_coordinate(size_type candidate_index, GeneratorType& generator) const`
  - Current notes:
    - Cell selection is weighted by the same midpoint physical-volume estimate used by
      field candidate sets.
    - The coordinate inside a selected cell is sampled uniformly in coordinate space.
      For strongly curved or very coarse spherical/polar grids this is an approximation;
      exact intra-cell physical-volume sampling can be added later as a coordinate-system
      specialization.
- Current visible free functions:
  - `double injection_kinetic_energy(const ParticleProperties& properties, const ParticleInjectionKinematics<SpaceDim>& kinematics)`
  - `void validate_particle_injection_kinematics(const ParticleInjectionKinematics<SpaceDim>& kinematics)`
  - `ParticleInjectionPlan make_particle_injection_plan(std::uint64_t first_particle_index, std::uint64_t particle_count, int max_attempts)`
  - `ParticleInjectionPlan make_particle_append_plan(const ParticleSystemType& particles, std::uint64_t particle_count, int max_attempts)`
  - `AxisAlignedParticleRegion<SpaceDim> make_particle_region(const Kokkos::Array<double, SpaceDim>& lower, const Kokkos::Array<double, SpaceDim>& upper)`
  - `AxisAlignedParticleRegion<GridType::space_dim> make_particle_region_from_grid(const GridType& grid, GridDomain domain)`
  - `FieldThresholdParticlePredicate<FieldType> make_field_threshold_particle_predicate(const FieldType& field, double threshold, FieldPredicateComparison comparison, int component, GridDomain domain)`
  - `double approximate_grid_cell_physical_volume(const GridType& grid, const GridType::index_array_type& cell_indices)`
  - `int clamp_grid_cell_index_for_volume(const GridType& grid, int dim, int index, GridDomain domain)`
  - `FieldType::index_array_type field_indices_to_volume_cell_indices(const FieldType& field, const FieldType::index_array_type& field_indices, GridDomain domain)`
  - `SizeType sample_cumulative_weight_index(...)`
  - `FieldParticleCandidateSet<FieldType> make_field_particle_candidate_set_in_region(...)`
  - `FieldParticleCandidateSet<FieldType> make_field_particle_candidate_set(...)`
  - `GridParticleCandidateSet<GridType> make_grid_particle_candidate_set(const GridType& grid, GridDomain domain)`
  - `void validate_particle_injection_plan(const ParticleSystemType& particles, const ParticleInjectionPlan& plan)`
  - `void inject_particles_uniform_if(ParticleSystemType& particles, const RandomManagerType& random_manager, const AxisAlignedParticleRegion<ParticleSystemType::space_dim>& region, const ParticleInjectionPlan& plan, const ParticleInjectionKinematics<ParticleSystemType::space_dim>& kinematics, PredicateType predicate)`
  - `void inject_particles_from_field_candidates(...)`
  - `void inject_particles_from_grid_candidates(...)`
  - `void inject_particles_uniform(...)`
  - `void inject_particles_uniform_in_grid(...)`
  - `void inject_particles_uniform_where_field(...)`
  - `void recycle_particles_uniform_if(...)`
  - `void recycle_particles_uniform(...)`
  - `void recycle_particles_uniform_in_grid(...)`
  - `void recycle_particles_from_grid_candidates(...)`
  - `void recycle_particles_from_field_candidates(...)`
  - `void recycle_particles_uniform_where_field(...)`
- Current notes:
  - Full-space random injection over a coordinate grid uses
    `make_grid_particle_candidate_set`, chooses physical cells by midpoint physical
    volume, and samples coordinates inside the selected cell. This path works with
    stored, analytic, and mapped coordinate grids.
  - Explicit `AxisAlignedParticleRegion` injection remains coordinate-uniform and uses
    rejection sampling for custom predicates because no grid metric is available in that
    API.
  - Generic custom-predicate injection uses bounded rejection sampling and writes
    `ParticleStatus::Inactive` when a candidate cannot be accepted within
    `max_attempts`.
  - Field-conditioned injection first compacts satisfying field grid points with
    `parallel_scan`, computes cumulative physical-volume weights, then samples from that
    candidate set directly. No attempt limit is involved in this path.
  - Injection writes `particle_id`, `position`, `previous_position`,
    `previous_step_position`, scalar `momentum`, `mu`, `status`, and `sort_key`. If the
    plan reaches beyond the current particle count but stays within capacity, injection
    calls `ParticleSystem::commit_particle_range` to append the new range.
  - Recycling kernels reinitialize selected `Escaped` or `Inactive` slots in place and
    keep `particle_id` unchanged so storage slots remain stable.

### Placeholder Headers

The following files currently contain only `#pragma once` and no public interface:

- `include/QLTModel.hpp`

## What To Update After Code Changes

When modifying code, update this document if any of the following change:

- a new struct, class, alias, or free function is added;
- an existing constructor or kernel call point changes;
- ownership of data between solver, field, grid, RNG, or particle modules changes;
- host/device assumptions or execution-space usage changes;
- placeholder files become active implementations;
- a previously documented mismatch or TODO is resolved.
