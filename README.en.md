# Kokkos Particle Transport

A Kokkos-based prototype for energetic-particle transport and microwave emission synthesis. The code targets solar eruptive and reconnection-region applications, using Monte Carlo stochastic differential equations for Parker transport, focused transport, and particle-snapshot-driven radiation products.

Chinese documentation is available in [readme.md](readme.md).

## Highlights

- Kokkos backend: OpenMP CPU runs for debugging and benchmarking, with CUDA/HIP portability in mind.
- Parker transport: solar-wind advection, diffusion tensor terms, drift, and adiabatic momentum change.
- Focused transport: field-aligned motion, pitch-angle scattering, magnetic focusing, and momentum evolution.
- Athena++ / Fortran MHD input: 2D background fields can be converted into compact solver inputs.
- Particle diagnostics: summary CSV files, momentum histograms, and version-4 binary particle snapshots.
- Emission synthesis: compact fields, particle snapshots, and thermal background maps can be converted into microwave images, spectra, and ProRes movies.
- Post-processing tools: HDF5/XDMF conversion, benchmark timing/spectrum plots, and a Streamlit emission viewer.

## Repository Layout

```text
include/                 Core grid, field, particle, solver, coefficient, and unit headers
src/main.cpp             Reconnection 2D calibration driver for Parker/Focused transport
apps/                    Field conversion, particle conversion, and benchmark plotting tools
particleEmission/        Microwave emission synthesis, validation, and viewer utilities
scripts/                 CPU/GPU benchmark, validation, and movie pipeline wrappers
docs/code_map.md         Current code ownership and interface map
emissionValidation/      Tracked synthetic emission validation products
kokkos_cpu_format_example_frame180_262k/
                          Tracked compact-field plus particle-output example
pygsfit_cp-main/         External gyrosynchrotron backend source bundled for reference use
```

Local generated results are written under `benchmark_runs/`, which is intentionally ignored by Git.

## Requirements

### C++ / Kokkos

- CMake 3.24+
- C++20 compiler
- Kokkos installation with OpenMP for CPU runs
- Optional CUDA Kokkos installation for GPU runs
- Optional MPI/Fortran/HDF5 stack for the Li Xiaocan Fortran reference comparison

Typical local CPU Kokkos path:

```bash
/usr/local/kokkos_cpu/lib/cmake/Kokkos
```

### Python

The post-processing environment should provide:

```bash
numpy
h5py
matplotlib
streamlit
imageio-ffmpeg
```

Example setup:

```bash
python3 -m venv /tmp/kpt_postprocess_venv
/tmp/kpt_postprocess_venv/bin/python -m pip install -r particleEmission/requirements.txt
```

`imageio-ffmpeg` provides a bundled ffmpeg binary when system `ffmpeg` is unavailable.

## Build

CPU build:

```bash
cmake -S . -B cmake-build-benchmark-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native" \
  -DKokkos_DIR=/usr/local/kokkos_cpu/lib/cmake/Kokkos

cmake --build cmake-build-benchmark-cpu -j 16
```

CUDA build, when a CUDA-enabled Kokkos installation is available:

```bash
cmake -S . -B cmake-build-benchmark-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/local/kokkos/bin/nvcc_wrapper \
  -DKokkos_DIR=/usr/local/kokkos/lib/cmake/Kokkos

cmake --build cmake-build-benchmark-cuda -j 16
```

## Input Data

The calibration driver consumes compact 2D background fields:

```text
int32 nx
int32 ny
double x_edges[nx + 1]
double y_edges[ny + 1]
double Bx[nx * ny]
double By[nx * ny]
double Bz[nx * ny]
double Vx[nx * ny]
double Vy[nx * ny]
double Vz[nx * ny]
```

Two converters are available:

```bash
# Athena++ .athdf -> compact field, with optional emission background maps
/tmp/kpt_postprocess_venv/bin/python apps/athena2bin.py \
  --input-glob "/path/to/reconnection.prim.*.athdf" \
  --output-dir benchmark_runs/compact_field \
  --emission-background-dir benchmark_runs/emission_background \
  --start-frame 0 \
  --end-frame 200

# Li Xiaocan Fortran MHD binary -> compact field
cmake-build-benchmark-cpu/fortran_mhd_to_compact_field \
  --input-dir /path/to/bin_data \
  --output-dir benchmark_runs/compact_field \
  --start-frame 0 \
  --end-frame 200
```

Use `apps/athena2bin.py --skip-compact-output` when only thermal/background emission maps are needed.

## Running Transport

Smoke run:

```bash
cmake-build-benchmark-cpu/kokkos_particle_transport_app \
  --profile smoke \
  --transport parker \
  --field-dir benchmark_runs/compact_field \
  --output-dir benchmark_runs/smoke \
  --frames 1
```

Parker CPU run with per-frame particle snapshots:

```bash
OMP_NUM_THREADS=16 KOKKOS_NUM_THREADS=16 OMP_PROC_BIND=spread OMP_PLACES=threads \
cmake-build-benchmark-cpu/kokkos_particle_transport_app \
  --profile fortran-global \
  --transport parker \
  --field-dir benchmark_runs/compact_field \
  --output-dir benchmark_runs/kokkos_cpu \
  --start-frame 0 \
  --end-frame 200 \
  --particles-per-frame 1600 \
  --rank-scale 16 \
  --capacity 16000000 \
  --histogram-interval 10 \
  --particle-snapshot-interval 1
```

Focused transport uses:

```bash
--transport focused
```

Restart from a completed-frame particle snapshot:

```bash
cmake-build-benchmark-cpu/kokkos_particle_transport_app \
  --profile fortran-global \
  --transport parker \
  --restart-particle-snapshot benchmark_runs/kokkos_cpu/particles_00100.bin \
  --field-dir benchmark_runs/compact_field \
  --output-dir benchmark_runs/kokkos_cpu \
  --end-frame 200
```

When the snapshot name matches `particles_XXXXX.bin`, the driver infers `--start-frame XXXXX`; shell wrappers expose the same path through `KPT_RESTART_PARTICLE_SNAPSHOT=/path/to/particles_00100.bin`.

## Walltime Control

The Kokkos driver can stop cleanly at a completed frame:

```bash
--walltime-hours 10 \
--walltime-reserve-minutes 30
```

Shell wrappers expose the same control through:

```bash
KPT_KOKKOS_WALLTIME_HOURS=10
KPT_KOKKOS_WALLTIME_RESERVE_MINUTES=30
```

This is a frame-boundary diagnostic stop, not a restart/checkpoint system.

## Benchmark Wrappers

Common wrappers:

```bash
scripts/run_reconnection_three_cases.sh
scripts/run_reconnection_optimized_comparison.sh
scripts/run_reconnection_focused_comparison.sh
scripts/run_local_parker_focused_speed_accuracy_test.sh
```

These scripts can run Fortran, Kokkos CPU, and Kokkos GPU comparisons and generate timing/spectrum products under `benchmark_runs/`.

## Emission Synthesis and Movie Pipeline

For a single Parker particle snapshot:

```bash
scripts/run_parker_emission_synthesis.sh
```

For a full CPU Parker rerun followed by movie rendering:

```bash
KPT_RUN_ROOT=benchmark_runs/kokkos_cpu_parker_movie_200 \
KPT_FIELD_DIR=benchmark_runs/local_parker_speed_accuracy/compact_field \
KPT_PREPARE_COMPACT_FIELD=0 \
KPT_START_FRAME=0 \
KPT_END_FRAME=200 \
KPT_MOVIE_START_FRAME=1 \
KPT_MOVIE_END_FRAME=200 \
KPT_PARTICLE_SNAPSHOT_INTERVAL=1 \
KPT_CPU_CORES=16 \
KPT_KOKKOS_CPU_THREADS=16 \
scripts/run_kokkos_cpu_parker_movie_pipeline.sh
```

The movie renderer creates a 2x4 panel per frame:

1. `Bz`
2. `Jz = dBy/dx - dBx/dy`
3. particle weight below the configured energy threshold
4. particle weight above the configured energy threshold
5. beam-convolved 1 GHz image
6. beam-convolved 3 GHz image
7. beam-convolved 5 GHz image
8. full-domain spectrum

Default movie settings:

```text
image grid: 128 x 128
beam FWHM: 3 pixels
codec: ProRes HQ through ffmpeg or imageio-ffmpeg
```

Per-frame particle snapshots are large. A 200-frame Parker run at the local default size can write `50+ GB` of particle data.

## Physics Summary

### Parker Transport

The Parker SDE is represented schematically as:

$$d\mathbf{X} = \mathbf{u}_X dt + \sum_\sigma \mathbf{A}_\sigma dW_\sigma .$$

$$\mathbf{u}_X = \mathbf{V}_{sw} + \mathbf{V}_d + \nabla \cdot \mathbf{K}.$$

$$dp = -\frac{p}{3}\left(\nabla \cdot \mathbf{V}_{sw}\right)dt .$$

with

$$\kappa_{ij} = \kappa_\perp \delta_{ij} - (\kappa_\perp - \kappa_\parallel)b_i b_j .$$

$$\sum_\sigma \mathbf{A}_\sigma \mathbf{A}_\sigma^T = 2\mathbf{K}.$$

A convenient drift representation is:

$$\mathbf{G} = \frac{\mathbf{B}}{B^2}.$$

$$\mathbf{V}_d = \frac{pvc}{3q}\nabla \times \mathbf{G}.$$

### Focused Transport

The focused transport branch evolves position, momentum, and pitch-angle cosine:

$$d\mathbf{X} = \mathbf{u}_X dt + \sum_\sigma \mathbf{A}_\sigma dW_\sigma .$$

$$\mathbf{u}_X = v\mu\mathbf{b} + \mathbf{V}_{sw} + \mathbf{V}_d + \nabla \cdot \mathbf{K}_\perp .$$

The focused drift velocity contains gradient and curvature drift. The generic `FocusTransportSolver` precomputes two drift bases:

$$\mathbf{G}_B = \frac{\mathbf{b}\times\nabla\ln B}{B},\quad \mathbf{G}_c = \frac{\mathbf{b}\times(\mathbf{b}\cdot\nabla)\mathbf{b}}{B}.$$

$$\mathbf{V}_d = \frac{pvc}{q}\left[\frac{1-\mu^2}{2}\mathbf{G}_B + \mu^2\mathbf{G}_c\right].$$

In older Fortran notation, `G_mumu` corresponds to the pitch-angle diffusion coefficient `D_mumu` used here. The generic scattering model uses:

$$d\mu = \left(F_\mu + \frac{\partial D_{\mu\mu}}{\partial\mu}\right)dt + \sqrt{2D_{\mu\mu}}dW_\mu.$$

$$\Omega = \frac{|q|B}{\gamma mc},\quad \xi = \frac{\Omega L_c}{v},\quad \mu_h = |\mu| + h_0,\quad C_\gamma = \frac{\gamma_k\sin(\pi/\gamma_k)}{\pi}.$$

$$D_{\mu\mu} = \frac{\pi\Omega\sigma^2}{4}C_\gamma(1-\mu^2)\frac{\xi\mu_h^{\gamma_k-1}}{\mu_h^{\gamma_k}+\xi^{\gamma_k}}.$$

The current 2D reconnection calibration driver in `src/main.cpp` uses a reduced form to match the Fortran local example:

$$D_{\mu\mu}^{2D} = d_{uu0}(1-\mu^2)(|\mu|^{\gamma_{turb}-1}+h_0)B^{2-\gamma_{turb}}\left(\frac{p}{p_0}\right)^{\gamma_{turb}-1}.$$

The calibration-driver defaults are `duu0=5578.445`, `h0=0.2`, `gamma_turb=5/3`, `p0=0.1`, and `mu_max=0.99`. In the same calibration kernel, the focused-drift `pvc/q` prefactor is replaced by the nondimensional `1/(q R_d)`:

$$R_d(p)=\sqrt{\left(a_1\frac{p_0}{p}\right)^2+\left(a_2\frac{p_0^2}{p^2}\right)^2},\quad a_1=850964.408,\quad a_2=13575468.975.$$

The corresponding implementation is in `include/LegencyModel.hpp`, `include/FocusCoefficient.hpp`, and `src/main.cpp`.

## Current Limitations

- The reconnection calibration driver supports frame-boundary particle-state restart from `particles_XXXXX.bin`, but it is not a general production checkpoint system.
- Kokkos walltime stopping writes final diagnostics and optional particle snapshots at a completed frame; restart restores particle state but does not restore a bitwise-reproducible RNG state.
- Current reduced reconnection particle snapshots store scalar momentum magnitude but not full momentum direction.
- Parker emission reconstruction treats stored `mu` as unavailable; focused transport is the branch that carries physical pitch-angle information.
- Absolute particle energy calibration for emission thresholds still depends on the transport-to-CGS normalization selected for a given run.

## Documentation

- [docs/code_map.md](docs/code_map.md): detailed code map and interface status.
- [particleEmission/workflow_plan.md](particleEmission/workflow_plan.md): staged emission workflow design.
- [emissionValidation/README.md](emissionValidation/README.md): synthetic emission validation products.

## License and Attribution

This repository is maintained by Yihan Liu / SDU.

The repository includes reference material and source code from external workflows, including the Li Xiaocan stochastic Parker example and `pygsfit_cp-main` gyrosynchrotron backend material. Keep provenance and license notes intact when redistributing derived work.
