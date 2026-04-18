# Particle Emission Workflow Plan

## Goal

Build a post-processing workflow on top of the current transport solver outputs to:

1. reconstruct or assign local gyrosynchrotron source parameters for each simulation region;
2. forward-synthesize the microwave spectrum of each region, each image pixel, and selected ROIs;
3. report total intensity, circular polarization, and linear polarization when available;
4. express the results both as specific intensity and brightness temperature;
5. export binary products that can later be opened by Python scripts for imaging, beam convolution, and spectral analysis.

The first target is gyrosynchrotron emission, with explicit support for self-absorption, Razin suppression, and a configurable background radiation term.

## Current Repository Context

### Available solver-side inputs

- `kokkos_cpu_format_example_frame180_262k/compact_field/field00180.bin`
  - 2D Cartesian grid
  - `x_edges`, `y_edges`
  - `Bx`, `By`, `Bz`
  - `Vx`, `Vy`, `Vz`
- `kokkos_cpu_format_example_frame180_262k/kokkos_cpu/particles_00181.bin`
  - particle snapshot version 4
  - `position`
  - `previous_position`
  - `previous_step_position`
  - `momentum_magnitude`
  - `mu`
  - `weight`
  - `status`, `split_level`, `sort_key`

### Available external microwave engine

- `pygsfit_cp-main/`
  - Python wrapper around the Fleishman/Kuznetsov microwave spectrum code
  - already supports fitting and forward spectrum evaluation through the shared library
  - current Python wrapper returns two polarization channels that are summed for total model flux
  - deeper Fortran code contains more internal polarization and transfer information than the current Python wrapper exposes

## Important Constraints Identified Now

### Constraint 1: the current particle snapshot does not store full momentum direction

The current particle container stores scalar momentum magnitude plus pitch-angle cosine `mu`, but not the full particle momentum vector or gyrophase. This is enough for gyrotropic distribution reconstruction, but it is not enough for an exact single-particle directional radiation treatment.

Implication:

- a distribution-based emissivity workflow is feasible now;
- a true single-particle directional emission workflow would require new particle outputs later.
- for Parker transport outputs, `mu` must currently be treated as unavailable because the
  saved value is expected to be zero everywhere; any emission workflow that depends on
  pitch-angle information must therefore branch on transport model and avoid using
  Parker `mu` as physical anisotropy data.

### Constraint 2: the current field example does not include thermal plasma state

The compact field example currently provides magnetic and flow fields, but not:

- thermal electron density `n_th`;
- thermal temperature `T_e`;
- explicit LOS depth;
- plasma density or plasma frequency needed to model Razin suppression directly from the field file alone.

Implication:

- a fully constrained physical microwave forward model cannot be closed from the current field file alone;
- the workflow must support additional background parameter sources:
  - user-provided maps or constants,
  - reconstructed parameter maps,
  - fitted parameters from observations through `pygsfit_cp-main`,
  - future solver-side plasma outputs.
- the workflow should explicitly reserve input interfaces for `n_th`, `T_e`, and related
  background variables so a later MHD-to-emission converter can fill them directly.

### Constraint 3: field input must support both file-backed and analytic representations

The emission workflow should not assume that the magnetic and background fields always
come from a binary snapshot. The repository already contains both stored-grid and
analytic-grid abstractions, so the emission side should keep the same flexibility.

Implication:

- field access should be abstracted behind a common source interface;
- the first implementation should support at least:
  - compact field files or later MHD-converted files;
  - analytic field/background providers for controlled tests and idealized models.

### Constraint 4: inverse and forward problems must be separated clearly

The current solver outputs can support local nonthermal distribution reconstruction, but they do not by themselves uniquely determine all microwave source parameters. Therefore the workflow should distinguish:

- **source reconstruction** from simulation outputs and priors;
- **forward emission synthesis** from a completed source description;
- **observational fitting/inversion** when real spectra are available.

## Recommended Development Strategy

Develop the workflow in three layers:

1. **reference CPU workflow**
   - Python driven
   - reuse `pygsfit_cp-main`
   - validate physics, units, and data products first
2. **repository-native production workflow**
   - C++/Kokkos data preparation
   - structured binary output
   - optional Python visualization front-end
3. **performance upgrade**
   - move the heavy loops to Kokkos only after the reference workflow is validated

This reduces physics risk before optimizing performance.

## Proposed Workflow Architecture

## Phase 0: Define the physical closure

Before writing the production pipeline, define one explicit source model for each simulation cell or synthesis element.

Required local quantities:

- magnetic field magnitude `B`
- viewing angle `theta` between LOS and `B`
- nonthermal electron density or normalization `n_nth`
- thermal electron density `n_th`
- electron temperature `T_e`
- nonthermal spectral shape parameters such as `delta`, `E_min`, `E_max`
- LOS depth `L`
- optional background brightness temperature or external incident intensity

Recommended input-provider split:

- **Field provider**
  - file-backed field snapshots
  - analytic field definitions
- **Background plasma provider**
  - constants
  - tabulated maps
  - future MHD-converted inputs
  - fitted parameter maps
- **Particle distribution provider**
  - solver particle snapshots
  - future reduced distribution products

Recommended closure modes:

1. **Synthetic forward mode**
   - use solver outputs plus user-defined background parameter recipes
   - best for initial validation
2. **Hybrid map mode**
   - use solver outputs for nonthermal particles
   - use external maps or fitted maps for `n_th`, `T_e`, `L`, and background
3. **Observation-constrained mode**
   - fit observed pixel spectra with `pygsfit_cp-main`
   - compare fitted parameter maps against solver-derived distributions

## Phase 1: Build a common emission input representation

Create a common source description independent of the final emission backend.

Recommended structure:

- field-source metadata
- grid metadata
- observer definition
- frequency grid
- per-cell source parameters
- optional per-cell particle distribution descriptors
- optional per-pixel or per-ROI aggregation masks

The representation should carry explicit provenance tags for:

- field source kind: `file` or `analytic`
- background source kind: `constant`, `map`, `mhd_converted`, or `fit`
- particle source kind: `focused`, `parker`, or future reduced products

This layer should support two synthesis modes.

### Mode A: cell-based synthesis

1. bin weighted particles into simulation cells;
2. reconstruct local nonthermal electron distributions;
3. combine them with background plasma parameters;
4. compute local emissivity/absorption spectra;
5. integrate to pixels and ROIs.

This should be the primary production path.

### Mode B: particle-accumulated diagnostic synthesis

1. accumulate weighted particle contributions into pixel or cell histograms;
2. use this mode only as a diagnostic or cross-check;
3. do not treat it as the default production method because it will be noisier and more expensive.

## Phase 2: Reconstruct local nonthermal distributions

From the current particle snapshot, reconstruct local distributions in each cell or pixel:

- spatial binning by cell or projected pixel;
- energy binning from `momentum_magnitude`;
- pitch-angle binning from `mu` only when the transport model provides meaningful `mu`;
- weighted accumulation using `weight`;
- optional time interpolation using `previous_position` and `previous_step_position`.

Transport-model-specific policy:

- **Focused transport**
  - use `mu` as a physical pitch-angle coordinate
  - allow anisotropic distribution reconstruction
- **Parker transport**
  - treat `mu` as unavailable for emission purposes
  - reconstruct only isotropic or angle-parameterized reduced distributions unless a
    new directional output is added later

Outputs of this stage:

- local particle counts and weights;
- local `f(E, mu)` or reduced moments;
- effective nonthermal density `n_nth`;
- spectral slope estimates for a power-law approximation where needed;
- diagnostic quality flags for cells with insufficient particle statistics.

This stage is essential because the current particle data naturally support distribution reconstruction, not direct exact single-particle radiation.

## Phase 3: Couple to the microwave emission engine

### Short-term recommendation

Use `pygsfit_cp-main` as the reference backend first.

Why:

- it already encapsulates a tested gyrosynchrotron spectrum engine;
- it already handles fitting-oriented parameterization that is close to the required source closure;
- it is the fastest path to a trustworthy reference implementation.

### Required wrapper work

The current Python wrapper is not yet sufficient for the full target product set. It will need to be extended to expose:

- right and left polarization outputs explicitly;
- total intensity `I = R + L`;
- circular polarization `V = R - L`;
- linear polarization terms if the backend can provide `Q` and possibly `U`, with
  `L = sqrt(Q^2 + U^2)`;
- internal transfer diagnostics needed to interpret self-absorption and Razin effects.

If the backend naturally returns flux density, the workflow must convert it to specific
intensity using the pixel or source solid angle before deriving brightness temperature.

### Important note on spectral component labeling

The requested spectral labels

- gyrosynchrotron emission,
- self-absorption,
- Razin effect,
- background radiation

should not be treated as simple additive positive components in all cases. Self-absorption and Razin suppression modify the transfer solution.

A physically cleaner design is to store several controlled spectra:

- full physics;
- optically thin reference;
- no-Razin reference;
- background-only reference;
- optional thermal-only and nonthermal-only references.

Then the plotting script can display diagnostic differences such as:

- absorption impact = optically thin reference minus full spectrum;
- Razin impact = no-Razin reference minus full spectrum.

This avoids mislabeling non-additive transfer effects as independent conserved intensities.

## Phase 4: LOS transfer, pixel mapping, and image synthesis

Define how local source elements map to image pixels.

Recommended baseline assumptions:

- start with a 2D simulation plane plus an assumed LOS depth;
- treat each simulation cell as a homogeneous slab for the first implementation;
- integrate the transfer equation along the selected LOS ordering;
- build frequency-dependent image cubes.

Per frequency, store:

- specific intensity `I_nu`
- brightness temperature `T_B`
- circular polarization intensity or Stokes `V`
- circular polarization degree `V / I`
- linear polarization intensity and degree when available

Use the Rayleigh-Jeans conversion

- `T_B = c^2 I_nu / (2 k_B nu^2)`

for every stored Stokes-like intensity that is reported as a brightness temperature.

For the first version, if the backend does not expose linear polarization robustly, mark it as unavailable rather than inventing a proxy.

## Phase 5: ROI and beam-aware spectrum extraction

The analysis script should support three ROI selectors:

1. fixed pixel selector;
2. fixed geometric region selector;
3. beam-centered selector with frequency-dependent area.

The beam-centered selector should support:

- center position in simulation or image coordinates;
- Gaussian beam with user-defined FWHM;
- frequency-dependent beam size, so the integrated region changes with wavelength.

For each ROI, the script should return:

- full spectrum over the requested frequency grid;
- total intensity spectrum;
- brightness temperature spectrum;
- circular polarization spectrum and degree;
- linear polarization spectrum and degree when available;
- diagnostic component/reference curves.

## Phase 6: Binary product design

The output must be binary and easy to reopen from Python.

### Recommended first implementation

Use an HDF5 container as the first binary product format.

Reasons:

- binary by definition;
- simple Python access with `h5py`;
- natural support for multidimensional cubes and metadata;
- easier than a custom raw binary format for early iteration.

### Suggested layout

- `/meta`
  - code version
  - frame id
  - unit metadata
  - observer metadata
  - closure mode
  - transport model
  - field source kind
  - background source kind
- `/grid`
  - coordinates or edges
  - pixel geometry
- `/freq_hz`
- `/cell`
  - reconstructed source parameters
  - quality flags
- `/image/full`
  - `I_specific`
  - `I_brightness_temperature`
  - `V_specific`
  - `V_brightness_temperature`
  - optional `Q`, `U`, `L`
- `/image/references`
  - optically thin
  - no Razin
  - background only
  - optional thermal only
- `/roi`
  - saved ROI definitions
  - extracted spectra

If repository consistency later requires a custom raw binary format, a second-stage writer can be added after the HDF5 contract stabilizes.

## Phase 7: Python analysis and visualization tools

Provide one Python entry point that can:

1. open the emission binary product;
2. save a radio image at one selected frequency;
3. save both native-resolution and beam-convolved images;
4. extract one-pixel, one-region, or one-beam spectrum;
5. overlay diagnostic component/reference curves;
6. export figures and optionally NumPy tables for downstream work.

Required plotting outputs:

- fine-structure image at one frequency;
- blurred image after beam convolution;
- selected ROI spectrum over all frequencies;
- polarization panels or overlays;
- component/reference decomposition panels.

## Kokkos Assessment

## Kokkos is not required for the first implementation

For the first working version, Kokkos is **not mandatory** if the goal is:

- validate the physical closure;
- verify unit conversions;
- prove the binary product format;
- prototype image and ROI analysis;
- reuse the existing `pygsfit_cp-main` backend.

Python plus the existing Fortran backend is the lowest-risk path here.

## Kokkos becomes valuable in later stages

Kokkos should be considered once the reference workflow is validated and one or more of the following loops become dominant:

- particle-to-cell histogramming over very large particle sets;
- per-cell and per-frequency emissivity evaluation over large grids;
- LOS transfer over many frequencies and many image pixels;
- repeated synthesis for parameter scans or fitting loops.

Recommended policy:

1. keep orchestration, plotting, and interactive ROI analysis in Python;
2. move heavy deposition and transfer loops into repository-native C++/Kokkos when profiling shows they dominate;
3. keep the backend interface modular so the same binary format works with both CPU-reference and Kokkos-accelerated producers.

## Proposed Milestones

### Milestone A: planning and contracts

- define source closure modes
- define units and observer geometry
- define HDF5 schema
- define required backend outputs

### Milestone B: reference forward synthesis

- reconstruct local particle distributions
- prepare per-cell source parameters
- call the microwave backend on a small test subset
- generate spectra for selected cells and pixels

### Milestone C: imaging pipeline

- integrate cell spectra into image cubes
- convert to specific intensity and brightness temperature
- implement beam convolution
- implement ROI extraction

### Milestone D: component-aware analysis

- add controlled reference spectra
- add component-aware plotting
- add quality flags and diagnostics

### Milestone E: performance pass

- profile the reference implementation
- decide which kernels should move to Kokkos
- keep the binary product contract unchanged

## Definition of Done for the First Useful Version

The first useful version should be able to:

1. read one solver field snapshot and one particle snapshot;
2. reconstruct local nonthermal distributions on the simulation grid;
3. combine them with user-specified or externally supplied thermal/background parameters;
4. synthesize microwave spectra on a chosen frequency grid;
5. output binary image cubes and ROI spectra;
6. produce one native image, one beam-convolved image, and one ROI spectrum plot from Python;
7. report total intensity and circular polarization, with linear polarization included only if the backend exposes it reliably.

## Open Technical Questions to Resolve Before Implementation

1. What physical unit system should map the current solver outputs to CGS for emission?
2. What LOS geometry should be assumed relative to the 2D reconnection plane?
3. Where will `n_th`, `T_e`, and LOS depth come from in the first physics-meaningful run?
4. Should the first version treat each simulation cell as a homogeneous source?
5. Do we want HDF5 as the official first binary product, or do we want a repository-native raw binary from the start?
6. Is exposing linear polarization from the Fleishman/Kuznetsov backend a first-phase requirement or a second-phase requirement?

## Immediate Next Step

After this planning document is accepted, the next implementation discussion should focus on:

1. the physical closure of one cell;
2. the minimum viable binary schema;
3. the smallest end-to-end prototype dataset to validate the workflow.
