#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "FieldInterpolator.hpp"
#include "ParkerCoefficient.hpp"
#include "RandomManager.hpp"
#include "SolverBase.hpp"

/**
 * Host/device snapshot of the Parker transport terms for one particle.
 */
template <int SpaceDim, int VecDim>
struct ParkerDeterministicParticleTerms {
    using coordinate_array_type = Kokkos::Array<double, SpaceDim>;
    using vector_array_type = Kokkos::Array<double, VecDim>;

    std::uint64_t particle_id{0};
    coordinate_array_type position{};
    double momentum_magnitude{0.0};
    double gamma{1.0};
    double speed{0.0};
    vector_array_type magnetic_field{};
    vector_array_type magnetic_direction{};
    double magnetic_field_magnitude{0.0};
    vector_array_type solar_wind_velocity{};
    vector_array_type drift_velocity{};
    vector_array_type diffusion_advection{};
    vector_array_type total_advection{};
    coordinate_array_type coordinate_rate{};
    double solar_wind_divergence{0.0};
    double momentum_rate{0.0};
    double kappa_parallel{0.0};
    double kappa_perpendicular{0.0};
    coordinate_array_type kappa_effective{};
    double diffusion_time_step{0.0};
    double advection_time_step{0.0};
    double momentum_time_step{0.0};
    double stable_time_step{0.0};
};

template <
    typename GridType,
    int VecDim = GridType::vec_dim,
    typename DeviceType = typename GridType::device_type
>
struct ParkerSolverDebugger;

/**
 * Parker transport solver with deterministic and Euler-Maruyama stochastic stepping.
 *
 * Coefficient construction remains in ParkerCoefficient.hpp. This solver interpolates
 * precomputed coefficient fields, combines particle-gamma factors, and advances active
 * particles with deterministic drift/advection, adiabatic momentum change, and optional
 * magnetic-field-aligned stochastic diffusion increments.
 */
template <
    typename GridType,
    int VecDim = GridType::vec_dim,
    typename DeviceType = typename GridType::device_type
>
struct ParkerSolver : public SolverBase<GridType, VecDim, DeviceType> {
    using base_type = SolverBase<GridType, VecDim, DeviceType>;
    using grid_type = typename base_type::grid_type;
    using device_type = typename base_type::device_type;
    using execution_space = typename base_type::execution_space;
    using memory_space = typename base_type::memory_space;
    using layout_type = typename base_type::layout_type;
    using vector_field_type = typename base_type::vector_field_type;
    using scalar_field_type = ScalarField<grid_type, layout_type>;
    using particle_system_type = typename base_type::particle_system_type;
    using size_type = typename particle_system_type::size_type;
    using coordinate_array_type = typename grid_type::coordinate_array_type;
    using vector_array_type = Kokkos::Array<double, VecDim>;
    using random_manager_type = RandomManager<DeviceType>;
    using stochastic_sampler_type = StochasticSampler<DeviceType>;
    using coefficient_field_set_type =
        ParkerCoefficient::TransportCoefficientFieldSet<grid_type, layout_type>;
    using debug_terms_type = ParkerDeterministicParticleTerms<GridType::space_dim, VecDim>;
    using debug_terms_view_type = Kokkos::View<debug_terms_type*, layout_type, memory_space>;

    static constexpr int space_dim = base_type::space_dim;
    static constexpr int vec_dim = base_type::vec_dim;
    static_assert(VecDim > 0, "ParkerSolver VecDim must be positive.");
    static_assert(VecDim <= 3, "ParkerSolver currently supports VecDim <= 3.");
    static_assert(space_dim <= VecDim,
                  "ParkerSolver requires VecDim to cover all coordinate directions.");

    coefficient_field_set_type coefficients;

    explicit ParkerSolver(grid_type input_grid,
                          vector_field_type input_B_vec,
                          vector_field_type input_V_body)
        : base_type(input_grid, input_B_vec, input_V_body) {}

    ParkerSolver(grid_type input_grid,
                 vector_field_type input_B_vec,
                 vector_field_type input_V_body,
                 coefficient_field_set_type input_coefficients)
        : base_type(input_grid, input_B_vec, input_V_body),
          coefficients(input_coefficients) {}

    ParkerSolver(particle_system_type input_particles,
                 grid_type input_grid,
                 vector_field_type input_B_vec,
                 vector_field_type input_V_body)
        : base_type(input_particles, input_grid, input_B_vec, input_V_body) {}

    ParkerSolver(particle_system_type input_particles,
                 grid_type input_grid,
                 vector_field_type input_B_vec,
                 vector_field_type input_V_body,
                 coefficient_field_set_type input_coefficients)
        : base_type(input_particles, input_grid, input_B_vec, input_V_body),
          coefficients(input_coefficients) {}

    /**
     * Assign precomputed coefficient fields after construction.
     */
    void set_coefficient_fields(coefficient_field_set_type input_coefficients) {
        coefficients = input_coefficients;
    }

    KOKKOS_INLINE_FUNCTION
    void initializer() {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        (void)i;
    }

    /**
     * Compute the adaptive Parker time step in code units.
     */
    double compute_adaptive_time_step() const {
        validate_for_deterministic_step();
        if (coefficients.perpendicular_model ==
            LegencyModel::PerpendicularDiffusionModelKind::ConstantParallelRatio) {
            return compute_adaptive_time_step_impl<true>();
        }
        return compute_adaptive_time_step_impl<false>();
    }

    /**
     * Advance all active particles by one deterministic Parker step.
     *
     * The returned value is the dimensionless time step actually used by the update.
     */
    double advance_deterministic() {
        validate_for_deterministic_step();
        this->particles.apply_grid_boundary_conditions(this->grid,
                                                       GridDomain::PhysicalDomain);
        const double step_dt = compute_adaptive_time_step();
        if (!std::isfinite(step_dt) || step_dt <= 0.0) {
            throw std::runtime_error(
                "ParkerSolver: adaptive deterministic time step is not positive and finite.");
        }

        this->dT = typename base_type::dimensionless_value_type(step_dt);
        this->particles.capture_previous_positions();
        if (coefficients.perpendicular_model ==
            LegencyModel::PerpendicularDiffusionModelKind::ConstantParallelRatio) {
            advance_deterministic_impl<true>(step_dt);
        } else {
            advance_deterministic_impl<false>(step_dt);
        }
        this->particles.apply_grid_boundary_conditions(this->grid,
                                                       GridDomain::PhysicalDomain);
        return step_dt;
    }

    /**
     * Advance all active particles by one Euler-Maruyama Parker SDE step.
     *
     * The stochastic spatial increment is sampled analytically from the local magnetic
     * field direction without constructing a stored diffusion matrix.
     */
    double advance_stochastic(const random_manager_type& random_manager) {
        validate_for_deterministic_step();
        this->particles.apply_grid_boundary_conditions(this->grid,
                                                       GridDomain::PhysicalDomain);
        const double step_dt = compute_adaptive_time_step();
        if (!std::isfinite(step_dt) || step_dt <= 0.0) {
            throw std::runtime_error(
                "ParkerSolver: adaptive stochastic time step is not positive and finite.");
        }

        this->dT = typename base_type::dimensionless_value_type(step_dt);
        this->particles.capture_previous_positions();
        if (coefficients.perpendicular_model ==
            LegencyModel::PerpendicularDiffusionModelKind::ConstantParallelRatio) {
            advance_stochastic_impl<true>(step_dt, random_manager);
        } else {
            advance_stochastic_impl<false>(step_dt, random_manager);
        }
        this->particles.apply_grid_boundary_conditions(this->grid,
                                                       GridDomain::PhysicalDomain);
        return step_dt;
    }

    /**
     * Compatibility name for one full Parker SDE step.
     */
    double advance_sde(const random_manager_type& random_manager) {
        return advance_stochastic(random_manager);
    }

private:
    static constexpr double large_time_step = 1.0e300;
    static constexpr double diffusion_displacement_fraction = 0.5;
    static constexpr double advection_displacement_fraction = 1.0;
    static constexpr double momentum_relative_change_fraction = 1.0;

    /**
     * Validate field and scalar state needed by Parker stepping.
     */
    void validate_for_deterministic_step() const {
        static_assert(requires(const grid_type& input_grid,
                               typename grid_type::index_array_type indices,
                               typename vector_field_type::centering_array_type centerings) {
                          input_grid.cell_index(0, 0.0, GridDomain::PhysicalDomain);
                          input_grid.physical_cell_width(0, indices, centerings);
                          input_grid.contains_coordinate(0, 0.0,
                                                         GridDomain::PhysicalDomain);
                          input_grid.wrap_coordinate(0, 0.0);
                      },
                      "ParkerSolver stepping requires a concrete coordinate grid.");

        if (!coefficients.configured()) {
            throw std::runtime_error(
                "ParkerSolver: coefficient fields are not configured.");
        }
        if (!std::isfinite(this->courant_number()) || this->courant_number() <= 0.0) {
            throw std::runtime_error(
                "ParkerSolver: Courant scale must be positive and finite.");
        }
        if (!std::isfinite(this->particles.properties.charge) ||
            this->particles.properties.charge == 0.0) {
            throw std::runtime_error(
                "ParkerSolver: Parker drift requires a finite nonzero particle charge.");
        }
        validate_field_point_count(this->B_vec, "magnetic field");
        validate_field_point_count(this->V_body, "solar wind velocity");
        validate_field_point_count(coefficients.magnetic_drift_curl,
                                   "magnetic drift curl");
        validate_field_point_count(coefficients.kappa_parallel_gamma_one,
                                   "kappa_parallel_gamma_one");
        validate_field_point_count(coefficients.kappa_perpendicular_gamma_one,
                                   "kappa_perpendicular_gamma_one");
        validate_field_point_count(coefficients.solar_wind_divergence,
                                   "solar wind divergence");
        if (coefficients.perpendicular_model ==
            LegencyModel::PerpendicularDiffusionModelKind::ConstantParallelRatio) {
            validate_field_point_count(
                coefficients.diffusion_tensor_divergence.constant_ratio_basis,
                "constant-ratio diffusion-tensor divergence");
        } else {
            validate_field_point_count(
                coefficients.diffusion_tensor_divergence.parallel_basis,
                "parallel diffusion-tensor divergence basis");
            validate_field_point_count(
                coefficients.diffusion_tensor_divergence.perpendicular_basis,
                "perpendicular diffusion-tensor divergence basis");
        }
    }

    /**
     * Validate one same-grid field against the solar-wind field storage size.
     */
    template <typename FieldType>
    void validate_field_point_count(const FieldType& field,
                                    const char* field_name) const {
        if (field.point_count() == 0) {
            throw std::runtime_error(std::string("ParkerSolver: missing ") +
                                     field_name + " field.");
        }
        if (field.point_count() != this->V_body.point_count()) {
            throw std::runtime_error(std::string("ParkerSolver: ") + field_name +
                                     " field point count does not match V_sw.");
        }
        for (int dim = 0; dim < space_dim; ++dim) {
            if (field.centering(dim) != this->V_body.centering(dim) ||
                field.lower_index(dim, GridDomain::GhostedDomain) !=
                    this->V_body.lower_index(dim, GridDomain::GhostedDomain) ||
                field.upper_index(dim, GridDomain::GhostedDomain) !=
                    this->V_body.upper_index(dim, GridDomain::GhostedDomain)) {
                throw std::runtime_error(std::string("ParkerSolver: ") + field_name +
                                         " storage domain does not match V_sw.");
            }
        }
    }

    /**
     * Return a device-safe absolute value.
     */
    KOKKOS_INLINE_FUNCTION
    static double absolute_value(const double value) {
        return value >= 0.0 ? value : -value;
    }

    /**
     * Return a device-safe finite positive check.
     */
    KOKKOS_INLINE_FUNCTION
    static bool finite_positive(const double value) {
        return Kokkos::isfinite(value) && value > 0.0;
    }

    /**
     * Return a device-safe finite non-negative check.
     */
    KOKKOS_INLINE_FUNCTION
    static bool finite_nonnegative(const double value) {
        return Kokkos::isfinite(value) && value >= 0.0;
    }

    /**
     * Return the smaller of two positive time-step candidates.
     */
    KOKKOS_INLINE_FUNCTION
    static double min_time_step(const double left, const double right) {
        return left < right ? left : right;
    }

    /**
     * Sample a physical stochastic increment with covariance 2 K dt.
     *
     * The implementation draws an isotropic Gaussian vector, removes its parallel
     * component, and replaces that component with an independent field-aligned draw.
     */
    template <typename GeneratorType>
    KOKKOS_INLINE_FUNCTION
    static vector_array_type sample_magnetic_diffusion_increment(
        const debug_terms_type& terms,
        const double step_dt,
        GeneratorType& generator) {
        vector_array_type gaussian_vector{};
        double isotropic_parallel_component = 0.0;
        for (int component = 0; component < VecDim; ++component) {
            gaussian_vector[component] =
                stochastic_sampler_type::normal(generator);
            isotropic_parallel_component +=
                gaussian_vector[component] * terms.magnetic_direction[component];
        }

        const double parallel_gaussian =
            stochastic_sampler_type::normal(generator);
        const double perpendicular_scale =
            terms.kappa_perpendicular > 0.0
                ? Kokkos::sqrt(2.0 * terms.kappa_perpendicular * step_dt)
                : 0.0;
        const double parallel_scale =
            terms.kappa_parallel > 0.0
                ? Kokkos::sqrt(2.0 * terms.kappa_parallel * step_dt)
                : 0.0;
        const double parallel_replacement =
            parallel_scale * parallel_gaussian -
            perpendicular_scale * isotropic_parallel_component;

        vector_array_type increment{};
        for (int component = 0; component < VecDim; ++component) {
            increment[component] =
                perpendicular_scale * gaussian_vector[component] +
                parallel_replacement * terms.magnetic_direction[component];
        }
        return increment;
    }

    /**
     * Build a cell-centered metric-centering array.
     */
    KOKKOS_INLINE_FUNCTION
    static typename vector_field_type::centering_array_type cell_centerings() {
        typename vector_field_type::centering_array_type centerings{};
        for (int dim = 0; dim < space_dim; ++dim) {
            centerings[dim] = GridCentering::CellCentered;
        }
        return centerings;
    }

    /**
     * Return whether a coordinate tuple lies in the physical domain after wrapping periodic dimensions.
     */
    KOKKOS_INLINE_FUNCTION
    bool prepare_physical_position(coordinate_array_type& position) const {
        for (int dim = 0; dim < space_dim; ++dim) {
            if (this->grid.is_periodic(dim)) {
                position[dim] = this->grid.wrap_coordinate(dim, position[dim]);
            }
            if (!this->grid.contains_coordinate(dim, position[dim],
                                                GridDomain::PhysicalDomain)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Convert a physical-vector component to a coordinate-time derivative.
     */
    KOKKOS_INLINE_FUNCTION
    double coordinate_rate_component(const coordinate_array_type& position,
                                     const int component,
                                     const double physical_component) const {
        if (component >= space_dim) {
            return 0.0;
        }
        const double h_inv = this->grid.h_inv(component, position);
        if (!Kokkos::isfinite(h_inv)) {
            Kokkos::abort("ParkerSolver: invalid inverse Lame coefficient.");
        }
        return physical_component * h_inv;
    }

    /**
     * Interpolate fields and assemble all deterministic Parker terms for one particle.
     */
    template <
        bool UseConstantRatio,
        typename MagneticFieldInterpolator,
        typename SolarWindInterpolator,
        typename DriftCurlInterpolator,
        typename DiffusionPrimaryInterpolator,
        typename DiffusionSecondaryInterpolator,
        typename KappaParallelInterpolator,
        typename KappaPerpendicularInterpolator,
        typename SolarWindDivergenceInterpolator
    >
    KOKKOS_INLINE_FUNCTION
    debug_terms_type evaluate_terms(
        coordinate_array_type position,
        const double input_momentum,
        const MagneticFieldInterpolator& magnetic_field_interpolator,
        const SolarWindInterpolator& solar_wind_interpolator,
        const DriftCurlInterpolator& drift_curl_interpolator,
        const DiffusionPrimaryInterpolator& diffusion_primary_interpolator,
        const DiffusionSecondaryInterpolator& diffusion_secondary_interpolator,
        const KappaParallelInterpolator& kappa_parallel_interpolator,
        const KappaPerpendicularInterpolator& kappa_perpendicular_interpolator,
        const SolarWindDivergenceInterpolator& solar_wind_divergence_interpolator) const {
        debug_terms_type terms;
        terms.position = position;
        terms.momentum_magnitude = absolute_value(input_momentum);
        terms.gamma = this->particles.properties.gamma_from_momentum_magnitude(
            terms.momentum_magnitude);
        terms.speed = this->particles.properties.speed_from_momentum_magnitude(
            terms.momentum_magnitude);

        const vector_array_type magnetic_field =
            magnetic_field_interpolator.sample(position);
        double magnetic_field_magnitude_squared = 0.0;
        for (int component = 0; component < VecDim; ++component) {
            terms.magnetic_field[component] = magnetic_field[component];
            magnetic_field_magnitude_squared +=
                magnetic_field[component] * magnetic_field[component];
        }
        terms.magnetic_field_magnitude =
            Kokkos::sqrt(magnetic_field_magnitude_squared);
        if (!finite_positive(terms.magnetic_field_magnitude)) {
            Kokkos::abort("ParkerSolver: stochastic diffusion requires |B| > 0.");
        }
        const double inverse_magnetic_field_magnitude =
            1.0 / terms.magnetic_field_magnitude;
        for (int component = 0; component < VecDim; ++component) {
            terms.magnetic_direction[component] =
                magnetic_field[component] * inverse_magnetic_field_magnitude;
        }

        const vector_array_type solar_wind = solar_wind_interpolator.sample(position);
        const vector_array_type drift_curl = drift_curl_interpolator.sample(position);
        vector_array_type diffusion_advection{};
        if constexpr (UseConstantRatio) {
            const vector_array_type constant_ratio_basis =
                diffusion_primary_interpolator.sample(position);
            diffusion_advection =
                ParkerCoefficient::apply_constant_ratio_diffusion_tensor_divergence_gamma(
                    constant_ratio_basis, terms.gamma);
        } else {
            const vector_array_type parallel_basis =
                diffusion_primary_interpolator.sample(position);
            const vector_array_type perpendicular_basis =
                diffusion_secondary_interpolator.sample(position);
            diffusion_advection =
                ParkerCoefficient::apply_readme_diffusion_tensor_divergence_gamma(
                    parallel_basis, perpendicular_basis, terms.gamma);
        }

        const double kappa_parallel_gamma_one =
            kappa_parallel_interpolator.sample_component(position, 0);
        const double kappa_perpendicular_gamma_one =
            kappa_perpendicular_interpolator.sample_component(position, 0);
        const auto diffusion_coefficients = LegencyModel::apply_particle_gamma(
            kappa_parallel_gamma_one, kappa_perpendicular_gamma_one, terms.gamma,
            coefficients.perpendicular_model);
        terms.kappa_parallel = diffusion_coefficients.kappa_parallel;
        terms.kappa_perpendicular = diffusion_coefficients.kappa_perpendicular;
        if (!finite_nonnegative(terms.kappa_parallel) ||
            !finite_nonnegative(terms.kappa_perpendicular)) {
            Kokkos::abort(
                "ParkerSolver: diffusion coefficients must be finite and non-negative.");
        }
        for (int dim = 0; dim < space_dim; ++dim) {
            const double b_i = terms.magnetic_direction[dim];
            terms.kappa_effective[dim] =
                terms.kappa_perpendicular +
                (terms.kappa_parallel - terms.kappa_perpendicular) * b_i * b_i;
            if (!finite_nonnegative(terms.kappa_effective[dim])) {
                Kokkos::abort(
                    "ParkerSolver: projected diffusion coefficient is invalid.");
            }
        }
        terms.solar_wind_divergence =
            solar_wind_divergence_interpolator.sample_component(position, 0);

        const double charge = this->particles.properties.charge;
        if (charge == 0.0) {
            Kokkos::abort("ParkerSolver: Parker drift requires nonzero charge.");
        }
        const double drift_prefactor =
            terms.momentum_magnitude * terms.speed *
            this->particles.properties.speed_of_light / (3.0 * charge);

        for (int component = 0; component < VecDim; ++component) {
            terms.solar_wind_velocity[component] = solar_wind[component];
            terms.drift_velocity[component] = drift_prefactor * drift_curl[component];
            terms.diffusion_advection[component] = diffusion_advection[component];
            terms.total_advection[component] =
                terms.solar_wind_velocity[component] +
                terms.drift_velocity[component] +
                terms.diffusion_advection[component];
        }
        for (int dim = 0; dim < space_dim; ++dim) {
            terms.coordinate_rate[dim] = coordinate_rate_component(
                position, dim, terms.total_advection[dim]);
        }
        terms.momentum_rate =
            -(terms.momentum_magnitude / 3.0) * terms.solar_wind_divergence;
        return terms;
    }

    /**
     * Evaluate all terms for one particle index.
     */
    template <
        bool UseConstantRatio,
        bool ComputeTimeStep,
        typename MagneticFieldInterpolator,
        typename SolarWindInterpolator,
        typename DriftCurlInterpolator,
        typename DiffusionPrimaryInterpolator,
        typename DiffusionSecondaryInterpolator,
        typename KappaParallelInterpolator,
        typename KappaPerpendicularInterpolator,
        typename SolarWindDivergenceInterpolator
    >
    KOKKOS_INLINE_FUNCTION
    debug_terms_type evaluate_particle_terms(
        const size_type particle_index,
        const MagneticFieldInterpolator& magnetic_field_interpolator,
        const SolarWindInterpolator& solar_wind_interpolator,
        const DriftCurlInterpolator& drift_curl_interpolator,
        const DiffusionPrimaryInterpolator& diffusion_primary_interpolator,
        const DiffusionSecondaryInterpolator& diffusion_secondary_interpolator,
        const KappaParallelInterpolator& kappa_parallel_interpolator,
        const KappaPerpendicularInterpolator& kappa_perpendicular_interpolator,
        const SolarWindDivergenceInterpolator& solar_wind_divergence_interpolator) const {
        coordinate_array_type position{};
        for (int dim = 0; dim < space_dim; ++dim) {
            position[dim] = this->particles.position(particle_index, dim);
        }
        if (!prepare_physical_position(position)) {
            return debug_terms_type{};
        }
        debug_terms_type terms = evaluate_terms<UseConstantRatio>(
            position, this->particles.momentum(particle_index),
            magnetic_field_interpolator, solar_wind_interpolator,
            drift_curl_interpolator, diffusion_primary_interpolator,
            diffusion_secondary_interpolator, kappa_parallel_interpolator,
            kappa_perpendicular_interpolator, solar_wind_divergence_interpolator);
        terms.particle_id = this->particles.particle_id(particle_index);
        if constexpr (ComputeTimeStep) {
            terms.stable_time_step = local_stable_time_step(terms);
        }
        return terms;
    }

    /**
     * Return the local adaptive time-step bound for one particle term snapshot.
     */
    KOKKOS_INLINE_FUNCTION
    double local_stable_time_step(debug_terms_type& terms) const {
        coordinate_array_type position = terms.position;
        if (!prepare_physical_position(position)) {
            return large_time_step;
        }

        typename grid_type::index_array_type cell_indices{};
        for (int dim = 0; dim < space_dim; ++dim) {
            cell_indices[dim] = this->grid.cell_index(dim, position[dim],
                                                      GridDomain::PhysicalDomain);
        }
        const auto metric_centerings = cell_centerings();
        double diffusion_dt_min = large_time_step;
        double advection_dt_min = large_time_step;
        for (int dim = 0; dim < space_dim; ++dim) {
            const double physical_dx = absolute_value(
                this->grid.physical_cell_width(dim, cell_indices, metric_centerings));
            if (!finite_positive(physical_dx)) {
                Kokkos::abort("ParkerSolver: local physical cell width must be positive.");
            }

            double diffusion_dt = large_time_step;
            if (terms.kappa_effective[dim] > 0.0) {
                const double limited_dx =
                    diffusion_displacement_fraction * physical_dx;
                diffusion_dt = limited_dx * limited_dx /
                               (2.0 * terms.kappa_effective[dim]);
            }

            const double advection_component = absolute_value(terms.total_advection[dim]);
            double advection_dt = large_time_step;
            if (advection_component > 0.0) {
                advection_dt =
                    advection_displacement_fraction * physical_dx /
                    advection_component;
            }
            diffusion_dt_min = min_time_step(diffusion_dt_min, diffusion_dt);
            advection_dt_min = min_time_step(advection_dt_min, advection_dt);
        }
        double momentum_dt = large_time_step;
        const double divergence_magnitude = absolute_value(terms.solar_wind_divergence);
        if (divergence_magnitude > 0.0) {
            momentum_dt = 3.0 * momentum_relative_change_fraction /
                          divergence_magnitude;
        }

        terms.diffusion_time_step = diffusion_dt_min;
        terms.advection_time_step = advection_dt_min;
        terms.momentum_time_step = momentum_dt;
        return min_time_step(min_time_step(diffusion_dt_min, advection_dt_min),
                             momentum_dt);
    }

    /**
     * Build a common interpolator boundary policy for particle sampling.
     */
    static constexpr FieldInterpolator::InterpolationBoundaryPolicy interpolation_policy() {
        return FieldInterpolator::InterpolationBoundaryPolicy::WrapPeriodic;
    }

    /**
     * Build a common interpolator coordinate domain for particle sampling.
     */
    static constexpr GridDomain interpolation_domain() {
        return GridDomain::GhostedDomain;
    }

    /**
     * Compute the adaptive time step for one diffusion-divergence branch.
     */
    template <bool UseConstantRatio>
    double compute_adaptive_time_step_impl() const {
        const auto magnetic_field_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                this->B_vec, interpolation_domain(), interpolation_policy());
        const auto solar_wind_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                this->V_body, interpolation_domain(), interpolation_policy());
        const auto drift_curl_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.magnetic_drift_curl, interpolation_domain(),
                interpolation_policy());
        const auto kappa_parallel_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.kappa_parallel_gamma_one, interpolation_domain(),
                interpolation_policy());
        const auto kappa_perpendicular_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.kappa_perpendicular_gamma_one, interpolation_domain(),
                interpolation_policy());
        const auto solar_wind_divergence_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.solar_wind_divergence, interpolation_domain(),
                interpolation_policy());

        if constexpr (UseConstantRatio) {
            const auto diffusion_primary_interpolator =
                FieldInterpolator::make_linear_position_interpolator(
                    coefficients.diffusion_tensor_divergence.constant_ratio_basis,
                    interpolation_domain(), interpolation_policy());
            return compute_adaptive_time_step_with_interpolators<UseConstantRatio>(
                magnetic_field_interpolator, solar_wind_interpolator,
                drift_curl_interpolator,
                diffusion_primary_interpolator, diffusion_primary_interpolator,
                kappa_parallel_interpolator, kappa_perpendicular_interpolator,
                solar_wind_divergence_interpolator);
        } else {
            const auto diffusion_primary_interpolator =
                FieldInterpolator::make_linear_position_interpolator(
                    coefficients.diffusion_tensor_divergence.parallel_basis,
                    interpolation_domain(), interpolation_policy());
            const auto diffusion_secondary_interpolator =
                FieldInterpolator::make_linear_position_interpolator(
                    coefficients.diffusion_tensor_divergence.perpendicular_basis,
                    interpolation_domain(), interpolation_policy());
            return compute_adaptive_time_step_with_interpolators<UseConstantRatio>(
                magnetic_field_interpolator, solar_wind_interpolator,
                drift_curl_interpolator,
                diffusion_primary_interpolator, diffusion_secondary_interpolator,
                kappa_parallel_interpolator, kappa_perpendicular_interpolator,
                solar_wind_divergence_interpolator);
        }
    }

    /**
     * Compute the adaptive time step from fully constructed interpolators.
     */
    template <
        bool UseConstantRatio,
        typename MagneticFieldInterpolator,
        typename SolarWindInterpolator,
        typename DriftCurlInterpolator,
        typename DiffusionPrimaryInterpolator,
        typename DiffusionSecondaryInterpolator,
        typename KappaParallelInterpolator,
        typename KappaPerpendicularInterpolator,
        typename SolarWindDivergenceInterpolator
    >
    double compute_adaptive_time_step_with_interpolators(
        const MagneticFieldInterpolator& magnetic_field_interpolator,
        const SolarWindInterpolator& solar_wind_interpolator,
        const DriftCurlInterpolator& drift_curl_interpolator,
        const DiffusionPrimaryInterpolator& diffusion_primary_interpolator,
        const DiffusionSecondaryInterpolator& diffusion_secondary_interpolator,
        const KappaParallelInterpolator& kappa_parallel_interpolator,
        const KappaPerpendicularInterpolator& kappa_perpendicular_interpolator,
        const SolarWindDivergenceInterpolator& solar_wind_divergence_interpolator) const {
        const ParkerSolver local_solver = *this;
        double minimum_dt = large_time_step;
        Kokkos::parallel_reduce(
            "ParkerSolver::compute_adaptive_time_step",
            Kokkos::RangePolicy<execution_space>(0, this->particles.particle_count()),
            KOKKOS_LAMBDA(const size_type particle_index, double& update) {
                if (!particle_status_is_alive(local_solver.particles.status(particle_index))) {
                    return;
                }
                const debug_terms_type terms =
                    local_solver.template evaluate_particle_terms<UseConstantRatio, true>(
                        particle_index, magnetic_field_interpolator,
                        solar_wind_interpolator,
                        drift_curl_interpolator, diffusion_primary_interpolator,
                        diffusion_secondary_interpolator, kappa_parallel_interpolator,
                        kappa_perpendicular_interpolator,
                        solar_wind_divergence_interpolator);
                update = min_time_step(update, terms.stable_time_step);
            },
            Kokkos::Min<double>(minimum_dt));
        Kokkos::fence("ParkerSolver::compute_adaptive_time_step");

        if (!std::isfinite(minimum_dt) || minimum_dt >= 0.5 * large_time_step) {
            return 0.0;
        }
        return this->courant_number() * minimum_dt;
    }

    /**
     * Advance particles for one diffusion-divergence branch.
     */
    template <bool UseConstantRatio>
    void advance_deterministic_impl(const double step_dt) {
        const auto magnetic_field_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                this->B_vec, interpolation_domain(), interpolation_policy());
        const auto solar_wind_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                this->V_body, interpolation_domain(), interpolation_policy());
        const auto drift_curl_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.magnetic_drift_curl, interpolation_domain(),
                interpolation_policy());
        const auto kappa_parallel_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.kappa_parallel_gamma_one, interpolation_domain(),
                interpolation_policy());
        const auto kappa_perpendicular_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.kappa_perpendicular_gamma_one, interpolation_domain(),
                interpolation_policy());
        const auto solar_wind_divergence_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.solar_wind_divergence, interpolation_domain(),
                interpolation_policy());

        if constexpr (UseConstantRatio) {
            const auto diffusion_primary_interpolator =
                FieldInterpolator::make_linear_position_interpolator(
                    coefficients.diffusion_tensor_divergence.constant_ratio_basis,
                    interpolation_domain(), interpolation_policy());
            advance_deterministic_with_interpolators<UseConstantRatio>(
                step_dt, magnetic_field_interpolator, solar_wind_interpolator,
                drift_curl_interpolator,
                diffusion_primary_interpolator, diffusion_primary_interpolator,
                kappa_parallel_interpolator, kappa_perpendicular_interpolator,
                solar_wind_divergence_interpolator);
        } else {
            const auto diffusion_primary_interpolator =
                FieldInterpolator::make_linear_position_interpolator(
                    coefficients.diffusion_tensor_divergence.parallel_basis,
                    interpolation_domain(), interpolation_policy());
            const auto diffusion_secondary_interpolator =
                FieldInterpolator::make_linear_position_interpolator(
                    coefficients.diffusion_tensor_divergence.perpendicular_basis,
                    interpolation_domain(), interpolation_policy());
            advance_deterministic_with_interpolators<UseConstantRatio>(
                step_dt, magnetic_field_interpolator, solar_wind_interpolator,
                drift_curl_interpolator,
                diffusion_primary_interpolator, diffusion_secondary_interpolator,
                kappa_parallel_interpolator, kappa_perpendicular_interpolator,
                solar_wind_divergence_interpolator);
        }
    }

    /**
     * Advance particles from fully constructed interpolators.
     */
    template <
        bool UseConstantRatio,
        typename MagneticFieldInterpolator,
        typename SolarWindInterpolator,
        typename DriftCurlInterpolator,
        typename DiffusionPrimaryInterpolator,
        typename DiffusionSecondaryInterpolator,
        typename KappaParallelInterpolator,
        typename KappaPerpendicularInterpolator,
        typename SolarWindDivergenceInterpolator
    >
    void advance_deterministic_with_interpolators(
        const double step_dt,
        const MagneticFieldInterpolator& magnetic_field_interpolator,
        const SolarWindInterpolator& solar_wind_interpolator,
        const DriftCurlInterpolator& drift_curl_interpolator,
        const DiffusionPrimaryInterpolator& diffusion_primary_interpolator,
        const DiffusionSecondaryInterpolator& diffusion_secondary_interpolator,
        const KappaParallelInterpolator& kappa_parallel_interpolator,
        const KappaPerpendicularInterpolator& kappa_perpendicular_interpolator,
        const SolarWindDivergenceInterpolator& solar_wind_divergence_interpolator) {
        const ParkerSolver local_solver = *this;
        auto positions = this->particles.position;
        auto momenta = this->particles.momentum;
        auto statuses = this->particles.status;
        Kokkos::parallel_for(
            "ParkerSolver::advance_deterministic",
            Kokkos::RangePolicy<execution_space>(0, this->particles.particle_count()),
            KOKKOS_LAMBDA(const size_type particle_index) {
                if (!particle_status_is_alive(statuses(particle_index))) {
                    return;
                }
                const debug_terms_type terms =
                    local_solver.template evaluate_particle_terms<UseConstantRatio, false>(
                        particle_index, magnetic_field_interpolator,
                        solar_wind_interpolator,
                        drift_curl_interpolator, diffusion_primary_interpolator,
                        diffusion_secondary_interpolator, kappa_parallel_interpolator,
                        kappa_perpendicular_interpolator,
                        solar_wind_divergence_interpolator);
                for (int dim = 0; dim < space_dim; ++dim) {
                    positions(particle_index, dim) += step_dt * terms.coordinate_rate[dim];
                }
                const double updated_momentum =
                    terms.momentum_magnitude + step_dt * terms.momentum_rate;
                momenta(particle_index) = updated_momentum > 0.0 ? updated_momentum : 0.0;
            });
        Kokkos::fence("ParkerSolver::advance_deterministic");
    }

    /**
     * Advance particles with deterministic terms and stochastic diffusion.
     */
    template <bool UseConstantRatio>
    void advance_stochastic_impl(const double step_dt,
                                 const random_manager_type& random_manager) {
        const auto magnetic_field_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                this->B_vec, interpolation_domain(), interpolation_policy());
        const auto solar_wind_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                this->V_body, interpolation_domain(), interpolation_policy());
        const auto drift_curl_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.magnetic_drift_curl, interpolation_domain(),
                interpolation_policy());
        const auto kappa_parallel_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.kappa_parallel_gamma_one, interpolation_domain(),
                interpolation_policy());
        const auto kappa_perpendicular_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.kappa_perpendicular_gamma_one, interpolation_domain(),
                interpolation_policy());
        const auto solar_wind_divergence_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.solar_wind_divergence, interpolation_domain(),
                interpolation_policy());

        if constexpr (UseConstantRatio) {
            const auto diffusion_primary_interpolator =
                FieldInterpolator::make_linear_position_interpolator(
                    coefficients.diffusion_tensor_divergence.constant_ratio_basis,
                    interpolation_domain(), interpolation_policy());
            advance_stochastic_with_interpolators<UseConstantRatio>(
                step_dt, random_manager, magnetic_field_interpolator,
                solar_wind_interpolator, drift_curl_interpolator,
                diffusion_primary_interpolator, diffusion_primary_interpolator,
                kappa_parallel_interpolator, kappa_perpendicular_interpolator,
                solar_wind_divergence_interpolator);
        } else {
            const auto diffusion_primary_interpolator =
                FieldInterpolator::make_linear_position_interpolator(
                    coefficients.diffusion_tensor_divergence.parallel_basis,
                    interpolation_domain(), interpolation_policy());
            const auto diffusion_secondary_interpolator =
                FieldInterpolator::make_linear_position_interpolator(
                    coefficients.diffusion_tensor_divergence.perpendicular_basis,
                    interpolation_domain(), interpolation_policy());
            advance_stochastic_with_interpolators<UseConstantRatio>(
                step_dt, random_manager, magnetic_field_interpolator,
                solar_wind_interpolator, drift_curl_interpolator,
                diffusion_primary_interpolator, diffusion_secondary_interpolator,
                kappa_parallel_interpolator, kappa_perpendicular_interpolator,
                solar_wind_divergence_interpolator);
        }
    }

    /**
     * Advance stochastic Parker particles from fully constructed interpolators.
     */
    template <
        bool UseConstantRatio,
        typename MagneticFieldInterpolator,
        typename SolarWindInterpolator,
        typename DriftCurlInterpolator,
        typename DiffusionPrimaryInterpolator,
        typename DiffusionSecondaryInterpolator,
        typename KappaParallelInterpolator,
        typename KappaPerpendicularInterpolator,
        typename SolarWindDivergenceInterpolator
    >
    void advance_stochastic_with_interpolators(
        const double step_dt,
        const random_manager_type& random_manager,
        const MagneticFieldInterpolator& magnetic_field_interpolator,
        const SolarWindInterpolator& solar_wind_interpolator,
        const DriftCurlInterpolator& drift_curl_interpolator,
        const DiffusionPrimaryInterpolator& diffusion_primary_interpolator,
        const DiffusionSecondaryInterpolator& diffusion_secondary_interpolator,
        const KappaParallelInterpolator& kappa_parallel_interpolator,
        const KappaPerpendicularInterpolator& kappa_perpendicular_interpolator,
        const SolarWindDivergenceInterpolator& solar_wind_divergence_interpolator) {
        const ParkerSolver local_solver = *this;
        const random_manager_type local_random_manager = random_manager;
        auto positions = this->particles.position;
        auto momenta = this->particles.momentum;
        auto statuses = this->particles.status;
        Kokkos::parallel_for(
            "ParkerSolver::advance_stochastic",
            Kokkos::RangePolicy<execution_space>(0, this->particles.particle_count()),
            KOKKOS_LAMBDA(const size_type particle_index) {
                if (!particle_status_is_alive(statuses(particle_index))) {
                    return;
                }
                const debug_terms_type terms =
                    local_solver.template evaluate_particle_terms<UseConstantRatio, false>(
                        particle_index, magnetic_field_interpolator,
                        solar_wind_interpolator,
                        drift_curl_interpolator, diffusion_primary_interpolator,
                        diffusion_secondary_interpolator, kappa_parallel_interpolator,
                        kappa_perpendicular_interpolator,
                        solar_wind_divergence_interpolator);
                auto generator = local_random_manager.get_state();
                const vector_array_type physical_random_increment =
                    sample_magnetic_diffusion_increment(terms, step_dt, generator);
                local_random_manager.free_state(generator);

                for (int dim = 0; dim < space_dim; ++dim) {
                    const double deterministic_increment =
                        step_dt * terms.coordinate_rate[dim];
                    const double stochastic_increment =
                        local_solver.coordinate_rate_component(
                            terms.position, dim, physical_random_increment[dim]);
                    positions(particle_index, dim) +=
                        deterministic_increment + stochastic_increment;
                }
                const double updated_momentum =
                    terms.momentum_magnitude + step_dt * terms.momentum_rate;
                momenta(particle_index) = updated_momentum > 0.0 ? updated_momentum : 0.0;
            });
        Kokkos::fence("ParkerSolver::advance_stochastic");
    }

    /**
     * Evaluate debug terms for one particle on the host.
     */
    debug_terms_type debug_terms_for_particle(const size_type particle_index) const {
        validate_for_deterministic_step();
        if (particle_index >= this->particles.particle_count()) {
            throw std::runtime_error("ParkerSolver: debug particle index is out of range.");
        }
        if (coefficients.perpendicular_model ==
            LegencyModel::PerpendicularDiffusionModelKind::ConstantParallelRatio) {
            return debug_terms_for_particle_impl<true>(particle_index);
        }
        return debug_terms_for_particle_impl<false>(particle_index);
    }

    /**
     * Evaluate debug terms for one particle and one diffusion-divergence branch.
     */
    template <bool UseConstantRatio>
    debug_terms_type debug_terms_for_particle_impl(const size_type particle_index) const {
        const auto magnetic_field_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                this->B_vec, interpolation_domain(), interpolation_policy());
        const auto solar_wind_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                this->V_body, interpolation_domain(), interpolation_policy());
        const auto drift_curl_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.magnetic_drift_curl, interpolation_domain(),
                interpolation_policy());
        const auto kappa_parallel_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.kappa_parallel_gamma_one, interpolation_domain(),
                interpolation_policy());
        const auto kappa_perpendicular_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.kappa_perpendicular_gamma_one, interpolation_domain(),
                interpolation_policy());
        const auto solar_wind_divergence_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.solar_wind_divergence, interpolation_domain(),
                interpolation_policy());

        if constexpr (UseConstantRatio) {
            const auto diffusion_primary_interpolator =
                FieldInterpolator::make_linear_position_interpolator(
                    coefficients.diffusion_tensor_divergence.constant_ratio_basis,
                    interpolation_domain(), interpolation_policy());
            return debug_terms_for_particle_with_interpolators<UseConstantRatio>(
                particle_index, magnetic_field_interpolator, solar_wind_interpolator,
                drift_curl_interpolator,
                diffusion_primary_interpolator, diffusion_primary_interpolator,
                kappa_parallel_interpolator, kappa_perpendicular_interpolator,
                solar_wind_divergence_interpolator);
        } else {
            const auto diffusion_primary_interpolator =
                FieldInterpolator::make_linear_position_interpolator(
                    coefficients.diffusion_tensor_divergence.parallel_basis,
                    interpolation_domain(), interpolation_policy());
            const auto diffusion_secondary_interpolator =
                FieldInterpolator::make_linear_position_interpolator(
                    coefficients.diffusion_tensor_divergence.perpendicular_basis,
                    interpolation_domain(), interpolation_policy());
            return debug_terms_for_particle_with_interpolators<UseConstantRatio>(
                particle_index, magnetic_field_interpolator, solar_wind_interpolator,
                drift_curl_interpolator,
                diffusion_primary_interpolator, diffusion_secondary_interpolator,
                kappa_parallel_interpolator, kappa_perpendicular_interpolator,
                solar_wind_divergence_interpolator);
        }
    }

    /**
     * Evaluate debug terms for one particle from fully constructed interpolators.
     */
    template <
        bool UseConstantRatio,
        typename MagneticFieldInterpolator,
        typename SolarWindInterpolator,
        typename DriftCurlInterpolator,
        typename DiffusionPrimaryInterpolator,
        typename DiffusionSecondaryInterpolator,
        typename KappaParallelInterpolator,
        typename KappaPerpendicularInterpolator,
        typename SolarWindDivergenceInterpolator
    >
    debug_terms_type debug_terms_for_particle_with_interpolators(
        const size_type particle_index,
        const MagneticFieldInterpolator& magnetic_field_interpolator,
        const SolarWindInterpolator& solar_wind_interpolator,
        const DriftCurlInterpolator& drift_curl_interpolator,
        const DiffusionPrimaryInterpolator& diffusion_primary_interpolator,
        const DiffusionSecondaryInterpolator& diffusion_secondary_interpolator,
        const KappaParallelInterpolator& kappa_parallel_interpolator,
        const KappaPerpendicularInterpolator& kappa_perpendicular_interpolator,
        const SolarWindDivergenceInterpolator& solar_wind_divergence_interpolator) const {
        debug_terms_view_type terms_view("parker_debug_terms", 1);
        const ParkerSolver local_solver = *this;
        Kokkos::parallel_for(
            "ParkerSolver::debug_terms_for_particle",
            Kokkos::RangePolicy<execution_space>(0, 1),
            KOKKOS_LAMBDA(const int) {
                terms_view(0) =
                    local_solver.template evaluate_particle_terms<UseConstantRatio, true>(
                        particle_index, magnetic_field_interpolator,
                        solar_wind_interpolator,
                        drift_curl_interpolator, diffusion_primary_interpolator,
                        diffusion_secondary_interpolator, kappa_parallel_interpolator,
                        kappa_perpendicular_interpolator,
                        solar_wind_divergence_interpolator);
            });
        Kokkos::fence("ParkerSolver::debug_terms_for_particle");
        const auto terms_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, terms_view);
        return terms_host(0);
    }

    friend struct ParkerSolverDebugger<GridType, VecDim, DeviceType>;
};

/**
 * Metadata-only Parker solver alias kept for early scaffolding and smoke tests.
 */
template <int SpaceDim,
          int VecDim,
          typename DeviceType = Device,
          typename LayoutType = typename DeviceTraits<DeviceType>::array_layout>
using MetadataParkerSolver =
    ParkerSolver<Grid<SpaceDim, DeviceType, LayoutType, VecDim>, VecDim, DeviceType>;
