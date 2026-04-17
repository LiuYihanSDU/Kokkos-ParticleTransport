#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "FieldInterpolator.hpp"
#include "FocusCoefficient.hpp"
#include "RandomManager.hpp"
#include "SolverBase.hpp"

/**
 * Host/device snapshot of focused transport terms for one particle.
 */
template <int SpaceDim, int VecDim>
struct FocusTransportParticleTerms {
    using coordinate_array_type = Kokkos::Array<double, SpaceDim>;
    using vector_array_type = Kokkos::Array<double, VecDim>;

    std::uint64_t particle_id{0};
    coordinate_array_type position{};
    double momentum_magnitude{0.0};
    double mu{0.0};
    double gamma{1.0};
    double speed{0.0};
    vector_array_type magnetic_field{};
    vector_array_type magnetic_direction{};
    double magnetic_field_magnitude{0.0};
    vector_array_type solar_wind_velocity{};
    vector_array_type streaming_velocity{};
    vector_array_type drift_velocity{};
    vector_array_type diffusion_advection{};
    vector_array_type total_advection{};
    coordinate_array_type coordinate_rate{};
    double solar_wind_divergence{0.0};
    double bb_grad_solar_wind{0.0};
    double b_dot_solar_wind_total_derivative{0.0};
    double magnetic_focusing{0.0};
    double momentum_rate{0.0};
    double focusing_mu_rate{0.0};
    double D_mumu{0.0};
    double dD_mumu_dmu{0.0};
    double mu_rate{0.0};
    double scattering_sigma2{0.0};
    double scattering_correlation_length{0.0};
    double kappa_parallel{0.0};
    double kappa_perpendicular{0.0};
    coordinate_array_type kappa_perpendicular_effective{};
    double diffusion_time_step{0.0};
    double advection_time_step{0.0};
    double momentum_time_step{0.0};
    double pitch_angle_advection_time_step{0.0};
    double pitch_angle_diffusion_time_step{0.0};
    double pitch_angle_balance_time_step{0.0};
    double stable_time_step{0.0};
};

template <
    typename GridType,
    int VecDim = GridType::vec_dim,
    typename DeviceType = typename GridType::device_type
>
struct FocusTransportSolverDebugger;

/**
 * Focused transport solver with Euler-Maruyama spatial diffusion and semi-implicit
 * Picard pitch-angle stepping.
 */
template <
    typename GridType,
    int VecDim = GridType::vec_dim,
    typename DeviceType = typename GridType::device_type
>
struct FocusTransportSolver : public SolverBase<GridType, VecDim, DeviceType> {
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
        FocusCoefficient::TransportCoefficientFieldSet<grid_type, layout_type>;
    using debug_terms_type = FocusTransportParticleTerms<GridType::space_dim, VecDim>;
    using debug_terms_view_type = Kokkos::View<debug_terms_type*, layout_type, memory_space>;

    static constexpr int space_dim = base_type::space_dim;
    static constexpr int vec_dim = base_type::vec_dim;
    static_assert(VecDim > 0, "FocusTransportSolver VecDim must be positive.");
    static_assert(VecDim <= 3, "FocusTransportSolver currently supports VecDim <= 3.");
    static_assert(space_dim <= VecDim,
                  "FocusTransportSolver requires VecDim to cover all coordinate directions.");

    coefficient_field_set_type coefficients;
    double maximum_pitch_angle_cosine{1.0};

    explicit FocusTransportSolver(grid_type input_grid,
                                  vector_field_type input_B_vec,
                                  vector_field_type input_V_body)
        : base_type(input_grid, input_B_vec, input_V_body) {}

    FocusTransportSolver(grid_type input_grid,
                         vector_field_type input_B_vec,
                         vector_field_type input_V_body,
                         coefficient_field_set_type input_coefficients)
        : base_type(input_grid, input_B_vec, input_V_body),
          coefficients(input_coefficients) {}

    FocusTransportSolver(particle_system_type input_particles,
                         grid_type input_grid,
                         vector_field_type input_B_vec,
                         vector_field_type input_V_body)
        : base_type(input_particles, input_grid, input_B_vec, input_V_body) {}

    FocusTransportSolver(particle_system_type input_particles,
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

    /**
     * Set the maximum absolute pitch-angle cosine used after each update.
     */
    void set_maximum_pitch_angle_cosine(const double input_maximum_mu) {
        if (!std::isfinite(input_maximum_mu) || input_maximum_mu <= 0.0 ||
            input_maximum_mu > 1.0) {
            throw std::runtime_error(
                "FocusTransportSolver: maximum |mu| must satisfy 0 < value <= 1.");
        }
        maximum_pitch_angle_cosine = input_maximum_mu;
    }

    KOKKOS_INLINE_FUNCTION
    void initializer() {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        (void)i;
    }

    /**
     * Compute the adaptive focused-transport time step in code units.
     */
    double compute_adaptive_time_step() const {
        validate_for_step();
        return compute_adaptive_time_step_impl();
    }

    /**
     * Advance active particles without stochastic increments.
     */
    double advance_deterministic() {
        validate_for_step();
        this->particles.apply_grid_boundary_conditions(this->grid,
                                                       GridDomain::PhysicalDomain);
        const double step_dt = compute_adaptive_time_step();
        if (!std::isfinite(step_dt) || step_dt <= 0.0) {
            throw std::runtime_error(
                "FocusTransportSolver: adaptive deterministic time step is not positive and finite.");
        }

        this->dT = typename base_type::dimensionless_value_type(step_dt);
        this->particles.capture_previous_positions();
        advance_deterministic_impl(step_dt);
        this->particles.apply_grid_boundary_conditions(this->grid,
                                                       GridDomain::PhysicalDomain);
        return step_dt;
    }

    /**
     * Advance active particles by one focused-transport SDE step.
     */
    double advance_stochastic(const random_manager_type& random_manager) {
        validate_for_step();
        this->particles.apply_grid_boundary_conditions(this->grid,
                                                       GridDomain::PhysicalDomain);
        const double step_dt = compute_adaptive_time_step();
        if (!std::isfinite(step_dt) || step_dt <= 0.0) {
            throw std::runtime_error(
                "FocusTransportSolver: adaptive stochastic time step is not positive and finite.");
        }

        this->dT = typename base_type::dimensionless_value_type(step_dt);
        this->particles.capture_previous_positions();
        advance_stochastic_impl(step_dt, random_manager);
        this->particles.apply_grid_boundary_conditions(this->grid,
                                                       GridDomain::PhysicalDomain);
        return step_dt;
    }

    /**
     * Compatibility name for one full focused-transport SDE step.
     */
    double advance_sde(const random_manager_type& random_manager) {
        return advance_stochastic(random_manager);
    }

private:
    static constexpr double large_time_step = 1.0e300;
    static constexpr double diffusion_displacement_fraction = 0.5;
    static constexpr double advection_displacement_fraction = 1.0;
    static constexpr double momentum_relative_change_fraction = 1.0;
    static constexpr double pitch_angle_change_fraction = 0.1;
    static constexpr double pitch_angle_diffusion_fraction = 0.1;
    static constexpr double pitch_angle_courant_relaxation = 10.0;
    static constexpr int pitch_angle_picard_iterations = 3;

    struct PitchAngleEvaluation {
        double focusing_mu_rate{0.0};
        double D_mumu{0.0};
        double dD_mumu_dmu{0.0};
        double mu_rate{0.0};
    };

    /**
     * Validate field and scalar state needed by focused stepping.
     */
    void validate_for_step() const {
        static_assert(requires(const grid_type& input_grid,
                               typename grid_type::index_array_type indices,
                               typename vector_field_type::centering_array_type centerings) {
                          input_grid.cell_index(0, 0.0, GridDomain::PhysicalDomain);
                          input_grid.physical_cell_width(0, indices, centerings);
                          input_grid.contains_coordinate(0, 0.0,
                                                         GridDomain::PhysicalDomain);
                          input_grid.wrap_coordinate(0, 0.0);
                      },
                      "FocusTransportSolver stepping requires a concrete coordinate grid.");

        if (!coefficients.configured()) {
            throw std::runtime_error(
                "FocusTransportSolver: coefficient fields are not configured.");
        }
        if (!std::isfinite(this->courant_number()) || this->courant_number() <= 0.0) {
            throw std::runtime_error(
                "FocusTransportSolver: Courant scale must be positive and finite.");
        }
        if (!std::isfinite(maximum_pitch_angle_cosine) ||
            maximum_pitch_angle_cosine <= 0.0 ||
            maximum_pitch_angle_cosine > 1.0) {
            throw std::runtime_error(
                "FocusTransportSolver: maximum pitch-angle cosine must satisfy 0 < value <= 1.");
        }
        if (!std::isfinite(this->particles.properties.charge) ||
            this->particles.properties.charge == 0.0) {
            throw std::runtime_error(
                "FocusTransportSolver: focused drift requires a finite nonzero particle charge.");
        }
        coefficients.pitch_angle_scattering.validate();

        validate_field_point_count(this->B_vec, "magnetic field");
        validate_field_point_count(this->V_body, "solar wind velocity");
        validate_field_point_count(coefficients.magnetic_direction, "magnetic direction");
        validate_field_point_count(coefficients.magnetic_field_magnitude,
                                   "magnetic field magnitude");
        validate_field_point_count(coefficients.grad_log_magnetic_field,
                                   "grad log magnetic field");
        validate_field_point_count(coefficients.gradient_drift_basis_over_magnetic_field,
                                   "gradient drift basis");
        validate_field_point_count(coefficients.curvature_drift_basis_over_magnetic_field,
                                   "curvature drift basis");
        validate_field_point_count(coefficients.perpendicular_diffusion_advection_gamma_one,
                                   "perpendicular diffusion advection");
        validate_field_point_count(coefficients.kappa_parallel_gamma_one,
                                   "kappa_parallel_gamma_one");
        validate_field_point_count(coefficients.kappa_perpendicular_gamma_one,
                                   "kappa_perpendicular_gamma_one");
        validate_field_point_count(coefficients.solar_wind_divergence,
                                   "solar wind divergence");
        validate_field_point_count(coefficients.magnetic_focusing,
                                   "magnetic focusing");
        validate_field_point_count(coefficients.bb_grad_solar_wind,
                                   "bb grad solar wind");
        validate_field_point_count(coefficients.solar_wind_time_derivative,
                                   "solar wind time derivative");
        validate_field_point_count(coefficients.b_dot_solar_wind_total_derivative,
                                   "b dot solar wind total derivative");
        validate_field_point_count(coefficients.scattering_sigma2,
                                   "scattering sigma2");
        validate_field_point_count(coefficients.scattering_correlation_length,
                                   "scattering correlation length");
    }

    /**
     * Validate one same-grid field against the solar-wind field storage size.
     */
    template <typename FieldType>
    void validate_field_point_count(const FieldType& field,
                                    const char* field_name) const {
        if (field.point_count() == 0) {
            throw std::runtime_error(std::string("FocusTransportSolver: missing ") +
                                     field_name + " field.");
        }
        if (field.point_count() != this->V_body.point_count()) {
            throw std::runtime_error(std::string("FocusTransportSolver: ") +
                                     field_name +
                                     " field point count does not match V_sw.");
        }
        for (int dim = 0; dim < space_dim; ++dim) {
            if (field.centering(dim) != this->V_body.centering(dim) ||
                field.lower_index(dim, GridDomain::GhostedDomain) !=
                    this->V_body.lower_index(dim, GridDomain::GhostedDomain) ||
                field.upper_index(dim, GridDomain::GhostedDomain) !=
                    this->V_body.upper_index(dim, GridDomain::GhostedDomain)) {
                throw std::runtime_error(std::string("FocusTransportSolver: ") +
                                         field_name +
                                         " storage domain does not match V_sw.");
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    static double absolute_value(const double value) {
        return value >= 0.0 ? value : -value;
    }

    KOKKOS_INLINE_FUNCTION
    static bool finite_positive(const double value) {
        return Kokkos::isfinite(value) && value > 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    static bool finite_nonnegative(const double value) {
        return Kokkos::isfinite(value) && value >= 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    static double min_time_step(const double left, const double right) {
        return left < right ? left : right;
    }

    KOKKOS_INLINE_FUNCTION
    double reflected_pitch_angle_cosine(const double input_mu) const {
        if (!Kokkos::isfinite(input_mu)) {
            Kokkos::abort("FocusTransportSolver: pitch-angle cosine is not finite.");
        }
        double reflected_mu = input_mu;
        const double upper = maximum_pitch_angle_cosine;
        const double lower = -maximum_pitch_angle_cosine;
        while (reflected_mu > upper || reflected_mu < lower) {
            if (reflected_mu > upper) {
                reflected_mu = 2.0 * upper - reflected_mu;
            }
            if (reflected_mu < lower) {
                reflected_mu = 2.0 * lower - reflected_mu;
            }
        }
        return reflected_mu;
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
     * Return whether a coordinate tuple lies in the physical domain after periodic wrapping.
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
     * Convert a physical vector component or displacement to coordinate space.
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
            Kokkos::abort(
                "FocusTransportSolver: invalid inverse Lame coefficient.");
        }
        return physical_component * h_inv;
    }

    /**
     * Sample a physical perpendicular stochastic increment with covariance
     * 2 kappa_perp (I - bb) dt.
     */
    template <typename GeneratorType>
    KOKKOS_INLINE_FUNCTION
    static vector_array_type sample_perpendicular_diffusion_increment(
        const debug_terms_type& terms,
        const double step_dt,
        GeneratorType& generator) {
        vector_array_type gaussian_vector{};
        double parallel_component = 0.0;
        for (int component = 0; component < VecDim; ++component) {
            gaussian_vector[component] =
                stochastic_sampler_type::normal(generator);
            parallel_component += gaussian_vector[component] *
                                  terms.magnetic_direction[component];
        }

        const double scale =
            terms.kappa_perpendicular > 0.0
                ? Kokkos::sqrt(2.0 * terms.kappa_perpendicular * step_dt)
                : 0.0;
        vector_array_type increment{};
        for (int component = 0; component < VecDim; ++component) {
            increment[component] =
                scale * (gaussian_vector[component] -
                         parallel_component * terms.magnetic_direction[component]);
        }
        return increment;
    }

    /**
     * Evaluate pitch-angle drift and diffusion terms at a trial mu.
     */
    KOKKOS_INLINE_FUNCTION
    PitchAngleEvaluation evaluate_pitch_angle_at_mu(
        const debug_terms_type& terms,
        const double trial_mu) const {
        const double mu = reflected_pitch_angle_cosine(trial_mu);
        const double mu2 = mu * mu;
        const double one_minus_mu2 = 1.0 - mu2;
        if (terms.speed <= 0.0) {
            Kokkos::abort(
                "FocusTransportSolver: pitch-angle stepping requires positive particle speed.");
        }

        PitchAngleEvaluation result;
        result.focusing_mu_rate =
            0.5 * one_minus_mu2 *
            (-terms.speed * terms.magnetic_focusing +
             mu * terms.solar_wind_divergence -
             3.0 * mu * terms.bb_grad_solar_wind -
             2.0 * terms.b_dot_solar_wind_total_derivative /
                 terms.speed);
        const auto scattering =
            LegencyModel::evaluate_pitch_angle_scattering(
                this->particles.properties, terms.magnetic_field_magnitude,
                terms.speed, terms.gamma, mu, terms.scattering_sigma2,
                terms.scattering_correlation_length,
                coefficients.pitch_angle_scattering);
        result.D_mumu = scattering.D_mumu;
        result.dD_mumu_dmu = scattering.dD_mumu_dmu;
        result.mu_rate = result.focusing_mu_rate + result.dD_mumu_dmu;
        return result;
    }

    /**
     * Return the Picard-updated pitch-angle cosine for one stochastic increment.
     */
    KOKKOS_INLINE_FUNCTION
    double picard_pitch_angle_update(const debug_terms_type& terms,
                                     const double step_dt,
                                     const double normal_increment) const {
        const double initial_mu = reflected_pitch_angle_cosine(terms.mu);
        double trial_mu = initial_mu;
        for (int iteration = 0; iteration < pitch_angle_picard_iterations;
             ++iteration) {
            const PitchAngleEvaluation pitch =
                evaluate_pitch_angle_at_mu(terms, trial_mu);
            const double stochastic_scale =
                pitch.D_mumu > 0.0
                    ? Kokkos::sqrt(2.0 * pitch.D_mumu * step_dt)
                    : 0.0;
            trial_mu =
                initial_mu + step_dt * pitch.mu_rate +
                stochastic_scale * normal_increment;
            trial_mu = reflected_pitch_angle_cosine(trial_mu);
        }
        return trial_mu;
    }

    /**
     * Interpolate fields and assemble all focused transport terms for one particle.
     */
    template <
        typename MagneticFieldInterpolator,
        typename MagneticDirectionInterpolator,
        typename MagneticMagnitudeInterpolator,
        typename SolarWindInterpolator,
        typename GradientDriftBasisInterpolator,
        typename CurvatureDriftBasisInterpolator,
        typename PerpendicularDiffusionAdvectionInterpolator,
        typename KappaParallelInterpolator,
        typename KappaPerpendicularInterpolator,
        typename SolarWindDivergenceInterpolator,
        typename MagneticFocusingInterpolator,
        typename BBGradSolarWindInterpolator,
        typename ConvectiveDerivativeInterpolator,
        typename ScatteringSigmaInterpolator,
        typename ScatteringCorrelationLengthInterpolator
    >
    KOKKOS_INLINE_FUNCTION
    debug_terms_type evaluate_terms(
        coordinate_array_type position,
        const double input_momentum,
        const double input_mu,
        const MagneticFieldInterpolator& magnetic_field_interpolator,
        const MagneticDirectionInterpolator& magnetic_direction_interpolator,
        const MagneticMagnitudeInterpolator& magnetic_magnitude_interpolator,
        const SolarWindInterpolator& solar_wind_interpolator,
        const GradientDriftBasisInterpolator& gradient_drift_basis_interpolator,
        const CurvatureDriftBasisInterpolator& curvature_drift_basis_interpolator,
        const PerpendicularDiffusionAdvectionInterpolator&
            perpendicular_diffusion_advection_interpolator,
        const KappaParallelInterpolator& kappa_parallel_interpolator,
        const KappaPerpendicularInterpolator& kappa_perpendicular_interpolator,
        const SolarWindDivergenceInterpolator& solar_wind_divergence_interpolator,
        const MagneticFocusingInterpolator& magnetic_focusing_interpolator,
        const BBGradSolarWindInterpolator& bb_grad_solar_wind_interpolator,
        const ConvectiveDerivativeInterpolator& convective_derivative_interpolator,
        const ScatteringSigmaInterpolator& scattering_sigma_interpolator,
        const ScatteringCorrelationLengthInterpolator&
            scattering_correlation_length_interpolator) const {
        debug_terms_type terms;
        terms.position = position;
        terms.momentum_magnitude = absolute_value(input_momentum);
        terms.mu = reflected_pitch_angle_cosine(input_mu);
        terms.gamma = this->particles.properties.gamma_from_momentum_magnitude(
            terms.momentum_magnitude);
        terms.speed = this->particles.properties.speed_from_momentum_magnitude(
            terms.momentum_magnitude);

        const vector_array_type magnetic_field =
            magnetic_field_interpolator.sample(position);
        const vector_array_type magnetic_direction =
            magnetic_direction_interpolator.sample(position);
        const vector_array_type solar_wind =
            solar_wind_interpolator.sample(position);
        const vector_array_type gradient_drift_basis =
            gradient_drift_basis_interpolator.sample(position);
        const vector_array_type curvature_drift_basis =
            curvature_drift_basis_interpolator.sample(position);
        const vector_array_type perpendicular_diffusion_advection_gamma_one =
            perpendicular_diffusion_advection_interpolator.sample(position);

        terms.magnetic_field_magnitude =
            magnetic_magnitude_interpolator.sample_component(position, 0);
        if (!finite_positive(terms.magnetic_field_magnitude)) {
            Kokkos::abort(
                "FocusTransportSolver: focused transport requires |B| > 0.");
        }
        double sampled_direction_magnitude_squared = 0.0;
        for (int component = 0; component < VecDim; ++component) {
            terms.magnetic_field[component] = magnetic_field[component];
            sampled_direction_magnitude_squared +=
                magnetic_direction[component] * magnetic_direction[component];
            terms.solar_wind_velocity[component] = solar_wind[component];
        }
        const double sampled_direction_magnitude =
            Kokkos::sqrt(sampled_direction_magnitude_squared);
        if (!finite_positive(sampled_direction_magnitude)) {
            Kokkos::abort(
                "FocusTransportSolver: interpolated magnetic direction is invalid.");
        }
        const double inverse_sampled_direction_magnitude =
            1.0 / sampled_direction_magnitude;
        for (int component = 0; component < VecDim; ++component) {
            terms.magnetic_direction[component] =
                magnetic_direction[component] *
                inverse_sampled_direction_magnitude;
        }

        const double kappa_parallel_gamma_one =
            kappa_parallel_interpolator.sample_component(position, 0);
        const double kappa_perpendicular_gamma_one =
            kappa_perpendicular_interpolator.sample_component(position, 0);
        const double perpendicular_gamma_factor =
            LegencyModel::perpendicular_gamma_factor(
                terms.gamma, coefficients.perpendicular_model);
        terms.kappa_parallel =
            kappa_parallel_gamma_one * Kokkos::pow(terms.gamma, 1.0 / 3.0);
        terms.kappa_perpendicular =
            kappa_perpendicular_gamma_one * perpendicular_gamma_factor;
        if (!finite_nonnegative(terms.kappa_parallel) ||
            !finite_nonnegative(terms.kappa_perpendicular)) {
            Kokkos::abort(
                "FocusTransportSolver: diffusion coefficients must be finite and non-negative.");
        }

        const double charge = this->particles.properties.charge;
        if (charge == 0.0) {
            Kokkos::abort(
                "FocusTransportSolver: focused drift requires nonzero charge.");
        }
        const double mu2 = terms.mu * terms.mu;
        const double gradient_drift_factor = 0.5 * (1.0 - mu2);
        const double curvature_drift_factor = mu2;
        const double drift_prefactor =
            terms.momentum_magnitude * terms.speed *
            this->particles.properties.speed_of_light / charge;

        for (int component = 0; component < VecDim; ++component) {
            terms.streaming_velocity[component] =
                terms.speed * terms.mu * terms.magnetic_direction[component];
            terms.drift_velocity[component] =
                drift_prefactor *
                (gradient_drift_factor * gradient_drift_basis[component] +
                 curvature_drift_factor * curvature_drift_basis[component]);
            terms.diffusion_advection[component] =
                perpendicular_gamma_factor *
                perpendicular_diffusion_advection_gamma_one[component];
            terms.total_advection[component] =
                terms.streaming_velocity[component] +
                terms.solar_wind_velocity[component] +
                terms.drift_velocity[component] +
                terms.diffusion_advection[component];
        }
        for (int dim = 0; dim < space_dim; ++dim) {
            terms.coordinate_rate[dim] = coordinate_rate_component(
                position, dim, terms.total_advection[dim]);
            const double b_i = terms.magnetic_direction[dim];
            const double projection = 1.0 - b_i * b_i;
            terms.kappa_perpendicular_effective[dim] =
                projection > 0.0 ? terms.kappa_perpendicular * projection : 0.0;
        }

        terms.solar_wind_divergence =
            solar_wind_divergence_interpolator.sample_component(position, 0);
        terms.bb_grad_solar_wind =
            bb_grad_solar_wind_interpolator.sample_component(position, 0);
        terms.b_dot_solar_wind_total_derivative =
            convective_derivative_interpolator.sample_component(position, 0);
        terms.magnetic_focusing =
            magnetic_focusing_interpolator.sample_component(position, 0);
        terms.scattering_sigma2 =
            scattering_sigma_interpolator.sample_component(position, 0);
        terms.scattering_correlation_length =
            scattering_correlation_length_interpolator.sample_component(position, 0);

        const double one_minus_mu2 = 1.0 - mu2;
        double acceleration_term = 0.0;
        if (terms.speed > 0.0) {
            acceleration_term =
                terms.mu * terms.b_dot_solar_wind_total_derivative /
                terms.speed;
        }
        const double momentum_factor =
            0.5 * one_minus_mu2 *
                (terms.solar_wind_divergence - terms.bb_grad_solar_wind) +
            mu2 * terms.bb_grad_solar_wind + acceleration_term;
        terms.momentum_rate = -terms.momentum_magnitude * momentum_factor;

        const PitchAngleEvaluation pitch =
            evaluate_pitch_angle_at_mu(terms, terms.mu);
        terms.focusing_mu_rate = pitch.focusing_mu_rate;
        terms.D_mumu = pitch.D_mumu;
        terms.dD_mumu_dmu = pitch.dD_mumu_dmu;
        terms.mu_rate = pitch.mu_rate;
        return terms;
    }

    /**
     * Evaluate all terms for one particle index.
     */
    template <
        bool ComputeTimeStep,
        typename MagneticFieldInterpolator,
        typename MagneticDirectionInterpolator,
        typename MagneticMagnitudeInterpolator,
        typename SolarWindInterpolator,
        typename GradientDriftBasisInterpolator,
        typename CurvatureDriftBasisInterpolator,
        typename PerpendicularDiffusionAdvectionInterpolator,
        typename KappaParallelInterpolator,
        typename KappaPerpendicularInterpolator,
        typename SolarWindDivergenceInterpolator,
        typename MagneticFocusingInterpolator,
        typename BBGradSolarWindInterpolator,
        typename ConvectiveDerivativeInterpolator,
        typename ScatteringSigmaInterpolator,
        typename ScatteringCorrelationLengthInterpolator
    >
    KOKKOS_INLINE_FUNCTION
    debug_terms_type evaluate_particle_terms(
        const size_type particle_index,
        const MagneticFieldInterpolator& magnetic_field_interpolator,
        const MagneticDirectionInterpolator& magnetic_direction_interpolator,
        const MagneticMagnitudeInterpolator& magnetic_magnitude_interpolator,
        const SolarWindInterpolator& solar_wind_interpolator,
        const GradientDriftBasisInterpolator& gradient_drift_basis_interpolator,
        const CurvatureDriftBasisInterpolator& curvature_drift_basis_interpolator,
        const PerpendicularDiffusionAdvectionInterpolator&
            perpendicular_diffusion_advection_interpolator,
        const KappaParallelInterpolator& kappa_parallel_interpolator,
        const KappaPerpendicularInterpolator& kappa_perpendicular_interpolator,
        const SolarWindDivergenceInterpolator& solar_wind_divergence_interpolator,
        const MagneticFocusingInterpolator& magnetic_focusing_interpolator,
        const BBGradSolarWindInterpolator& bb_grad_solar_wind_interpolator,
        const ConvectiveDerivativeInterpolator& convective_derivative_interpolator,
        const ScatteringSigmaInterpolator& scattering_sigma_interpolator,
        const ScatteringCorrelationLengthInterpolator&
            scattering_correlation_length_interpolator) const {
        coordinate_array_type position{};
        for (int dim = 0; dim < space_dim; ++dim) {
            position[dim] = this->particles.position(particle_index, dim);
        }
        if (!prepare_physical_position(position)) {
            return debug_terms_type{};
        }

        debug_terms_type terms = evaluate_terms(
            position, this->particles.momentum(particle_index),
            this->particles.mu(particle_index), magnetic_field_interpolator,
            magnetic_direction_interpolator, magnetic_magnitude_interpolator,
            solar_wind_interpolator, gradient_drift_basis_interpolator,
            curvature_drift_basis_interpolator,
            perpendicular_diffusion_advection_interpolator,
            kappa_parallel_interpolator, kappa_perpendicular_interpolator,
            solar_wind_divergence_interpolator, magnetic_focusing_interpolator,
            bb_grad_solar_wind_interpolator, convective_derivative_interpolator,
            scattering_sigma_interpolator,
            scattering_correlation_length_interpolator);
        terms.particle_id = this->particles.particle_id(particle_index);
        if constexpr (ComputeTimeStep) {
            terms.stable_time_step = local_stable_time_step(terms);
        }
        return terms;
    }

    /**
     * Return the local adaptive time-step bound for one focused term snapshot.
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
                this->grid.physical_cell_width(dim, cell_indices,
                                               metric_centerings));
            if (!finite_positive(physical_dx)) {
                Kokkos::abort(
                    "FocusTransportSolver: local physical cell width must be positive.");
            }

            double diffusion_dt = large_time_step;
            if (terms.kappa_perpendicular_effective[dim] > 0.0) {
                const double limited_dx =
                    diffusion_displacement_fraction * physical_dx;
                diffusion_dt =
                    limited_dx * limited_dx /
                    (2.0 * terms.kappa_perpendicular_effective[dim]);
            }

            const double advection_component =
                absolute_value(terms.total_advection[dim]);
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
        const double momentum_rate_magnitude = absolute_value(terms.momentum_rate);
        if (terms.momentum_magnitude > 0.0 && momentum_rate_magnitude > 0.0) {
            momentum_dt = momentum_relative_change_fraction *
                          terms.momentum_magnitude / momentum_rate_magnitude;
        }

        double pitch_advection_dt = large_time_step;
        const double mu_rate_magnitude = absolute_value(terms.mu_rate);
        if (mu_rate_magnitude > 0.0) {
            pitch_advection_dt = pitch_angle_courant_relaxation *
                                 pitch_angle_change_fraction / mu_rate_magnitude;
        }

        double pitch_diffusion_dt = large_time_step;
        if (terms.D_mumu > 0.0) {
            pitch_diffusion_dt = pitch_angle_courant_relaxation *
                                 pitch_angle_diffusion_fraction *
                                 pitch_angle_diffusion_fraction /
                                 (2.0 * terms.D_mumu);
        }

        double pitch_balance_dt = large_time_step;
        if (terms.D_mumu > 0.0 && mu_rate_magnitude > 0.0) {
            pitch_balance_dt = pitch_angle_courant_relaxation *
                2.0 * terms.D_mumu / (mu_rate_magnitude * mu_rate_magnitude);
        }

        terms.diffusion_time_step = diffusion_dt_min;
        terms.advection_time_step = advection_dt_min;
        terms.momentum_time_step = momentum_dt;
        terms.pitch_angle_advection_time_step = pitch_advection_dt;
        terms.pitch_angle_diffusion_time_step = pitch_diffusion_dt;
        terms.pitch_angle_balance_time_step = pitch_balance_dt;
        return min_time_step(
            min_time_step(min_time_step(diffusion_dt_min, advection_dt_min),
                          momentum_dt),
            min_time_step(pitch_advection_dt,
                          min_time_step(pitch_diffusion_dt, pitch_balance_dt)));
    }

    static constexpr FieldInterpolator::InterpolationBoundaryPolicy interpolation_policy() {
        return FieldInterpolator::InterpolationBoundaryPolicy::WrapPeriodic;
    }

    static constexpr GridDomain interpolation_domain() {
        return GridDomain::GhostedDomain;
    }

    /**
     * Compute the adaptive time step.
     */
    double compute_adaptive_time_step_impl() const {
        const auto magnetic_field_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                this->B_vec, interpolation_domain(), interpolation_policy());
        const auto magnetic_direction_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.magnetic_direction, interpolation_domain(),
                interpolation_policy());
        const auto magnetic_magnitude_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.magnetic_field_magnitude, interpolation_domain(),
                interpolation_policy());
        const auto solar_wind_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                this->V_body, interpolation_domain(), interpolation_policy());
        const auto gradient_drift_basis_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.gradient_drift_basis_over_magnetic_field,
                interpolation_domain(), interpolation_policy());
        const auto curvature_drift_basis_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.curvature_drift_basis_over_magnetic_field,
                interpolation_domain(), interpolation_policy());
        const auto perpendicular_diffusion_advection_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.perpendicular_diffusion_advection_gamma_one,
                interpolation_domain(), interpolation_policy());
        const auto kappa_parallel_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.kappa_parallel_gamma_one, interpolation_domain(),
                interpolation_policy());
        const auto kappa_perpendicular_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.kappa_perpendicular_gamma_one,
                interpolation_domain(), interpolation_policy());
        const auto solar_wind_divergence_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.solar_wind_divergence, interpolation_domain(),
                interpolation_policy());
        const auto magnetic_focusing_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.magnetic_focusing, interpolation_domain(),
                interpolation_policy());
        const auto bb_grad_solar_wind_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.bb_grad_solar_wind, interpolation_domain(),
                interpolation_policy());
        const auto convective_derivative_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.b_dot_solar_wind_total_derivative,
                interpolation_domain(), interpolation_policy());
        const auto scattering_sigma_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.scattering_sigma2, interpolation_domain(),
                interpolation_policy());
        const auto scattering_correlation_length_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.scattering_correlation_length,
                interpolation_domain(), interpolation_policy());

        return compute_adaptive_time_step_with_interpolators(
            magnetic_field_interpolator, magnetic_direction_interpolator,
            magnetic_magnitude_interpolator, solar_wind_interpolator,
            gradient_drift_basis_interpolator, curvature_drift_basis_interpolator,
            perpendicular_diffusion_advection_interpolator,
            kappa_parallel_interpolator, kappa_perpendicular_interpolator,
            solar_wind_divergence_interpolator, magnetic_focusing_interpolator,
            bb_grad_solar_wind_interpolator, convective_derivative_interpolator,
            scattering_sigma_interpolator,
            scattering_correlation_length_interpolator);
    }

    /**
     * Compute the adaptive time step from fully constructed interpolators.
     */
    template <
        typename MagneticFieldInterpolator,
        typename MagneticDirectionInterpolator,
        typename MagneticMagnitudeInterpolator,
        typename SolarWindInterpolator,
        typename GradientDriftBasisInterpolator,
        typename CurvatureDriftBasisInterpolator,
        typename PerpendicularDiffusionAdvectionInterpolator,
        typename KappaParallelInterpolator,
        typename KappaPerpendicularInterpolator,
        typename SolarWindDivergenceInterpolator,
        typename MagneticFocusingInterpolator,
        typename BBGradSolarWindInterpolator,
        typename ConvectiveDerivativeInterpolator,
        typename ScatteringSigmaInterpolator,
        typename ScatteringCorrelationLengthInterpolator
    >
    double compute_adaptive_time_step_with_interpolators(
        const MagneticFieldInterpolator& magnetic_field_interpolator,
        const MagneticDirectionInterpolator& magnetic_direction_interpolator,
        const MagneticMagnitudeInterpolator& magnetic_magnitude_interpolator,
        const SolarWindInterpolator& solar_wind_interpolator,
        const GradientDriftBasisInterpolator& gradient_drift_basis_interpolator,
        const CurvatureDriftBasisInterpolator& curvature_drift_basis_interpolator,
        const PerpendicularDiffusionAdvectionInterpolator&
            perpendicular_diffusion_advection_interpolator,
        const KappaParallelInterpolator& kappa_parallel_interpolator,
        const KappaPerpendicularInterpolator& kappa_perpendicular_interpolator,
        const SolarWindDivergenceInterpolator& solar_wind_divergence_interpolator,
        const MagneticFocusingInterpolator& magnetic_focusing_interpolator,
        const BBGradSolarWindInterpolator& bb_grad_solar_wind_interpolator,
        const ConvectiveDerivativeInterpolator& convective_derivative_interpolator,
        const ScatteringSigmaInterpolator& scattering_sigma_interpolator,
        const ScatteringCorrelationLengthInterpolator&
            scattering_correlation_length_interpolator) const {
        const FocusTransportSolver local_solver = *this;
        double minimum_dt = large_time_step;
        Kokkos::parallel_reduce(
            "FocusTransportSolver::compute_adaptive_time_step",
            Kokkos::RangePolicy<execution_space>(0, this->particles.particle_count()),
            KOKKOS_LAMBDA(const size_type particle_index, double& update) {
                if (!particle_status_is_alive(
                        local_solver.particles.status(particle_index))) {
                    return;
                }
                const debug_terms_type terms =
                    local_solver.template evaluate_particle_terms<true>(
                        particle_index, magnetic_field_interpolator,
                        magnetic_direction_interpolator,
                        magnetic_magnitude_interpolator, solar_wind_interpolator,
                        gradient_drift_basis_interpolator,
                        curvature_drift_basis_interpolator,
                        perpendicular_diffusion_advection_interpolator,
                        kappa_parallel_interpolator,
                        kappa_perpendicular_interpolator,
                        solar_wind_divergence_interpolator,
                        magnetic_focusing_interpolator,
                        bb_grad_solar_wind_interpolator,
                        convective_derivative_interpolator,
                        scattering_sigma_interpolator,
                        scattering_correlation_length_interpolator);
                update = min_time_step(update, terms.stable_time_step);
            },
            Kokkos::Min<double>(minimum_dt));
        Kokkos::fence("FocusTransportSolver::compute_adaptive_time_step");

        if (!std::isfinite(minimum_dt) || minimum_dt >= 0.5 * large_time_step) {
            return 0.0;
        }
        return this->courant_number() * minimum_dt;
    }

    /**
     * Advance particles without stochastic increments.
     */
    void advance_deterministic_impl(const double step_dt) {
        advance_with_random_impl(step_dt, nullptr);
    }

    /**
     * Advance particles with stochastic increments.
     */
    void advance_stochastic_impl(const double step_dt,
                                 const random_manager_type& random_manager) {
        advance_with_random_impl(step_dt, &random_manager);
    }

    /**
     * Shared host-side interpolator construction for deterministic/stochastic updates.
     */
    void advance_with_random_impl(const double step_dt,
                                  const random_manager_type* random_manager) {
        const auto magnetic_field_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                this->B_vec, interpolation_domain(), interpolation_policy());
        const auto magnetic_direction_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.magnetic_direction, interpolation_domain(),
                interpolation_policy());
        const auto magnetic_magnitude_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.magnetic_field_magnitude, interpolation_domain(),
                interpolation_policy());
        const auto solar_wind_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                this->V_body, interpolation_domain(), interpolation_policy());
        const auto gradient_drift_basis_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.gradient_drift_basis_over_magnetic_field,
                interpolation_domain(), interpolation_policy());
        const auto curvature_drift_basis_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.curvature_drift_basis_over_magnetic_field,
                interpolation_domain(), interpolation_policy());
        const auto perpendicular_diffusion_advection_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.perpendicular_diffusion_advection_gamma_one,
                interpolation_domain(), interpolation_policy());
        const auto kappa_parallel_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.kappa_parallel_gamma_one, interpolation_domain(),
                interpolation_policy());
        const auto kappa_perpendicular_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.kappa_perpendicular_gamma_one,
                interpolation_domain(), interpolation_policy());
        const auto solar_wind_divergence_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.solar_wind_divergence, interpolation_domain(),
                interpolation_policy());
        const auto magnetic_focusing_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.magnetic_focusing, interpolation_domain(),
                interpolation_policy());
        const auto bb_grad_solar_wind_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.bb_grad_solar_wind, interpolation_domain(),
                interpolation_policy());
        const auto convective_derivative_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.b_dot_solar_wind_total_derivative,
                interpolation_domain(), interpolation_policy());
        const auto scattering_sigma_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.scattering_sigma2, interpolation_domain(),
                interpolation_policy());
        const auto scattering_correlation_length_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.scattering_correlation_length,
                interpolation_domain(), interpolation_policy());

        if (random_manager == nullptr) {
            advance_with_interpolators<false>(
                step_dt, random_manager_type{1}, magnetic_field_interpolator,
                magnetic_direction_interpolator, magnetic_magnitude_interpolator,
                solar_wind_interpolator, gradient_drift_basis_interpolator,
                curvature_drift_basis_interpolator,
                perpendicular_diffusion_advection_interpolator,
                kappa_parallel_interpolator, kappa_perpendicular_interpolator,
                solar_wind_divergence_interpolator, magnetic_focusing_interpolator,
                bb_grad_solar_wind_interpolator,
                convective_derivative_interpolator, scattering_sigma_interpolator,
                scattering_correlation_length_interpolator);
        } else {
            advance_with_interpolators<true>(
                step_dt, *random_manager, magnetic_field_interpolator,
                magnetic_direction_interpolator, magnetic_magnitude_interpolator,
                solar_wind_interpolator, gradient_drift_basis_interpolator,
                curvature_drift_basis_interpolator,
                perpendicular_diffusion_advection_interpolator,
                kappa_parallel_interpolator, kappa_perpendicular_interpolator,
                solar_wind_divergence_interpolator, magnetic_focusing_interpolator,
                bb_grad_solar_wind_interpolator,
                convective_derivative_interpolator, scattering_sigma_interpolator,
                scattering_correlation_length_interpolator);
        }
    }

    /**
     * Advance particles from fully constructed interpolators.
     */
    template <
        bool UseRandom,
        typename MagneticFieldInterpolator,
        typename MagneticDirectionInterpolator,
        typename MagneticMagnitudeInterpolator,
        typename SolarWindInterpolator,
        typename GradientDriftBasisInterpolator,
        typename CurvatureDriftBasisInterpolator,
        typename PerpendicularDiffusionAdvectionInterpolator,
        typename KappaParallelInterpolator,
        typename KappaPerpendicularInterpolator,
        typename SolarWindDivergenceInterpolator,
        typename MagneticFocusingInterpolator,
        typename BBGradSolarWindInterpolator,
        typename ConvectiveDerivativeInterpolator,
        typename ScatteringSigmaInterpolator,
        typename ScatteringCorrelationLengthInterpolator
    >
    void advance_with_interpolators(
        const double step_dt,
        const random_manager_type& random_manager,
        const MagneticFieldInterpolator& magnetic_field_interpolator,
        const MagneticDirectionInterpolator& magnetic_direction_interpolator,
        const MagneticMagnitudeInterpolator& magnetic_magnitude_interpolator,
        const SolarWindInterpolator& solar_wind_interpolator,
        const GradientDriftBasisInterpolator& gradient_drift_basis_interpolator,
        const CurvatureDriftBasisInterpolator& curvature_drift_basis_interpolator,
        const PerpendicularDiffusionAdvectionInterpolator&
            perpendicular_diffusion_advection_interpolator,
        const KappaParallelInterpolator& kappa_parallel_interpolator,
        const KappaPerpendicularInterpolator& kappa_perpendicular_interpolator,
        const SolarWindDivergenceInterpolator& solar_wind_divergence_interpolator,
        const MagneticFocusingInterpolator& magnetic_focusing_interpolator,
        const BBGradSolarWindInterpolator& bb_grad_solar_wind_interpolator,
        const ConvectiveDerivativeInterpolator& convective_derivative_interpolator,
        const ScatteringSigmaInterpolator& scattering_sigma_interpolator,
        const ScatteringCorrelationLengthInterpolator&
            scattering_correlation_length_interpolator) {
        const FocusTransportSolver local_solver = *this;
        const random_manager_type local_random_manager = random_manager;
        auto positions = this->particles.position;
        auto momenta = this->particles.momentum;
        auto mus = this->particles.mu;
        auto statuses = this->particles.status;
        Kokkos::parallel_for(
            UseRandom ? "FocusTransportSolver::advance_stochastic"
                      : "FocusTransportSolver::advance_deterministic",
            Kokkos::RangePolicy<execution_space>(0, this->particles.particle_count()),
            KOKKOS_LAMBDA(const size_type particle_index) {
                if (!particle_status_is_alive(statuses(particle_index))) {
                    return;
                }
                const debug_terms_type terms =
                    local_solver.template evaluate_particle_terms<false>(
                        particle_index, magnetic_field_interpolator,
                        magnetic_direction_interpolator,
                        magnetic_magnitude_interpolator, solar_wind_interpolator,
                        gradient_drift_basis_interpolator,
                        curvature_drift_basis_interpolator,
                        perpendicular_diffusion_advection_interpolator,
                        kappa_parallel_interpolator,
                        kappa_perpendicular_interpolator,
                        solar_wind_divergence_interpolator,
                        magnetic_focusing_interpolator,
                        bb_grad_solar_wind_interpolator,
                        convective_derivative_interpolator,
                        scattering_sigma_interpolator,
                        scattering_correlation_length_interpolator);

                vector_array_type physical_random_increment{};
                double pitch_angle_normal = 0.0;
                if constexpr (UseRandom) {
                    auto generator = local_random_manager.get_state();
                    physical_random_increment =
                        sample_perpendicular_diffusion_increment(
                            terms, step_dt, generator);
                    pitch_angle_normal =
                        stochastic_sampler_type::normal(generator);
                    local_random_manager.free_state(generator);
                }

                for (int dim = 0; dim < space_dim; ++dim) {
                    const double deterministic_increment =
                        step_dt * terms.coordinate_rate[dim];
                    const double stochastic_increment =
                        local_solver.coordinate_rate_component(
                            terms.position, dim,
                            physical_random_increment[dim]);
                    positions(particle_index, dim) +=
                        deterministic_increment + stochastic_increment;
                }

                const double updated_momentum =
                    terms.momentum_magnitude + step_dt * terms.momentum_rate;
                momenta(particle_index) =
                    updated_momentum > 0.0 ? updated_momentum : 0.0;
                mus(particle_index) = local_solver.picard_pitch_angle_update(
                    terms, step_dt, pitch_angle_normal);
            });
        Kokkos::fence(UseRandom ? "FocusTransportSolver::advance_stochastic"
                                : "FocusTransportSolver::advance_deterministic");
    }

    /**
     * Evaluate debug terms for one particle on the host.
     */
    debug_terms_type debug_terms_for_particle(const size_type particle_index) const {
        validate_for_step();
        if (particle_index >= this->particles.particle_count()) {
            throw std::runtime_error(
                "FocusTransportSolver: debug particle index is out of range.");
        }
        return debug_terms_for_particle_impl(particle_index);
    }

    /**
     * Evaluate debug terms for one particle.
     */
    debug_terms_type debug_terms_for_particle_impl(
        const size_type particle_index) const {
        const auto magnetic_field_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                this->B_vec, interpolation_domain(), interpolation_policy());
        const auto magnetic_direction_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.magnetic_direction, interpolation_domain(),
                interpolation_policy());
        const auto magnetic_magnitude_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.magnetic_field_magnitude, interpolation_domain(),
                interpolation_policy());
        const auto solar_wind_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                this->V_body, interpolation_domain(), interpolation_policy());
        const auto gradient_drift_basis_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.gradient_drift_basis_over_magnetic_field,
                interpolation_domain(), interpolation_policy());
        const auto curvature_drift_basis_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.curvature_drift_basis_over_magnetic_field,
                interpolation_domain(), interpolation_policy());
        const auto perpendicular_diffusion_advection_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.perpendicular_diffusion_advection_gamma_one,
                interpolation_domain(), interpolation_policy());
        const auto kappa_parallel_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.kappa_parallel_gamma_one, interpolation_domain(),
                interpolation_policy());
        const auto kappa_perpendicular_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.kappa_perpendicular_gamma_one,
                interpolation_domain(), interpolation_policy());
        const auto solar_wind_divergence_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.solar_wind_divergence, interpolation_domain(),
                interpolation_policy());
        const auto magnetic_focusing_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.magnetic_focusing, interpolation_domain(),
                interpolation_policy());
        const auto bb_grad_solar_wind_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.bb_grad_solar_wind, interpolation_domain(),
                interpolation_policy());
        const auto convective_derivative_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.b_dot_solar_wind_total_derivative,
                interpolation_domain(), interpolation_policy());
        const auto scattering_sigma_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.scattering_sigma2, interpolation_domain(),
                interpolation_policy());
        const auto scattering_correlation_length_interpolator =
            FieldInterpolator::make_linear_position_interpolator(
                coefficients.scattering_correlation_length,
                interpolation_domain(), interpolation_policy());

        debug_terms_view_type terms_view("focus_debug_terms", 1);
        const FocusTransportSolver local_solver = *this;
        Kokkos::parallel_for(
            "FocusTransportSolver::debug_terms_for_particle",
            Kokkos::RangePolicy<execution_space>(0, 1),
            KOKKOS_LAMBDA(const int) {
                terms_view(0) =
                    local_solver.template evaluate_particle_terms<true>(
                        particle_index, magnetic_field_interpolator,
                        magnetic_direction_interpolator,
                        magnetic_magnitude_interpolator, solar_wind_interpolator,
                        gradient_drift_basis_interpolator,
                        curvature_drift_basis_interpolator,
                        perpendicular_diffusion_advection_interpolator,
                        kappa_parallel_interpolator,
                        kappa_perpendicular_interpolator,
                        solar_wind_divergence_interpolator,
                        magnetic_focusing_interpolator,
                        bb_grad_solar_wind_interpolator,
                        convective_derivative_interpolator,
                        scattering_sigma_interpolator,
                        scattering_correlation_length_interpolator);
            });
        Kokkos::fence("FocusTransportSolver::debug_terms_for_particle");
        const auto terms_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, terms_view);
        return terms_host(0);
    }

    friend struct FocusTransportSolverDebugger<GridType, VecDim, DeviceType>;
};

template <
    typename GridType,
    int VecDim = GridType::vec_dim,
    typename DeviceType = typename GridType::device_type
>
using FocusSolver = FocusTransportSolver<GridType, VecDim, DeviceType>;

/**
 * Metadata-only focused solver alias kept for early scaffolding and smoke tests.
 */
template <int SpaceDim,
          int VecDim,
          typename DeviceType = Device,
          typename LayoutType = typename DeviceTraits<DeviceType>::array_layout>
using MetadataFocusTransportSolver =
    FocusTransportSolver<Grid<SpaceDim, DeviceType, LayoutType, VecDim>,
                         VecDim,
                         DeviceType>;
