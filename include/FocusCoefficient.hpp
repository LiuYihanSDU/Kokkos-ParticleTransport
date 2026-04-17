#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Field.hpp"
#include "FieldOperator.hpp"
#include "LegencyModel.hpp"
#include "ParkerCoefficient.hpp"

/**
 * Focused transport coefficient helpers and specialized orthogonal-coordinate
 * operators.
 */
namespace FocusCoefficient {

/**
 * Precomputed focused-transport coefficient fields consumed by FocusTransportSolver.
 */
template <
    typename GridType,
    typename LayoutType = typename GridType::layout_type
>
struct TransportCoefficientFieldSet {
    using grid_type = GridType;
    using layout_type = LayoutType;
    using scalar_field_type = ScalarField<grid_type, layout_type>;
    using vector_field_type = VectorField<grid_type, grid_type::vec_dim, layout_type>;

    LegencyModel::PerpendicularDiffusionModelKind perpendicular_model{
        LegencyModel::PerpendicularDiffusionModelKind::ReadmeFormula
    };
    double perpendicular_parallel_ratio{0.0};
    LegencyModel::PitchAngleScatteringParameters pitch_angle_scattering{};
    vector_field_type magnetic_direction;
    scalar_field_type magnetic_field_magnitude;
    scalar_field_type log_magnetic_field;
    vector_field_type grad_log_magnetic_field;
    vector_field_type magnetic_curvature;
    vector_field_type gradient_drift_basis_over_magnetic_field;
    vector_field_type curvature_drift_basis_over_magnetic_field;
    vector_field_type perpendicular_diffusion_advection_gamma_one;
    scalar_field_type kappa_parallel_gamma_one;
    scalar_field_type kappa_perpendicular_gamma_one;
    scalar_field_type solar_wind_divergence;
    scalar_field_type magnetic_focusing;
    scalar_field_type bb_grad_solar_wind;
    vector_field_type solar_wind_time_derivative;
    scalar_field_type b_dot_solar_wind_total_derivative;
    scalar_field_type scattering_sigma2;
    scalar_field_type scattering_correlation_length;

    TransportCoefficientFieldSet() = default;

    /**
     * Build a field set from already precomputed coefficient fields.
     */
    TransportCoefficientFieldSet(
        const vector_field_type& input_magnetic_direction,
        const scalar_field_type& input_magnetic_field_magnitude,
        const scalar_field_type& input_log_magnetic_field,
        const vector_field_type& input_grad_log_magnetic_field,
        const vector_field_type& input_magnetic_curvature,
        const vector_field_type& input_gradient_drift_basis_over_magnetic_field,
        const vector_field_type& input_curvature_drift_basis_over_magnetic_field,
        const vector_field_type& input_perpendicular_diffusion_advection_gamma_one,
        const scalar_field_type& input_kappa_parallel_gamma_one,
        const scalar_field_type& input_kappa_perpendicular_gamma_one,
        const scalar_field_type& input_solar_wind_divergence,
        const scalar_field_type& input_magnetic_focusing,
        const scalar_field_type& input_bb_grad_solar_wind,
        const vector_field_type& input_solar_wind_time_derivative,
        const scalar_field_type& input_b_dot_solar_wind_total_derivative,
        const scalar_field_type& input_scattering_sigma2,
        const scalar_field_type& input_scattering_correlation_length,
        const LegencyModel::DiffusionModelParameters& diffusion_parameters =
            LegencyModel::DiffusionModelParameters{},
        const LegencyModel::PitchAngleScatteringParameters&
            scattering_parameters = LegencyModel::PitchAngleScatteringParameters{})
        : perpendicular_model(diffusion_parameters.perpendicular_model),
          perpendicular_parallel_ratio(
              diffusion_parameters.perpendicular_parallel_ratio),
          pitch_angle_scattering(scattering_parameters),
          magnetic_direction(input_magnetic_direction),
          magnetic_field_magnitude(input_magnetic_field_magnitude),
          log_magnetic_field(input_log_magnetic_field),
          grad_log_magnetic_field(input_grad_log_magnetic_field),
          magnetic_curvature(input_magnetic_curvature),
          gradient_drift_basis_over_magnetic_field(
              input_gradient_drift_basis_over_magnetic_field),
          curvature_drift_basis_over_magnetic_field(
              input_curvature_drift_basis_over_magnetic_field),
          perpendicular_diffusion_advection_gamma_one(
              input_perpendicular_diffusion_advection_gamma_one),
          kappa_parallel_gamma_one(input_kappa_parallel_gamma_one),
          kappa_perpendicular_gamma_one(input_kappa_perpendicular_gamma_one),
          solar_wind_divergence(input_solar_wind_divergence),
          magnetic_focusing(input_magnetic_focusing),
          bb_grad_solar_wind(input_bb_grad_solar_wind),
          solar_wind_time_derivative(input_solar_wind_time_derivative),
          b_dot_solar_wind_total_derivative(
              input_b_dot_solar_wind_total_derivative),
          scattering_sigma2(input_scattering_sigma2),
          scattering_correlation_length(input_scattering_correlation_length) {}

    /**
     * Return true when all solver-consumed fields have been assigned.
     */
    bool configured() const {
        return magnetic_direction.point_count() > 0 &&
               magnetic_field_magnitude.point_count() > 0 &&
               log_magnetic_field.point_count() > 0 &&
               grad_log_magnetic_field.point_count() > 0 &&
               magnetic_curvature.point_count() > 0 &&
               gradient_drift_basis_over_magnetic_field.point_count() > 0 &&
               curvature_drift_basis_over_magnetic_field.point_count() > 0 &&
               perpendicular_diffusion_advection_gamma_one.point_count() > 0 &&
               kappa_parallel_gamma_one.point_count() > 0 &&
               kappa_perpendicular_gamma_one.point_count() > 0 &&
               solar_wind_divergence.point_count() > 0 &&
               magnetic_focusing.point_count() > 0 &&
               bb_grad_solar_wind.point_count() > 0 &&
               solar_wind_time_derivative.point_count() > 0 &&
               b_dot_solar_wind_total_derivative.point_count() > 0 &&
               scattering_sigma2.point_count() > 0 &&
               scattering_correlation_length.point_count() > 0;
    }
};

/**
 * Return whether logical field indices lie inside a selected domain.
 */
template <typename FieldType>
KOKKOS_INLINE_FUNCTION
bool field_indices_inside_domain(
    const FieldType& field,
    const typename FieldType::index_array_type& indices,
    const GridDomain domain) {
    for (int dim = 0; dim < FieldType::space_dim; ++dim) {
        if (indices[dim] < field.lower_index(dim, domain) ||
            indices[dim] > field.upper_index(dim, domain)) {
            return false;
        }
    }
    return true;
}

/**
 * Return a component value, treating components absent from the field as zero.
 */
template <typename FieldType>
KOKKOS_INLINE_FUNCTION
double field_component_or_zero(
    const FieldType& field,
    const typename FieldType::index_array_type& indices,
    const int component) {
    if (component < 0 || component >= FieldType::component_dim) {
        return 0.0;
    }
    return field(indices, component);
}

/**
 * Return the active vector-calculus basis dimension.
 */
template <typename GridType>
KOKKOS_INLINE_FUNCTION
int vector_calculus_basis_dim(const GridType&) {
    return GridType::vec_dim < 3 ? GridType::vec_dim : 3;
}

/**
 * Return a device-safe absolute value.
 */
KOKKOS_INLINE_FUNCTION
double absolute_value(const double value) {
    return value >= 0.0 ? value : -value;
}

/**
 * Validate same-grid field storage for focused coefficient operators.
 */
template <typename ReferenceFieldType, typename CheckedFieldType>
void validate_same_storage(const ReferenceFieldType& reference_field,
                           const CheckedFieldType& checked_field,
                           const char* operation_name) {
    static_assert(std::is_same_v<typename ReferenceFieldType::grid_type,
                                 typename CheckedFieldType::grid_type>,
                  "Focused transport coefficient fields must use the same grid type.");

    if (reference_field.point_count() != checked_field.point_count()) {
        throw std::runtime_error(
            std::string("FocusCoefficient: point-count mismatch in ") +
            operation_name + ".");
    }

    for (int dim = 0; dim < ReferenceFieldType::space_dim; ++dim) {
        if (reference_field.centering(dim) != checked_field.centering(dim) ||
            reference_field.lower_index(dim, GridDomain::GhostedDomain) !=
                checked_field.lower_index(dim, GridDomain::GhostedDomain) ||
            reference_field.upper_index(dim, GridDomain::GhostedDomain) !=
                checked_field.upper_index(dim, GridDomain::GhostedDomain)) {
            throw std::runtime_error(
                std::string("FocusCoefficient: storage-domain mismatch in ") +
                operation_name + ".");
        }
    }
}

/**
 * Validate derivative support for focused coefficient operators.
 */
template <typename FieldType>
void validate_derivative_field(const FieldType& field,
                               const char* operation_name) {
    static_assert(FieldType::space_dim <= 3,
                  "Focused transport operators support SpaceDim <= 3.");
    for (int dim = 0; dim < FieldType::space_dim; ++dim) {
        if (field.grid.extent(dim) > 1 &&
            (field.grid.ghost_extent(dim, LowerBoundary) < 1 ||
             field.grid.ghost_extent(dim, UpperBoundary) < 1)) {
            throw std::runtime_error(
                std::string("FocusCoefficient: ") + operation_name +
                " requires at least one ghost layer.");
        }
    }
}

/**
 * Return one physical-basis vector-gradient component in an orthogonal coordinate system.
 *
 * The returned value is (nabla_j A)_i, where i is output_component and j is
 * derivative_component, both expressed in the local physical orthonormal basis.
 */
template <typename VectorFieldType>
KOKKOS_INLINE_FUNCTION
double physical_vector_gradient_component(
    const VectorFieldType& vector_field,
    const typename VectorFieldType::index_array_type& indices,
    const int output_component,
    const int derivative_component) {
    const int basis_dim = vector_calculus_basis_dim(vector_field.grid);
    if (output_component < 0 || output_component >= basis_dim ||
        derivative_component < 0 || derivative_component >= basis_dim) {
        return 0.0;
    }

    const auto q = vector_field.grid.coordinate_tuple(indices,
                                                      vector_field.centerings);
    double result = 0.0;
    if (derivative_component < VectorFieldType::space_dim) {
        const double h_j = vector_field.grid.h(derivative_component, q);
        if (h_j == 0.0) {
            Kokkos::abort(
                "FocusCoefficient: vector gradient encountered a metric singularity.");
        }
        result = FieldOperator::centered_component_derivative(
                     vector_field, indices, derivative_component,
                     output_component) /
                 h_j;
    }

    const auto coordinate_system = vector_field.grid.coordinate_system();
    if (coordinate_system == OrthogonalCoordinateSystem::Cartesian) {
        return result;
    }

    const double r = q[0];
    if (r == 0.0) {
        Kokkos::abort(
            "FocusCoefficient: vector gradient encountered r = 0.");
    }
    const double inv_r = 1.0 / r;

    if (coordinate_system == OrthogonalCoordinateSystem::Polar) {
        if (derivative_component == 1) {
            if (output_component == 0) {
                result -= field_component_or_zero(vector_field, indices, 1) * inv_r;
            } else if (output_component == 1) {
                result += field_component_or_zero(vector_field, indices, 0) * inv_r;
            }
        }
        return result;
    }

    double cot_theta_over_r = 0.0;
    if constexpr (VectorFieldType::space_dim > 1) {
        const double sin_theta = Kokkos::sin(q[1]);
        if (absolute_value(sin_theta) <=
            OrthogonalGeometry<VectorFieldType::space_dim,
                               VectorFieldType::component_dim>::singularity_tolerance) {
            Kokkos::abort(
                "FocusCoefficient: vector gradient encountered a polar singularity.");
        }
        cot_theta_over_r = Kokkos::cos(q[1]) / sin_theta * inv_r;
    }

    if (derivative_component == 1) {
        if (output_component == 0) {
            result -= field_component_or_zero(vector_field, indices, 1) * inv_r;
        } else if (output_component == 1) {
            result += field_component_or_zero(vector_field, indices, 0) * inv_r;
        }
    } else if (derivative_component == 2) {
        if (output_component == 0) {
            result -= field_component_or_zero(vector_field, indices, 2) * inv_r;
        } else if (output_component == 1) {
            result -= field_component_or_zero(vector_field, indices, 2) *
                      cot_theta_over_r;
        } else if (output_component == 2) {
            result += field_component_or_zero(vector_field, indices, 0) * inv_r +
                      field_component_or_zero(vector_field, indices, 1) *
                          cot_theta_over_r;
        }
    }

    return result;
}

/**
 * Return b_i b_j (nabla_j V)_i.
 */
template <typename DirectionFieldType, typename VectorFieldType>
KOKKOS_INLINE_FUNCTION
double magnetic_direction_solar_wind_gradient_contraction(
    const DirectionFieldType& magnetic_direction,
    const VectorFieldType& solar_wind_velocity,
    const typename DirectionFieldType::index_array_type& indices) {
    const int basis_dim = vector_calculus_basis_dim(magnetic_direction.grid);
    double result = 0.0;
    for (int output_component = 0; output_component < basis_dim;
         ++output_component) {
        const double b_i =
            field_component_or_zero(magnetic_direction, indices, output_component);
        for (int derivative_component = 0; derivative_component < basis_dim;
             ++derivative_component) {
            const double b_j = field_component_or_zero(
                magnetic_direction, indices, derivative_component);
            result += b_i * b_j *
                      physical_vector_gradient_component(
                          solar_wind_velocity, indices, output_component,
                          derivative_component);
        }
    }
    return result;
}

/**
 * Return b_i V_j (nabla_j V)_i for the steady-flow material derivative projection.
 */
template <typename DirectionFieldType, typename VectorFieldType>
KOKKOS_INLINE_FUNCTION
double magnetic_direction_solar_wind_convective_derivative(
    const DirectionFieldType& magnetic_direction,
    const VectorFieldType& solar_wind_velocity,
    const typename DirectionFieldType::index_array_type& indices) {
    const int basis_dim = vector_calculus_basis_dim(magnetic_direction.grid);
    double result = 0.0;
    for (int output_component = 0; output_component < basis_dim;
         ++output_component) {
        const double b_i =
            field_component_or_zero(magnetic_direction, indices, output_component);
        for (int derivative_component = 0; derivative_component < basis_dim;
             ++derivative_component) {
            const double v_j = field_component_or_zero(
                solar_wind_velocity, indices, derivative_component);
            result += b_i * v_j *
                      physical_vector_gradient_component(
                          solar_wind_velocity, indices, output_component,
                          derivative_component);
        }
    }
    return result;
}

/**
 * Return b dot dV/dt = b dot partial_t V + b dot (V dot nabla) V.
 */
template <
    typename DirectionFieldType,
    typename SolarWindVelocityFieldType,
    typename SolarWindTimeDerivativeFieldType
>
KOKKOS_INLINE_FUNCTION
double magnetic_direction_solar_wind_total_derivative(
    const DirectionFieldType& magnetic_direction,
    const SolarWindVelocityFieldType& solar_wind_velocity,
    const SolarWindTimeDerivativeFieldType& solar_wind_time_derivative,
    const typename DirectionFieldType::index_array_type& indices) {
    const int basis_dim = vector_calculus_basis_dim(magnetic_direction.grid);
    double partial_time_projection = 0.0;
    for (int component = 0; component < basis_dim; ++component) {
        partial_time_projection +=
            field_component_or_zero(magnetic_direction, indices, component) *
            field_component_or_zero(solar_wind_time_derivative, indices, component);
    }
    return partial_time_projection +
           magnetic_direction_solar_wind_convective_derivative(
               magnetic_direction, solar_wind_velocity, indices);
}

/**
 * Return b dot grad(ln |B|).
 */
template <typename DirectionFieldType, typename GradientFieldType>
KOKKOS_INLINE_FUNCTION
double magnetic_focusing_value(
    const DirectionFieldType& magnetic_direction,
    const GradientFieldType& grad_log_magnetic_field,
    const typename DirectionFieldType::index_array_type& indices) {
    const int basis_dim = vector_calculus_basis_dim(magnetic_direction.grid);
    double result = 0.0;
    for (int component = 0; component < basis_dim; ++component) {
        result += field_component_or_zero(magnetic_direction, indices, component) *
                  field_component_or_zero(grad_log_magnetic_field, indices, component);
    }
    return result;
}

/**
 * Return one component of (b dot nabla) b.
 */
template <typename DirectionFieldType>
KOKKOS_INLINE_FUNCTION
double magnetic_curvature_component(
    const DirectionFieldType& magnetic_direction,
    const typename DirectionFieldType::index_array_type& indices,
    const int output_component) {
    const int basis_dim = vector_calculus_basis_dim(magnetic_direction.grid);
    if (output_component < 0 || output_component >= basis_dim) {
        return 0.0;
    }

    double result = 0.0;
    for (int derivative_component = 0; derivative_component < basis_dim;
         ++derivative_component) {
        const double b_j = field_component_or_zero(
            magnetic_direction, indices, derivative_component);
        result += b_j * physical_vector_gradient_component(
                            magnetic_direction, indices, output_component,
                            derivative_component);
    }
    return result;
}

/**
 * Return one component of a physical cross product.
 */
KOKKOS_INLINE_FUNCTION
double cross_product_component(const double ax,
                               const double ay,
                               const double az,
                               const double bx,
                               const double by,
                               const double bz,
                               const int component) {
    if (component == 0) {
        return ay * bz - az * by;
    }
    if (component == 1) {
        return az * bx - ax * bz;
    }
    if (component == 2) {
        return ax * by - ay * bx;
    }
    return 0.0;
}

/**
 * Return |B| at one field point.
 */
template <typename MagneticFieldType>
KOKKOS_INLINE_FUNCTION
double magnetic_field_magnitude_value(
    const MagneticFieldType& magnetic_field,
    const typename MagneticFieldType::index_array_type& indices) {
    double magnitude_squared = 0.0;
    for (int component = 0; component < MagneticFieldType::component_dim;
         ++component) {
        const double value = magnetic_field(indices, component);
        magnitude_squared += value * value;
    }
    return Kokkos::sqrt(magnitude_squared);
}

/**
 * Fill a scalar field with ln |B|.
 */
template <typename MagneticFieldType, typename LogFieldType>
void fill_log_magnetic_field(
    const MagneticFieldType& magnetic_field,
    LogFieldType& log_magnetic_field,
    const double minimum_magnetic_field_magnitude = 0.0,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    static_assert(LogFieldType::component_dim == 1,
                  "FocusCoefficient::fill_log_magnetic_field requires a scalar output.");
    if (!std::isfinite(minimum_magnetic_field_magnitude) ||
        minimum_magnetic_field_magnitude < 0.0) {
        throw std::runtime_error(
            "FocusCoefficient: magnetic-field magnitude floor must be finite and non-negative.");
    }
    validate_same_storage(magnetic_field, log_magnetic_field,
                          "fill_log_magnetic_field");

    const MagneticFieldType local_magnetic_field = magnetic_field;
    const LogFieldType local_log_field = log_magnetic_field;
    auto log_data = log_magnetic_field.data;
    Kokkos::parallel_for(
        "FocusCoefficient::fill_log_magnetic_field",
        Kokkos::RangePolicy<typename LogFieldType::execution_space>(
            0, log_magnetic_field.point_count()),
        KOKKOS_LAMBDA(const typename LogFieldType::size_type point_index) {
            const auto indices = local_log_field.logical_indices(point_index);
            const double magnitude = magnetic_field_magnitude_value(
                local_magnetic_field, indices);
            const double safe_magnitude =
                magnitude > minimum_magnetic_field_magnitude
                    ? magnitude
                    : minimum_magnetic_field_magnitude;
            if (safe_magnitude <= 0.0) {
                Kokkos::abort(
                    "FocusCoefficient: ln |B| requires a positive magnetic field.");
            }
            log_data(point_index, 0) = Kokkos::log(safe_magnitude);
        });
    Kokkos::fence("FocusCoefficient::fill_log_magnetic_field");
    log_magnetic_field.fill_ghost_cells(ghost_fill_mode);
}

/**
 * Allocate and fill a scalar field with ln |B|.
 */
template <typename MagneticFieldType>
auto make_log_magnetic_field(
    const MagneticFieldType& magnetic_field,
    const char* label = "log_magnetic_field",
    const double minimum_magnetic_field_magnitude = 0.0,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    using scalar_field_type =
        ScalarField<typename MagneticFieldType::grid_type,
                    typename MagneticFieldType::layout_type>;
    scalar_field_type log_magnetic_field(magnetic_field.grid, label,
                                         magnetic_field.centerings);
    fill_log_magnetic_field(magnetic_field, log_magnetic_field,
                            minimum_magnetic_field_magnitude, ghost_fill_mode);
    return log_magnetic_field;
}

/**
 * Fill a vector field with magnetic curvature (b dot nabla) b.
 */
template <typename DirectionFieldType, typename CurvatureFieldType>
void fill_magnetic_curvature(
    const DirectionFieldType& magnetic_direction,
    CurvatureFieldType& magnetic_curvature,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    static_assert(DirectionFieldType::component_dim == CurvatureFieldType::component_dim,
                  "FocusCoefficient::fill_magnetic_curvature requires matching components.");
    validate_same_storage(magnetic_direction, magnetic_curvature,
                          "fill_magnetic_curvature");
    validate_derivative_field(magnetic_direction, "fill_magnetic_curvature");

    const DirectionFieldType local_direction = magnetic_direction;
    const CurvatureFieldType local_curvature = magnetic_curvature;
    auto curvature_data = magnetic_curvature.data;
    Kokkos::parallel_for(
        "FocusCoefficient::fill_magnetic_curvature",
        Kokkos::RangePolicy<typename CurvatureFieldType::execution_space>(
            0, magnetic_curvature.point_count()),
        KOKKOS_LAMBDA(const typename CurvatureFieldType::size_type point_index) {
            const auto indices = local_curvature.logical_indices(point_index);
            if (!field_indices_inside_domain(local_curvature, indices,
                                             GridDomain::PhysicalDomain)) {
                return;
            }
            for (int component = 0; component < CurvatureFieldType::component_dim;
                 ++component) {
                curvature_data(point_index, component) =
                    magnetic_curvature_component(local_direction, indices, component);
            }
        });
    Kokkos::fence("FocusCoefficient::fill_magnetic_curvature");
    magnetic_curvature.fill_ghost_cells(ghost_fill_mode);
}

/**
 * Allocate and fill a vector field with magnetic curvature.
 */
template <typename DirectionFieldType>
auto make_magnetic_curvature_field(
    const DirectionFieldType& magnetic_direction,
    const char* label = "magnetic_curvature",
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    using curvature_field_type =
        VectorField<typename DirectionFieldType::grid_type,
                    DirectionFieldType::component_dim,
                    typename DirectionFieldType::layout_type>;
    curvature_field_type magnetic_curvature(magnetic_direction.grid, label,
                                            magnetic_direction.centerings);
    fill_magnetic_curvature(magnetic_direction, magnetic_curvature,
                            ghost_fill_mode);
    return magnetic_curvature;
}

/**
 * Fill focused drift basis fields divided by |B|.
 */
template <
    typename DirectionFieldType,
    typename GradientFieldType,
    typename CurvatureFieldType,
    typename MagneticMagnitudeFieldType,
    typename GradientBasisFieldType,
    typename CurvatureBasisFieldType
>
void fill_focused_drift_basis_fields(
    const DirectionFieldType& magnetic_direction,
    const GradientFieldType& grad_log_magnetic_field,
    const CurvatureFieldType& magnetic_curvature,
    const MagneticMagnitudeFieldType& magnetic_field_magnitude,
    GradientBasisFieldType& gradient_drift_basis_over_magnetic_field,
    CurvatureBasisFieldType& curvature_drift_basis_over_magnetic_field,
    const double minimum_magnetic_field_magnitude = 0.0,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    static_assert(DirectionFieldType::component_dim ==
                      GradientBasisFieldType::component_dim &&
                  DirectionFieldType::component_dim ==
                      CurvatureBasisFieldType::component_dim,
                  "FocusCoefficient::fill_focused_drift_basis_fields requires matching vector components.");
    validate_same_storage(magnetic_direction, grad_log_magnetic_field,
                          "fill_focused_drift_basis_fields");
    validate_same_storage(magnetic_direction, magnetic_curvature,
                          "fill_focused_drift_basis_fields");
    validate_same_storage(magnetic_direction, magnetic_field_magnitude,
                          "fill_focused_drift_basis_fields");
    validate_same_storage(magnetic_direction,
                          gradient_drift_basis_over_magnetic_field,
                          "fill_focused_drift_basis_fields");
    validate_same_storage(magnetic_direction,
                          curvature_drift_basis_over_magnetic_field,
                          "fill_focused_drift_basis_fields");

    const DirectionFieldType local_direction = magnetic_direction;
    const GradientFieldType local_gradient = grad_log_magnetic_field;
    const CurvatureFieldType local_curvature = magnetic_curvature;
    const MagneticMagnitudeFieldType local_magnitude = magnetic_field_magnitude;
    const GradientBasisFieldType local_gradient_basis =
        gradient_drift_basis_over_magnetic_field;
    auto gradient_basis_data = gradient_drift_basis_over_magnetic_field.data;
    auto curvature_basis_data = curvature_drift_basis_over_magnetic_field.data;
    Kokkos::parallel_for(
        "FocusCoefficient::fill_focused_drift_basis_fields",
        Kokkos::RangePolicy<typename GradientBasisFieldType::execution_space>(
            0, gradient_drift_basis_over_magnetic_field.point_count()),
        KOKKOS_LAMBDA(
            const typename GradientBasisFieldType::size_type point_index) {
            const auto indices = local_gradient_basis.logical_indices(point_index);
            const double magnetic_magnitude =
                local_magnitude(indices, 0) > minimum_magnetic_field_magnitude
                    ? local_magnitude(indices, 0)
                    : minimum_magnetic_field_magnitude;
            if (magnetic_magnitude <= 0.0) {
                Kokkos::abort(
                    "FocusCoefficient: focused drift requires |B| > 0.");
            }
            const double inverse_magnetic_magnitude = 1.0 / magnetic_magnitude;
            const double bx = field_component_or_zero(local_direction, indices, 0);
            const double by = field_component_or_zero(local_direction, indices, 1);
            const double bz = field_component_or_zero(local_direction, indices, 2);
            const double gx = field_component_or_zero(local_gradient, indices, 0);
            const double gy = field_component_or_zero(local_gradient, indices, 1);
            const double gz = field_component_or_zero(local_gradient, indices, 2);
            const double cx = field_component_or_zero(local_curvature, indices, 0);
            const double cy = field_component_or_zero(local_curvature, indices, 1);
            const double cz = field_component_or_zero(local_curvature, indices, 2);
            for (int component = 0;
                 component < GradientBasisFieldType::component_dim;
                 ++component) {
                gradient_basis_data(point_index, component) =
                    cross_product_component(bx, by, bz, gx, gy, gz, component) *
                    inverse_magnetic_magnitude;
                curvature_basis_data(point_index, component) =
                    cross_product_component(bx, by, bz, cx, cy, cz, component) *
                    inverse_magnetic_magnitude;
            }
        });
    Kokkos::fence("FocusCoefficient::fill_focused_drift_basis_fields");
    gradient_drift_basis_over_magnetic_field.fill_ghost_cells(ghost_fill_mode);
    curvature_drift_basis_over_magnetic_field.fill_ghost_cells(ghost_fill_mode);
}

/**
 * Fill scalar focused transport contraction fields.
 */
template <
    typename DirectionFieldType,
    typename GradientFieldType,
    typename SolarWindVelocityFieldType,
    typename SolarWindTimeDerivativeFieldType,
    typename FocusingFieldType,
    typename BBGradFieldType,
    typename TotalDerivativeFieldType
>
void fill_focused_transport_contraction_fields(
    const DirectionFieldType& magnetic_direction,
    const GradientFieldType& grad_log_magnetic_field,
    const SolarWindVelocityFieldType& solar_wind_velocity,
    const SolarWindTimeDerivativeFieldType& solar_wind_time_derivative,
    FocusingFieldType& magnetic_focusing,
    BBGradFieldType& bb_grad_solar_wind,
    TotalDerivativeFieldType& b_dot_solar_wind_total_derivative,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    static_assert(FocusingFieldType::component_dim == 1 &&
                  BBGradFieldType::component_dim == 1 &&
                  TotalDerivativeFieldType::component_dim == 1,
                  "FocusCoefficient contraction outputs must be scalar fields.");
    validate_same_storage(magnetic_direction, grad_log_magnetic_field,
                          "fill_focused_transport_contraction_fields");
    validate_same_storage(magnetic_direction, solar_wind_velocity,
                          "fill_focused_transport_contraction_fields");
    validate_same_storage(magnetic_direction, solar_wind_time_derivative,
                          "fill_focused_transport_contraction_fields");
    validate_same_storage(magnetic_direction, magnetic_focusing,
                          "fill_focused_transport_contraction_fields");
    validate_same_storage(magnetic_direction, bb_grad_solar_wind,
                          "fill_focused_transport_contraction_fields");
    validate_same_storage(magnetic_direction,
                          b_dot_solar_wind_total_derivative,
                          "fill_focused_transport_contraction_fields");
    validate_derivative_field(solar_wind_velocity,
                              "fill_focused_transport_contraction_fields");

    const DirectionFieldType local_direction = magnetic_direction;
    const GradientFieldType local_gradient = grad_log_magnetic_field;
    const SolarWindVelocityFieldType local_solar_wind = solar_wind_velocity;
    const SolarWindTimeDerivativeFieldType local_solar_wind_time_derivative =
        solar_wind_time_derivative;
    const FocusingFieldType local_focusing = magnetic_focusing;
    auto focusing_data = magnetic_focusing.data;
    auto bb_grad_data = bb_grad_solar_wind.data;
    auto total_derivative_data = b_dot_solar_wind_total_derivative.data;
    Kokkos::parallel_for(
        "FocusCoefficient::fill_focused_transport_contraction_fields",
        Kokkos::RangePolicy<typename FocusingFieldType::execution_space>(
            0, magnetic_focusing.point_count()),
        KOKKOS_LAMBDA(const typename FocusingFieldType::size_type point_index) {
            const auto indices = local_focusing.logical_indices(point_index);
            if (!field_indices_inside_domain(local_focusing, indices,
                                             GridDomain::PhysicalDomain)) {
                return;
            }
            focusing_data(point_index, 0) =
                magnetic_focusing_value(local_direction, local_gradient, indices);
            bb_grad_data(point_index, 0) =
                magnetic_direction_solar_wind_gradient_contraction(
                    local_direction, local_solar_wind, indices);
            total_derivative_data(point_index, 0) =
                magnetic_direction_solar_wind_total_derivative(
                    local_direction, local_solar_wind,
                    local_solar_wind_time_derivative, indices);
        });
    Kokkos::fence(
        "FocusCoefficient::fill_focused_transport_contraction_fields");
    magnetic_focusing.fill_ghost_cells(ghost_fill_mode);
    bb_grad_solar_wind.fill_ghost_cells(ghost_fill_mode);
    b_dot_solar_wind_total_derivative.fill_ghost_cells(ghost_fill_mode);
}

/**
 * Fill scalar turbulence fields used by pitch-angle scattering.
 */
template <
    typename ReferenceFieldType,
    typename TurbulencePropertiesType,
    typename SigmaFieldType,
    typename CorrelationLengthFieldType
>
void fill_scattering_turbulence_fields(
    const ReferenceFieldType& reference_field,
    const TurbulencePropertiesType& turbulence_properties,
    SigmaFieldType& scattering_sigma2,
    CorrelationLengthFieldType& scattering_correlation_length,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    static_assert(std::is_same_v<typename ReferenceFieldType::grid_type,
                                 typename TurbulencePropertiesType::grid_type>,
                  "Focused scattering turbulence fields must use the same grid type.");
    static_assert(SigmaFieldType::component_dim == 1 &&
                  CorrelationLengthFieldType::component_dim == 1,
                  "Focused scattering turbulence outputs must be scalar fields.");
    validate_same_storage(reference_field, scattering_sigma2,
                          "fill_scattering_turbulence_fields");
    validate_same_storage(reference_field, scattering_correlation_length,
                          "fill_scattering_turbulence_fields");

    const SigmaFieldType local_sigma = scattering_sigma2;
    auto sigma_data = scattering_sigma2.data;
    auto correlation_data = scattering_correlation_length.data;
    const auto turbulence = turbulence_properties.accessor();
    Kokkos::parallel_for(
        "FocusCoefficient::fill_scattering_turbulence_fields",
        Kokkos::RangePolicy<typename SigmaFieldType::execution_space>(
            0, scattering_sigma2.point_count()),
        KOKKOS_LAMBDA(const typename SigmaFieldType::size_type point_index) {
            const auto indices = local_sigma.logical_indices(point_index);
            const TurbulencePropertyState state = turbulence(indices);
            sigma_data(point_index, 0) = state.sigma2;
            correlation_data(point_index, 0) = state.lc;
        });
    Kokkos::fence("FocusCoefficient::fill_scattering_turbulence_fields");
    scattering_sigma2.fill_ghost_cells(ghost_fill_mode);
    scattering_correlation_length.fill_ghost_cells(ghost_fill_mode);
}

/**
 * Fill div(kappa_perp (I - bb)) for gamma = 1.
 */
template <
    typename MagneticFieldType,
    typename KappaPerpendicularFieldType,
    typename OutputFieldType
>
void fill_perpendicular_diffusion_advection_field(
    const MagneticFieldType& magnetic_field,
    const KappaPerpendicularFieldType& kappa_perpendicular_gamma_one,
    OutputFieldType& perpendicular_diffusion_advection_gamma_one,
    const double minimum_magnetic_field_magnitude = 0.0,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    ParkerCoefficient::fill_diffusion_tensor_divergence_basis_field(
        magnetic_field, kappa_perpendicular_gamma_one,
        kappa_perpendicular_gamma_one,
        perpendicular_diffusion_advection_gamma_one,
        ParkerCoefficient::DiffusionTensorDivergenceBasis::PerpendicularProjection,
        0.0, minimum_magnetic_field_magnitude, ghost_fill_mode);
}

/**
 * Allocate and fill all focused-transport coefficient fields.
 */
template <
    typename MagneticFieldType,
    typename SolarWindVelocityFieldType,
    typename SolarWindTimeDerivativeFieldType,
    typename TurbulencePropertiesType
>
auto make_legacy_transport_coefficient_fields_with_time_derivative(
    const MagneticFieldType& magnetic_field,
    const SolarWindVelocityFieldType& solar_wind_velocity,
    const SolarWindTimeDerivativeFieldType& solar_wind_time_derivative,
    const TurbulencePropertiesType& turbulence_properties,
    const ParticleProperties& particle_properties,
    const char* label = "focused_transport_coefficients",
    const LegencyModel::DiffusionModelParameters& diffusion_parameters =
        LegencyModel::DiffusionModelParameters{},
    const LegencyModel::PitchAngleScatteringParameters&
        scattering_parameters = LegencyModel::PitchAngleScatteringParameters{},
    const double minimum_magnetic_field_magnitude = 0.0,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    static_assert(std::is_same_v<typename MagneticFieldType::grid_type,
                                 typename SolarWindVelocityFieldType::grid_type>,
                  "Focused transport coefficients require B and Vsw on the same grid.");
    static_assert(std::is_same_v<typename MagneticFieldType::grid_type,
                                 typename SolarWindTimeDerivativeFieldType::grid_type>,
                  "Focused transport coefficients require partial_t Vsw on the same grid.");
    static_assert(std::is_same_v<typename MagneticFieldType::grid_type,
                                 typename TurbulencePropertiesType::grid_type>,
                  "Focused transport coefficients require turbulence on the same grid.");

    using grid_type = typename MagneticFieldType::grid_type;
    using layout_type = typename MagneticFieldType::layout_type;
    using scalar_field_type = ScalarField<grid_type, layout_type>;
    using vector_field_type = VectorField<grid_type, grid_type::vec_dim, layout_type>;
    using coefficient_field_set_type =
        TransportCoefficientFieldSet<grid_type, layout_type>;

    diffusion_parameters.validate();
    scattering_parameters.validate();
    particle_properties.validate();

    const std::string prefix(label);
    auto magnetic_field_magnitude =
        FieldOperator::make_magnitude_field(
            magnetic_field, (prefix + "_magnetic_field_magnitude").c_str());
    auto magnetic_direction =
        FieldOperator::make_direction_field(
            magnetic_field, (prefix + "_magnetic_direction").c_str(),
            minimum_magnetic_field_magnitude,
            FieldOperator::ZeroMagnitudePolicy::Abort);
    auto log_magnetic_field =
        make_log_magnetic_field(magnetic_field,
                                (prefix + "_log_magnetic_field").c_str(),
                                minimum_magnetic_field_magnitude,
                                ghost_fill_mode);
    auto grad_log_magnetic_field =
        FieldOperator::make_gradient_field(
            log_magnetic_field,
            (prefix + "_grad_log_magnetic_field").c_str(),
            ghost_fill_mode);
    auto magnetic_curvature =
        make_magnetic_curvature_field(
            magnetic_direction, (prefix + "_magnetic_curvature").c_str(),
            ghost_fill_mode);

    vector_field_type gradient_drift_basis_over_magnetic_field(
        magnetic_field.grid,
        (prefix + "_gradient_drift_basis_over_magnetic_field").c_str(),
        magnetic_field.centerings);
    vector_field_type curvature_drift_basis_over_magnetic_field(
        magnetic_field.grid,
        (prefix + "_curvature_drift_basis_over_magnetic_field").c_str(),
        magnetic_field.centerings);
    fill_focused_drift_basis_fields(
        magnetic_direction, grad_log_magnetic_field, magnetic_curvature,
        magnetic_field_magnitude, gradient_drift_basis_over_magnetic_field,
        curvature_drift_basis_over_magnetic_field,
        minimum_magnetic_field_magnitude, ghost_fill_mode);

    auto diffusion_fields =
        LegencyModel::make_gamma_independent_diffusion_fields(
            magnetic_field, solar_wind_velocity, turbulence_properties,
            particle_properties, (prefix + "_legacy_diffusion").c_str(),
            diffusion_parameters, ghost_fill_mode);

    vector_field_type perpendicular_diffusion_advection_gamma_one(
        magnetic_field.grid,
        (prefix + "_perpendicular_diffusion_advection_gamma_one").c_str(),
        magnetic_field.centerings);
    fill_perpendicular_diffusion_advection_field(
        magnetic_field, diffusion_fields.kappa_perpendicular_gamma_one,
        perpendicular_diffusion_advection_gamma_one,
        diffusion_parameters.minimum_magnetic_field_magnitude, ghost_fill_mode);

    auto solar_wind_divergence =
        FieldOperator::make_divergence_field(
            solar_wind_velocity, (prefix + "_solar_wind_divergence").c_str(),
            ghost_fill_mode);

    scalar_field_type magnetic_focusing(
        magnetic_field.grid, (prefix + "_magnetic_focusing").c_str(),
        magnetic_field.centerings);
    scalar_field_type bb_grad_solar_wind(
        magnetic_field.grid, (prefix + "_bb_grad_solar_wind").c_str(),
        magnetic_field.centerings);
    scalar_field_type b_dot_solar_wind_total_derivative(
        magnetic_field.grid,
        (prefix + "_b_dot_solar_wind_total_derivative").c_str(),
        magnetic_field.centerings);
    fill_focused_transport_contraction_fields(
        magnetic_direction, grad_log_magnetic_field, solar_wind_velocity,
        solar_wind_time_derivative, magnetic_focusing, bb_grad_solar_wind,
        b_dot_solar_wind_total_derivative, ghost_fill_mode);

    scalar_field_type scattering_sigma2(
        magnetic_field.grid, (prefix + "_scattering_sigma2").c_str(),
        magnetic_field.centerings);
    scalar_field_type scattering_correlation_length(
        magnetic_field.grid,
        (prefix + "_scattering_correlation_length").c_str(),
        magnetic_field.centerings);
    fill_scattering_turbulence_fields(
        magnetic_field, turbulence_properties, scattering_sigma2,
        scattering_correlation_length, ghost_fill_mode);

    return coefficient_field_set_type(
        magnetic_direction, magnetic_field_magnitude, log_magnetic_field,
        grad_log_magnetic_field, magnetic_curvature,
        gradient_drift_basis_over_magnetic_field,
        curvature_drift_basis_over_magnetic_field,
        perpendicular_diffusion_advection_gamma_one,
        diffusion_fields.kappa_parallel_gamma_one,
        diffusion_fields.kappa_perpendicular_gamma_one,
        solar_wind_divergence, magnetic_focusing, bb_grad_solar_wind,
        solar_wind_time_derivative, b_dot_solar_wind_total_derivative,
        scattering_sigma2,
        scattering_correlation_length, diffusion_parameters,
        scattering_parameters);
}

/**
 * Allocate and fill focused coefficient fields with partial_t V_sw = 0.
 */
template <
    typename MagneticFieldType,
    typename SolarWindVelocityFieldType,
    typename TurbulencePropertiesType
>
auto make_legacy_transport_coefficient_fields(
    const MagneticFieldType& magnetic_field,
    const SolarWindVelocityFieldType& solar_wind_velocity,
    const TurbulencePropertiesType& turbulence_properties,
    const ParticleProperties& particle_properties,
    const char* label = "focused_transport_coefficients",
    const LegencyModel::DiffusionModelParameters& diffusion_parameters =
        LegencyModel::DiffusionModelParameters{},
    const LegencyModel::PitchAngleScatteringParameters&
        scattering_parameters = LegencyModel::PitchAngleScatteringParameters{},
    const double minimum_magnetic_field_magnitude = 0.0,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    const std::string prefix(label);
    auto zero_solar_wind_time_derivative =
        FieldOperator::make_scaled_field(
            solar_wind_velocity,
            0.0,
            (prefix + "_zero_solar_wind_time_derivative").c_str());
    return make_legacy_transport_coefficient_fields_with_time_derivative(
        magnetic_field, solar_wind_velocity, zero_solar_wind_time_derivative,
        turbulence_properties, particle_properties, label, diffusion_parameters,
        scattering_parameters, minimum_magnetic_field_magnitude, ghost_fill_mode);
}

} // namespace FocusCoefficient
