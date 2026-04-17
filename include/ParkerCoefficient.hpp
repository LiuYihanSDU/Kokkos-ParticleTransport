#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Field.hpp"
#include "FieldOperator.hpp"
#include "LegencyModel.hpp"

/**
 * Parker transport coefficient helpers.
 */
namespace ParkerCoefficient {

/**
 * Basis branch used to build gamma-independent diffusion-tensor divergence fields.
 */
enum class DiffusionTensorDivergenceBasis : int {
    ParallelProjection = 0,
    PerpendicularProjection = 1,
    ConstantParallelRatio = 2
};

/**
 * Grid fields containing gamma-independent diffusion-tensor divergence bases.
 */
template <typename GridType>
struct DiffusionTensorDivergenceFieldSet {
    using grid_type = GridType;
    using vector_field_type = VectorField<grid_type, grid_type::vec_dim>;

    LegencyModel::PerpendicularDiffusionModelKind perpendicular_model{
        LegencyModel::PerpendicularDiffusionModelKind::ReadmeFormula
    };
    vector_field_type parallel_basis;
    vector_field_type perpendicular_basis;
    vector_field_type constant_ratio_basis;

    DiffusionTensorDivergenceFieldSet() = default;

    /**
     * Allocate the divergence basis fields required by the selected branch.
     */
    DiffusionTensorDivergenceFieldSet(
        const grid_type& grid,
        const char* label = "diffusion_tensor_divergence",
        const LegencyModel::PerpendicularDiffusionModelKind model_kind =
            LegencyModel::PerpendicularDiffusionModelKind::ReadmeFormula)
        : perpendicular_model(model_kind) {
        const std::string prefix(label);
        if (model_kind ==
            LegencyModel::PerpendicularDiffusionModelKind::ConstantParallelRatio) {
            constant_ratio_basis =
                vector_field_type(grid, (prefix + "_constant_ratio_basis").c_str());
        } else {
            parallel_basis =
                vector_field_type(grid, (prefix + "_parallel_basis").c_str());
            perpendicular_basis =
                vector_field_type(grid, (prefix + "_perpendicular_basis").c_str());
        }
    }

    /**
     * Return true when this set stores the two README-formula basis fields.
     */
    bool stores_readme_basis_fields() const {
        return perpendicular_model ==
               LegencyModel::PerpendicularDiffusionModelKind::ReadmeFormula;
    }

    /**
     * Return true when this set stores the single constant-ratio basis field.
     */
    bool stores_constant_ratio_basis_field() const {
        return perpendicular_model ==
               LegencyModel::PerpendicularDiffusionModelKind::ConstantParallelRatio;
    }
};

/**
 * Precomputed Parker transport coefficient fields consumed by ParkerSolver.
 *
 * The fields are gamma-independent where possible. Particle kernels interpolate these
 * fields and apply the particle gamma factors locally.
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
    vector_field_type magnetic_drift_curl;
    DiffusionTensorDivergenceFieldSet<grid_type> diffusion_tensor_divergence;
    scalar_field_type kappa_parallel_gamma_one;
    scalar_field_type kappa_perpendicular_gamma_one;
    scalar_field_type solar_wind_divergence;

    TransportCoefficientFieldSet() = default;

    /**
     * Build a field set from already precomputed coefficient fields.
     */
    TransportCoefficientFieldSet(
        const vector_field_type& input_magnetic_drift_curl,
        const DiffusionTensorDivergenceFieldSet<grid_type>& input_divergence_fields,
        const scalar_field_type& input_kappa_parallel_gamma_one,
        const scalar_field_type& input_kappa_perpendicular_gamma_one,
        const scalar_field_type& input_solar_wind_divergence,
        const LegencyModel::DiffusionModelParameters& parameters =
            LegencyModel::DiffusionModelParameters{})
        : perpendicular_model(parameters.perpendicular_model),
          perpendicular_parallel_ratio(parameters.perpendicular_parallel_ratio),
          magnetic_drift_curl(input_magnetic_drift_curl),
          diffusion_tensor_divergence(input_divergence_fields),
          kappa_parallel_gamma_one(input_kappa_parallel_gamma_one),
          kappa_perpendicular_gamma_one(input_kappa_perpendicular_gamma_one),
          solar_wind_divergence(input_solar_wind_divergence) {}

    /**
     * Return true when all solver-consumed fields have been assigned.
     */
    bool configured() const {
        const bool has_common_fields =
            magnetic_drift_curl.point_count() > 0 &&
            kappa_parallel_gamma_one.point_count() > 0 &&
            kappa_perpendicular_gamma_one.point_count() > 0 &&
            solar_wind_divergence.point_count() > 0;
        if (!has_common_fields) {
            return false;
        }
        if (perpendicular_model ==
            LegencyModel::PerpendicularDiffusionModelKind::ConstantParallelRatio) {
            return diffusion_tensor_divergence.constant_ratio_basis.point_count() > 0;
        }
        return diffusion_tensor_divergence.parallel_basis.point_count() > 0 &&
               diffusion_tensor_divergence.perpendicular_basis.point_count() > 0;
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
 * Return one vector component, treating components absent from the field as zero.
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
 * Return the active physical basis dimension used by vector/tensor calculus.
 */
template <typename GridType>
KOKKOS_INLINE_FUNCTION
int vector_calculus_basis_dim(const GridType&) {
    return GridType::vec_dim < 3 ? GridType::vec_dim : 3;
}

/**
 * Return the metric Jacobian for the active physical vector-calculus basis.
 */
template <typename GridType>
KOKKOS_INLINE_FUNCTION
double vector_calculus_jacobian(
    const GridType& grid,
    const typename GridType::coordinate_array_type& q) {
    double jacobian = 1.0;
    const int basis_dim = vector_calculus_basis_dim(grid);
    for (int component = 0; component < basis_dim; ++component) {
        jacobian *= grid.h(component, q);
    }
    return jacobian;
}

/**
 * Return a nonuniform-grid three-point centered derivative.
 */
KOKKOS_INLINE_FUNCTION
double centered_three_point_derivative(const double lower_coordinate,
                                       const double center_coordinate,
                                       const double upper_coordinate,
                                       const double lower_value,
                                       const double center_value,
                                       const double upper_value) {
    const double lower_width = center_coordinate - lower_coordinate;
    const double upper_width = upper_coordinate - center_coordinate;
    const double total_width = upper_coordinate - lower_coordinate;
    if (lower_width == 0.0 || upper_width == 0.0 || total_width == 0.0) {
        Kokkos::abort(
            "ParkerCoefficient: centered derivative requires distinct coordinates.");
    }

    return (-upper_width / (lower_width * total_width)) * lower_value +
           ((upper_width - lower_width) / (lower_width * upper_width)) * center_value +
           (lower_width / (upper_width * total_width)) * upper_value;
}

/**
 * Return the magnetic-field magnitude at one logical grid point.
 */
template <typename MagneticFieldType>
KOKKOS_INLINE_FUNCTION
double magnetic_field_magnitude(
    const MagneticFieldType& magnetic_field,
    const typename MagneticFieldType::index_array_type& indices) {
    double magnetic_field_squared = 0.0;
    for (int component = 0; component < MagneticFieldType::component_dim; ++component) {
        const double value = magnetic_field(indices, component);
        magnetic_field_squared += value * value;
    }
    return Kokkos::sqrt(magnetic_field_squared);
}

/**
 * Return one magnetic-field unit-vector component with optional magnitude floor.
 */
template <typename MagneticFieldType>
KOKKOS_INLINE_FUNCTION
double magnetic_direction_component(
    const MagneticFieldType& magnetic_field,
    const typename MagneticFieldType::index_array_type& indices,
    const int component,
    const double minimum_magnetic_field_magnitude) {
    const double magnitude = magnetic_field_magnitude(magnetic_field, indices);
    if (magnitude <= 0.0 && minimum_magnetic_field_magnitude <= 0.0) {
        Kokkos::abort(
            "ParkerCoefficient: diffusion tensor direction is undefined where |B| <= 0.");
    }
    const double denominator = magnitude > minimum_magnetic_field_magnitude
                                   ? magnitude
                                   : minimum_magnetic_field_magnitude;
    return field_component_or_zero(magnetic_field, indices, component) / denominator;
}

/**
 * Return B_i / max(B^2, floor) at one logical grid point.
 *
 * If the floor is zero or negative, non-positive B^2 is treated as a fatal field
 * singularity because Parker drift is undefined at a magnetic null.
 */
template <typename MagneticFieldType>
KOKKOS_INLINE_FUNCTION
double magnetic_field_over_magnitude_squared_component(
    const MagneticFieldType& magnetic_field,
    const typename MagneticFieldType::index_array_type& indices,
    const int component,
    const double minimum_magnetic_field_squared) {
    double magnetic_field_squared = 0.0;
    for (int magnetic_component = 0;
         magnetic_component < MagneticFieldType::component_dim;
         ++magnetic_component) {
        const double value =
            magnetic_field(indices, magnetic_component);
        magnetic_field_squared += value * value;
    }

    if (magnetic_field_squared <= 0.0 &&
        minimum_magnetic_field_squared <= 0.0) {
        Kokkos::abort(
            "ParkerCoefficient: curl(B/B^2) is undefined where B^2 <= 0.");
    }

    const double denominator =
        magnetic_field_squared > minimum_magnetic_field_squared
            ? magnetic_field_squared
            : minimum_magnetic_field_squared;
    return field_component_or_zero(magnetic_field, indices, component) /
           denominator;
}

/**
 * Return h_component * B_component / B^2 at one logical grid point.
 */
template <typename MagneticFieldType>
KOKKOS_INLINE_FUNCTION
double metric_weighted_magnetic_field_over_magnitude_squared_component(
    const MagneticFieldType& magnetic_field,
    const typename MagneticFieldType::index_array_type& indices,
    const int component,
    const double minimum_magnetic_field_squared) {
    const auto q = magnetic_field.grid.coordinate_tuple(indices,
                                                        magnetic_field.centerings);
    return magnetic_field.grid.h(component, q) *
           magnetic_field_over_magnitude_squared_component(
               magnetic_field, indices, component,
               minimum_magnetic_field_squared);
}

/**
 * Return a centered coordinate derivative of h_component * B_component / B^2.
 *
 * Missing dimensions are interpreted as ignorable 2.5D coordinates, so their
 * derivatives are zero.
 */
template <typename MagneticFieldType>
KOKKOS_INLINE_FUNCTION
double centered_metric_weighted_derivative(
    const MagneticFieldType& magnetic_field,
    const typename MagneticFieldType::index_array_type& indices,
    const int derivative_dim,
    const int component,
    const double minimum_magnetic_field_squared) {
    if (derivative_dim >= MagneticFieldType::space_dim ||
        magnetic_field.grid.extent(derivative_dim) <= 1) {
        return 0.0;
    }

    auto lower_indices = indices;
    auto upper_indices = indices;
    lower_indices[derivative_dim] -= 1;
    upper_indices[derivative_dim] += 1;

    if (!field_indices_inside_domain(magnetic_field, lower_indices,
                                     GridDomain::GhostedDomain) ||
        !field_indices_inside_domain(magnetic_field, upper_indices,
                                     GridDomain::GhostedDomain)) {
        Kokkos::abort(
            "ParkerCoefficient: curl(B/B^2) requires one valid ghost layer.");
    }

    const double lower_coordinate = magnetic_field.grid.coordinate(
        derivative_dim,
        lower_indices[derivative_dim],
        magnetic_field.centering(derivative_dim));
    const double upper_coordinate = magnetic_field.grid.coordinate(
        derivative_dim,
        upper_indices[derivative_dim],
        magnetic_field.centering(derivative_dim));
    const double coordinate_width = upper_coordinate - lower_coordinate;
    if (coordinate_width == 0.0) {
        Kokkos::abort(
            "ParkerCoefficient: curl(B/B^2) requires non-degenerate coordinates.");
    }

    const double lower_value =
        metric_weighted_magnetic_field_over_magnitude_squared_component(
            magnetic_field, lower_indices, component,
            minimum_magnetic_field_squared);
    const double upper_value =
        metric_weighted_magnetic_field_over_magnitude_squared_component(
            magnetic_field, upper_indices, component,
            minimum_magnetic_field_squared);
    return (upper_value - lower_value) / coordinate_width;
}

/**
 * Return one physical component of curl(B / B^2) in an orthogonal coordinate basis.
 *
 * The formula assumes that vector components are physical orthonormal-basis
 * components, for example (B_r, B_theta, B_phi) in spherical coordinates.
 */
template <typename MagneticFieldType>
KOKKOS_INLINE_FUNCTION
double magnetic_field_over_magnitude_squared_curl_component(
    const MagneticFieldType& magnetic_field,
    const typename MagneticFieldType::index_array_type& indices,
    const int curl_component,
    const double minimum_magnetic_field_squared = 0.0) {
    const int derivative_dim_a = (curl_component + 1) % 3;
    const int derivative_dim_b = (curl_component + 2) % 3;
    const int vector_component_a = derivative_dim_a;
    const int vector_component_b = derivative_dim_b;

    const double derivative_a = centered_metric_weighted_derivative(
        magnetic_field, indices, derivative_dim_a, vector_component_b,
        minimum_magnetic_field_squared);
    const double derivative_b = centered_metric_weighted_derivative(
        magnetic_field, indices, derivative_dim_b, vector_component_a,
        minimum_magnetic_field_squared);

    const auto q = magnetic_field.grid.coordinate_tuple(indices,
                                                        magnetic_field.centerings);
    const double h_a = magnetic_field.grid.h(vector_component_a, q);
    const double h_b = magnetic_field.grid.h(vector_component_b, q);
    const double denominator = h_a * h_b;
    if (denominator == 0.0) {
        Kokkos::abort(
            "ParkerCoefficient: curl(B/B^2) encountered a metric singularity.");
    }

    return (derivative_a - derivative_b) / denominator;
}

/**
 * Validate field compatibility for curl(B / B^2) precomputation.
 */
template <typename MagneticFieldType, typename CurlFieldType>
void validate_magnetic_field_curl_inputs(
    const MagneticFieldType& magnetic_field,
    const CurlFieldType& curl_field,
    const double minimum_magnetic_field_squared) {
    static_assert(std::is_same_v<typename MagneticFieldType::grid_type,
                                 typename CurlFieldType::grid_type>,
                  "Parker drift curl fields must use the same grid type.");
    static_assert(MagneticFieldType::component_dim <= 3,
                  "Magnetic field curl currently supports at most three vector components.");
    static_assert(CurlFieldType::component_dim <= 3,
                  "Magnetic drift curl output currently supports at most three components.");

    if (!std::isfinite(minimum_magnetic_field_squared) ||
        minimum_magnetic_field_squared < 0.0) {
        throw std::runtime_error(
            "ParkerCoefficient: magnetic field floor must be finite and non-negative.");
    }
    if (magnetic_field.point_count() != curl_field.point_count()) {
        throw std::runtime_error(
            "ParkerCoefficient: magnetic field and curl field storage sizes differ.");
    }

    for (int dim = 0; dim < MagneticFieldType::space_dim; ++dim) {
        if (magnetic_field.centering(dim) != curl_field.centering(dim)) {
            throw std::runtime_error(
                "ParkerCoefficient: magnetic field and curl field centerings differ.");
        }
        if (magnetic_field.lower_index(dim, GridDomain::GhostedDomain) !=
                curl_field.lower_index(dim, GridDomain::GhostedDomain) ||
            magnetic_field.upper_index(dim, GridDomain::GhostedDomain) !=
                curl_field.upper_index(dim, GridDomain::GhostedDomain)) {
            throw std::runtime_error(
                "ParkerCoefficient: magnetic field and curl field index domains differ.");
        }
        if (magnetic_field.grid.extent(dim) > 1 &&
            (magnetic_field.grid.ghost_extent(dim, LowerBoundary) < 1 ||
             magnetic_field.grid.ghost_extent(dim, UpperBoundary) < 1)) {
            throw std::runtime_error(
                "ParkerCoefficient: curl(B/B^2) requires at least one ghost layer.");
        }
    }
}

/**
 * Fill a preallocated field with curl(B / B^2).
 *
 * The magnetic field ghost cells must be current before this routine is called.
 * The output ghost cells are filled after physical-domain curl values are computed.
 */
template <typename MagneticFieldType, typename CurlFieldType>
void fill_magnetic_field_over_magnitude_squared_curl(
    const MagneticFieldType& magnetic_field,
    CurlFieldType& curl_field,
    const double minimum_magnetic_field_squared = 0.0,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    validate_magnetic_field_curl_inputs(
        magnetic_field, curl_field, minimum_magnetic_field_squared);

    const MagneticFieldType local_magnetic_field = magnetic_field;
    const CurlFieldType local_curl_field = curl_field;
    auto curl_data = curl_field.data;
    Kokkos::parallel_for(
        "ParkerCoefficient::fill_magnetic_field_over_magnitude_squared_curl",
        Kokkos::RangePolicy<typename CurlFieldType::execution_space>(
            0, curl_field.point_count()),
        KOKKOS_LAMBDA(const typename CurlFieldType::size_type point_index) {
            const auto indices = local_curl_field.logical_indices(point_index);
            if (!field_indices_inside_domain(local_curl_field, indices,
                                             GridDomain::PhysicalDomain)) {
                return;
            }

            for (int component = 0;
                 component < CurlFieldType::component_dim;
                 ++component) {
                curl_data(point_index, component) =
                    magnetic_field_over_magnitude_squared_curl_component(
                        local_magnetic_field, indices, component,
                        minimum_magnetic_field_squared);
            }
        });
    Kokkos::fence(
        "ParkerCoefficient::fill_magnetic_field_over_magnitude_squared_curl");

    curl_field.fill_ghost_cells(ghost_fill_mode);
}

/**
 * Allocate and fill a vector field containing curl(B / B^2).
 */
template <typename MagneticFieldType>
auto make_magnetic_field_over_magnitude_squared_curl_field(
    const MagneticFieldType& magnetic_field,
    const char* label = "magnetic_drift_curl",
    const double minimum_magnetic_field_squared = 0.0,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    using grid_type = typename MagneticFieldType::grid_type;
    using curl_field_type =
        VectorField<grid_type, grid_type::vec_dim,
                    typename MagneticFieldType::layout_type>;

    curl_field_type curl_field(magnetic_field.grid, label,
                               magnetic_field.centerings);
    fill_magnetic_field_over_magnitude_squared_curl(
        magnetic_field, curl_field, minimum_magnetic_field_squared,
        ghost_fill_mode);
    return curl_field;
}

/**
 * Validate same-grid, cell-centered storage for diffusion tensor divergence.
 */
template <typename ReferenceFieldType, typename CheckedFieldType>
void validate_diffusion_tensor_storage_compatibility(
    const ReferenceFieldType& reference_field,
    const CheckedFieldType& checked_field,
    const char* operation_name) {
    static_assert(std::is_same_v<typename ReferenceFieldType::grid_type,
                                 typename CheckedFieldType::grid_type>,
                  "Parker diffusion tensor divergence fields must use the same grid type.");

    if (reference_field.point_count() != checked_field.point_count()) {
        throw std::runtime_error(
            std::string("ParkerCoefficient: point-count mismatch in ") +
            operation_name + ".");
    }

    for (int dim = 0; dim < ReferenceFieldType::space_dim; ++dim) {
        if (reference_field.centering(dim) != GridCentering::CellCentered ||
            checked_field.centering(dim) != GridCentering::CellCentered ||
            reference_field.lower_index(dim, GridDomain::GhostedDomain) !=
                checked_field.lower_index(dim, GridDomain::GhostedDomain) ||
            reference_field.upper_index(dim, GridDomain::GhostedDomain) !=
                checked_field.upper_index(dim, GridDomain::GhostedDomain)) {
            throw std::runtime_error(
                std::string("ParkerCoefficient: storage-domain mismatch in ") +
                operation_name + ".");
        }
    }
}

/**
 * Validate common requirements for diffusion tensor divergence precomputation.
 */
template <typename MagneticFieldType, typename OutputFieldType>
void validate_diffusion_tensor_divergence_inputs(
    const MagneticFieldType& magnetic_field,
    const OutputFieldType& output_field,
    const double minimum_magnetic_field_magnitude,
    const char* operation_name) {
    static_assert(MagneticFieldType::component_dim <= 3,
                  "Parker diffusion tensor divergence supports at most three B components.");
    static_assert(OutputFieldType::component_dim <= 3,
                  "Parker diffusion tensor divergence supports at most three output components.");
    static_assert(MagneticFieldType::space_dim <= 3,
                  "Parker diffusion tensor divergence supports SpaceDim <= 3.");

    if (!std::isfinite(minimum_magnetic_field_magnitude) ||
        minimum_magnetic_field_magnitude < 0.0) {
        throw std::runtime_error(
            "ParkerCoefficient: magnetic-field magnitude floor must be finite and non-negative.");
    }

    validate_diffusion_tensor_storage_compatibility(magnetic_field, output_field,
                                                   operation_name);
    for (int dim = 0; dim < MagneticFieldType::space_dim; ++dim) {
        if (magnetic_field.grid.extent(dim) > 1 &&
            (magnetic_field.grid.ghost_extent(dim, LowerBoundary) < 1 ||
             magnetic_field.grid.ghost_extent(dim, UpperBoundary) < 1)) {
            throw std::runtime_error(
                std::string("ParkerCoefficient: ") + operation_name +
                " requires at least one ghost layer.");
        }
    }
}

/**
 * Return one gamma-independent diffusion tensor basis component.
 */
template <
    typename MagneticFieldType,
    typename KappaParallelFieldType,
    typename KappaPerpendicularFieldType
>
KOKKOS_INLINE_FUNCTION
double diffusion_tensor_basis_component(
    const MagneticFieldType& magnetic_field,
    const KappaParallelFieldType& kappa_parallel_gamma_one,
    const KappaPerpendicularFieldType& kappa_perpendicular_gamma_one,
    const typename MagneticFieldType::index_array_type& indices,
    const int output_component,
    const int derivative_component,
    const DiffusionTensorDivergenceBasis basis,
    const double perpendicular_parallel_ratio,
    const double minimum_magnetic_field_magnitude) {
    const double b_i = magnetic_direction_component(
        magnetic_field, indices, output_component,
        minimum_magnetic_field_magnitude);
    const double b_j = magnetic_direction_component(
        magnetic_field, indices, derivative_component,
        minimum_magnetic_field_magnitude);
    const double projection = b_i * b_j;
    const double identity = output_component == derivative_component ? 1.0 : 0.0;

    if (basis == DiffusionTensorDivergenceBasis::ParallelProjection) {
        return kappa_parallel_gamma_one(indices) * projection;
    }
    if (basis == DiffusionTensorDivergenceBasis::PerpendicularProjection) {
        return kappa_perpendicular_gamma_one(indices) * (identity - projection);
    }

    return kappa_parallel_gamma_one(indices) *
           (perpendicular_parallel_ratio * identity +
            (1.0 - perpendicular_parallel_ratio) * projection);
}

/**
 * Return J / h_j times one diffusion tensor basis component.
 */
template <
    typename MagneticFieldType,
    typename KappaParallelFieldType,
    typename KappaPerpendicularFieldType
>
KOKKOS_INLINE_FUNCTION
double metric_weighted_diffusion_tensor_basis_component(
    const MagneticFieldType& magnetic_field,
    const KappaParallelFieldType& kappa_parallel_gamma_one,
    const KappaPerpendicularFieldType& kappa_perpendicular_gamma_one,
    const typename MagneticFieldType::index_array_type& indices,
    const int output_component,
    const int derivative_component,
    const DiffusionTensorDivergenceBasis basis,
    const double perpendicular_parallel_ratio,
    const double minimum_magnetic_field_magnitude) {
    const auto q = magnetic_field.grid.coordinate_tuple(indices,
                                                        magnetic_field.centerings);
    const double h_j = magnetic_field.grid.h(derivative_component, q);
    if (h_j == 0.0) {
        Kokkos::abort(
            "ParkerCoefficient: diffusion tensor divergence encountered a metric singularity.");
    }
    return vector_calculus_jacobian(magnetic_field.grid, q) / h_j *
           diffusion_tensor_basis_component(
               magnetic_field, kappa_parallel_gamma_one,
               kappa_perpendicular_gamma_one, indices, output_component,
               derivative_component, basis, perpendicular_parallel_ratio,
               minimum_magnetic_field_magnitude);
}

/**
 * Return a centered derivative of J / h_j * kappa_ij.
 */
template <
    typename MagneticFieldType,
    typename KappaParallelFieldType,
    typename KappaPerpendicularFieldType
>
KOKKOS_INLINE_FUNCTION
double centered_metric_weighted_diffusion_tensor_basis_derivative(
    const MagneticFieldType& magnetic_field,
    const KappaParallelFieldType& kappa_parallel_gamma_one,
    const KappaPerpendicularFieldType& kappa_perpendicular_gamma_one,
    const typename MagneticFieldType::index_array_type& indices,
    const int output_component,
    const int derivative_dim,
    const DiffusionTensorDivergenceBasis basis,
    const double perpendicular_parallel_ratio,
    const double minimum_magnetic_field_magnitude) {
    if (derivative_dim >= MagneticFieldType::space_dim ||
        magnetic_field.grid.extent(derivative_dim) <= 1) {
        return 0.0;
    }

    auto lower_indices = indices;
    auto upper_indices = indices;
    lower_indices[derivative_dim] -= 1;
    upper_indices[derivative_dim] += 1;

    if (!field_indices_inside_domain(magnetic_field, lower_indices,
                                     GridDomain::GhostedDomain) ||
        !field_indices_inside_domain(magnetic_field, upper_indices,
                                     GridDomain::GhostedDomain)) {
        Kokkos::abort(
            "ParkerCoefficient: diffusion tensor divergence requires one valid ghost layer.");
    }

    const double lower_coordinate = magnetic_field.grid.coordinate(
        derivative_dim, lower_indices[derivative_dim],
        magnetic_field.centering(derivative_dim));
    const double center_coordinate = magnetic_field.grid.coordinate(
        derivative_dim, indices[derivative_dim],
        magnetic_field.centering(derivative_dim));
    const double upper_coordinate = magnetic_field.grid.coordinate(
        derivative_dim, upper_indices[derivative_dim],
        magnetic_field.centering(derivative_dim));

    return centered_three_point_derivative(
        lower_coordinate, center_coordinate, upper_coordinate,
        metric_weighted_diffusion_tensor_basis_component(
            magnetic_field, kappa_parallel_gamma_one,
            kappa_perpendicular_gamma_one, lower_indices, output_component,
            derivative_dim, basis, perpendicular_parallel_ratio,
            minimum_magnetic_field_magnitude),
        metric_weighted_diffusion_tensor_basis_component(
            magnetic_field, kappa_parallel_gamma_one,
            kappa_perpendicular_gamma_one, indices, output_component,
            derivative_dim, basis, perpendicular_parallel_ratio,
            minimum_magnetic_field_magnitude),
        metric_weighted_diffusion_tensor_basis_component(
            magnetic_field, kappa_parallel_gamma_one,
            kappa_perpendicular_gamma_one, upper_indices, output_component,
            derivative_dim, basis, perpendicular_parallel_ratio,
            minimum_magnetic_field_magnitude));
}

/**
 * Return a centered coordinate derivative of one Lame coefficient.
 */
template <typename FieldType>
KOKKOS_INLINE_FUNCTION
double centered_lame_derivative(
    const FieldType& field,
    const typename FieldType::index_array_type& indices,
    const int lame_component,
    const int derivative_dim) {
    if (derivative_dim >= FieldType::space_dim ||
        field.grid.extent(derivative_dim) <= 1) {
        return 0.0;
    }

    auto lower_indices = indices;
    auto upper_indices = indices;
    lower_indices[derivative_dim] -= 1;
    upper_indices[derivative_dim] += 1;

    if (!field_indices_inside_domain(field, lower_indices,
                                     GridDomain::GhostedDomain) ||
        !field_indices_inside_domain(field, upper_indices,
                                     GridDomain::GhostedDomain)) {
        Kokkos::abort(
            "ParkerCoefficient: Lame derivative requires one valid ghost layer.");
    }

    const double lower_coordinate = field.grid.coordinate(
        derivative_dim, lower_indices[derivative_dim],
        field.centering(derivative_dim));
    const double center_coordinate = field.grid.coordinate(
        derivative_dim, indices[derivative_dim],
        field.centering(derivative_dim));
    const double upper_coordinate = field.grid.coordinate(
        derivative_dim, upper_indices[derivative_dim],
        field.centering(derivative_dim));
    const auto lower_q = field.grid.coordinate_tuple(lower_indices, field.centerings);
    const auto center_q = field.grid.coordinate_tuple(indices, field.centerings);
    const auto upper_q = field.grid.coordinate_tuple(upper_indices, field.centerings);

    return centered_three_point_derivative(
        lower_coordinate, center_coordinate, upper_coordinate,
        field.grid.h(lame_component, lower_q),
        field.grid.h(lame_component, center_q),
        field.grid.h(lame_component, upper_q));
}

/**
 * Return one physical component of div(kappa_ij) for one basis tensor.
 */
template <
    typename MagneticFieldType,
    typename KappaParallelFieldType,
    typename KappaPerpendicularFieldType
>
KOKKOS_INLINE_FUNCTION
double diffusion_tensor_divergence_basis_component(
    const MagneticFieldType& magnetic_field,
    const KappaParallelFieldType& kappa_parallel_gamma_one,
    const KappaPerpendicularFieldType& kappa_perpendicular_gamma_one,
    const typename MagneticFieldType::index_array_type& indices,
    const int output_component,
    const DiffusionTensorDivergenceBasis basis,
    const double perpendicular_parallel_ratio,
    const double minimum_magnetic_field_magnitude) {
    const int basis_dim = vector_calculus_basis_dim(magnetic_field.grid);
    if (output_component >= basis_dim) {
        return 0.0;
    }

    const auto q = magnetic_field.grid.coordinate_tuple(indices,
                                                        magnetic_field.centerings);
    const double jacobian = vector_calculus_jacobian(magnetic_field.grid, q);
    const double h_i = magnetic_field.grid.h(output_component, q);
    if (jacobian == 0.0 || h_i == 0.0) {
        Kokkos::abort(
            "ParkerCoefficient: diffusion tensor divergence encountered a metric singularity.");
    }

    double result = 0.0;
    for (int derivative_dim = 0;
         derivative_dim < MagneticFieldType::space_dim;
         ++derivative_dim) {
        result += centered_metric_weighted_diffusion_tensor_basis_derivative(
            magnetic_field, kappa_parallel_gamma_one,
            kappa_perpendicular_gamma_one, indices, output_component,
            derivative_dim, basis, perpendicular_parallel_ratio,
            minimum_magnetic_field_magnitude);
    }
    result /= jacobian;

    for (int component = 0; component < basis_dim; ++component) {
        if (component == output_component) {
            continue;
        }

        const double h_component = magnetic_field.grid.h(component, q);
        if (h_component == 0.0) {
            Kokkos::abort(
                "ParkerCoefficient: diffusion tensor divergence encountered a metric singularity.");
        }

        if (component < MagneticFieldType::space_dim) {
            const double dh_i_dq_component =
                centered_lame_derivative(magnetic_field, indices,
                                         output_component, component);
            const double tensor_i_component =
                diffusion_tensor_basis_component(
                    magnetic_field, kappa_parallel_gamma_one,
                    kappa_perpendicular_gamma_one, indices, output_component,
                    component, basis, perpendicular_parallel_ratio,
                    minimum_magnetic_field_magnitude);
            result += tensor_i_component * dh_i_dq_component /
                      (h_i * h_component);
        }

        if (output_component < MagneticFieldType::space_dim) {
            const double dh_component_dq_i =
                centered_lame_derivative(magnetic_field, indices, component,
                                         output_component);
            const double tensor_component_component =
                diffusion_tensor_basis_component(
                    magnetic_field, kappa_parallel_gamma_one,
                    kappa_perpendicular_gamma_one, indices, component,
                    component, basis, perpendicular_parallel_ratio,
                    minimum_magnetic_field_magnitude);
            result -= tensor_component_component * dh_component_dq_i /
                      (h_i * h_component);
        }
    }

    return result;
}

/**
 * Fill one vector field with a selected gamma-independent div(kappa_ij) basis.
 */
template <
    typename MagneticFieldType,
    typename KappaParallelFieldType,
    typename KappaPerpendicularFieldType,
    typename OutputFieldType
>
void fill_diffusion_tensor_divergence_basis_field(
    const MagneticFieldType& magnetic_field,
    const KappaParallelFieldType& kappa_parallel_gamma_one,
    const KappaPerpendicularFieldType& kappa_perpendicular_gamma_one,
    OutputFieldType& output_field,
    const DiffusionTensorDivergenceBasis basis,
    const double perpendicular_parallel_ratio = 0.0,
    const double minimum_magnetic_field_magnitude = 0.0,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    static_assert(KappaParallelFieldType::component_dim == 1,
                  "Parker diffusion tensor divergence requires scalar kappa_parallel.");
    static_assert(KappaPerpendicularFieldType::component_dim == 1,
                  "Parker diffusion tensor divergence requires scalar kappa_perpendicular.");

    if (!std::isfinite(perpendicular_parallel_ratio) ||
        perpendicular_parallel_ratio < 0.0) {
        throw std::runtime_error(
            "ParkerCoefficient: perpendicular/parallel ratio must be finite and non-negative.");
    }
    if (basis == DiffusionTensorDivergenceBasis::ConstantParallelRatio &&
        perpendicular_parallel_ratio <= 0.0) {
        throw std::runtime_error(
            "ParkerCoefficient: constant-ratio diffusion divergence requires a positive ratio.");
    }

    validate_diffusion_tensor_divergence_inputs(
        magnetic_field, output_field, minimum_magnetic_field_magnitude,
        "fill_diffusion_tensor_divergence_basis_field");
    validate_diffusion_tensor_storage_compatibility(
        magnetic_field, kappa_parallel_gamma_one,
        "fill_diffusion_tensor_divergence_basis_field");
    validate_diffusion_tensor_storage_compatibility(
        magnetic_field, kappa_perpendicular_gamma_one,
        "fill_diffusion_tensor_divergence_basis_field");

    const MagneticFieldType local_magnetic_field = magnetic_field;
    const KappaParallelFieldType local_kappa_parallel = kappa_parallel_gamma_one;
    const KappaPerpendicularFieldType local_kappa_perpendicular =
        kappa_perpendicular_gamma_one;
    const OutputFieldType local_output = output_field;
    auto output_data = output_field.data;
    Kokkos::parallel_for(
        "ParkerCoefficient::fill_diffusion_tensor_divergence_basis_field",
        Kokkos::RangePolicy<typename OutputFieldType::execution_space>(
            0, output_field.point_count()),
        KOKKOS_LAMBDA(const typename OutputFieldType::size_type point_index) {
            const auto indices = local_output.logical_indices(point_index);
            if (!field_indices_inside_domain(local_output, indices,
                                             GridDomain::PhysicalDomain)) {
                return;
            }

            for (int component = 0;
                 component < OutputFieldType::component_dim;
                 ++component) {
                output_data(point_index, component) =
                    diffusion_tensor_divergence_basis_component(
                        local_magnetic_field, local_kappa_parallel,
                        local_kappa_perpendicular, indices, component, basis,
                        perpendicular_parallel_ratio,
                        minimum_magnetic_field_magnitude);
            }
        });
    Kokkos::fence(
        "ParkerCoefficient::fill_diffusion_tensor_divergence_basis_field");

    output_field.fill_ghost_cells(ghost_fill_mode);
}

/**
 * Fill the two README-formula divergence basis fields.
 */
template <
    typename MagneticFieldType,
    typename KappaParallelFieldType,
    typename KappaPerpendicularFieldType,
    typename ParallelDivergenceFieldType,
    typename PerpendicularDivergenceFieldType
>
void fill_readme_diffusion_tensor_divergence_basis_fields(
    const MagneticFieldType& magnetic_field,
    const KappaParallelFieldType& kappa_parallel_gamma_one,
    const KappaPerpendicularFieldType& kappa_perpendicular_gamma_one,
    ParallelDivergenceFieldType& parallel_divergence_basis,
    PerpendicularDivergenceFieldType& perpendicular_divergence_basis,
    const double minimum_magnetic_field_magnitude = 0.0,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    fill_diffusion_tensor_divergence_basis_field(
        magnetic_field, kappa_parallel_gamma_one, kappa_perpendicular_gamma_one,
        parallel_divergence_basis,
        DiffusionTensorDivergenceBasis::ParallelProjection, 0.0,
        minimum_magnetic_field_magnitude, ghost_fill_mode);
    fill_diffusion_tensor_divergence_basis_field(
        magnetic_field, kappa_parallel_gamma_one, kappa_perpendicular_gamma_one,
        perpendicular_divergence_basis,
        DiffusionTensorDivergenceBasis::PerpendicularProjection, 0.0,
        minimum_magnetic_field_magnitude, ghost_fill_mode);
}

/**
 * Fill the single constant-ratio divergence basis field.
 */
template <
    typename MagneticFieldType,
    typename KappaParallelFieldType,
    typename OutputFieldType
>
void fill_constant_ratio_diffusion_tensor_divergence_basis_field(
    const MagneticFieldType& magnetic_field,
    const KappaParallelFieldType& kappa_parallel_gamma_one,
    OutputFieldType& constant_ratio_divergence_basis,
    const double perpendicular_parallel_ratio,
    const double minimum_magnetic_field_magnitude = 0.0,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    fill_diffusion_tensor_divergence_basis_field(
        magnetic_field, kappa_parallel_gamma_one, kappa_parallel_gamma_one,
        constant_ratio_divergence_basis,
        DiffusionTensorDivergenceBasis::ConstantParallelRatio,
        perpendicular_parallel_ratio, minimum_magnetic_field_magnitude,
        ghost_fill_mode);
}

/**
 * Allocate and fill README-formula divergence basis fields.
 */
template <
    typename MagneticFieldType,
    typename KappaParallelFieldType,
    typename KappaPerpendicularFieldType
>
auto make_readme_diffusion_tensor_divergence_basis_fields(
    const MagneticFieldType& magnetic_field,
    const KappaParallelFieldType& kappa_parallel_gamma_one,
    const KappaPerpendicularFieldType& kappa_perpendicular_gamma_one,
    const char* label = "diffusion_tensor_divergence",
    const double minimum_magnetic_field_magnitude = 0.0,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    using grid_type = typename MagneticFieldType::grid_type;
    DiffusionTensorDivergenceFieldSet<grid_type> fields(
        magnetic_field.grid, label,
        LegencyModel::PerpendicularDiffusionModelKind::ReadmeFormula);
    fill_readme_diffusion_tensor_divergence_basis_fields(
        magnetic_field, kappa_parallel_gamma_one,
        kappa_perpendicular_gamma_one, fields.parallel_basis,
        fields.perpendicular_basis, minimum_magnetic_field_magnitude,
        ghost_fill_mode);
    return fields;
}

/**
 * Allocate and fill a constant-ratio divergence basis field.
 */
template <typename MagneticFieldType, typename KappaParallelFieldType>
auto make_constant_ratio_diffusion_tensor_divergence_basis_field(
    const MagneticFieldType& magnetic_field,
    const KappaParallelFieldType& kappa_parallel_gamma_one,
    const double perpendicular_parallel_ratio,
    const char* label = "diffusion_tensor_divergence",
    const double minimum_magnetic_field_magnitude = 0.0,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    using grid_type = typename MagneticFieldType::grid_type;
    using output_field_type =
        VectorField<grid_type, grid_type::vec_dim,
                    typename MagneticFieldType::layout_type>;
    output_field_type output_field(magnetic_field.grid, label,
                                   GridCentering::CellCentered);
    fill_constant_ratio_diffusion_tensor_divergence_basis_field(
        magnetic_field, kappa_parallel_gamma_one, output_field,
        perpendicular_parallel_ratio, minimum_magnetic_field_magnitude,
        ghost_fill_mode);
    return output_field;
}

/**
 * Allocate and fill the divergence fields required by the selected legacy branch.
 */
template <typename MagneticFieldType, typename DiffusionFieldSetType>
auto make_diffusion_tensor_divergence_fields(
    const MagneticFieldType& magnetic_field,
    const DiffusionFieldSetType& diffusion_fields,
    const LegencyModel::DiffusionModelParameters& parameters,
    const char* label = "diffusion_tensor_divergence",
    const double minimum_magnetic_field_magnitude = 0.0,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    parameters.validate();
    using grid_type = typename MagneticFieldType::grid_type;
    DiffusionTensorDivergenceFieldSet<grid_type> fields(
        magnetic_field.grid, label, parameters.perpendicular_model);

    if (parameters.perpendicular_model ==
        LegencyModel::PerpendicularDiffusionModelKind::ConstantParallelRatio) {
        fill_constant_ratio_diffusion_tensor_divergence_basis_field(
            magnetic_field, diffusion_fields.kappa_parallel_gamma_one,
            fields.constant_ratio_basis,
            parameters.perpendicular_parallel_ratio,
            minimum_magnetic_field_magnitude, ghost_fill_mode);
    } else {
        fill_readme_diffusion_tensor_divergence_basis_fields(
            magnetic_field, diffusion_fields.kappa_parallel_gamma_one,
            diffusion_fields.kappa_perpendicular_gamma_one,
            fields.parallel_basis, fields.perpendicular_basis,
            minimum_magnetic_field_magnitude, ghost_fill_mode);
    }
    return fields;
}

/**
 * Allocate and fill the coefficient fields needed by deterministic Parker stepping.
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
    const char* label = "parker_transport_coefficients",
    const LegencyModel::DiffusionModelParameters& parameters =
        LegencyModel::DiffusionModelParameters{},
    const double minimum_magnetic_field_squared_for_drift = 0.0,
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    static_assert(std::is_same_v<typename MagneticFieldType::grid_type,
                                 typename SolarWindVelocityFieldType::grid_type>,
                  "Parker transport coefficients require B and Vsw on the same grid.");
    using grid_type = typename MagneticFieldType::grid_type;
    using coefficient_field_set_type =
        TransportCoefficientFieldSet<grid_type,
                                     typename MagneticFieldType::layout_type>;

    parameters.validate();
    const std::string prefix(label);
    auto magnetic_drift_curl =
        make_magnetic_field_over_magnitude_squared_curl_field(
            magnetic_field, (prefix + "_magnetic_drift_curl").c_str(),
            minimum_magnetic_field_squared_for_drift, ghost_fill_mode);
    auto diffusion_fields =
        LegencyModel::make_gamma_independent_diffusion_fields(
            magnetic_field, solar_wind_velocity, turbulence_properties,
            particle_properties, (prefix + "_legacy_diffusion").c_str(),
            parameters, ghost_fill_mode);
    auto diffusion_tensor_divergence =
        make_diffusion_tensor_divergence_fields(
            magnetic_field, diffusion_fields, parameters,
            (prefix + "_diffusion_tensor_divergence").c_str(),
            parameters.minimum_magnetic_field_magnitude, ghost_fill_mode);
    auto solar_wind_divergence =
        FieldOperator::make_divergence_field(
            solar_wind_velocity, (prefix + "_solar_wind_divergence").c_str(),
            ghost_fill_mode);

    return coefficient_field_set_type(
        magnetic_drift_curl, diffusion_tensor_divergence,
        diffusion_fields.kappa_parallel_gamma_one,
        diffusion_fields.kappa_perpendicular_gamma_one,
        solar_wind_divergence, parameters);
}

/**
 * Apply gamma factors to interpolated README-formula divergence basis values.
 */
template <std::size_t VecDim>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<double, VecDim> apply_readme_diffusion_tensor_divergence_gamma(
    const Kokkos::Array<double, VecDim>& parallel_basis,
    const Kokkos::Array<double, VecDim>& perpendicular_basis,
    const double gamma) {
    if (gamma <= 0.0) {
        Kokkos::abort(
            "ParkerCoefficient: diffusion tensor divergence gamma correction requires gamma > 0.");
    }

    Kokkos::Array<double, VecDim> result{};
    const double parallel_factor = Kokkos::pow(gamma, 1.0 / 3.0);
    const double perpendicular_factor = Kokkos::pow(gamma, 1.0 / 9.0);
    for (std::size_t component = 0; component < VecDim; ++component) {
        result[component] = parallel_factor * parallel_basis[component] +
                            perpendicular_factor * perpendicular_basis[component];
    }
    return result;
}

/**
 * Apply gamma factors to an interpolated constant-ratio divergence basis value.
 */
template <std::size_t VecDim>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<double, VecDim> apply_constant_ratio_diffusion_tensor_divergence_gamma(
    const Kokkos::Array<double, VecDim>& constant_ratio_basis,
    const double gamma) {
    if (gamma <= 0.0) {
        Kokkos::abort(
            "ParkerCoefficient: diffusion tensor divergence gamma correction requires gamma > 0.");
    }

    Kokkos::Array<double, VecDim> result{};
    const double gamma_factor = Kokkos::pow(gamma, 1.0 / 3.0);
    for (std::size_t component = 0; component < VecDim; ++component) {
        result[component] = gamma_factor * constant_ratio_basis[component];
    }
    return result;
}

} // namespace ParkerCoefficient
