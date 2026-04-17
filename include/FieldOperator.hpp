#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Field.hpp"

/**
 * Generic field operators for orthogonal coordinate grids.
 */
namespace FieldOperator {

/**
 * Behavior used when a vector direction is requested at zero magnitude.
 */
enum class ZeroMagnitudePolicy : int {
    ZeroVector = 0,
    Abort = 1
};

/**
 * Return whether logical field indices lie inside a selected domain.
 */
template <typename FieldType>
KOKKOS_INLINE_FUNCTION
bool indices_inside_domain(const FieldType& field,
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
double component_or_zero(const FieldType& field,
                         const typename FieldType::index_array_type& indices,
                         const int component) {
    if (component < 0 || component >= FieldType::component_dim) {
        return 0.0;
    }
    return field(indices, component);
}

/**
 * Return the number of physical basis vector components used by vector calculus.
 */
template <typename GridType>
KOKKOS_INLINE_FUNCTION
int vector_calculus_basis_dim(const GridType&) {
    return GridType::vec_dim < 3 ? GridType::vec_dim : 3;
}

/**
 * Return the vector-calculus metric Jacobian used by divergence operators.
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
        Kokkos::abort("FieldOperator: centered derivative requires distinct coordinates.");
    }

    return (-upper_width / (lower_width * total_width)) * lower_value +
           ((upper_width - lower_width) / (lower_width * upper_width)) * center_value +
           (lower_width / (upper_width * total_width)) * upper_value;
}

/**
 * Return a centered coordinate derivative of one field component.
 */
template <typename FieldType>
KOKKOS_INLINE_FUNCTION
double centered_component_derivative(
    const FieldType& field,
    const typename FieldType::index_array_type& indices,
    const int derivative_dim,
    const int component) {
    if (derivative_dim >= FieldType::space_dim ||
        field.grid.extent(derivative_dim) <= 1) {
        return 0.0;
    }

    auto lower_indices = indices;
    auto upper_indices = indices;
    lower_indices[derivative_dim] -= 1;
    upper_indices[derivative_dim] += 1;

    if (!indices_inside_domain(field, lower_indices, GridDomain::GhostedDomain) ||
        !indices_inside_domain(field, upper_indices, GridDomain::GhostedDomain)) {
        Kokkos::abort("FieldOperator: derivative requires one valid ghost layer.");
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

    return centered_three_point_derivative(
        lower_coordinate, center_coordinate, upper_coordinate,
        component_or_zero(field, lower_indices, component),
        component_or_zero(field, indices, component),
        component_or_zero(field, upper_indices, component));
}

/**
 * Return one physical component of the gradient of a scalar field.
 */
template <typename ScalarFieldType>
KOKKOS_INLINE_FUNCTION
double gradient_component(const ScalarFieldType& scalar_field,
                          const typename ScalarFieldType::index_array_type& indices,
                          const int component) {
    if (component >= ScalarFieldType::space_dim) {
        return 0.0;
    }

    const auto q = scalar_field.grid.coordinate_tuple(indices,
                                                      scalar_field.centerings);
    const double h = scalar_field.grid.h(component, q);
    if (h == 0.0) {
        Kokkos::abort("FieldOperator: gradient encountered a metric singularity.");
    }

    return centered_component_derivative(scalar_field, indices, component, 0) / h;
}

/**
 * Return the pointwise magnitude of a vector field.
 */
template <typename VectorFieldType>
KOKKOS_INLINE_FUNCTION
double magnitude_value(const VectorFieldType& vector_field,
                       const typename VectorFieldType::index_array_type& indices) {
    double magnitude_squared = 0.0;
    for (int component = 0; component < VectorFieldType::component_dim; ++component) {
        const double value = vector_field(indices, component);
        magnitude_squared += value * value;
    }
    return Kokkos::sqrt(magnitude_squared);
}

/**
 * Return the pointwise dot product of two vector fields.
 */
template <typename LeftFieldType, typename RightFieldType>
KOKKOS_INLINE_FUNCTION
double dot_product_value(const LeftFieldType& left_field,
                         const RightFieldType& right_field,
                         const typename LeftFieldType::index_array_type& indices) {
    double result = 0.0;
    for (int component = 0; component < LeftFieldType::component_dim; ++component) {
        result += left_field(indices, component) * right_field(indices, component);
    }
    return result;
}

/**
 * Return J / h_i * A_i at one point for divergence calculations.
 */
template <typename VectorFieldType>
KOKKOS_INLINE_FUNCTION
double divergence_weighted_component(
    const VectorFieldType& vector_field,
    const typename VectorFieldType::index_array_type& indices,
    const int component) {
    const auto q = vector_field.grid.coordinate_tuple(indices,
                                                      vector_field.centerings);
    const double h = vector_field.grid.h(component, q);
    if (h == 0.0) {
        Kokkos::abort("FieldOperator: divergence encountered a metric singularity.");
    }
    return vector_calculus_jacobian(vector_field.grid, q) /
           h * component_or_zero(vector_field, indices, component);
}

/**
 * Return a centered derivative of J / h_i * A_i.
 */
template <typename VectorFieldType>
KOKKOS_INLINE_FUNCTION
double centered_divergence_weighted_derivative(
    const VectorFieldType& vector_field,
    const typename VectorFieldType::index_array_type& indices,
    const int derivative_dim) {
    if (derivative_dim >= VectorFieldType::space_dim ||
        vector_field.grid.extent(derivative_dim) <= 1) {
        return 0.0;
    }

    auto lower_indices = indices;
    auto upper_indices = indices;
    lower_indices[derivative_dim] -= 1;
    upper_indices[derivative_dim] += 1;

    if (!indices_inside_domain(vector_field, lower_indices, GridDomain::GhostedDomain) ||
        !indices_inside_domain(vector_field, upper_indices, GridDomain::GhostedDomain)) {
        Kokkos::abort("FieldOperator: divergence requires one valid ghost layer.");
    }

    const double lower_coordinate = vector_field.grid.coordinate(
        derivative_dim, lower_indices[derivative_dim],
        vector_field.centering(derivative_dim));
    const double center_coordinate = vector_field.grid.coordinate(
        derivative_dim, indices[derivative_dim],
        vector_field.centering(derivative_dim));
    const double upper_coordinate = vector_field.grid.coordinate(
        derivative_dim, upper_indices[derivative_dim],
        vector_field.centering(derivative_dim));

    return centered_three_point_derivative(
        lower_coordinate, center_coordinate, upper_coordinate,
        divergence_weighted_component(vector_field, lower_indices, derivative_dim),
        divergence_weighted_component(vector_field, indices, derivative_dim),
        divergence_weighted_component(vector_field, upper_indices, derivative_dim));
}

/**
 * Return the divergence of a vector field in an orthogonal coordinate basis.
 */
template <typename VectorFieldType>
KOKKOS_INLINE_FUNCTION
double divergence_value(const VectorFieldType& vector_field,
                        const typename VectorFieldType::index_array_type& indices) {
    const auto q = vector_field.grid.coordinate_tuple(indices,
                                                      vector_field.centerings);
    const double jacobian = vector_calculus_jacobian(vector_field.grid, q);
    if (jacobian == 0.0) {
        Kokkos::abort("FieldOperator: divergence encountered a metric singularity.");
    }

    double result = 0.0;
    for (int dim = 0; dim < VectorFieldType::space_dim; ++dim) {
        result += centered_divergence_weighted_derivative(vector_field, indices, dim);
    }
    return result / jacobian;
}

/**
 * Return h_component * A_component at one logical grid point.
 */
template <typename VectorFieldType>
KOKKOS_INLINE_FUNCTION
double metric_weighted_component(
    const VectorFieldType& vector_field,
    const typename VectorFieldType::index_array_type& indices,
    const int component) {
    const auto q = vector_field.grid.coordinate_tuple(indices,
                                                      vector_field.centerings);
    return vector_field.grid.h(component, q) *
           component_or_zero(vector_field, indices, component);
}

/**
 * Return a centered derivative of h_component * A_component.
 */
template <typename VectorFieldType>
KOKKOS_INLINE_FUNCTION
double centered_metric_weighted_derivative(
    const VectorFieldType& vector_field,
    const typename VectorFieldType::index_array_type& indices,
    const int derivative_dim,
    const int component) {
    if (derivative_dim >= VectorFieldType::space_dim ||
        vector_field.grid.extent(derivative_dim) <= 1) {
        return 0.0;
    }

    auto lower_indices = indices;
    auto upper_indices = indices;
    lower_indices[derivative_dim] -= 1;
    upper_indices[derivative_dim] += 1;

    if (!indices_inside_domain(vector_field, lower_indices, GridDomain::GhostedDomain) ||
        !indices_inside_domain(vector_field, upper_indices, GridDomain::GhostedDomain)) {
        Kokkos::abort("FieldOperator: curl requires one valid ghost layer.");
    }

    const double lower_coordinate = vector_field.grid.coordinate(
        derivative_dim, lower_indices[derivative_dim],
        vector_field.centering(derivative_dim));
    const double center_coordinate = vector_field.grid.coordinate(
        derivative_dim, indices[derivative_dim],
        vector_field.centering(derivative_dim));
    const double upper_coordinate = vector_field.grid.coordinate(
        derivative_dim, upper_indices[derivative_dim],
        vector_field.centering(derivative_dim));

    return centered_three_point_derivative(
        lower_coordinate, center_coordinate, upper_coordinate,
        metric_weighted_component(vector_field, lower_indices, component),
        metric_weighted_component(vector_field, indices, component),
        metric_weighted_component(vector_field, upper_indices, component));
}

/**
 * Return one physical component of the curl of a vector field.
 */
template <typename VectorFieldType>
KOKKOS_INLINE_FUNCTION
double curl_component(const VectorFieldType& vector_field,
                      const typename VectorFieldType::index_array_type& indices,
                      const int component) {
    const int derivative_dim_a = (component + 1) % 3;
    const int derivative_dim_b = (component + 2) % 3;
    const int vector_component_a = derivative_dim_a;
    const int vector_component_b = derivative_dim_b;

    const double derivative_a = centered_metric_weighted_derivative(
        vector_field, indices, derivative_dim_a, vector_component_b);
    const double derivative_b = centered_metric_weighted_derivative(
        vector_field, indices, derivative_dim_b, vector_component_a);

    const auto q = vector_field.grid.coordinate_tuple(indices,
                                                      vector_field.centerings);
    const double h_a = vector_field.grid.h(vector_component_a, q);
    const double h_b = vector_field.grid.h(vector_component_b, q);
    const double denominator = h_a * h_b;
    if (denominator == 0.0) {
        Kokkos::abort("FieldOperator: curl encountered a metric singularity.");
    }

    return (derivative_a - derivative_b) / denominator;
}

/**
 * Validate that two same-grid fields have compatible storage domains.
 */
template <typename InputFieldType, typename OutputFieldType>
void validate_unary_operation_fields(const InputFieldType& input_field,
                                     const OutputFieldType& output_field,
                                     const char* operation_name) {
    static_assert(std::is_same_v<typename InputFieldType::grid_type,
                                 typename OutputFieldType::grid_type>,
                  "Field operators require the same grid type.");

    if (input_field.point_count() != output_field.point_count()) {
        throw std::runtime_error(
            std::string("FieldOperator: point-count mismatch in ") + operation_name + ".");
    }

    for (int dim = 0; dim < InputFieldType::space_dim; ++dim) {
        if (input_field.centering(dim) != output_field.centering(dim) ||
            input_field.lower_index(dim, GridDomain::GhostedDomain) !=
                output_field.lower_index(dim, GridDomain::GhostedDomain) ||
            input_field.upper_index(dim, GridDomain::GhostedDomain) !=
                output_field.upper_index(dim, GridDomain::GhostedDomain)) {
            throw std::runtime_error(
                std::string("FieldOperator: storage-domain mismatch in ") +
                operation_name + ".");
        }
    }
}

/**
 * Validate that two input fields and one output field have compatible domains.
 */
template <typename LeftFieldType, typename RightFieldType, typename OutputFieldType>
void validate_binary_operation_fields(const LeftFieldType& left_field,
                                      const RightFieldType& right_field,
                                      const OutputFieldType& output_field,
                                      const char* operation_name) {
    validate_unary_operation_fields(left_field, right_field, operation_name);
    validate_unary_operation_fields(left_field, output_field, operation_name);
}

/**
 * Validate derivative support for centered-difference field operators.
 */
template <typename FieldType>
void validate_derivative_field(const FieldType& field,
                               const char* operation_name) {
    static_assert(FieldType::space_dim <= 3,
                  "FieldOperator derivative operators currently support SpaceDim <= 3.");
    for (int dim = 0; dim < FieldType::space_dim; ++dim) {
        if (field.grid.extent(dim) > 1 &&
            (field.grid.ghost_extent(dim, LowerBoundary) < 1 ||
             field.grid.ghost_extent(dim, UpperBoundary) < 1)) {
            throw std::runtime_error(
                std::string("FieldOperator: ") + operation_name +
                " requires at least one ghost layer.");
        }
    }
}

/**
 * Fill a vector field with the gradient of a scalar field.
 */
template <typename ScalarFieldType, typename GradientFieldType>
void fill_gradient(const ScalarFieldType& scalar_field,
                   GradientFieldType& gradient_field,
                   const FieldGhostFillMode ghost_fill_mode =
                       FieldGhostFillMode::PeriodicOrClamped) {
    static_assert(ScalarFieldType::component_dim == 1,
                  "FieldOperator::fill_gradient requires a scalar input field.");
    static_assert(GradientFieldType::component_dim <= 3,
                  "FieldOperator::fill_gradient supports at most three output components.");
    validate_unary_operation_fields(scalar_field, gradient_field, "fill_gradient");
    validate_derivative_field(scalar_field, "fill_gradient");

    const ScalarFieldType local_scalar = scalar_field;
    const GradientFieldType local_gradient = gradient_field;
    auto gradient_data = gradient_field.data;
    Kokkos::parallel_for(
        "FieldOperator::fill_gradient",
        Kokkos::RangePolicy<typename GradientFieldType::execution_space>(
            0, gradient_field.point_count()),
        KOKKOS_LAMBDA(const typename GradientFieldType::size_type point_index) {
            const auto indices = local_gradient.logical_indices(point_index);
            if (!indices_inside_domain(local_gradient, indices,
                                       GridDomain::PhysicalDomain)) {
                return;
            }
            for (int component = 0;
                 component < GradientFieldType::component_dim;
                 ++component) {
                gradient_data(point_index, component) =
                    gradient_component(local_scalar, indices, component);
            }
        });
    Kokkos::fence("FieldOperator::fill_gradient");
    gradient_field.fill_ghost_cells(ghost_fill_mode);
}

/**
 * Allocate and fill a vector field with the gradient of a scalar field.
 */
template <typename ScalarFieldType>
auto make_gradient_field(const ScalarFieldType& scalar_field,
                         const char* label = "gradient",
                         const FieldGhostFillMode ghost_fill_mode =
                             FieldGhostFillMode::PeriodicOrClamped) {
    using grid_type = typename ScalarFieldType::grid_type;
    using gradient_field_type =
        VectorField<grid_type, grid_type::vec_dim,
                    typename ScalarFieldType::layout_type>;
    gradient_field_type gradient_field(scalar_field.grid, label,
                                       scalar_field.centerings);
    fill_gradient(scalar_field, gradient_field, ghost_fill_mode);
    return gradient_field;
}

/**
 * Fill a scalar field with the divergence of a vector field.
 */
template <typename VectorFieldType, typename DivergenceFieldType>
void fill_divergence(const VectorFieldType& vector_field,
                     DivergenceFieldType& divergence_field,
                     const FieldGhostFillMode ghost_fill_mode =
                         FieldGhostFillMode::PeriodicOrClamped) {
    static_assert(DivergenceFieldType::component_dim == 1,
                  "FieldOperator::fill_divergence requires a scalar output field.");
    static_assert(VectorFieldType::component_dim <= 3,
                  "FieldOperator::fill_divergence supports at most three input components.");
    validate_unary_operation_fields(vector_field, divergence_field, "fill_divergence");
    validate_derivative_field(vector_field, "fill_divergence");

    const VectorFieldType local_vector = vector_field;
    const DivergenceFieldType local_divergence = divergence_field;
    auto divergence_data = divergence_field.data;
    Kokkos::parallel_for(
        "FieldOperator::fill_divergence",
        Kokkos::RangePolicy<typename DivergenceFieldType::execution_space>(
            0, divergence_field.point_count()),
        KOKKOS_LAMBDA(const typename DivergenceFieldType::size_type point_index) {
            const auto indices = local_divergence.logical_indices(point_index);
            if (!indices_inside_domain(local_divergence, indices,
                                       GridDomain::PhysicalDomain)) {
                return;
            }
            divergence_data(point_index, 0) = divergence_value(local_vector, indices);
        });
    Kokkos::fence("FieldOperator::fill_divergence");
    divergence_field.fill_ghost_cells(ghost_fill_mode);
}

/**
 * Allocate and fill a scalar field with the divergence of a vector field.
 */
template <typename VectorFieldType>
auto make_divergence_field(const VectorFieldType& vector_field,
                           const char* label = "divergence",
                           const FieldGhostFillMode ghost_fill_mode =
                               FieldGhostFillMode::PeriodicOrClamped) {
    using divergence_field_type =
        ScalarField<typename VectorFieldType::grid_type,
                    typename VectorFieldType::layout_type>;
    divergence_field_type divergence_field(vector_field.grid, label,
                                           vector_field.centerings);
    fill_divergence(vector_field, divergence_field, ghost_fill_mode);
    return divergence_field;
}

/**
 * Fill a vector field with the curl of a vector field.
 */
template <typename VectorFieldType, typename CurlFieldType>
void fill_curl(const VectorFieldType& vector_field,
               CurlFieldType& curl_field,
               const FieldGhostFillMode ghost_fill_mode =
                   FieldGhostFillMode::PeriodicOrClamped) {
    static_assert(VectorFieldType::component_dim <= 3,
                  "FieldOperator::fill_curl supports at most three input components.");
    static_assert(CurlFieldType::component_dim <= 3,
                  "FieldOperator::fill_curl supports at most three output components.");
    validate_unary_operation_fields(vector_field, curl_field, "fill_curl");
    validate_derivative_field(vector_field, "fill_curl");

    const VectorFieldType local_vector = vector_field;
    const CurlFieldType local_curl = curl_field;
    auto curl_data = curl_field.data;
    Kokkos::parallel_for(
        "FieldOperator::fill_curl",
        Kokkos::RangePolicy<typename CurlFieldType::execution_space>(
            0, curl_field.point_count()),
        KOKKOS_LAMBDA(const typename CurlFieldType::size_type point_index) {
            const auto indices = local_curl.logical_indices(point_index);
            if (!indices_inside_domain(local_curl, indices,
                                       GridDomain::PhysicalDomain)) {
                return;
            }
            for (int component = 0; component < CurlFieldType::component_dim; ++component) {
                curl_data(point_index, component) =
                    curl_component(local_vector, indices, component);
            }
        });
    Kokkos::fence("FieldOperator::fill_curl");
    curl_field.fill_ghost_cells(ghost_fill_mode);
}

/**
 * Allocate and fill a vector field with the curl of a vector field.
 */
template <typename VectorFieldType>
auto make_curl_field(const VectorFieldType& vector_field,
                     const char* label = "curl",
                     const FieldGhostFillMode ghost_fill_mode =
                         FieldGhostFillMode::PeriodicOrClamped) {
    using grid_type = typename VectorFieldType::grid_type;
    using curl_field_type =
        VectorField<grid_type, grid_type::vec_dim,
                    typename VectorFieldType::layout_type>;
    curl_field_type curl_field(vector_field.grid, label, vector_field.centerings);
    fill_curl(vector_field, curl_field, ghost_fill_mode);
    return curl_field;
}

/**
 * Fill a scalar field with vector magnitudes.
 */
template <typename VectorFieldType, typename MagnitudeFieldType>
void fill_magnitude(const VectorFieldType& vector_field,
                    MagnitudeFieldType& magnitude_field) {
    static_assert(MagnitudeFieldType::component_dim == 1,
                  "FieldOperator::fill_magnitude requires a scalar output field.");
    validate_unary_operation_fields(vector_field, magnitude_field, "fill_magnitude");

    const VectorFieldType local_vector = vector_field;
    const MagnitudeFieldType local_magnitude = magnitude_field;
    auto magnitude_data = magnitude_field.data;
    Kokkos::parallel_for(
        "FieldOperator::fill_magnitude",
        Kokkos::RangePolicy<typename MagnitudeFieldType::execution_space>(
            0, magnitude_field.point_count()),
        KOKKOS_LAMBDA(const typename MagnitudeFieldType::size_type point_index) {
            const auto indices = local_magnitude.logical_indices(point_index);
            magnitude_data(point_index, 0) = magnitude_value(local_vector, indices);
        });
    Kokkos::fence("FieldOperator::fill_magnitude");
}

/**
 * Allocate and fill a scalar field with vector magnitudes.
 */
template <typename VectorFieldType>
auto make_magnitude_field(const VectorFieldType& vector_field,
                          const char* label = "magnitude") {
    using magnitude_field_type =
        ScalarField<typename VectorFieldType::grid_type,
                    typename VectorFieldType::layout_type>;
    magnitude_field_type magnitude_field(vector_field.grid, label,
                                         vector_field.centerings);
    fill_magnitude(vector_field, magnitude_field);
    return magnitude_field;
}

/**
 * Fill a field with alpha times an input field.
 */
template <typename InputFieldType, typename OutputFieldType>
void fill_scaled(const InputFieldType& input_field,
                 OutputFieldType& output_field,
                 const double alpha) {
    static_assert(InputFieldType::component_dim == OutputFieldType::component_dim,
                  "FieldOperator::fill_scaled requires matching component dimensions.");
    if (!std::isfinite(alpha)) {
        throw std::runtime_error("FieldOperator: fill_scaled requires a finite scalar.");
    }
    validate_unary_operation_fields(input_field, output_field, "fill_scaled");

    const InputFieldType local_input = input_field;
    const OutputFieldType local_output = output_field;
    auto output_data = output_field.data;
    Kokkos::parallel_for(
        "FieldOperator::fill_scaled",
        Kokkos::RangePolicy<typename OutputFieldType::execution_space>(
            0, output_field.point_count()),
        KOKKOS_LAMBDA(const typename OutputFieldType::size_type point_index) {
            const auto indices = local_output.logical_indices(point_index);
            for (int component = 0; component < OutputFieldType::component_dim; ++component) {
                output_data(point_index, component) =
                    alpha * local_input(indices, component);
            }
        });
    Kokkos::fence("FieldOperator::fill_scaled");
}

/**
 * Scale a field in place.
 */
template <typename FieldType>
void scale_in_place(FieldType& field, const double alpha) {
    fill_scaled(field, field, alpha);
}

/**
 * Allocate and fill a field with alpha times an input field.
 */
template <typename InputFieldType>
auto make_scaled_field(const InputFieldType& input_field,
                       const double alpha,
                       const char* label = "scaled") {
    using output_field_type =
        Field<typename InputFieldType::grid_type,
              InputFieldType::component_dim,
              typename InputFieldType::layout_type>;
    output_field_type output_field(input_field.grid, label, input_field.centerings);
    fill_scaled(input_field, output_field, alpha);
    return output_field;
}

/**
 * Fill a scalar field with the dot product of two vector fields.
 */
template <typename LeftFieldType, typename RightFieldType, typename DotFieldType>
void fill_dot_product(const LeftFieldType& left_field,
                      const RightFieldType& right_field,
                      DotFieldType& dot_field) {
    static_assert(LeftFieldType::component_dim == RightFieldType::component_dim,
                  "FieldOperator::fill_dot_product requires matching vector dimensions.");
    static_assert(DotFieldType::component_dim == 1,
                  "FieldOperator::fill_dot_product requires a scalar output field.");
    validate_binary_operation_fields(left_field, right_field, dot_field,
                                     "fill_dot_product");

    const LeftFieldType local_left = left_field;
    const RightFieldType local_right = right_field;
    const DotFieldType local_dot = dot_field;
    auto dot_data = dot_field.data;
    Kokkos::parallel_for(
        "FieldOperator::fill_dot_product",
        Kokkos::RangePolicy<typename DotFieldType::execution_space>(
            0, dot_field.point_count()),
        KOKKOS_LAMBDA(const typename DotFieldType::size_type point_index) {
            const auto indices = local_dot.logical_indices(point_index);
            dot_data(point_index, 0) =
                dot_product_value(local_left, local_right, indices);
        });
    Kokkos::fence("FieldOperator::fill_dot_product");
}

/**
 * Allocate and fill a scalar field with the dot product of two vector fields.
 */
template <typename LeftFieldType, typename RightFieldType>
auto make_dot_product_field(const LeftFieldType& left_field,
                            const RightFieldType& right_field,
                            const char* label = "dot_product") {
    using dot_field_type =
        ScalarField<typename LeftFieldType::grid_type,
                    typename LeftFieldType::layout_type>;
    dot_field_type dot_field(left_field.grid, label, left_field.centerings);
    fill_dot_product(left_field, right_field, dot_field);
    return dot_field;
}

/**
 * Fill a vector field with normalized directions of an input vector field.
 */
template <typename VectorFieldType, typename DirectionFieldType>
void fill_direction(const VectorFieldType& vector_field,
                    DirectionFieldType& direction_field,
                    const double minimum_magnitude = 0.0,
                    const ZeroMagnitudePolicy zero_policy =
                        ZeroMagnitudePolicy::ZeroVector) {
    static_assert(VectorFieldType::component_dim == DirectionFieldType::component_dim,
                  "FieldOperator::fill_direction requires matching vector dimensions.");
    if (!std::isfinite(minimum_magnitude) || minimum_magnitude < 0.0) {
        throw std::runtime_error(
            "FieldOperator: fill_direction requires a finite non-negative floor.");
    }
    validate_unary_operation_fields(vector_field, direction_field, "fill_direction");

    const VectorFieldType local_vector = vector_field;
    const DirectionFieldType local_direction = direction_field;
    auto direction_data = direction_field.data;
    Kokkos::parallel_for(
        "FieldOperator::fill_direction",
        Kokkos::RangePolicy<typename DirectionFieldType::execution_space>(
            0, direction_field.point_count()),
        KOKKOS_LAMBDA(const typename DirectionFieldType::size_type point_index) {
            const auto indices = local_direction.logical_indices(point_index);
            const double magnitude = magnitude_value(local_vector, indices);
            if (magnitude <= minimum_magnitude) {
                if (zero_policy == ZeroMagnitudePolicy::Abort) {
                    Kokkos::abort(
                        "FieldOperator: direction is undefined at zero magnitude.");
                }
                for (int component = 0;
                     component < DirectionFieldType::component_dim;
                     ++component) {
                    direction_data(point_index, component) = 0.0;
                }
                return;
            }

            const double inverse_magnitude = 1.0 / magnitude;
            for (int component = 0;
                 component < DirectionFieldType::component_dim;
                 ++component) {
                direction_data(point_index, component) =
                    local_vector(indices, component) * inverse_magnitude;
            }
        });
    Kokkos::fence("FieldOperator::fill_direction");
}

/**
 * Allocate and fill a vector field with normalized directions.
 */
template <typename VectorFieldType>
auto make_direction_field(const VectorFieldType& vector_field,
                          const char* label = "direction",
                          const double minimum_magnitude = 0.0,
                          const ZeroMagnitudePolicy zero_policy =
                              ZeroMagnitudePolicy::ZeroVector) {
    using direction_field_type =
        Field<typename VectorFieldType::grid_type,
              VectorFieldType::component_dim,
              typename VectorFieldType::layout_type>;
    direction_field_type direction_field(vector_field.grid, label,
                                         vector_field.centerings);
    fill_direction(vector_field, direction_field, minimum_magnitude, zero_policy);
    return direction_field;
}

} // namespace FieldOperator
