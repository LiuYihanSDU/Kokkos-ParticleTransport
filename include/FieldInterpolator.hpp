#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Field.hpp"

/**
 * Field interpolation utilities for time interpolation and particle-position sampling.
 */
namespace FieldInterpolator {

/**
 * Controls whether time interpolation may extrapolate outside the two input times.
 */
enum class TimeInterpolationBoundsPolicy : int {
    RequireBracketed = 0,
    AllowExtrapolation = 1
};

/**
 * Boundary handling used before position-space interpolation.
 */
enum class InterpolationBoundaryPolicy : int {
    RequireInside = 0,
    WrapPeriodic = 1,
    ClampToDomain = 2
};

/**
 * Validate coordinate-grid requirements for position-space interpolation.
 */
template <typename GridType>
void validate_coordinate_grid_for_interpolation() {
    static_assert(requires(const GridType& input_grid) {
                      input_grid.coordinate(0, 0, GridCentering::CellCentered);
                      input_grid.grid_index(0, 0.0, GridCentering::CellCentered);
                      input_grid.contains_coordinate(0, 0.0, GridDomain::PhysicalDomain);
                      input_grid.wrap_coordinate(0, 0.0);
                  },
                  "FieldInterpolator requires a coordinate grid with coordinate, "
                  "grid_index, contains_coordinate, and wrap_coordinate.");
    static_assert(GridType::space_dim <= 3,
                  "FieldInterpolator currently supports SpaceDim <= 3.");
}

/**
 * Validate that fields have compatible storage domains and component counts.
 */
template <typename LeftFieldType, typename RightFieldType>
void validate_compatible_fields(const LeftFieldType& left_field,
                                const RightFieldType& right_field,
                                const char* operation_name) {
    static_assert(std::is_same_v<typename LeftFieldType::grid_type,
                                 typename RightFieldType::grid_type>,
                  "Field interpolation requires matching grid types.");
    static_assert(LeftFieldType::component_dim == RightFieldType::component_dim,
                  "Field interpolation requires matching component dimensions.");

    if (left_field.point_count() != right_field.point_count()) {
        throw std::runtime_error(
            std::string("FieldInterpolator: point-count mismatch in ") +
            operation_name + ".");
    }

    for (int dim = 0; dim < LeftFieldType::space_dim; ++dim) {
        if (left_field.centering(dim) != right_field.centering(dim) ||
            left_field.lower_index(dim, GridDomain::GhostedDomain) !=
                right_field.lower_index(dim, GridDomain::GhostedDomain) ||
            left_field.upper_index(dim, GridDomain::GhostedDomain) !=
                right_field.upper_index(dim, GridDomain::GhostedDomain)) {
            throw std::runtime_error(
                std::string("FieldInterpolator: storage-domain mismatch in ") +
                operation_name + ".");
        }
    }
}

/**
 * Validate finite, distinct time stamps.
 */
inline void validate_time_pair(const double lower_time,
                               const double upper_time,
                               const char* operation_name) {
    if (!std::isfinite(lower_time) || !std::isfinite(upper_time) ||
        lower_time == upper_time) {
        throw std::runtime_error(
            std::string("FieldInterpolator: ") + operation_name +
            " requires finite distinct times.");
    }
}

/**
 * Return the linear interpolation weight for a target time.
 */
inline double time_interpolation_weight(
    const double lower_time,
    const double upper_time,
    const double target_time,
    const TimeInterpolationBoundsPolicy bounds_policy =
        TimeInterpolationBoundsPolicy::RequireBracketed) {
    validate_time_pair(lower_time, upper_time, "time_interpolation_weight");
    if (!std::isfinite(target_time)) {
        throw std::runtime_error(
            "FieldInterpolator: time interpolation requires a finite target time.");
    }

    const double weight = (target_time - lower_time) / (upper_time - lower_time);
    if (bounds_policy == TimeInterpolationBoundsPolicy::RequireBracketed) {
        const double tolerance = 64.0 * std::abs(std::numeric_limits<double>::epsilon());
        if (weight < -tolerance || weight > 1.0 + tolerance) {
            throw std::runtime_error(
                "FieldInterpolator: target time is outside the interpolation bracket.");
        }
    }
    return weight;
}

/**
 * Fill an output field with a linear interpolation between two time-adjacent fields.
 */
template <typename LowerFieldType, typename UpperFieldType, typename OutputFieldType>
void fill_time_interpolated_field(
    const LowerFieldType& lower_field,
    const UpperFieldType& upper_field,
    OutputFieldType& output_field,
    const double lower_time,
    const double upper_time,
    const double target_time,
    const TimeInterpolationBoundsPolicy bounds_policy =
        TimeInterpolationBoundsPolicy::RequireBracketed) {
    validate_compatible_fields(lower_field, upper_field,
                               "fill_time_interpolated_field");
    validate_compatible_fields(lower_field, output_field,
                               "fill_time_interpolated_field");

    const double upper_weight =
        time_interpolation_weight(lower_time, upper_time, target_time, bounds_policy);
    const double lower_weight = 1.0 - upper_weight;

    const auto lower_data = lower_field.data;
    const auto upper_data = upper_field.data;
    auto output_data = output_field.data;
    const auto count = output_field.point_count();
    Kokkos::parallel_for(
        "FieldInterpolator::fill_time_interpolated_field",
        Kokkos::RangePolicy<typename OutputFieldType::execution_space>(0, count),
        KOKKOS_LAMBDA(const typename OutputFieldType::size_type point_index) {
            for (int component = 0; component < OutputFieldType::component_dim;
                 ++component) {
                output_data(point_index, component) =
                    lower_weight * lower_data(point_index, component) +
                    upper_weight * upper_data(point_index, component);
            }
        });
    Kokkos::fence("FieldInterpolator::fill_time_interpolated_field");
}

/**
 * Allocate and fill a field with a linear time interpolation.
 */
template <typename LowerFieldType, typename UpperFieldType>
auto make_time_interpolated_field(
    const LowerFieldType& lower_field,
    const UpperFieldType& upper_field,
    const double lower_time,
    const double upper_time,
    const double target_time,
    const char* label = "time_interpolated_field",
    const TimeInterpolationBoundsPolicy bounds_policy =
        TimeInterpolationBoundsPolicy::RequireBracketed) {
    using output_field_type =
        Field<typename LowerFieldType::grid_type,
              LowerFieldType::component_dim,
              typename LowerFieldType::layout_type>;
    output_field_type output_field(lower_field.grid, label, lower_field.centerings);
    fill_time_interpolated_field(lower_field, upper_field, output_field,
                                 lower_time, upper_time, target_time,
                                 bounds_policy);
    return output_field;
}

/**
 * Fill an output field with the finite-difference time derivative between two fields.
 */
template <typename LowerFieldType, typename UpperFieldType, typename OutputFieldType>
void fill_time_derivative_field(const LowerFieldType& lower_field,
                                const UpperFieldType& upper_field,
                                OutputFieldType& output_field,
                                const double lower_time,
                                const double upper_time) {
    validate_compatible_fields(lower_field, upper_field,
                               "fill_time_derivative_field");
    validate_compatible_fields(lower_field, output_field,
                               "fill_time_derivative_field");
    validate_time_pair(lower_time, upper_time, "fill_time_derivative_field");

    const double inverse_dt = 1.0 / (upper_time - lower_time);
    const auto lower_data = lower_field.data;
    const auto upper_data = upper_field.data;
    auto output_data = output_field.data;
    const auto count = output_field.point_count();
    Kokkos::parallel_for(
        "FieldInterpolator::fill_time_derivative_field",
        Kokkos::RangePolicy<typename OutputFieldType::execution_space>(0, count),
        KOKKOS_LAMBDA(const typename OutputFieldType::size_type point_index) {
            for (int component = 0; component < OutputFieldType::component_dim;
                 ++component) {
                output_data(point_index, component) =
                    (upper_data(point_index, component) -
                     lower_data(point_index, component)) *
                    inverse_dt;
            }
        });
    Kokkos::fence("FieldInterpolator::fill_time_derivative_field");
}

/**
 * Allocate and fill a field with a finite-difference time derivative.
 */
template <typename LowerFieldType, typename UpperFieldType>
auto make_time_derivative_field(const LowerFieldType& lower_field,
                                const UpperFieldType& upper_field,
                                const double lower_time,
                                const double upper_time,
                                const char* label = "time_derivative_field") {
    using output_field_type =
        Field<typename LowerFieldType::grid_type,
              LowerFieldType::component_dim,
              typename LowerFieldType::layout_type>;
    output_field_type output_field(lower_field.grid, label, lower_field.centerings);
    fill_time_derivative_field(lower_field, upper_field, output_field,
                               lower_time, upper_time);
    return output_field;
}

/**
 * Device-callable multilinear field interpolator for particle-position sampling.
 */
template <typename FieldType>
struct LinearPositionInterpolator {
    using field_type = FieldType;
    using grid_type = typename field_type::grid_type;
    using index_array_type = typename field_type::index_array_type;
    using coordinate_array_type = typename field_type::coordinate_array_type;
    using centering_array_type = typename field_type::centering_array_type;
    using value_array_type = Kokkos::Array<double, field_type::component_dim>;
    using field_accessor_type = typename field_type::RandomAccessAccessor;

    static constexpr int space_dim = field_type::space_dim;
    static constexpr int component_dim = field_type::component_dim;

    grid_type grid;
    centering_array_type centerings{};
    field_accessor_type values;
    GridDomain domain{GridDomain::GhostedDomain};
    InterpolationBoundaryPolicy boundary_policy{
        InterpolationBoundaryPolicy::WrapPeriodic};

    LinearPositionInterpolator() = default;

    /**
     * Build a device-callable interpolator from a field.
     */
    LinearPositionInterpolator(
        const field_type& input_field,
        const GridDomain input_domain = GridDomain::GhostedDomain,
        const InterpolationBoundaryPolicy input_boundary_policy =
            InterpolationBoundaryPolicy::WrapPeriodic)
        : grid(input_field.grid),
          centerings(input_field.centerings),
          values(input_field.random_access()),
          domain(input_domain),
          boundary_policy(input_boundary_policy) {
        validate_coordinate_grid_for_interpolation<grid_type>();
    }

    /**
     * Return true if the position can be sampled under the configured boundary policy.
     */
    KOKKOS_INLINE_FUNCTION
    bool contains_position(const coordinate_array_type& position) const {
        coordinate_array_type adjusted_position{};
        for (int dim = 0; dim < space_dim; ++dim) {
            if (!prepare_coordinate(dim, position[dim], adjusted_position[dim])) {
                return false;
            }
        }
        return true;
    }

    /**
     * Interpolate one field component at a particle position.
     */
    KOKKOS_INLINE_FUNCTION
    double sample_component(const coordinate_array_type& position,
                            const int component) const {
        if (component < 0 || component >= component_dim) {
            Kokkos::abort(
                "FieldInterpolator: interpolated component is out of range.");
        }

        coordinate_array_type adjusted_position{};
        for (int dim = 0; dim < space_dim; ++dim) {
            if (!prepare_coordinate(dim, position[dim], adjusted_position[dim])) {
                Kokkos::abort(
                    "FieldInterpolator: position lies outside the interpolation domain.");
            }
        }

        index_array_type lower_indices{};
        index_array_type upper_indices{};
        coordinate_array_type upper_weights{};
        for (int dim = 0; dim < space_dim; ++dim) {
            compute_axis_stencil(dim, adjusted_position[dim],
                                 lower_indices[dim], upper_indices[dim],
                                 upper_weights[dim]);
        }

        double result = 0.0;
        constexpr int corner_count = 1 << space_dim;
        for (int corner = 0; corner < corner_count; ++corner) {
            index_array_type indices{};
            double weight = 1.0;
            for (int dim = 0; dim < space_dim; ++dim) {
                const bool use_upper = ((corner >> dim) & 1) != 0;
                indices[dim] = use_upper ? upper_indices[dim] : lower_indices[dim];
                weight *= use_upper ? upper_weights[dim] : (1.0 - upper_weights[dim]);
            }
            result += weight * values(indices, component);
        }
        return result;
    }

    /**
     * Interpolate all field components at a particle position.
     */
    KOKKOS_INLINE_FUNCTION
    value_array_type sample(const coordinate_array_type& position) const {
        value_array_type result{};

        coordinate_array_type adjusted_position{};
        for (int dim = 0; dim < space_dim; ++dim) {
            if (!prepare_coordinate(dim, position[dim], adjusted_position[dim])) {
                Kokkos::abort(
                    "FieldInterpolator: position lies outside the interpolation domain.");
            }
        }

        index_array_type lower_indices{};
        index_array_type upper_indices{};
        coordinate_array_type upper_weights{};
        for (int dim = 0; dim < space_dim; ++dim) {
            compute_axis_stencil(dim, adjusted_position[dim],
                                 lower_indices[dim], upper_indices[dim],
                                 upper_weights[dim]);
        }

        constexpr int corner_count = 1 << space_dim;
        for (int corner = 0; corner < corner_count; ++corner) {
            index_array_type indices{};
            double weight = 1.0;
            for (int dim = 0; dim < space_dim; ++dim) {
                const bool use_upper = ((corner >> dim) & 1) != 0;
                indices[dim] = use_upper ? upper_indices[dim] : lower_indices[dim];
                weight *= use_upper ? upper_weights[dim] : (1.0 - upper_weights[dim]);
            }
            for (int component = 0; component < component_dim; ++component) {
                result[component] += weight * values(indices, component);
            }
        }
        return result;
    }

    /**
     * Compatibility alias for sampling all field components at once.
     */
    KOKKOS_INLINE_FUNCTION
    value_array_type sample_all_components(
        const coordinate_array_type& position) const {
        return sample(position);
    }

    /**
     * Interpolate all field components at a particle position by component.
     */
    KOKKOS_INLINE_FUNCTION
    value_array_type sample_by_component(
        const coordinate_array_type& position) const {
        value_array_type result{};
        for (int component = 0; component < component_dim; ++component) {
            result[component] = sample_component(position, component);
        }
        return result;
    }

    /**
     * Interpolate one field component at a particle position.
     */
    KOKKOS_INLINE_FUNCTION
    double operator()(const coordinate_array_type& position,
                      const int component) const {
        return sample_component(position, component);
    }

    /**
     * Interpolate all field components at a particle position.
     */
    KOKKOS_INLINE_FUNCTION
    value_array_type operator()(const coordinate_array_type& position) const {
        return sample(position);
    }

private:
    /**
     * Return the lower logical field index in one direction.
     */
    KOKKOS_INLINE_FUNCTION
    int lower_index(const int dim) const {
        return centerings[dim] == GridCentering::CellCentered
                   ? grid.lower_cell_index(dim, domain)
                   : grid.lower_face_index(dim, domain);
    }

    /**
     * Return the upper logical field index in one direction.
     */
    KOKKOS_INLINE_FUNCTION
    int upper_index(const int dim) const {
        return centerings[dim] == GridCentering::CellCentered
                   ? grid.upper_cell_index(dim, domain)
                   : grid.upper_face_index(dim, domain);
    }

    /**
     * Return one face-coordinate bound for the interpolation coordinate domain.
     */
    KOKKOS_INLINE_FUNCTION
    double coordinate_domain_bound(const int dim, const int face_index) const {
        return grid.coordinate(dim, face_index, GridCentering::FaceCentered);
    }

    /**
     * Clamp a coordinate to the selected coordinate domain.
     */
    KOKKOS_INLINE_FUNCTION
    double clamp_coordinate_to_domain(const int dim, const double coordinate) const {
        const double lower =
            coordinate_domain_bound(dim, grid.lower_face_index(dim, domain));
        const double upper =
            coordinate_domain_bound(dim, grid.upper_face_index(dim, domain));
        if (upper >= lower) {
            if (coordinate < lower) {
                return lower;
            }
            if (coordinate > upper) {
                return upper;
            }
            return coordinate;
        }

        if (coordinate > lower) {
            return lower;
        }
        if (coordinate < upper) {
            return upper;
        }
        return coordinate;
    }

    /**
     * Apply the configured boundary policy to one coordinate.
     */
    KOKKOS_INLINE_FUNCTION
    bool prepare_coordinate(const int dim,
                            const double input_coordinate,
                            double& output_coordinate) const {
        double coordinate = input_coordinate;
        if ((boundary_policy == InterpolationBoundaryPolicy::WrapPeriodic ||
             boundary_policy == InterpolationBoundaryPolicy::ClampToDomain) &&
            grid.is_periodic(dim)) {
            coordinate = grid.wrap_coordinate(dim, coordinate);
        }

        if (boundary_policy == InterpolationBoundaryPolicy::ClampToDomain) {
            output_coordinate = clamp_coordinate_to_domain(dim, coordinate);
            return true;
        }

        if (!grid.contains_coordinate(dim, coordinate, domain)) {
            return false;
        }

        output_coordinate = coordinate;
        return true;
    }

    /**
     * Compute lower/upper interpolation indices and upper weight along one axis.
     */
    KOKKOS_INLINE_FUNCTION
    void compute_axis_stencil(const int dim,
                              const double coordinate,
                              int& lower_stencil_index,
                              int& upper_stencil_index,
                              double& upper_weight) const {
        const int lower_bound = lower_index(dim);
        const int upper_bound = upper_index(dim);
        if (upper_bound < lower_bound) {
            Kokkos::abort("FieldInterpolator: invalid interpolation index domain.");
        }
        if (upper_bound == lower_bound) {
            lower_stencil_index = lower_bound;
            upper_stencil_index = lower_bound;
            upper_weight = 0.0;
            return;
        }

        const double logical =
            grid.grid_index(dim, coordinate, centerings[dim]);
        if (logical != logical) {
            Kokkos::abort("FieldInterpolator: coordinate inversion produced NaN.");
        }

        if (logical <= static_cast<double>(lower_bound)) {
            lower_stencil_index = lower_bound;
            upper_stencil_index = lower_bound;
            upper_weight = 0.0;
            return;
        }
        if (logical >= static_cast<double>(upper_bound)) {
            lower_stencil_index = upper_bound;
            upper_stencil_index = upper_bound;
            upper_weight = 0.0;
            return;
        }

        const int lower_candidate = static_cast<int>(Kokkos::floor(logical));
        lower_stencil_index = lower_candidate;
        upper_stencil_index = lower_candidate + 1;
        upper_weight = logical - static_cast<double>(lower_candidate);

        if (lower_stencil_index < lower_bound) {
            lower_stencil_index = lower_bound;
            upper_stencil_index = lower_bound;
            upper_weight = 0.0;
        } else if (upper_stencil_index > upper_bound) {
            lower_stencil_index = upper_bound;
            upper_stencil_index = upper_bound;
            upper_weight = 0.0;
        }
    }
};

/**
 * Build a device-callable multilinear position interpolator.
 */
template <typename FieldType>
LinearPositionInterpolator<FieldType> make_linear_position_interpolator(
    const FieldType& field,
    const GridDomain domain = GridDomain::GhostedDomain,
    const InterpolationBoundaryPolicy boundary_policy =
        InterpolationBoundaryPolicy::WrapPeriodic) {
    return LinearPositionInterpolator<FieldType>(field, domain, boundary_policy);
}

/**
 * Fill an output view with field values sampled at particle positions.
 */
template <typename FieldType, typename PositionViewType, typename OutputViewType>
void fill_interpolated_values_at_positions(
    const FieldType& field,
    const PositionViewType& positions,
    OutputViewType& output_values,
    const std::size_t count,
    const GridDomain domain = GridDomain::GhostedDomain,
    const InterpolationBoundaryPolicy boundary_policy =
        InterpolationBoundaryPolicy::WrapPeriodic) {
    validate_coordinate_grid_for_interpolation<typename FieldType::grid_type>();

    if (positions.extent(0) < count ||
        positions.extent(1) < static_cast<std::size_t>(FieldType::space_dim) ||
        output_values.extent(0) < count ||
        output_values.extent(1) <
            static_cast<std::size_t>(FieldType::component_dim)) {
        throw std::runtime_error(
            "FieldInterpolator: position or output view extents are too small.");
    }

    const auto interpolator =
        make_linear_position_interpolator(field, domain, boundary_policy);
    auto output = output_values;
    Kokkos::parallel_for(
        "FieldInterpolator::fill_interpolated_values_at_positions",
        Kokkos::RangePolicy<typename FieldType::execution_space>(
            0, static_cast<typename FieldType::size_type>(count)),
        KOKKOS_LAMBDA(const typename FieldType::size_type sample_index) {
            typename FieldType::coordinate_array_type position{};
            for (int dim = 0; dim < FieldType::space_dim; ++dim) {
                position[dim] = positions(sample_index, dim);
            }
            for (int component = 0; component < FieldType::component_dim; ++component) {
                output(sample_index, component) =
                    interpolator.sample_component(position, component);
            }
        });
    Kokkos::fence("FieldInterpolator::fill_interpolated_values_at_positions");
}

/**
 * Fill an output view with field values sampled at all provided positions.
 */
template <typename FieldType, typename PositionViewType, typename OutputViewType>
void fill_interpolated_values_at_positions(
    const FieldType& field,
    const PositionViewType& positions,
    OutputViewType& output_values,
    const GridDomain domain = GridDomain::GhostedDomain,
    const InterpolationBoundaryPolicy boundary_policy =
        InterpolationBoundaryPolicy::WrapPeriodic) {
    fill_interpolated_values_at_positions(
        field, positions, output_values, positions.extent(0), domain,
        boundary_policy);
}

} // namespace FieldInterpolator
