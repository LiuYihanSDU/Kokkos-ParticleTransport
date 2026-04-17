#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "AnalyticCoordinateGrid.hpp"
#include "FieldOutputTypes.hpp"
#include "Grid.hpp"
#include "MappedCoordinateGrid.hpp"
#include "StoredCoordinateGrid.hpp"

/**
 * Host-side field snapshot construction for binary output and offline conversion.
 */
namespace FieldOutput {

inline double analytic_face_coordinate_on_host(AnalyticGridDiscretization discretization,
                                               double start,
                                               double step,
                                               double step_growth,
                                               double face_index);

inline double analytic_coordinate_on_host(AnalyticGridDiscretization discretization,
                                          double start,
                                          double step,
                                          double step_growth,
                                          double logical_index,
                                          GridCentering centering);

template <typename FieldType, typename Visitor>
void for_each_output_index(const FieldOutputSnapshot& snapshot, Visitor visitor);

template <typename FieldType>
std::array<double, 3> cartesian_point_from_coordinates(
    const FieldType& field,
    const typename FieldType::coordinate_array_type& q);

template <typename FieldType, typename HostDataView>
std::array<double, 3> cartesian_vector_from_components(
    const FieldType& field,
    const typename FieldType::coordinate_array_type& q,
    const HostDataView& data_host,
    typename FieldType::size_type point_index);

template <typename FieldType, typename HostDataView, typename CoordinateAt>
FieldOutputSnapshot build_snapshot_from_coordinate_provider(
    const FieldType& field,
    const std::string& field_name,
    GridDomain domain,
    const HostDataView& data_host,
    CoordinateAt coordinate_at);

/**
 * Convert logical field coordinates and values into a host-side visualization snapshot.
 */
template <typename FieldType>
FieldOutputSnapshot make_field_output_snapshot(const FieldType& field,
                                               const std::string& field_name = "field",
                                               const GridDomain domain =
                                                   GridDomain::PhysicalDomain) {
    if constexpr (FieldType::space_dim < 1 || FieldType::space_dim > 3) {
        throw std::runtime_error(
            "FieldOutput::make_field_output_snapshot supports SpaceDim from 1 to 3.");
    } else {
        auto data_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                             field.data);

        if constexpr (requires { FieldType::grid_type::coordinate_representation; }) {
            if constexpr (FieldType::grid_type::coordinate_representation ==
                          GridCoordinateRepresentation::StoredCoordinates) {
                auto cell_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                                     field.grid.cell_coordinates);
                auto face_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                                     field.grid.face_coordinates);
                const auto coordinate_at =
                    [&field, cell_host, face_host](const int dim,
                                                   const int logical_index,
                                                   const GridCentering centering) {
                        const int storage_index =
                            logical_index + field.grid.ghost_extent(dim, LowerBoundary);
                        return centering == GridCentering::CellCentered
                                   ? cell_host(dim, storage_index)
                                   : face_host(dim, storage_index);
                    };
                return build_snapshot_from_coordinate_provider(field, field_name, domain,
                                                               data_host, coordinate_at);
            } else if constexpr (FieldType::grid_type::coordinate_representation ==
                                 GridCoordinateRepresentation::AnalyticCoordinates) {
                auto discretization_host = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, field.grid.mapping.discretization);
                auto start_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                                      field.grid.mapping.start);
                auto step_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                                     field.grid.mapping.step);
                auto step_growth_host = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, field.grid.mapping.step_growth);
                const auto coordinate_at =
                    [discretization_host, start_host, step_host, step_growth_host](
                        const int dim,
                        const int logical_index,
                        const GridCentering centering) {
                        return analytic_coordinate_on_host(
                            static_cast<AnalyticGridDiscretization>(
                                discretization_host(dim)),
                            start_host(dim), step_host(dim), step_growth_host(dim),
                            static_cast<double>(logical_index), centering);
                };
                return build_snapshot_from_coordinate_provider(field, field_name, domain,
                                                               data_host, coordinate_at);
            } else if constexpr (FieldType::grid_type::coordinate_representation ==
                                 GridCoordinateRepresentation::MappedCoordinates) {
                const auto coordinate_at =
                    MappedCoordinateHostAccess::create_coordinate_provider_from_grid(field.grid);
                return build_snapshot_from_coordinate_provider(field, field_name, domain,
                                                               data_host, coordinate_at);
            } else {
                const auto coordinate_at = [&field](const int dim,
                                                    const int logical_index,
                                                    const GridCentering centering) {
                    return field.grid.coordinate(dim, logical_index, centering);
                };
                return build_snapshot_from_coordinate_provider(field, field_name, domain,
                                                               data_host, coordinate_at);
            }
        } else {
            if constexpr (requires(const typename FieldType::grid_type& input_grid) {
                              input_grid.coordinate(0, 0, GridCentering::CellCentered);
                          }) {
                const auto coordinate_at = [&field](const int dim,
                                                    const int logical_index,
                                                    const GridCentering centering) {
                    return field.grid.coordinate(dim, logical_index, centering);
                };
                return build_snapshot_from_coordinate_provider(field, field_name, domain,
                                                               data_host, coordinate_at);
            } else {
                throw std::runtime_error(
                    "FieldOutput::make_field_output_snapshot requires a coordinate grid.");
            }
        }
    }
}

/**
 * Evaluate an analytic coordinate expression on the host.
 */
inline double analytic_coordinate_on_host(const AnalyticGridDiscretization discretization,
                                          const double start,
                                          const double step,
                                          const double step_growth,
                                          const double logical_index,
                                          const GridCentering centering) {
    const double left_face =
        analytic_face_coordinate_on_host(discretization, start, step, step_growth,
                                         logical_index);
    if (centering == GridCentering::FaceCentered) {
        return left_face;
    }
    const double right_face =
        analytic_face_coordinate_on_host(discretization, start, step, step_growth,
                                         logical_index + 1.0);
    return 0.5 * (left_face + right_face);
}

/**
 * Evaluate an analytic face-coordinate expression on the host.
 */
inline double analytic_face_coordinate_on_host(const AnalyticGridDiscretization discretization,
                                               const double start,
                                               const double step,
                                               const double step_growth,
                                               const double face_index) {
    switch (discretization) {
        case AnalyticGridDiscretization::UniformSpacing:
            return start + face_index * step;
        case AnalyticGridDiscretization::LogarithmicSpacing:
            return std::exp(start + face_index * step);
        case AnalyticGridDiscretization::LinearWidthSpacing:
            return start + face_index * step +
                   0.5 * step_growth * face_index * (face_index - 1.0);
    }
    return start;
}

/**
 * Build a visualization snapshot using a host coordinate callback.
 */
template <typename FieldType, typename HostDataView, typename CoordinateAt>
FieldOutputSnapshot build_snapshot_from_coordinate_provider(
    const FieldType& field,
    const std::string& field_name,
    const GridDomain domain,
    const HostDataView& data_host,
    CoordinateAt coordinate_at) {
    FieldOutputSnapshot snapshot;
    snapshot.field_name = field_name;
    snapshot.space_dim = FieldType::space_dim;
    snapshot.component_dim = FieldType::component_dim;
    snapshot.output_component_dim = FieldType::component_dim == 1 ? 1 : 3;
    snapshot.coordinate_system = static_cast<std::int32_t>(field.grid.coordinate_system());
    snapshot.domain = static_cast<std::int32_t>(domain);

    snapshot.dimensions[0] = field.domain_extent(0, domain);
    snapshot.lower_indices[0] = field.lower_index(0, domain);
    snapshot.upper_indices[0] = field.upper_index(0, domain);
    snapshot.centerings[0] = static_cast<std::int32_t>(field.centering(0));
    if constexpr (FieldType::space_dim > 1) {
        snapshot.dimensions[1] = field.domain_extent(1, domain);
        snapshot.lower_indices[1] = field.lower_index(1, domain);
        snapshot.upper_indices[1] = field.upper_index(1, domain);
        snapshot.centerings[1] = static_cast<std::int32_t>(field.centering(1));
    }
    if constexpr (FieldType::space_dim > 2) {
        snapshot.dimensions[2] = field.domain_extent(2, domain);
        snapshot.lower_indices[2] = field.lower_index(2, domain);
        snapshot.upper_indices[2] = field.upper_index(2, domain);
        snapshot.centerings[2] = static_cast<std::int32_t>(field.centering(2));
    }

    const std::size_t point_count = snapshot.point_count();
    snapshot.points.resize(point_count * 3);
    snapshot.values.resize(point_count * static_cast<std::size_t>(
                                           snapshot.output_component_dim));

    std::size_t output_index = 0;
    for_each_output_index<FieldType>(snapshot, [&](const typename FieldType::index_array_type& indices) {
        typename FieldType::coordinate_array_type q{};
        for (int dim = 0; dim < FieldType::space_dim; ++dim) {
            q[dim] = coordinate_at(dim, indices[dim], field.centering(dim));
        }

        const auto point = cartesian_point_from_coordinates<FieldType>(field, q);
        snapshot.points[3 * output_index + 0] = point[0];
        snapshot.points[3 * output_index + 1] = point[1];
        snapshot.points[3 * output_index + 2] = point[2];

        const auto flat_index = field.flatten(indices);
        if constexpr (FieldType::component_dim == 1) {
            snapshot.values[output_index] = data_host(flat_index, 0);
        } else {
            const auto vector =
                cartesian_vector_from_components<FieldType>(field, q, data_host, flat_index);
            snapshot.values[3 * output_index + 0] = vector[0];
            snapshot.values[3 * output_index + 1] = vector[1];
            snapshot.values[3 * output_index + 2] = vector[2];
        }
        ++output_index;
    });

    return snapshot;
}

/**
 * Visit snapshot logical indices in structured-grid output order.
 */
template <typename FieldType, typename Visitor>
void for_each_output_index(const FieldOutputSnapshot& snapshot, Visitor visitor) {
    const int nx = snapshot.dimensions[0];
    const int ny = snapshot.dimensions[1];
    const int nz = snapshot.dimensions[2];
    const int lower_x = snapshot.lower_indices[0];
    const int lower_y = snapshot.lower_indices[1];
    const int lower_z = snapshot.lower_indices[2];

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                typename FieldType::index_array_type indices{};
                indices[0] = lower_x + i;
                if constexpr (FieldType::space_dim > 1) {
                    indices[1] = lower_y + j;
                }
                if constexpr (FieldType::space_dim > 2) {
                    indices[2] = lower_z + k;
                }
                visitor(indices);
            }
        }
    }
}

/**
 * Convert coordinate-basis point coordinates to Cartesian coordinates.
 */
template <typename FieldType>
std::array<double, 3> cartesian_point_from_coordinates(
    const FieldType& field,
    const typename FieldType::coordinate_array_type& q) {
    std::array<double, 3> point{0.0, 0.0, 0.0};
    switch (field.grid.coordinate_system()) {
        case OrthogonalCoordinateSystem::Cartesian:
            point[0] = q[0];
            if constexpr (FieldType::space_dim > 1) {
                point[1] = q[1];
            }
            if constexpr (FieldType::space_dim > 2) {
                point[2] = q[2];
            }
            break;
        case OrthogonalCoordinateSystem::Polar: {
            const double radius = q[0];
            const double phi = FieldType::space_dim > 1 ? q[1] : 0.0;
            point[0] = radius * std::cos(phi);
            point[1] = radius * std::sin(phi);
            if constexpr (FieldType::space_dim > 2) {
                point[2] = q[2];
            }
            break;
        }
        case OrthogonalCoordinateSystem::Spherical: {
            const double radius = q[0];
            if constexpr (FieldType::space_dim > 1) {
                const double theta = q[1];
                const double phi = FieldType::space_dim > 2 ? q[2] : 0.0;
                point[0] = radius * std::sin(theta) * std::cos(phi);
                point[1] = radius * std::sin(theta) * std::sin(phi);
                point[2] = radius * std::cos(theta);
            } else {
                point[0] = radius;
            }
            break;
        }
    }
    return point;
}

/**
 * Return a host field component, padding short vectors with zero.
 */
template <typename FieldType, typename HostDataView>
double component_value_on_host(const HostDataView& data_host,
                               const typename FieldType::size_type point_index,
                               const int component) {
    if (component >= FieldType::component_dim) {
        return 0.0;
    }
    return data_host(point_index, component);
}

/**
 * Convert coordinate-basis vector components to Cartesian vector components.
 */
template <typename FieldType, typename HostDataView>
std::array<double, 3> cartesian_vector_from_components(
    const FieldType& field,
    const typename FieldType::coordinate_array_type& q,
    const HostDataView& data_host,
    const typename FieldType::size_type point_index) {
    std::array<double, 3> vector{0.0, 0.0, 0.0};
    const double v0 = component_value_on_host<FieldType>(data_host, point_index, 0);
    const double v1 = component_value_on_host<FieldType>(data_host, point_index, 1);
    const double v2 = component_value_on_host<FieldType>(data_host, point_index, 2);

    switch (field.grid.coordinate_system()) {
        case OrthogonalCoordinateSystem::Cartesian:
            vector[0] = v0;
            vector[1] = v1;
            vector[2] = v2;
            break;
        case OrthogonalCoordinateSystem::Polar: {
            const double phi = FieldType::space_dim > 1 ? q[1] : 0.0;
            vector[0] = v0 * std::cos(phi) - v1 * std::sin(phi);
            vector[1] = v0 * std::sin(phi) + v1 * std::cos(phi);
            vector[2] = v2;
            break;
        }
        case OrthogonalCoordinateSystem::Spherical: {
            if constexpr (FieldType::space_dim > 1) {
                const double theta = q[1];
                const double phi = FieldType::space_dim > 2 ? q[2] : 0.0;
                vector[0] = v0 * std::sin(theta) * std::cos(phi) +
                            v1 * std::cos(theta) * std::cos(phi) -
                            v2 * std::sin(phi);
                vector[1] = v0 * std::sin(theta) * std::sin(phi) +
                            v1 * std::cos(theta) * std::sin(phi) +
                            v2 * std::cos(phi);
                vector[2] = v0 * std::cos(theta) - v1 * std::sin(theta);
            } else {
                vector[0] = v0;
            }
            break;
        }
    }
    return vector;
}

} // namespace FieldOutput
