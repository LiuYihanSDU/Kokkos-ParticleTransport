#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "AnalyticCoordinateGrid.hpp"
#include "StoredCoordinateGrid.hpp"

/**
 * Host-side grid validation helpers used before launching device kernels.
 */
namespace GridValidation {

/**
 * Runtime options for host-side grid validation.
 */
template <int SpaceDim>
struct GridValidationOptions {
    int required_ghost_layers{Grid<SpaceDim>::default_ghost_cell_layers};
    bool require_positive_extents{true};
    bool require_non_negative_ghost_cells{true};
    bool require_periodic_flags_binary{true};
    bool require_cell_coordinates{true};
    bool require_face_coordinates{true};
    bool require_monotonic_coordinates{true};
    bool require_periodic_ascending{true};
    bool reject_metric_singularities{true};
};

/**
 * Host-side copy of grid metadata used by validation routines.
 */
template <int SpaceDim>
struct GridMetadataSnapshot {
    Kokkos::Array<int, SpaceDim> extents{};
    Kokkos::Array<Kokkos::Array<int, 2>, SpaceDim> ghost_cells{};
    Kokkos::Array<int, SpaceDim> periodic{};
    OrthogonalCoordinateSystem coordinate_system{OrthogonalCoordinateSystem::Cartesian};
};

namespace Impl {

[[noreturn]] inline void fail(const std::string& message) {
    throw std::runtime_error(message);
}

inline std::string dimension_label(const int dim) {
    return "dim " + std::to_string(dim);
}

inline void validate_cached_double(const double expected,
                                   const double cached,
                                   const std::string& message) {
    const double scale = std::abs(expected) > 1.0 ? std::abs(expected) : 1.0;
    if (std::abs(expected - cached) > 1.0e-12 * scale) {
        fail(message);
    }
}

inline const char* centering_name(const GridCentering centering) {
    return centering == GridCentering::CellCentered ? "cell-centered" : "face-centered";
}

template <int SpaceDim>
inline int total_cell_count(const GridMetadataSnapshot<SpaceDim>& metadata, const int dim) {
    return metadata.extents[dim] + metadata.ghost_cells[dim][LowerBoundary] +
           metadata.ghost_cells[dim][UpperBoundary];
}

template <int SpaceDim>
inline int total_coordinate_count(const GridMetadataSnapshot<SpaceDim>& metadata,
                                  const int dim,
                                  const GridCentering centering) {
    return total_cell_count(metadata, dim) +
           (centering == GridCentering::FaceCentered ? 1 : 0);
}

template <int SpaceDim>
inline int lower_coordinate_index(const GridMetadataSnapshot<SpaceDim>& metadata,
                                  const int dim,
                                  const GridCentering centering) {
    (void) centering;
    return -metadata.ghost_cells[dim][LowerBoundary];
}

template <typename ViewType>
inline bool has_coordinate_view(const ViewType& view) {
    return view.extent(0) > 0 && view.extent(1) > 0;
}

template <typename ViewType, int SpaceDim>
inline void validate_coordinate_view_shape(const ViewType& view,
                                           const GridMetadataSnapshot<SpaceDim>& metadata,
                                           const GridCentering centering,
                                           const bool required) {
    if (!has_coordinate_view(view)) {
        if (required) {
            fail(std::string("GridValidation: missing ") + centering_name(centering) +
                 " coordinate view.");
        }
        return;
    }

    if (view.extent(0) < static_cast<std::size_t>(SpaceDim)) {
        fail(std::string("GridValidation: ") + centering_name(centering) +
             " coordinate view has fewer dimensions than SpaceDim.");
    }

    for (int dim = 0; dim < SpaceDim; ++dim) {
        const int required_count = total_coordinate_count(metadata, dim, centering);
        if (view.extent(1) < static_cast<std::size_t>(required_count)) {
            fail(std::string("GridValidation: ") + centering_name(centering) +
                 " coordinate view is too short in " + dimension_label(dim) + ".");
        }
    }
}

template <typename HostViewType, int SpaceDim>
inline void validate_stored_coordinate_values(const HostViewType& coordinates,
                                              const GridMetadataSnapshot<SpaceDim>& metadata,
                                              const GridCentering centering,
                                              const GridValidationOptions<SpaceDim>& options) {
    if (!has_coordinate_view(coordinates)) {
        return;
    }

    for (int dim = 0; dim < SpaceDim; ++dim) {
        const int count = total_coordinate_count(metadata, dim, centering);
        if (count <= 0) {
            fail(std::string("GridValidation: invalid coordinate count in ") +
                 dimension_label(dim) + ".");
        }

        for (int i = 0; i < count; ++i) {
            const double value = coordinates(dim, i);
            if (!std::isfinite(value)) {
                fail(std::string("GridValidation: non-finite ") + centering_name(centering) +
                     " coordinate in " + dimension_label(dim) + ".");
            }
        }

        if (options.require_monotonic_coordinates && count > 1) {
            const double first_dx = coordinates(dim, 1) - coordinates(dim, 0);
            if (first_dx == 0.0) {
                fail(std::string("GridValidation: duplicate adjacent ") +
                     centering_name(centering) + " coordinates in " + dimension_label(dim) +
                     ".");
            }
            if (first_dx < 0.0) {
                fail(std::string("GridValidation: descending ") + centering_name(centering) +
                     " coordinates are not supported in " + dimension_label(dim) + ".");
            }
            for (int i = 1; i < count - 1; ++i) {
                const double dx = coordinates(dim, i + 1) - coordinates(dim, i);
                if (dx == 0.0) {
                    fail(std::string("GridValidation: duplicate adjacent ") +
                         centering_name(centering) + " coordinates in " +
                         dimension_label(dim) + ".");
                }
                if (dx < 0.0) {
                    fail(std::string("GridValidation: non-monotonic ") +
                         centering_name(centering) + " coordinates in " +
                         dimension_label(dim) + ".");
                }
            }
        }
    }
}

template <int SpaceDim>
inline void validate_radial_singularity(const double coordinate,
                                        const char* coordinate_source,
                                        const int dim) {
    if (coordinate <= OrthogonalGeometry<SpaceDim>::singularity_tolerance) {
        fail(std::string("GridValidation: ") + coordinate_source +
             " radial coordinate touches a metric singularity in " + dimension_label(dim) +
             ".");
    }
}

template <int SpaceDim>
inline void validate_polar_angle_singularity(const double coordinate,
                                             const char* coordinate_source,
                                             const int dim) {
    const double sin_theta = std::sin(coordinate);
    if (std::abs(sin_theta) <= OrthogonalGeometry<SpaceDim>::singularity_tolerance) {
        fail(std::string("GridValidation: ") + coordinate_source +
             " polar angle touches a metric singularity in " + dimension_label(dim) + ".");
    }
}

template <typename HostViewType, int SpaceDim>
inline void validate_stored_metric_singularities(const HostViewType& coordinates,
                                                 const GridMetadataSnapshot<SpaceDim>& metadata,
                                                 const GridCentering centering) {
    if (!has_coordinate_view(coordinates)) {
        return;
    }

    if (metadata.coordinate_system == OrthogonalCoordinateSystem::Polar ||
        metadata.coordinate_system == OrthogonalCoordinateSystem::Spherical) {
        const int count = total_coordinate_count(metadata, 0, centering);
        for (int i = 0; i < count; ++i) {
            validate_radial_singularity<SpaceDim>(coordinates(0, i), centering_name(centering), 0);
        }
    }

    if constexpr (SpaceDim > 1) {
        if (metadata.coordinate_system == OrthogonalCoordinateSystem::Spherical) {
            const int count = total_coordinate_count(metadata, 1, centering);
            for (int i = 0; i < count; ++i) {
                validate_polar_angle_singularity<SpaceDim>(coordinates(1, i),
                                                           centering_name(centering), 1);
            }
        }
    }
}

template <typename HostViewType, int SpaceDim>
inline void validate_periodic_ascending_faces(const HostViewType& face_coordinates,
                                              const GridMetadataSnapshot<SpaceDim>& metadata,
                                              const GridValidationOptions<SpaceDim>& options) {
    if (!options.require_periodic_ascending) {
        return;
    }

    const bool has_faces = has_coordinate_view(face_coordinates);
    for (int dim = 0; dim < SpaceDim; ++dim) {
        if (metadata.periodic[dim] == 0) {
            continue;
        }
        if (!has_faces) {
            fail("GridValidation: periodic coordinate validation requires face coordinates.");
        }

        const int lower_storage = metadata.ghost_cells[dim][LowerBoundary];
        const int upper_storage = metadata.extents[dim] + metadata.ghost_cells[dim][LowerBoundary];
        const double lower_x = face_coordinates(dim, lower_storage);
        const double upper_x = face_coordinates(dim, upper_storage);
        if (!(upper_x > lower_x)) {
            fail(std::string("GridValidation: periodic wrapping requires ascending physical ") +
                 "face coordinates in " + dimension_label(dim) + ".");
        }
    }
}

inline AnalyticGridDiscretization analytic_discretization_from_int(const int value) {
    switch (value) {
        case static_cast<int>(AnalyticGridDiscretization::UniformSpacing):
            return AnalyticGridDiscretization::UniformSpacing;
        case static_cast<int>(AnalyticGridDiscretization::LogarithmicSpacing):
            return AnalyticGridDiscretization::LogarithmicSpacing;
        case static_cast<int>(AnalyticGridDiscretization::LinearWidthSpacing):
            return AnalyticGridDiscretization::LinearWidthSpacing;
    }

    fail("GridValidation: invalid analytic grid discretization value.");
}

inline double analytic_face_coordinate(const AnalyticGridDiscretization discretization,
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

    fail("GridValidation: unreachable analytic discretization branch.");
}

template <typename DiscretizationHostType, typename ScalarHostType>
inline void validate_analytic_parameters(const DiscretizationHostType& discretization,
                                         const ScalarHostType& start,
                                         const ScalarHostType& step,
                                         const ScalarHostType& step_growth,
                                         const int dim) {
    (void) analytic_discretization_from_int(discretization(dim));
    if (!std::isfinite(start(dim)) || !std::isfinite(step(dim)) ||
        !std::isfinite(step_growth(dim))) {
        fail(std::string("GridValidation: non-finite analytic mapping coefficient in ") +
             dimension_label(dim) + ".");
    }
    if (step(dim) == 0.0) {
        fail(std::string("GridValidation: analytic mapping step must be non-zero in ") +
             dimension_label(dim) + ".");
    }
}

template <int SpaceDim, typename DiscretizationHostType, typename ScalarHostType>
inline double analytic_coordinate_at(const DiscretizationHostType& discretization,
                                     const ScalarHostType& start,
                                     const ScalarHostType& step,
                                     const ScalarHostType& step_growth,
                                     const int dim,
                                     const double index,
                                     const GridCentering centering) {
    const AnalyticGridDiscretization type = analytic_discretization_from_int(discretization(dim));
    const double left_face = analytic_face_coordinate(type, start(dim), step(dim),
                                                      step_growth(dim), index);
    if (centering == GridCentering::FaceCentered) {
        return left_face;
    }
    const double right_face = analytic_face_coordinate(type, start(dim), step(dim),
                                                       step_growth(dim), index + 1.0);
    return 0.5 * (left_face + right_face);
}

template <int SpaceDim, typename DiscretizationHostType, typename ScalarHostType>
inline void validate_analytic_centering(const GridMetadataSnapshot<SpaceDim>& metadata,
                                        const DiscretizationHostType& discretization,
                                        const ScalarHostType& start,
                                        const ScalarHostType& step,
                                        const ScalarHostType& step_growth,
                                        const GridCentering centering,
                                        const GridValidationOptions<SpaceDim>& options) {
    for (int dim = 0; dim < SpaceDim; ++dim) {
        const int count = total_coordinate_count(metadata, dim, centering);
        const int lower_index = lower_coordinate_index(metadata, dim, centering);
        for (int i = 0; i < count; ++i) {
            const double logical_index = static_cast<double>(lower_index + i);
            const double coordinate = analytic_coordinate_at<SpaceDim>(
                discretization, start, step, step_growth, dim, logical_index, centering);
            if (!std::isfinite(coordinate)) {
                fail(std::string("GridValidation: non-finite analytic ") +
                     centering_name(centering) + " coordinate in " + dimension_label(dim) +
                     ".");
            }
        }

        if (options.require_monotonic_coordinates && count > 1) {
            const double first = analytic_coordinate_at<SpaceDim>(
                discretization, start, step, step_growth, dim, lower_index, centering);
            const double second = analytic_coordinate_at<SpaceDim>(
                discretization, start, step, step_growth, dim, lower_index + 1, centering);
            const double first_dx = second - first;
            if (first_dx == 0.0) {
                fail(std::string("GridValidation: duplicate adjacent analytic ") +
                     centering_name(centering) + " coordinates in " + dimension_label(dim) +
                     ".");
            }
            if (first_dx < 0.0) {
                fail(std::string("GridValidation: descending analytic ") +
                     centering_name(centering) + " coordinates are not supported in " +
                     dimension_label(dim) + ".");
            }
            for (int i = 1; i < count - 1; ++i) {
                const double left = analytic_coordinate_at<SpaceDim>(
                    discretization, start, step, step_growth, dim, lower_index + i, centering);
                const double right = analytic_coordinate_at<SpaceDim>(
                    discretization, start, step, step_growth, dim, lower_index + i + 1,
                    centering);
                const double dx = right - left;
                if (dx == 0.0) {
                    fail(std::string("GridValidation: duplicate adjacent analytic ") +
                         centering_name(centering) + " coordinates in " + dimension_label(dim) +
                         ".");
                }
                if (dx < 0.0) {
                    fail(std::string("GridValidation: non-monotonic analytic ") +
                         centering_name(centering) + " coordinates in " +
                         dimension_label(dim) + ".");
                }
            }
        }
    }
}

template <int SpaceDim, typename DiscretizationHostType, typename ScalarHostType>
inline void validate_analytic_metric_singularities(const GridMetadataSnapshot<SpaceDim>& metadata,
                                                   const DiscretizationHostType& discretization,
                                                   const ScalarHostType& start,
                                                   const ScalarHostType& step,
                                                   const ScalarHostType& step_growth,
                                                   const GridCentering centering) {
    if (metadata.coordinate_system == OrthogonalCoordinateSystem::Polar ||
        metadata.coordinate_system == OrthogonalCoordinateSystem::Spherical) {
        const int count = total_coordinate_count(metadata, 0, centering);
        const int lower_index = lower_coordinate_index(metadata, 0, centering);
        for (int i = 0; i < count; ++i) {
            const double coordinate = analytic_coordinate_at<SpaceDim>(
                discretization, start, step, step_growth, 0, lower_index + i, centering);
            validate_radial_singularity<SpaceDim>(coordinate, centering_name(centering), 0);
        }
    }

    if constexpr (SpaceDim > 1) {
        if (metadata.coordinate_system == OrthogonalCoordinateSystem::Spherical) {
            const int count = total_coordinate_count(metadata, 1, centering);
            const int lower_index = lower_coordinate_index(metadata, 1, centering);
            for (int i = 0; i < count; ++i) {
                const double coordinate = analytic_coordinate_at<SpaceDim>(
                    discretization, start, step, step_growth, 1, lower_index + i, centering);
                validate_polar_angle_singularity<SpaceDim>(coordinate, centering_name(centering),
                                                           1);
            }
        }
    }
}

template <int SpaceDim, typename DiscretizationHostType, typename ScalarHostType>
inline void validate_analytic_periodic_ascending(const GridMetadataSnapshot<SpaceDim>& metadata,
                                                 const DiscretizationHostType& discretization,
                                                 const ScalarHostType& start,
                                                 const ScalarHostType& step,
                                                 const ScalarHostType& step_growth,
                                                 const GridValidationOptions<SpaceDim>& options) {
    if (!options.require_periodic_ascending) {
        return;
    }

    for (int dim = 0; dim < SpaceDim; ++dim) {
        if (metadata.periodic[dim] == 0) {
            continue;
        }
        const double lower_x = analytic_coordinate_at<SpaceDim>(
            discretization, start, step, step_growth, dim, 0.0, GridCentering::FaceCentered);
        const double upper_x = analytic_coordinate_at<SpaceDim>(
            discretization, start, step, step_growth, dim, static_cast<double>(metadata.extents[dim]),
            GridCentering::FaceCentered);
        if (!(upper_x > lower_x)) {
            fail(std::string("GridValidation: periodic analytic wrapping requires ascending ") +
                 "physical face coordinates in " + dimension_label(dim) + ".");
        }
    }
}

template <int SpaceDim, typename DiscretizationHostType, typename ScalarHostType>
inline void validate_analytic_mapping_host_state(
    const GridMetadataSnapshot<SpaceDim>& metadata,
    const DiscretizationHostType& discretization,
    const ScalarHostType& start,
    const ScalarHostType& step,
    const ScalarHostType& step_growth,
    const GridValidationOptions<SpaceDim>& options) {
    if (discretization.extent(0) < static_cast<std::size_t>(SpaceDim) ||
        start.extent(0) < static_cast<std::size_t>(SpaceDim) ||
        step.extent(0) < static_cast<std::size_t>(SpaceDim) ||
        step_growth.extent(0) < static_cast<std::size_t>(SpaceDim)) {
        fail("GridValidation: analytic mapping coefficient views are too short.");
    }

    for (int dim = 0; dim < SpaceDim; ++dim) {
        validate_analytic_parameters(discretization, start, step, step_growth, dim);
    }

    validate_analytic_centering(metadata, discretization, start, step, step_growth,
                                GridCentering::FaceCentered, options);
    validate_analytic_centering(metadata, discretization, start, step, step_growth,
                                GridCentering::CellCentered, options);
    validate_analytic_periodic_ascending(metadata, discretization, start, step,
                                         step_growth, options);

    if (options.reject_metric_singularities) {
        validate_analytic_metric_singularities(metadata, discretization, start, step,
                                               step_growth, GridCentering::FaceCentered);
        validate_analytic_metric_singularities(metadata, discretization, start, step,
                                               step_growth, GridCentering::CellCentered);
    }
}

}  // namespace Impl

/**
 * Copy grid metadata to host memory for validation or diagnostics.
 */
template <typename GridType>
GridMetadataSnapshot<GridType::space_dim> metadata_snapshot_on_host(const GridType& grid) {
    constexpr int space_dim = GridType::space_dim;
    GridMetadataSnapshot<space_dim> metadata{};

    auto n_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid.n);
    auto ghost_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid.ghost_cells);
    auto periodic_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, grid.periodic);

    for (int dim = 0; dim < space_dim; ++dim) {
        metadata.extents[dim] = n_host(dim);
        metadata.ghost_cells[dim][LowerBoundary] = ghost_host(dim, LowerBoundary);
        metadata.ghost_cells[dim][UpperBoundary] = ghost_host(dim, UpperBoundary);
        metadata.periodic[dim] = periodic_host(dim);
    }
    metadata.coordinate_system = grid.coordinate_system();
    return metadata;
}

/**
 * Validate common grid metadata on the host before launching kernels.
 */
template <typename GridType>
void validate_grid_metadata_on_host(
    const GridType& grid,
    const GridValidationOptions<GridType::space_dim>& options =
        GridValidationOptions<GridType::space_dim>{}) {
    constexpr int space_dim = GridType::space_dim;
    const auto metadata = metadata_snapshot_on_host(grid);

    for (int dim = 0; dim < space_dim; ++dim) {
        if (metadata.extents[dim] != grid.n_cache[dim] ||
            metadata.ghost_cells[dim][LowerBoundary] !=
                grid.ghost_cells_cache[dim][LowerBoundary] ||
            metadata.ghost_cells[dim][UpperBoundary] !=
                grid.ghost_cells_cache[dim][UpperBoundary] ||
            metadata.periodic[dim] != grid.periodic_cache[dim]) {
            Impl::fail("GridValidation: grid metadata cache is stale in " +
                       Impl::dimension_label(dim) + ".");
        }

        if (options.require_positive_extents && metadata.extents[dim] <= 0) {
            Impl::fail("GridValidation: grid extent must be positive in " +
                       Impl::dimension_label(dim) + ".");
        }

        for (int side = LowerBoundary; side <= UpperBoundary; ++side) {
            const int ghost_count = metadata.ghost_cells[dim][side];
            if (options.require_non_negative_ghost_cells && ghost_count < 0) {
                Impl::fail("GridValidation: ghost cell count must be non-negative in " +
                           Impl::dimension_label(dim) + ".");
            }
            if (ghost_count < options.required_ghost_layers) {
                Impl::fail("GridValidation: not enough ghost cells for requested stencil in " +
                           Impl::dimension_label(dim) + ".");
            }
        }

        if (options.require_periodic_flags_binary && metadata.periodic[dim] != 0 &&
            metadata.periodic[dim] != 1) {
            Impl::fail("GridValidation: periodic flag must be 0 or 1 in " +
                       Impl::dimension_label(dim) + ".");
        }
    }
}

/**
 * Validate stored face-coordinate input before constructing a stored-coordinate grid.
 */
template <typename GridType, typename CoordinateViewType>
void validate_stored_face_coordinate_input_on_host(
    const GridType& metadata_grid,
    const CoordinateViewType& face_coordinates,
    const GridValidationOptions<GridType::space_dim>& options =
        GridValidationOptions<GridType::space_dim>{}) {
    validate_grid_metadata_on_host(metadata_grid, options);
    const auto metadata = metadata_snapshot_on_host(metadata_grid);

    Impl::validate_coordinate_view_shape(face_coordinates, metadata,
                                         GridCentering::FaceCentered, true);
    auto face_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                         face_coordinates);
    Impl::validate_stored_coordinate_values(face_host, metadata, GridCentering::FaceCentered,
                                            options);
    Impl::validate_periodic_ascending_faces(face_host, metadata, options);
    if (options.reject_metric_singularities) {
        Impl::validate_stored_metric_singularities(face_host, metadata,
                                                   GridCentering::FaceCentered);
    }
}

/**
 * Validate explicit stored coordinate views, monotonicity, periodic wrapping, and metrics.
 */
template <typename StoredGridType>
void validate_stored_coordinate_grid_on_host(
    const StoredGridType& grid,
    const GridValidationOptions<StoredGridType::space_dim>& options =
        GridValidationOptions<StoredGridType::space_dim>{}) {
    constexpr int space_dim = StoredGridType::space_dim;
    validate_grid_metadata_on_host(grid, options);
    const auto metadata = metadata_snapshot_on_host(grid);

    Impl::validate_coordinate_view_shape(grid.cell_coordinates, metadata,
                                         GridCentering::CellCentered,
                                         options.require_cell_coordinates);
    Impl::validate_coordinate_view_shape(grid.face_coordinates, metadata,
                                         GridCentering::FaceCentered,
                                         options.require_face_coordinates);

    if (Impl::has_coordinate_view(grid.cell_coordinates)) {
        auto cell_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                            grid.cell_coordinates);
        Impl::validate_stored_coordinate_values(cell_host, metadata, GridCentering::CellCentered,
                                                options);
        if (options.reject_metric_singularities) {
            Impl::validate_stored_metric_singularities(cell_host, metadata,
                                                       GridCentering::CellCentered);
        }
    }

    if (Impl::has_coordinate_view(grid.face_coordinates)) {
        auto face_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                            grid.face_coordinates);
        Impl::validate_stored_coordinate_values(face_host, metadata, GridCentering::FaceCentered,
                                                options);
        for (int dim = 0; dim < space_dim; ++dim) {
            const int lower_ghost = metadata.ghost_cells[dim][LowerBoundary];
            const int physical_lower_storage = lower_ghost;
            const int physical_upper_storage = metadata.extents[dim] + lower_ghost;
            const int ghost_lower_storage = 0;
            const int ghost_upper_storage =
                Impl::total_coordinate_count(metadata, dim, GridCentering::FaceCentered) - 1;

            Impl::validate_cached_double(
                face_host(dim, physical_lower_storage), grid.physical_lower_bound[dim],
                "GridValidation: stored physical lower-bound cache is stale in " +
                    Impl::dimension_label(dim) + ".");
            Impl::validate_cached_double(
                face_host(dim, physical_upper_storage), grid.physical_upper_bound[dim],
                "GridValidation: stored physical upper-bound cache is stale in " +
                    Impl::dimension_label(dim) + ".");
            Impl::validate_cached_double(
                face_host(dim, ghost_lower_storage), grid.ghost_lower_bound[dim],
                "GridValidation: stored ghost lower-bound cache is stale in " +
                    Impl::dimension_label(dim) + ".");
            Impl::validate_cached_double(
                face_host(dim, ghost_upper_storage), grid.ghost_upper_bound[dim],
                "GridValidation: stored ghost upper-bound cache is stale in " +
                    Impl::dimension_label(dim) + ".");
        }
        Impl::validate_periodic_ascending_faces(face_host, metadata, options);
        if (options.reject_metric_singularities) {
            Impl::validate_stored_metric_singularities(face_host, metadata,
                                                       GridCentering::FaceCentered);
        }
    } else {
        Impl::validate_periodic_ascending_faces(grid.face_coordinates, metadata, options);
    }
}

/**
 * Validate analytic mapping input before constructing an analytic-coordinate grid.
 */
template <typename GridType, typename AnalyticMappingType>
void validate_analytic_mapping_input_on_host(
    const GridType& metadata_grid,
    const AnalyticMappingType& mapping,
    const GridValidationOptions<GridType::space_dim>& options =
        GridValidationOptions<GridType::space_dim>{}) {
    validate_grid_metadata_on_host(metadata_grid, options);
    const auto metadata = metadata_snapshot_on_host(metadata_grid);

    auto discretization_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, mapping.discretization);
    auto start_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          mapping.start);
    auto step_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                         mapping.step);
    auto step_growth_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                                mapping.step_growth);

    Impl::validate_analytic_mapping_host_state(metadata, discretization_host, start_host,
                                               step_host, step_growth_host, options);
}

/**
 * Validate analytic coordinate coefficients and sampled ghosted coordinate ranges.
 */
template <typename AnalyticGridType>
void validate_analytic_coordinate_grid_on_host(
    const AnalyticGridType& grid,
    const GridValidationOptions<AnalyticGridType::space_dim>& options =
        GridValidationOptions<AnalyticGridType::space_dim>{}) {
    constexpr int space_dim = AnalyticGridType::space_dim;
    validate_grid_metadata_on_host(grid, options);
    const auto metadata = metadata_snapshot_on_host(grid);

    auto discretization_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, grid.mapping.discretization);
    auto start_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          grid.mapping.start);
    auto step_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                         grid.mapping.step);
    auto step_growth_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, grid.mapping.step_growth);

    Impl::validate_analytic_mapping_host_state(metadata, discretization_host, start_host,
                                               step_host, step_growth_host, options);

    for (int dim = 0; dim < space_dim; ++dim) {
        const double physical_lower = Impl::analytic_coordinate_at<space_dim>(
            discretization_host, start_host, step_host, step_growth_host, dim, 0.0,
            GridCentering::FaceCentered);
        const double physical_upper = Impl::analytic_coordinate_at<space_dim>(
            discretization_host, start_host, step_host, step_growth_host, dim,
            static_cast<double>(metadata.extents[dim]), GridCentering::FaceCentered);
        const double ghost_lower = Impl::analytic_coordinate_at<space_dim>(
            discretization_host, start_host, step_host, step_growth_host, dim,
            static_cast<double>(-metadata.ghost_cells[dim][LowerBoundary]),
            GridCentering::FaceCentered);
        const double ghost_upper = Impl::analytic_coordinate_at<space_dim>(
            discretization_host, start_host, step_host, step_growth_host, dim,
            static_cast<double>(metadata.extents[dim] +
                                metadata.ghost_cells[dim][UpperBoundary]),
            GridCentering::FaceCentered);

        Impl::validate_cached_double(
            physical_lower, grid.physical_lower_bound[dim],
            "GridValidation: analytic physical lower-bound cache is stale in " +
                Impl::dimension_label(dim) + ".");
        Impl::validate_cached_double(
            physical_upper, grid.physical_upper_bound[dim],
            "GridValidation: analytic physical upper-bound cache is stale in " +
                Impl::dimension_label(dim) + ".");
        Impl::validate_cached_double(
            ghost_lower, grid.ghost_lower_bound[dim],
            "GridValidation: analytic ghost lower-bound cache is stale in " +
                Impl::dimension_label(dim) + ".");
        Impl::validate_cached_double(
            ghost_upper, grid.ghost_upper_bound[dim],
            "GridValidation: analytic ghost upper-bound cache is stale in " +
                Impl::dimension_label(dim) + ".");
    }

}

/**
 * Build common grid metadata and validate it before returning the immutable grid object.
 */
template <
    int SpaceDim,
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout,
    int VecDim = SpaceDim
>
Grid<SpaceDim, DeviceType, LayoutType, VecDim> make_validated_grid_metadata(
    const typename Grid<SpaceDim, DeviceType, LayoutType, VecDim>::index_array_type& extents,
    const OrthogonalCoordinateSystem coordinate_system = OrthogonalCoordinateSystem::Cartesian,
    const GridValidationOptions<SpaceDim>& options = GridValidationOptions<SpaceDim>{}) {
    Grid<SpaceDim, DeviceType, LayoutType, VecDim> grid(extents, coordinate_system);
    validate_grid_metadata_on_host(grid, options);
    return grid;
}

/**
 * Build common grid metadata with explicit ghost cells and validate it before returning.
 */
template <
    int SpaceDim,
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout,
    int VecDim = SpaceDim
>
Grid<SpaceDim, DeviceType, LayoutType, VecDim> make_validated_grid_metadata(
    const typename Grid<SpaceDim, DeviceType, LayoutType, VecDim>::index_array_type& extents,
    const typename Grid<SpaceDim, DeviceType, LayoutType, VecDim>::ghost_array_type& ghosts,
    const typename Grid<SpaceDim, DeviceType, LayoutType, VecDim>::periodic_array_type&
        periodic_flags,
    const OrthogonalCoordinateSystem coordinate_system = OrthogonalCoordinateSystem::Cartesian,
    const GridValidationOptions<SpaceDim>& options = GridValidationOptions<SpaceDim>{}) {
    Grid<SpaceDim, DeviceType, LayoutType, VecDim> grid(extents, ghosts, periodic_flags,
                                                        coordinate_system);
    validate_grid_metadata_on_host(grid, options);
    return grid;
}

/**
 * Build a stored-coordinate grid from face coordinates and validate it before returning.
 */
template <
    int SpaceDim,
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout,
    int VecDim = SpaceDim
>
StoredCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>
make_validated_stored_coordinate_grid(
    const Grid<SpaceDim, DeviceType, LayoutType, VecDim>& metadata,
    typename StoredCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>::coordinate_view_type
        face_coordinates,
    const GridValidationOptions<SpaceDim>& options = GridValidationOptions<SpaceDim>{}) {
    validate_stored_face_coordinate_input_on_host(metadata, face_coordinates, options);
    StoredCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim> grid(metadata,
                                                                        face_coordinates);
    validate_stored_coordinate_grid_on_host(grid, options);
    return grid;
}

/**
 * Build an analytic grid from a mapping object and validate it before returning.
 */
template <
    int SpaceDim,
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout,
    int VecDim = SpaceDim
>
AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>
make_validated_analytic_coordinate_grid(
    const Grid<SpaceDim, DeviceType, LayoutType, VecDim>& metadata,
    typename AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>::mapping_type mapping,
    const GridValidationOptions<SpaceDim>& options = GridValidationOptions<SpaceDim>{}) {
    validate_analytic_mapping_input_on_host(metadata, mapping, options);
    AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim> grid(metadata, mapping);
    validate_analytic_coordinate_grid_on_host(grid, options);
    return grid;
}

/**
 * Build a uniform analytic grid from coefficient views and validate it before returning.
 */
template <
    int SpaceDim,
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout,
    int VecDim = SpaceDim
>
AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>
make_validated_analytic_coordinate_grid(
    const Grid<SpaceDim, DeviceType, LayoutType, VecDim>& metadata,
    typename AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>::scalar_view_type
        start,
    typename AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>::scalar_view_type
        step,
    const GridValidationOptions<SpaceDim>& options = GridValidationOptions<SpaceDim>{}) {
    using grid_type = AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>;
    typename grid_type::mapping_type mapping(start, step);
    validate_analytic_mapping_input_on_host(metadata, mapping, options);
    grid_type grid(metadata, mapping);
    validate_analytic_coordinate_grid_on_host(grid, options);
    return grid;
}

/**
 * Build an analytic grid from device-side coefficient views and validate it before returning.
 */
template <
    int SpaceDim,
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout,
    int VecDim = SpaceDim
>
AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>
make_validated_analytic_coordinate_grid(
    const Grid<SpaceDim, DeviceType, LayoutType, VecDim>& metadata,
    typename AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>::
        discretization_view_type discretization,
    typename AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>::scalar_view_type
        start,
    typename AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>::scalar_view_type
        step,
    typename AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>::scalar_view_type
        step_growth,
    const GridValidationOptions<SpaceDim>& options = GridValidationOptions<SpaceDim>{}) {
    using grid_type = AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>;
    typename grid_type::mapping_type mapping(discretization, start, step, step_growth);
    validate_analytic_mapping_input_on_host(metadata, mapping, options);
    grid_type grid(metadata, mapping);
    validate_analytic_coordinate_grid_on_host(grid, options);
    return grid;
}

/**
 * Build an analytic grid from host-side coefficient arrays and validate it before returning.
 */
template <
    int SpaceDim,
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout,
    int VecDim = SpaceDim
>
AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>
make_validated_analytic_coordinate_grid(
    const Grid<SpaceDim, DeviceType, LayoutType, VecDim>& metadata,
    const typename AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>::
        discretization_array_type& discretization,
    const typename AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>::
        scalar_array_type& start,
    const typename AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>::
        scalar_array_type& step,
    const typename AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>::
        scalar_array_type& step_growth,
    const GridValidationOptions<SpaceDim>& options = GridValidationOptions<SpaceDim>{}) {
    using grid_type = AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>;
    typename grid_type::mapping_type mapping(discretization, start, step, step_growth);
    validate_analytic_mapping_input_on_host(metadata, mapping, options);
    grid_type grid(metadata, mapping);
    validate_analytic_coordinate_grid_on_host(grid, options);
    return grid;
}

/**
 * Build an analytic grid from host-side arrays without width growth and validate it.
 */
template <
    int SpaceDim,
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout,
    int VecDim = SpaceDim
>
AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>
make_validated_analytic_coordinate_grid(
    const Grid<SpaceDim, DeviceType, LayoutType, VecDim>& metadata,
    const typename AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>::
        discretization_array_type& discretization,
    const typename AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>::
        scalar_array_type& start,
    const typename AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>::
        scalar_array_type& step,
    const GridValidationOptions<SpaceDim>& options = GridValidationOptions<SpaceDim>{}) {
    using grid_type = AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>;
    typename grid_type::mapping_type mapping(discretization, start, step);
    validate_analytic_mapping_input_on_host(metadata, mapping, options);
    grid_type grid(metadata, mapping);
    validate_analytic_coordinate_grid_on_host(grid, options);
    return grid;
}

}  // namespace GridValidation
