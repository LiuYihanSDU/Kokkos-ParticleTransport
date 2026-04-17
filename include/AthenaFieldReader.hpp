#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "Field.hpp"
#include "GridValidation.hpp"

/**
 * Host-side reader for the compact Athena++ background-field binary files produced by
 * apps/athena2bin.py.
 */
namespace AthenaFieldIO {

inline constexpr int athena_binary_space_dim = 2;
inline constexpr int athena_binary_vector_dim = 3;
inline constexpr int athena_binary_vector_field_count = 2;

/**
 * Runtime options used when loading one Athena++ background-field binary file.
 */
template <int SpaceDim = athena_binary_space_dim>
struct AthenaBinaryFieldReadOptions {
    using ghost_array_type = Kokkos::Array<Kokkos::Array<int, 2>, SpaceDim>;
    using periodic_array_type = Kokkos::Array<int, SpaceDim>;

    ghost_array_type ghost_cells{};
    periodic_array_type periodic{};
    OrthogonalCoordinateSystem coordinate_system{OrthogonalCoordinateSystem::Cartesian};
    FieldGhostFillMode ghost_fill_mode{FieldGhostFillMode::PeriodicOrClamped};
    bool close_periodic_faces{true};
    double coordinate_scale{1.0};
    double magnetic_field_scale{1.0};
    double velocity_scale{1.0};
    GridValidation::GridValidationOptions<SpaceDim> grid_validation{};

    /**
     * Use the repository default ghost-cell thickness and non-periodic boundaries.
     */
    AthenaBinaryFieldReadOptions() {
        for (int dim = 0; dim < SpaceDim; ++dim) {
            ghost_cells[dim][LowerBoundary] = Grid<SpaceDim>::default_ghost_cell_layers;
            ghost_cells[dim][UpperBoundary] = Grid<SpaceDim>::default_ghost_cell_layers;
            periodic[dim] = 0;
        }
    }
};

/**
 * Device-ready grid plus magnetic-field and flow-velocity fields loaded from one
 * Athena++ binary snapshot.
 */
template <
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout
>
struct AthenaBackgroundField {
    using device_type = DeviceType;
    using layout_type = LayoutType;
    using grid_type =
        StoredCoordinateGrid<athena_binary_space_dim, device_type, layout_type,
                             athena_binary_vector_dim>;
    using vector_field_type =
        VectorField<grid_type, athena_binary_vector_dim, layout_type>;

    grid_type grid;
    vector_field_type magnetic_field;
    vector_field_type velocity_field;
    std::int32_t nx{0};
    std::int32_t ny{0};
};

namespace Impl {

inline void validate_little_endian_host() {
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error(
            "AthenaFieldReader: athena2bin.py writes native-endian files; "
            "this reader currently expects little-endian host data.");
    }
}

inline void validate_positive_finite_scale(const double scale, const char* name) {
    if (!std::isfinite(scale) || scale <= 0.0) {
        throw std::runtime_error(std::string("AthenaFieldReader: ") + name +
                                 " must be positive and finite.");
    }
}

template <typename T>
T read_binary_scalar(std::ifstream& stream, const char* name) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    if (!stream) {
        throw std::runtime_error(std::string("AthenaFieldReader: failed to read ") +
                                 name + ".");
    }
    return value;
}

inline std::vector<double> read_double_vector(std::ifstream& stream,
                                              const std::size_t count,
                                              const char* name) {
    std::vector<double> values(count);
    if (!values.empty()) {
        const auto byte_count =
            static_cast<std::streamsize>(values.size() * sizeof(double));
        stream.read(reinterpret_cast<char*>(values.data()), byte_count);
        if (!stream) {
            throw std::runtime_error(std::string("AthenaFieldReader: failed to read ") +
                                     name + ".");
        }
    }
    return values;
}

inline std::size_t checked_cell_count(const std::int32_t nx, const std::int32_t ny) {
    if (nx <= 0 || ny <= 0) {
        throw std::runtime_error(
            "AthenaFieldReader: Athena binary dimensions must be positive.");
    }
    const auto x_count = static_cast<std::size_t>(nx);
    const auto y_count = static_cast<std::size_t>(ny);
    if (x_count > std::numeric_limits<std::size_t>::max() / y_count) {
        throw std::runtime_error("AthenaFieldReader: Athena binary grid is too large.");
    }
    return x_count * y_count;
}

inline void scale_coordinates(std::vector<double>& coordinates, const double scale) {
    for (double& coordinate : coordinates) {
        coordinate *= scale;
    }
}

inline void validate_face_edges(const std::vector<double>& edges, const char* axis_name) {
    if (edges.size() < 2) {
        throw std::runtime_error(std::string("AthenaFieldReader: ") + axis_name +
                                 " requires at least two face coordinates.");
    }
    for (std::size_t i = 0; i < edges.size(); ++i) {
        if (!std::isfinite(edges[i])) {
            throw std::runtime_error(std::string("AthenaFieldReader: non-finite ") +
                                     axis_name + " face coordinate.");
        }
        if (i > 0 && !(edges[i] > edges[i - 1])) {
            throw std::runtime_error(std::string("AthenaFieldReader: ") + axis_name +
                                     " face coordinates must be strictly ascending.");
        }
    }
}

inline double extended_face_coordinate(const std::vector<double>& physical_edges,
                                       const int logical_face_index) {
    const int physical_face_count = static_cast<int>(physical_edges.size());
    const int upper_physical_face = physical_face_count - 1;
    if (logical_face_index >= 0 && logical_face_index <= upper_physical_face) {
        return physical_edges[static_cast<std::size_t>(logical_face_index)];
    }

    if (logical_face_index < 0) {
        const double lower_width = physical_edges[1] - physical_edges[0];
        return physical_edges[0] + static_cast<double>(logical_face_index) * lower_width;
    }

    const double upper_width =
        physical_edges[static_cast<std::size_t>(upper_physical_face)] -
        physical_edges[static_cast<std::size_t>(upper_physical_face - 1)];
    return physical_edges[static_cast<std::size_t>(upper_physical_face)] +
           static_cast<double>(logical_face_index - upper_physical_face) * upper_width;
}

template <typename GridType>
int max_face_coordinate_count(const GridType& metadata) {
    int count = 0;
    for (int dim = 0; dim < GridType::space_dim; ++dim) {
        const int dim_count = metadata.total_extent(dim, GridCentering::FaceCentered);
        count = dim_count > count ? dim_count : count;
    }
    return count;
}

template <typename DeviceType, typename LayoutType>
typename AthenaBackgroundField<DeviceType, LayoutType>::grid_type make_grid(
    const std::int32_t nx,
    const std::int32_t ny,
    const std::vector<double>& x_edges,
    const std::vector<double>& y_edges,
    const AthenaBinaryFieldReadOptions<athena_binary_space_dim>& options) {
    using background_type = AthenaBackgroundField<DeviceType, LayoutType>;
    using grid_type = typename background_type::grid_type;
    using metadata_type =
        Grid<athena_binary_space_dim, DeviceType, LayoutType, athena_binary_vector_dim>;
    using coordinate_view_type = typename grid_type::coordinate_view_type;

    typename metadata_type::index_array_type extents{};
    extents[0] = nx;
    extents[1] = ny;

    metadata_type metadata(extents, options.ghost_cells, options.periodic,
                           options.coordinate_system);
    const int coordinate_count = max_face_coordinate_count(metadata);
    coordinate_view_type face_coordinates("athena_face_coordinates",
                                          athena_binary_space_dim,
                                          coordinate_count);
    auto face_host = Kokkos::create_mirror_view(face_coordinates);
    for (int dim = 0; dim < athena_binary_space_dim; ++dim) {
        for (int storage = 0; storage < coordinate_count; ++storage) {
            face_host(dim, storage) = 0.0;
        }
    }

    const std::vector<double>* edge_sets[athena_binary_space_dim] = {&x_edges, &y_edges};
    for (int dim = 0; dim < athena_binary_space_dim; ++dim) {
        const int lower_ghost = metadata.ghost_extent(dim, LowerBoundary);
        const int face_count = metadata.total_extent(dim, GridCentering::FaceCentered);
        for (int storage = 0; storage < face_count; ++storage) {
            const int logical_face_index = storage - lower_ghost;
            face_host(dim, storage) =
                extended_face_coordinate(*edge_sets[dim], logical_face_index);
        }
    }
    Kokkos::deep_copy(face_coordinates, face_host);

    return GridValidation::make_validated_stored_coordinate_grid<
        athena_binary_space_dim, DeviceType, LayoutType, athena_binary_vector_dim>(
        metadata, face_coordinates, options.grid_validation);
}

template <typename FieldType, typename HostDataView>
void copy_component_to_host(const std::vector<double>& component_values,
                            const std::int32_t nx,
                            const std::int32_t ny,
                            const double scale,
                            const int component,
                            const char* component_name,
                            const FieldType& field,
                            HostDataView& data_host) {
    const std::size_t expected_count =
        static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    if (component_values.size() != expected_count) {
        throw std::runtime_error(std::string("AthenaFieldReader: invalid payload size for ") +
                                 component_name + ".");
    }

    typename FieldType::index_array_type indices{};
    for (std::int32_t j = 0; j < ny; ++j) {
        indices[1] = j;
        for (std::int32_t i = 0; i < nx; ++i) {
            indices[0] = i;
            const std::size_t source_index =
                static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
                static_cast<std::size_t>(i);
            const double value = component_values[source_index];
            if (!std::isfinite(value)) {
                throw std::runtime_error(std::string("AthenaFieldReader: non-finite ") +
                                         component_name + " value.");
            }
            data_host(field.flatten(indices), component) = value * scale;
        }
    }
}

inline void validate_end_of_file(std::ifstream& stream) {
    const auto next = stream.peek();
    if (next != std::char_traits<char>::eof()) {
        throw std::runtime_error(
            "AthenaFieldReader: Athena binary file has trailing payload.");
    }
}

} // namespace Impl

/**
 * Read one athena2bin.py field snapshot as a solver-ready background field.
 *
 * File layout:
 * - int32 nx, int32 ny
 * - double x_edges[nx + 1], y_edges[ny + 1]
 * - double Bx[nx * ny], By[nx * ny], Bz[nx * ny]
 * - double Vx[nx * ny], Vy[nx * ny], Vz[nx * ny]
 *
 * Component arrays are stored in row-major order with x-index contiguous:
 * source_index = j * nx + i.
 */
template <
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout
>
AthenaBackgroundField<DeviceType, LayoutType> read_athena_binary_background_field(
    const std::string& file_path,
    const AthenaBinaryFieldReadOptions<athena_binary_space_dim>& options =
        AthenaBinaryFieldReadOptions<athena_binary_space_dim>{}) {
    Impl::validate_little_endian_host();
    Impl::validate_positive_finite_scale(options.coordinate_scale, "coordinate_scale");
    Impl::validate_positive_finite_scale(options.magnetic_field_scale,
                                         "magnetic_field_scale");
    Impl::validate_positive_finite_scale(options.velocity_scale, "velocity_scale");

    std::ifstream stream(file_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("AthenaFieldReader: failed to open Athena binary file.");
    }

    const auto nx = Impl::read_binary_scalar<std::int32_t>(stream, "nx");
    const auto ny = Impl::read_binary_scalar<std::int32_t>(stream, "ny");
    const std::size_t cell_count = Impl::checked_cell_count(nx, ny);

    auto x_edges =
        Impl::read_double_vector(stream, static_cast<std::size_t>(nx) + 1, "x_edges");
    auto y_edges =
        Impl::read_double_vector(stream, static_cast<std::size_t>(ny) + 1, "y_edges");
    Impl::scale_coordinates(x_edges, options.coordinate_scale);
    Impl::scale_coordinates(y_edges, options.coordinate_scale);
    Impl::validate_face_edges(x_edges, "x");
    Impl::validate_face_edges(y_edges, "y");

    AthenaBackgroundField<DeviceType, LayoutType> result;
    result.nx = nx;
    result.ny = ny;
    result.grid = Impl::make_grid<DeviceType, LayoutType>(nx, ny, x_edges, y_edges,
                                                          options);
    result.magnetic_field =
        typename AthenaBackgroundField<DeviceType, LayoutType>::vector_field_type(
            result.grid, "athena_magnetic_field", GridCentering::CellCentered);
    result.velocity_field =
        typename AthenaBackgroundField<DeviceType, LayoutType>::vector_field_type(
            result.grid, "athena_velocity_field", GridCentering::CellCentered);

    auto magnetic_host = Kokkos::create_mirror_view(result.magnetic_field.data);
    auto velocity_host = Kokkos::create_mirror_view(result.velocity_field.data);

    const char* magnetic_names[athena_binary_vector_dim] = {"Bx", "By", "Bz"};
    const char* velocity_names[athena_binary_vector_dim] = {"Vx", "Vy", "Vz"};
    for (int component = 0; component < athena_binary_vector_dim; ++component) {
        const auto values = Impl::read_double_vector(stream, cell_count,
                                                     magnetic_names[component]);
        Impl::copy_component_to_host(values, nx, ny, options.magnetic_field_scale,
                                     component, magnetic_names[component],
                                     result.magnetic_field, magnetic_host);
    }
    for (int component = 0; component < athena_binary_vector_dim; ++component) {
        const auto values = Impl::read_double_vector(stream, cell_count,
                                                     velocity_names[component]);
        Impl::copy_component_to_host(values, nx, ny, options.velocity_scale,
                                     component, velocity_names[component],
                                     result.velocity_field, velocity_host);
    }

    Impl::validate_end_of_file(stream);

    Kokkos::deep_copy(result.magnetic_field.data, magnetic_host);
    Kokkos::deep_copy(result.velocity_field.data, velocity_host);
    result.magnetic_field.fill_ghost_cells(options.ghost_fill_mode,
                                           options.close_periodic_faces);
    result.velocity_field.fill_ghost_cells(options.ghost_fill_mode,
                                           options.close_periodic_faces);

    return result;
}

} // namespace AthenaFieldIO
