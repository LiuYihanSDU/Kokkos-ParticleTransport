#pragma once

#include <cstddef>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "AnalyticCoordinateGrid.hpp"
#include "KokkosDevice.hpp"
#include "MappedCoordinateGrid.hpp"
#include "StoredCoordinateGrid.hpp"

/**
 * Ghost-cell fill policy for field storage.
 */
enum class FieldGhostFillMode : int {
    PeriodicOrClamped = 0,
    PeriodicOnly = 1
};

/**
 * Grid-bound scalar or vector field stored over the grid's ghosted coordinate domain.
 */
template <
    typename GridType,
    int ComponentDim,
    typename LayoutType = typename GridType::layout_type
>
struct Field {
    using grid_type = GridType;
    using execution_space = typename grid_type::execution_space;
    using memory_space = typename grid_type::memory_space;
    using layout_type = LayoutType;
    using index_array_type = typename grid_type::index_array_type;
    using coordinate_array_type = typename grid_type::coordinate_array_type;
    using centering_array_type = Kokkos::Array<GridCentering, grid_type::space_dim>;
    using view_type = Kokkos::View<double*[ComponentDim], layout_type, memory_space>;
    using const_random_access_view_type =
        Kokkos::View<const double*[ComponentDim], layout_type, memory_space,
                     Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using size_type = typename view_type::size_type;
    using stride_array_type = Kokkos::Array<size_type, grid_type::space_dim>;

    static constexpr int space_dim = grid_type::space_dim;
    static constexpr int component_dim = ComponentDim;
    static_assert(ComponentDim > 0, "Field component dimension must be positive.");

    /**
     * Device-side read-only accessor using Kokkos RandomAccess memory traits.
     */
    struct RandomAccessAccessor {
        const_random_access_view_type values;
        index_array_type lower_index_cache{};
        index_array_type extents_cache{};
        stride_array_type strides_cache{};

        KOKKOS_INLINE_FUNCTION
        double operator()(const size_type point_index, const int component) const {
            return values(point_index, component);
        }

        KOKKOS_INLINE_FUNCTION
        double operator()(const index_array_type& indices, const int component) const {
            return values(flatten(indices), component);
        }

        KOKKOS_INLINE_FUNCTION
        double operator()(const index_array_type& indices) const {
            return values(flatten(indices), 0);
        }

        KOKKOS_INLINE_FUNCTION
        size_type flatten(const index_array_type& indices) const {
            size_type linear_index = 0;
            for (int dim = 0; dim < space_dim; ++dim) {
                linear_index += static_cast<size_type>(
                    (indices[dim] - lower_index_cache[dim]) * strides_cache[dim]);
            }
            return linear_index;
        }
    };

    grid_type grid;
    centering_array_type centerings{};
    index_array_type lower_index_cache{};
    index_array_type upper_index_cache{};
    index_array_type extents_cache{};
    stride_array_type strides_cache{};
    size_type value_count_cache{0};
    view_type data;

    Field() = default;

    /**
     * Build a field using the same centering in every coordinate direction.
     */
    Field(const grid_type& input_grid,
          const char* label,
          const GridCentering centering = GridCentering::CellCentered)
        : Field(input_grid, label, make_centering_array(centering)) {}

    /**
     * Build a field with per-direction coordinate centering.
     */
    Field(const grid_type& input_grid,
          const char* label,
          const centering_array_type& input_centerings)
        : grid(input_grid),
          centerings(input_centerings),
          lower_index_cache(compute_lower_indices(input_grid, input_centerings)),
          upper_index_cache(compute_upper_indices(input_grid, input_centerings)),
          extents_cache(compute_extents(input_grid, input_centerings)),
          strides_cache(compute_strides(extents_cache)),
          value_count_cache(compute_value_count(extents_cache)),
          data(label, value_count_cache) {}

    /**
     * Return the total number of stored field points, including ghost points.
     */
    KOKKOS_INLINE_FUNCTION
    size_type point_count() const {
        return value_count_cache;
    }

    /**
     * Return the ghosted extent in one coordinate direction.
     */
    KOKKOS_INLINE_FUNCTION
    int extent(const int dim) const {
        return extents_cache[dim];
    }

    /**
     * Return the data view extent.
     */
    KOKKOS_INLINE_FUNCTION
    size_type extent(const std::size_t dim) const {
        return data.extent(dim);
    }

    /**
     * Return the field centering in one coordinate direction.
     */
    KOKKOS_INLINE_FUNCTION
    GridCentering centering(const int dim) const {
        return centerings[dim];
    }

    /**
     * Return the lower logical field index for a selected domain.
     */
    KOKKOS_INLINE_FUNCTION
    int lower_index(const int dim, const GridDomain domain) const {
        return centerings[dim] == GridCentering::CellCentered
                   ? grid.lower_cell_index(dim, domain)
                   : grid.lower_face_index(dim, domain);
    }

    /**
     * Return the upper logical field index for a selected domain.
     */
    KOKKOS_INLINE_FUNCTION
    int upper_index(const int dim, const GridDomain domain) const {
        return centerings[dim] == GridCentering::CellCentered
                   ? grid.upper_cell_index(dim, domain)
                   : grid.upper_face_index(dim, domain);
    }

    /**
     * Return the field coordinate count in a selected domain.
     */
    KOKKOS_INLINE_FUNCTION
    int domain_extent(const int dim, const GridDomain domain) const {
        return upper_index(dim, domain) - lower_index(dim, domain) + 1;
    }

    /**
     * Flatten logical grid indices into the field storage index.
     */
    KOKKOS_INLINE_FUNCTION
    size_type flatten(const index_array_type& indices) const {
        size_type linear_index = 0;
        for (int dim = 0; dim < space_dim; ++dim) {
            linear_index += static_cast<size_type>(
                (indices[dim] - lower_index_cache[dim]) * strides_cache[dim]);
        }
        return linear_index;
    }

    /**
     * Recover logical grid indices from a flattened field storage index.
     */
    KOKKOS_INLINE_FUNCTION
    index_array_type logical_indices(size_type linear_index) const {
        index_array_type indices{};
        for (int dim = 0; dim < space_dim; ++dim) {
            const int storage_index = static_cast<int>(linear_index % extents_cache[dim]);
            indices[dim] = lower_index_cache[dim] + storage_index;
            linear_index /= static_cast<size_type>(extents_cache[dim]);
        }
        return indices;
    }

    /**
     * Writable field access by flattened point index and component.
     */
    KOKKOS_INLINE_FUNCTION
    double& operator()(const size_type point_index, const int component) {
        return data(point_index, component);
    }

    /**
     * Read-only field access by flattened point index and component.
     */
    KOKKOS_INLINE_FUNCTION
    double operator()(const size_type point_index, const int component) const {
        return data(point_index, component);
    }

    /**
     * Writable field access by logical grid index and component.
     */
    KOKKOS_INLINE_FUNCTION
    double& operator()(const index_array_type& indices, const int component) {
        return data(flatten(indices), component);
    }

    /**
     * Read-only field access by logical grid index and component.
     */
    KOKKOS_INLINE_FUNCTION
    double operator()(const index_array_type& indices, const int component) const {
        return data(flatten(indices), component);
    }

    /**
     * Writable scalar access by logical grid index.
     */
    KOKKOS_INLINE_FUNCTION
    double& operator()(const index_array_type& indices) {
        return data(flatten(indices), 0);
    }

    /**
     * Read-only scalar access by logical grid index.
     */
    KOKKOS_INLINE_FUNCTION
    double operator()(const index_array_type& indices) const {
        return data(flatten(indices), 0);
    }

    /**
     * Return a const RandomAccess view for kernels that perform irregular lookups.
     */
    const_random_access_view_type random_access_view() const {
        return const_random_access_view_type(data);
    }

    /**
     * Return a device-side random-access accessor with logical-index support.
     */
    RandomAccessAccessor random_access() const {
        return RandomAccessAccessor{random_access_view(), lower_index_cache, extents_cache,
                                    strides_cache};
    }

    /**
     * Fill ghost points from periodic wrapped indices or clamped physical boundary values.
     */
    void fill_ghost_cells(
        const FieldGhostFillMode mode = FieldGhostFillMode::PeriodicOrClamped,
        const bool close_periodic_faces = true) {
        if (close_periodic_faces) {
            enforce_periodic_face_closure();
        }

        const Field field = *this;
        Kokkos::parallel_for(
            "Field::fill_ghost_cells",
            Kokkos::RangePolicy<execution_space>(0, value_count_cache),
            KOKKOS_LAMBDA(const size_type point_index) {
                const index_array_type destination = field.logical_indices(point_index);
                index_array_type source = destination;
                bool is_ghost_point = false;

                for (int dim = 0; dim < space_dim; ++dim) {
                    const int lower = field.lower_index(dim, GridDomain::PhysicalDomain);
                    const int upper = field.upper_index(dim, GridDomain::PhysicalDomain);
                    if (destination[dim] < lower || destination[dim] > upper) {
                        is_ghost_point = true;
                        if (field.grid.is_periodic(dim)) {
                            source[dim] = field.wrap_periodic_index(dim, destination[dim]);
                        } else {
                            if (mode == FieldGhostFillMode::PeriodicOnly) {
                                return;
                            }
                            source[dim] = destination[dim] < lower ? lower : upper;
                        }
                    }
                }

                if (!is_ghost_point) {
                    return;
                }

                const size_type source_index = field.flatten(source);
                for (int component = 0; component < ComponentDim; ++component) {
                    field.data(point_index, component) = field.data(source_index, component);
                }
            });
        Kokkos::fence("Field::fill_ghost_cells");
    }

    /**
     * Copy periodic lower physical faces onto redundant upper physical faces.
     */
    void enforce_periodic_face_closure() {
        const Field field = *this;
        for (int closure_dim = 0; closure_dim < space_dim; ++closure_dim) {
            if (centerings[closure_dim] != GridCentering::FaceCentered ||
                !grid.is_periodic(closure_dim)) {
                continue;
            }

            Kokkos::parallel_for(
                "Field::enforce_periodic_face_closure",
                Kokkos::RangePolicy<execution_space>(0, value_count_cache),
                KOKKOS_LAMBDA(const size_type point_index) {
                    const index_array_type destination = field.logical_indices(point_index);
                    for (int dim = 0; dim < space_dim; ++dim) {
                        if (destination[dim] <
                                field.lower_index(dim, GridDomain::PhysicalDomain) ||
                            destination[dim] >
                                field.upper_index(dim, GridDomain::PhysicalDomain)) {
                            return;
                        }
                    }

                    if (destination[closure_dim] !=
                        field.upper_index(closure_dim, GridDomain::PhysicalDomain)) {
                        return;
                    }

                    index_array_type source = destination;
                    source[closure_dim] =
                        field.lower_index(closure_dim, GridDomain::PhysicalDomain);
                    const size_type source_index = field.flatten(source);
                    for (int component = 0; component < ComponentDim; ++component) {
                        field.data(point_index, component) =
                            field.data(source_index, component);
                    }
                });
            Kokkos::fence("Field::enforce_periodic_face_closure");
        }
    }

    /**
     * Write the field to a legacy ASCII VTK structured-grid file.
     */
    void write_vtk(const std::string& file_path,
                   const std::string& field_name = "field",
                   const GridDomain domain = GridDomain::PhysicalDomain) const {
#ifdef NDEBUG
        (void)file_path;
        (void)field_name;
        (void)domain;
        throw std::runtime_error("Field::write_vtk is available only in debug builds.");
#else
        if constexpr (space_dim < 1 || space_dim > 3) {
            throw std::runtime_error("Field::write_vtk supports SpaceDim from 1 to 3.");
        } else {
            auto data_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);

            std::ofstream file(file_path);
            if (!file) {
                throw std::runtime_error("Field::write_vtk failed to open output file.");
            }
            file << std::setprecision(17);

            if constexpr (requires { grid_type::coordinate_representation; }) {
                if constexpr (grid_type::coordinate_representation ==
                              GridCoordinateRepresentation::StoredCoordinates) {
                    auto cell_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                                         grid.cell_coordinates);
                    auto face_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                                         grid.face_coordinates);
                    const auto coordinate_at =
                        [this, cell_host, face_host](const int dim,
                                                     const int logical_index,
                                                     const GridCentering centering) {
                            const int storage_index =
                                logical_index + grid.ghost_extent(dim, LowerBoundary);
                            return centering == GridCentering::CellCentered
                                       ? cell_host(dim, storage_index)
                                       : face_host(dim, storage_index);
                        };
                    write_vtk_payload(file, field_name, domain, data_host, coordinate_at);
                } else if constexpr (grid_type::coordinate_representation ==
                                     GridCoordinateRepresentation::AnalyticCoordinates) {
                    auto discretization_host = Kokkos::create_mirror_view_and_copy(
                        Kokkos::HostSpace{}, grid.mapping.discretization);
                    auto start_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                                          grid.mapping.start);
                    auto step_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                                         grid.mapping.step);
                    auto step_growth_host = Kokkos::create_mirror_view_and_copy(
                        Kokkos::HostSpace{}, grid.mapping.step_growth);
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
                    write_vtk_payload(file, field_name, domain, data_host, coordinate_at);
                } else if constexpr (grid_type::coordinate_representation ==
                                     GridCoordinateRepresentation::MappedCoordinates) {
                    const auto coordinate_at =
                        MappedCoordinateHostAccess::create_coordinate_provider_from_grid(grid);
                    write_vtk_payload(file, field_name, domain, data_host, coordinate_at);
                } else {
                    const auto coordinate_at = [this](const int dim,
                                                      const int logical_index,
                                                      const GridCentering centering) {
                        return grid.coordinate(dim, logical_index, centering);
                    };
                    write_vtk_payload(file, field_name, domain, data_host, coordinate_at);
                }
            } else {
                if constexpr (requires(const grid_type& input_grid) {
                                  input_grid.coordinate(0, 0, GridCentering::CellCentered);
                              }) {
                    const auto coordinate_at = [this](const int dim,
                                                      const int logical_index,
                                                      const GridCentering centering) {
                        return grid.coordinate(dim, logical_index, centering);
                    };
                    write_vtk_payload(file, field_name, domain, data_host, coordinate_at);
                } else {
                    throw std::runtime_error(
                        "Field::write_vtk requires a coordinate grid, not metadata-only Grid.");
                }
            }
        }
#endif
    }

private:
    /**
     * Build a same-centering array.
     */
    static centering_array_type make_centering_array(const GridCentering centering) {
        centering_array_type result{};
        for (int dim = 0; dim < space_dim; ++dim) {
            result[dim] = centering;
        }
        return result;
    }

    /**
     * Compute ghosted lower logical indices for field storage.
     */
    static index_array_type compute_lower_indices(const grid_type& input_grid,
                                                  const centering_array_type& input_centerings) {
        index_array_type result{};
        for (int dim = 0; dim < space_dim; ++dim) {
            result[dim] = input_centerings[dim] == GridCentering::CellCentered
                              ? input_grid.lower_cell_index(dim, GridDomain::GhostedDomain)
                              : input_grid.lower_face_index(dim, GridDomain::GhostedDomain);
        }
        return result;
    }

    /**
     * Compute ghosted upper logical indices for field storage.
     */
    static index_array_type compute_upper_indices(const grid_type& input_grid,
                                                  const centering_array_type& input_centerings) {
        index_array_type result{};
        for (int dim = 0; dim < space_dim; ++dim) {
            result[dim] = input_centerings[dim] == GridCentering::CellCentered
                              ? input_grid.upper_cell_index(dim, GridDomain::GhostedDomain)
                              : input_grid.upper_face_index(dim, GridDomain::GhostedDomain);
        }
        return result;
    }

    /**
     * Compute ghosted field extents.
     */
    static index_array_type compute_extents(const grid_type& input_grid,
                                            const centering_array_type& input_centerings) {
        index_array_type result{};
        for (int dim = 0; dim < space_dim; ++dim) {
            result[dim] = input_grid.total_extent(dim, input_centerings[dim]);
        }
        return result;
    }

    /**
     * Compute row-major strides with dimension zero contiguous.
     */
    static stride_array_type compute_strides(const index_array_type& input_extents) {
        stride_array_type result{};
        size_type stride = 1;
        for (int dim = 0; dim < space_dim; ++dim) {
            result[dim] = stride;
            stride *= static_cast<size_type>(input_extents[dim]);
        }
        return result;
    }

    /**
     * Compute the flattened field point count.
     */
    static size_type compute_value_count(const index_array_type& input_extents) {
        size_type count = 1;
        for (int dim = 0; dim < space_dim; ++dim) {
            count *= static_cast<size_type>(input_extents[dim]);
        }
        return count;
    }

    /**
     * Wrap a logical index into the physical periodic range for the field centering.
     */
    KOKKOS_INLINE_FUNCTION
    int wrap_periodic_index(const int dim, const int index) const {
        if (centerings[dim] == GridCentering::CellCentered) {
            return grid.wrap_cell_index(dim, index);
        }

        const int physical_extent = grid.extent(dim);
        if (physical_extent <= 0) {
            return index;
        }

        int wrapped = index % physical_extent;
        if (wrapped < 0) {
            wrapped += physical_extent;
        }
        return wrapped;
    }

    /**
     * Evaluate an analytic coordinate on the host.
     */
    static double analytic_coordinate_on_host(const AnalyticGridDiscretization discretization,
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
     * Evaluate an analytic face coordinate on the host.
     */
    static double analytic_face_coordinate_on_host(
        const AnalyticGridDiscretization discretization,
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
     * Return a field value from a host mirror, filling missing vector components with zero.
     */
    template <typename HostDataView>
    static double component_value_on_host(const HostDataView& data_host,
                                          const size_type point_index,
                                          const int component) {
        if (component >= ComponentDim) {
            return 0.0;
        }
        return data_host(point_index, component);
    }

    /**
     * Convert grid coordinates to Cartesian VTK point coordinates.
     */
    Kokkos::Array<double, 3> vtk_point_from_coordinates(
        const coordinate_array_type& q) const {
        Kokkos::Array<double, 3> point{};
        switch (grid.coordinate_system()) {
            case OrthogonalCoordinateSystem::Cartesian:
                point[0] = q[0];
                if constexpr (space_dim > 1) {
                    point[1] = q[1];
                }
                if constexpr (space_dim > 2) {
                    point[2] = q[2];
                }
                break;
            case OrthogonalCoordinateSystem::Polar: {
                const double radius = q[0];
                const double phi = space_dim > 1 ? q[1] : 0.0;
                point[0] = radius * std::cos(phi);
                point[1] = radius * std::sin(phi);
                if constexpr (space_dim > 2) {
                    point[2] = q[2];
                }
                break;
            }
            case OrthogonalCoordinateSystem::Spherical: {
                const double radius = q[0];
                if constexpr (space_dim > 1) {
                    const double theta = q[1];
                    const double phi = space_dim > 2 ? q[2] : 0.0;
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
     * Convert coordinate-basis vector components to Cartesian VTK vector components.
     */
    template <typename HostDataView>
    Kokkos::Array<double, 3> vtk_vector_from_components(
        const coordinate_array_type& q,
        const HostDataView& data_host,
        const size_type point_index) const {
        Kokkos::Array<double, 3> vector{};
        const double v0 = component_value_on_host(data_host, point_index, 0);
        const double v1 = component_value_on_host(data_host, point_index, 1);
        const double v2 = component_value_on_host(data_host, point_index, 2);

        switch (grid.coordinate_system()) {
            case OrthogonalCoordinateSystem::Cartesian:
                vector[0] = v0;
                vector[1] = v1;
                vector[2] = v2;
                break;
            case OrthogonalCoordinateSystem::Polar: {
                const double phi = space_dim > 1 ? q[1] : 0.0;
                vector[0] = v0 * std::cos(phi) - v1 * std::sin(phi);
                vector[1] = v0 * std::sin(phi) + v1 * std::cos(phi);
                vector[2] = v2;
                break;
            }
            case OrthogonalCoordinateSystem::Spherical: {
                if constexpr (space_dim > 1) {
                    const double theta = q[1];
                    const double phi = space_dim > 2 ? q[2] : 0.0;
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

    /**
     * Write VTK points and field payload using a host coordinate provider.
     */
    template <typename HostDataView, typename CoordinateAt>
    void write_vtk_payload(std::ofstream& file,
                           const std::string& field_name,
                           const GridDomain domain,
                           const HostDataView& data_host,
                           CoordinateAt coordinate_at) const {
        const int nx = domain_extent(0, domain);
        const int ny = space_dim > 1 ? domain_extent(1, domain) : 1;
        const int nz = space_dim > 2 ? domain_extent(2, domain) : 1;
        const size_type output_count = static_cast<size_type>(nx) *
                                       static_cast<size_type>(ny) *
                                       static_cast<size_type>(nz);

        file << "# vtk DataFile Version 3.0\n";
        file << field_name << "\n";
        file << "ASCII\n";
        file << "DATASET STRUCTURED_GRID\n";
        file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
        file << "POINTS " << output_count << " double\n";

        for_each_output_index(domain, nx, ny, nz, [&](const index_array_type& indices) {
            coordinate_array_type q{};
            for (int dim = 0; dim < space_dim; ++dim) {
                q[dim] = coordinate_at(dim, indices[dim], centerings[dim]);
            }
            const auto point = vtk_point_from_coordinates(q);
            file << point[0] << " " << point[1] << " " << point[2] << "\n";
        });

        file << "POINT_DATA " << output_count << "\n";
        if constexpr (ComponentDim == 1) {
            file << "SCALARS " << field_name << " double 1\n";
            file << "LOOKUP_TABLE default\n";
            for_each_output_index(domain, nx, ny, nz, [&](const index_array_type& indices) {
                file << data_host(flatten(indices), 0) << "\n";
            });
        } else {
            file << "VECTORS " << field_name << " double\n";
            for_each_output_index(domain, nx, ny, nz, [&](const index_array_type& indices) {
                coordinate_array_type q{};
                for (int dim = 0; dim < space_dim; ++dim) {
                    q[dim] = coordinate_at(dim, indices[dim], centerings[dim]);
                }
                const auto vector = vtk_vector_from_components(q, data_host, flatten(indices));
                file << vector[0] << " " << vector[1] << " " << vector[2] << "\n";
            });
        }
    }

    /**
     * Visit output logical indices in VTK structured-grid order.
     */
    template <typename Visitor>
    void for_each_output_index(const GridDomain domain,
                               const int nx,
                               const int ny,
                               const int nz,
                               Visitor visitor) const {
        const int lower_x = lower_index(0, domain);
        const int lower_y = space_dim > 1 ? lower_index(1, domain) : 0;
        const int lower_z = space_dim > 2 ? lower_index(2, domain) : 0;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    index_array_type indices{};
                    indices[0] = lower_x + i;
                    if constexpr (space_dim > 1) {
                        indices[1] = lower_y + j;
                    }
                    if constexpr (space_dim > 2) {
                        indices[2] = lower_z + k;
                    }
                    visitor(indices);
                }
            }
        }
    }
};

template <typename GridType,
          typename LayoutType = typename GridType::layout_type>
using ScalarField = Field<GridType, 1, LayoutType>;

template <typename GridType,
          int ComponentDim = GridType::vec_dim,
          typename LayoutType = typename GridType::layout_type>
using VectorField = Field<GridType, ComponentDim, LayoutType>;
