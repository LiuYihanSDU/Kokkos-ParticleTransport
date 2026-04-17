#pragma once

#include "KokkosDevice.hpp"
#include "OrthogonalCoordinate.hpp"

/**
 * Coordinate representation selected by a concrete grid type.
 */
enum class GridCoordinateRepresentation {
    StoredCoordinates,
    AnalyticCoordinates,
    MappedCoordinates
};

/**
 * Coordinate placement on a structured grid axis.
 */
enum class GridCentering : int {
    CellCentered = 0,
    FaceCentered = 1
};

/**
 * Coordinate domain used for boundary and ghost-cell queries.
 */
enum class GridDomain : int {
    PhysicalDomain = 0,
    GhostedDomain = 1
};

/**
 * Boundary side index used by ghost-cell metadata.
 */
enum GridBoundarySide : int {
    LowerBoundary = 0,
    UpperBoundary = 1
};

/**
 * Common grid metadata stored in the project device memory space.
 */
template <
    int SpaceDim,
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout,
    int VecDim = SpaceDim
>
struct Grid {
    using device_type = DeviceType;
    using execution_space = typename DeviceTraits<DeviceType>::execution_space;
    using memory_space = typename DeviceTraits<DeviceType>::memory_space;
    using layout_type = LayoutType;
    using index_array_type = Kokkos::Array<int, SpaceDim>;
    using ghost_array_type = Kokkos::Array<Kokkos::Array<int, 2>, SpaceDim>;
    using periodic_array_type = Kokkos::Array<int, SpaceDim>;
    using extent_view_type = Kokkos::View<int*, layout_type, memory_space>;
    using ghost_extent_view_type = Kokkos::View<int*[2], layout_type, memory_space>;
    using periodic_view_type = Kokkos::View<int*, layout_type, memory_space>;
    using coordinate_array_type = Kokkos::Array<double, SpaceDim>;
    using metric_array_type = Kokkos::Array<double, VecDim>;
    using metric_type = OrthogonalGeometry<SpaceDim, VecDim>;

    static constexpr int space_dim = SpaceDim;
    static constexpr int vec_dim = VecDim;
    static constexpr int default_ghost_cell_layers = 2;

    extent_view_type n;
    ghost_extent_view_type ghost_cells;
    periodic_view_type periodic;
    index_array_type n_cache{};
    ghost_array_type ghost_cells_cache{};
    periodic_array_type periodic_cache{};
    metric_type metric;

    Grid() = default;

    /**
     * Build grid metadata with default ghost cells and non-periodic boundaries.
     */
    explicit Grid(const index_array_type& extents,
                  const OrthogonalCoordinateSystem input_coordinate_system =
                      OrthogonalCoordinateSystem::Cartesian)
        : n("grid_n", SpaceDim),
          ghost_cells("grid_ghost_cells", SpaceDim),
          periodic("grid_periodic", SpaceDim),
          metric(input_coordinate_system) {
        ghost_array_type ghosts{};
        for (int dim = 0; dim < SpaceDim; ++dim) {
            ghosts[dim][LowerBoundary] = default_ghost_cell_layers;
            ghosts[dim][UpperBoundary] = default_ghost_cell_layers;
        }
        const periodic_array_type periodic_flags{};
        assign_metadata(extents, ghosts, periodic_flags);
    }

    /**
     * Build grid metadata from host-side extents, ghost cells, and periodic flags.
     */
    Grid(const index_array_type& extents,
         const ghost_array_type& ghosts,
         const periodic_array_type& periodic_flags,
         const OrthogonalCoordinateSystem input_coordinate_system =
             OrthogonalCoordinateSystem::Cartesian)
        : n("grid_n", SpaceDim),
          ghost_cells("grid_ghost_cells", SpaceDim),
          periodic("grid_periodic", SpaceDim),
          metric(input_coordinate_system) {
        assign_metadata(extents, ghosts, periodic_flags);
    }

    /**
     * Wrap preallocated device-side metadata views.
     */
    Grid(extent_view_type input_n,
         ghost_extent_view_type input_ghost_cells,
         periodic_view_type input_periodic,
         const OrthogonalCoordinateSystem input_coordinate_system =
             OrthogonalCoordinateSystem::Cartesian)
        : n(input_n),
          ghost_cells(input_ghost_cells),
          periodic(input_periodic),
          metric(input_coordinate_system) {
        refresh_metadata_cache_on_host();
    }

    /**
     * Refresh small host-built metadata caches used by device hot paths.
     */
    void refresh_metadata_cache_on_host() {
        auto n_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, n);
        auto ghost_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ghost_cells);
        auto periodic_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, periodic);

        for (int dim = 0; dim < SpaceDim; ++dim) {
            n_cache[dim] = n_host(dim);
            ghost_cells_cache[dim][LowerBoundary] = ghost_host(dim, LowerBoundary);
            ghost_cells_cache[dim][UpperBoundary] = ghost_host(dim, UpperBoundary);
            periodic_cache[dim] = periodic_host(dim);
        }
    }

    /**
     * Return the coordinate system used by this grid geometry.
     */
    KOKKOS_INLINE_FUNCTION
    OrthogonalCoordinateSystem coordinate_system() const {
        return metric.coordinate_system;
    }

    /**
     * Return the Lame coefficient h_i at a coordinate point.
     */
    KOKKOS_INLINE_FUNCTION
    double h(const int component, const coordinate_array_type& q) const {
        return metric.h(component, q);
    }

    /**
     * Return all Lame coefficients at a coordinate point.
     */
    KOKKOS_INLINE_FUNCTION
    metric_array_type h(const coordinate_array_type& q) const {
        return metric.h(q);
    }

    /**
     * Return the inverse Lame coefficient 1 / h_i at a coordinate point.
     */
    KOKKOS_INLINE_FUNCTION
    double h_inv(const int component, const coordinate_array_type& q) const {
        return metric.h_inv(component, q);
    }

    /**
     * Return all inverse Lame coefficients at a coordinate point.
     */
    KOKKOS_INLINE_FUNCTION
    metric_array_type h_inv(const coordinate_array_type& q) const {
        return metric.h_inv(q);
    }

    /**
     * Return the product of active-coordinate Lame coefficients at a coordinate point.
     */
    KOKKOS_INLINE_FUNCTION
    double jacobian(const coordinate_array_type& q) const {
        return metric.jacobian(q);
    }

    /**
     * Return the physical line element h_i dq_i along one coordinate direction.
     */
    KOKKOS_INLINE_FUNCTION
    double physical_distance(const int component,
                             const double coordinate_spacing,
                             const coordinate_array_type& q) const {
        return metric.physical_distance(component, coordinate_spacing, q);
    }

    /**
     * Return the number of physical cells in one dimension.
     */
    KOKKOS_INLINE_FUNCTION
    int extent(const int dim) const {
        return n_cache[dim];
    }

    /**
     * Return the physical coordinate count for one centering on an axis.
     */
    KOKKOS_INLINE_FUNCTION
    int extent(const int dim, const GridCentering centering) const {
        return n_cache[dim] + face_offset(centering);
    }

    /**
     * Return the number of ghost cells on one boundary side.
     */
    KOKKOS_INLINE_FUNCTION
    int ghost_extent(const int dim, const int side) const {
        return ghost_cells_cache[dim][side];
    }

    /**
     * Return the physical plus ghost-cell extent in one dimension.
     */
    KOKKOS_INLINE_FUNCTION
    int total_extent(const int dim) const {
        return n_cache[dim] + ghost_cells_cache[dim][LowerBoundary] +
               ghost_cells_cache[dim][UpperBoundary];
    }

    /**
     * Return the ghosted coordinate count for one centering on an axis.
     */
    KOKKOS_INLINE_FUNCTION
    int total_extent(const int dim, const GridCentering centering) const {
        return total_extent(dim) + face_offset(centering);
    }

    /**
     * Return whether one dimension uses periodic boundary indexing.
     */
    KOKKOS_INLINE_FUNCTION
    bool is_periodic(const int dim) const {
        return periodic_cache[dim] != 0;
    }

    /**
     * Return the lower logical cell index for the selected domain.
     */
    KOKKOS_INLINE_FUNCTION
    int lower_cell_index(const int dim, const GridDomain domain) const {
        return domain == GridDomain::GhostedDomain ? -ghost_cells_cache[dim][LowerBoundary] : 0;
    }

    /**
     * Return the upper logical cell index for the selected domain.
     */
    KOKKOS_INLINE_FUNCTION
    int upper_cell_index(const int dim, const GridDomain domain) const {
        return domain == GridDomain::GhostedDomain
                   ? n_cache[dim] + ghost_cells_cache[dim][UpperBoundary] - 1
                   : n_cache[dim] - 1;
    }

    /**
     * Return the lower logical face index for the selected domain.
     */
    KOKKOS_INLINE_FUNCTION
    int lower_face_index(const int dim, const GridDomain domain) const {
        return lower_cell_index(dim, domain);
    }

    /**
     * Return the upper logical face index for the selected domain.
     */
    KOKKOS_INLINE_FUNCTION
    int upper_face_index(const int dim, const GridDomain domain) const {
        return upper_cell_index(dim, domain) + 1;
    }

    /**
     * Return whether a logical cell index lies inside the selected domain.
     */
    KOKKOS_INLINE_FUNCTION
    bool contains_cell_index(const int dim,
                             const int index,
                             const GridDomain domain) const {
        return index >= lower_cell_index(dim, domain) &&
               index <= upper_cell_index(dim, domain);
    }

    /**
     * Wrap a logical cell index back into the physical periodic index range.
     */
    KOKKOS_INLINE_FUNCTION
    int wrap_cell_index(const int dim, const int index) const {
        const int physical_extent = n_cache[dim];
        if (!is_periodic(dim) || physical_extent <= 0) {
            return index;
        }

        int wrapped = index % physical_extent;
        if (wrapped < 0) {
            wrapped += physical_extent;
        }
        return wrapped;
    }

private:
    /**
     * Return the extra coordinate slot needed by face-centered geometry.
     */
    KOKKOS_INLINE_FUNCTION
    static int face_offset(const GridCentering centering) {
        return centering == GridCentering::FaceCentered ? 1 : 0;
    }

    /**
     * Copy host-side metadata arrays into device-side Views.
     */
    void assign_metadata(const index_array_type& extents,
                         const ghost_array_type& ghosts,
                         const periodic_array_type& periodic_flags) {
        auto n_host = Kokkos::create_mirror_view(n);
        auto ghost_host = Kokkos::create_mirror_view(ghost_cells);
        auto periodic_host = Kokkos::create_mirror_view(periodic);

        for (int dim = 0; dim < SpaceDim; ++dim) {
            n_host(dim) = extents[dim];
            ghost_host(dim, LowerBoundary) = ghosts[dim][LowerBoundary];
            ghost_host(dim, UpperBoundary) = ghosts[dim][UpperBoundary];
            periodic_host(dim) = periodic_flags[dim];
            n_cache[dim] = extents[dim];
            ghost_cells_cache[dim][LowerBoundary] = ghosts[dim][LowerBoundary];
            ghost_cells_cache[dim][UpperBoundary] = ghosts[dim][UpperBoundary];
            periodic_cache[dim] = periodic_flags[dim];
        }

        Kokkos::deep_copy(n, n_host);
        Kokkos::deep_copy(ghost_cells, ghost_host);
        Kokkos::deep_copy(periodic, periodic_host);
    }
};
