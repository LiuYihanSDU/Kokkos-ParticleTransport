#pragma once

#include <cstddef>
#include <stdexcept>

#include <Kokkos_MathematicalFunctions.hpp>

#include "Grid.hpp"

/**
 * Grid whose coordinates are explicitly stored in device memory.
 */
template <
    int SpaceDim,
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout,
    int VecDim = SpaceDim
>
struct StoredCoordinateGrid : public Grid<SpaceDim, DeviceType, LayoutType, VecDim> {
    using base_type = Grid<SpaceDim, DeviceType, LayoutType, VecDim>;
    using execution_space = typename base_type::execution_space;
    using memory_space = typename base_type::memory_space;
    using layout_type = typename base_type::layout_type;
    using index_array_type = typename base_type::index_array_type;
    using coordinate_array_type = typename base_type::coordinate_array_type;
    using metric_array_type = typename base_type::metric_array_type;
    using centering_array_type = Kokkos::Array<GridCentering, SpaceDim>;
    using coordinate_view_type = Kokkos::View<double**, layout_type, memory_space>;
    using bound_array_type = Kokkos::Array<double, SpaceDim>;

    using base_type::h;
    using base_type::h_inv;
    using base_type::jacobian;
    using base_type::physical_distance;

    static constexpr GridCoordinateRepresentation coordinate_representation =
        GridCoordinateRepresentation::StoredCoordinates;

    coordinate_view_type cell_coordinates;
    coordinate_view_type face_coordinates;
    bound_array_type physical_lower_bound{};
    bound_array_type physical_upper_bound{};
    bound_array_type ghost_lower_bound{};
    bound_array_type ghost_upper_bound{};
    bound_array_type periodic_width{};
    bound_array_type periodic_inv_width{};

    StoredCoordinateGrid() = default;

    /**
     * Build an explicit-coordinate grid from common metadata and face coordinates.
     */
    StoredCoordinateGrid(const base_type& metadata,
                         coordinate_view_type input_face_coordinates)
        : base_type(metadata),
          cell_coordinates("stored_cell_coordinates",
                           SpaceDim,
                           max_coordinate_count(metadata, GridCentering::CellCentered)),
          face_coordinates(input_face_coordinates) {
        validate_face_coordinates_for_construction();
        refresh_cell_coordinates_from_faces_on_host();
        refresh_bounds_cache_on_host();
    }

    /**
     * Centering-tagged input is intentionally not accepted for stored grids.
     */
    StoredCoordinateGrid(const base_type&, coordinate_view_type, GridCentering) = delete;

    /**
     * Independent cell and face coordinate input is intentionally not accepted.
     */
    StoredCoordinateGrid(const base_type&, coordinate_view_type, coordinate_view_type) = delete;

    /**
     * Refresh coordinate bounds cached for boundary checks and periodic wrapping.
     */
    void refresh_bounds_cache_on_host() {
        validate_face_coordinates_for_construction();

        auto face_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                             face_coordinates);
        for (int dim = 0; dim < SpaceDim; ++dim) {
            const int lower_ghost = this->ghost_extent(dim, LowerBoundary);
            const int physical_lower_storage = lower_ghost;
            const int physical_upper_storage = this->extent(dim) + lower_ghost;
            const int ghost_lower_storage =
                this->lower_face_index(dim, GridDomain::GhostedDomain) + lower_ghost;
            const int ghost_upper_storage =
                this->upper_face_index(dim, GridDomain::GhostedDomain) + lower_ghost;

            physical_lower_bound[dim] = face_host(dim, physical_lower_storage);
            physical_upper_bound[dim] = face_host(dim, physical_upper_storage);
            ghost_lower_bound[dim] = face_host(dim, ghost_lower_storage);
            ghost_upper_bound[dim] = face_host(dim, ghost_upper_storage);
            if (physical_upper_bound[dim] <= physical_lower_bound[dim]) {
                throw std::runtime_error(
                    "StoredCoordinateGrid: face coordinates must be strictly ascending.");
            }
            periodic_width[dim] = physical_upper_bound[dim] - physical_lower_bound[dim];
            periodic_inv_width[dim] =
                periodic_width[dim] != 0.0 ? 1.0 / periodic_width[dim] : 0.0;
        }
    }

    /**
     * Return a stored coordinate value from the selected centering.
     */
    KOKKOS_INLINE_FUNCTION
    double coordinate(const int dim, const int index, const GridCentering centering) const {
        return centering == GridCentering::CellCentered
                   ? cell_coordinates(dim, storage_index(dim, index))
                   : face_coordinates(dim, storage_index(dim, index));
    }

    /**
     * Return the continuous logical grid coordinate for a physical coordinate.
     */
    KOKKOS_INLINE_FUNCTION
    double grid_index(const int dim,
                      const double x,
                      const GridCentering centering) const {
        return grid_index(dim, x, centering, GridDomain::GhostedDomain);
    }

    /**
     * Return the continuous logical grid coordinate within the selected domain.
     */
    KOKKOS_INLINE_FUNCTION
    double grid_index(const int dim,
                      const double x,
                      const GridCentering centering,
                      const GridDomain domain) const {
        const int lower_index = lower_coordinate_index(dim, centering, domain);
        const int upper_index = upper_coordinate_index(dim, centering, domain);
        if (upper_index <= lower_index) {
            Kokkos::abort("StoredCoordinateGrid: grid_index requires at least two coordinates.");
        }

        const double lower_x = coordinate(dim, lower_index, centering);
        const double upper_x = coordinate(dim, upper_index, centering);
        const bool ascending = upper_x >= lower_x;

        if ((ascending && x <= lower_x) || (!ascending && x >= lower_x)) {
            return interpolate_grid_index(dim, x, centering, lower_index, lower_index + 1);
        }
        if ((ascending && x >= upper_x) || (!ascending && x <= upper_x)) {
            return interpolate_grid_index(dim, x, centering, upper_index - 1, upper_index);
        }

        int left = lower_index;
        int right = upper_index - 1;
        while (left <= right) {
            const int mid = left + (right - left) / 2;
            const double x_left = coordinate(dim, mid, centering);
            const double x_right = coordinate(dim, mid + 1, centering);
            const bool in_interval = ascending
                                         ? (x >= x_left && x < x_right)
                                         : (x <= x_left && x > x_right);

            if (in_interval) {
                return interpolate_grid_index(dim, x, centering, mid, mid + 1);
            }

            if ((ascending && x < x_left) || (!ascending && x > x_left)) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        Kokkos::abort("StoredCoordinateGrid: grid_index requires monotonic coordinates.");
        return 0.0;
    }

    /**
     * Return all coordinates at a logical grid point with one centering.
     */
    KOKKOS_INLINE_FUNCTION
    coordinate_array_type coordinate_tuple(const index_array_type& indices,
                                           const GridCentering centering) const {
        centering_array_type centerings{};
        for (int dim = 0; dim < SpaceDim; ++dim) {
            centerings[dim] = centering;
        }
        return coordinate_tuple(indices, centerings);
    }

    /**
     * Return all coordinates at a logical grid point with per-dimension centerings.
     */
    KOKKOS_INLINE_FUNCTION
    coordinate_array_type coordinate_tuple(const index_array_type& indices,
                                           const centering_array_type& centerings) const {
        coordinate_array_type q{};
        for (int dim = 0; dim < SpaceDim; ++dim) {
            q[dim] = coordinate(dim, indices[dim], centerings[dim]);
        }
        return q;
    }

    /**
     * Return the Lame coefficient h_i at a logical grid point.
     */
    KOKKOS_INLINE_FUNCTION
    double h(const int component,
             const index_array_type& indices,
             const centering_array_type& centerings) const {
        return base_type::h(component, coordinate_tuple(indices, centerings));
    }

    /**
     * Return all Lame coefficients at a logical grid point.
     */
    KOKKOS_INLINE_FUNCTION
    metric_array_type h(const index_array_type& indices,
                        const centering_array_type& centerings) const {
        return base_type::h(coordinate_tuple(indices, centerings));
    }

    /**
     * Return the inverse Lame coefficient 1 / h_i at a logical grid point.
     */
    KOKKOS_INLINE_FUNCTION
    double h_inv(const int component,
                 const index_array_type& indices,
                 const centering_array_type& centerings) const {
        return base_type::h_inv(component, coordinate_tuple(indices, centerings));
    }

    /**
     * Return all inverse Lame coefficients at a logical grid point.
     */
    KOKKOS_INLINE_FUNCTION
    metric_array_type h_inv(const index_array_type& indices,
                            const centering_array_type& centerings) const {
        return base_type::h_inv(coordinate_tuple(indices, centerings));
    }

    /**
     * Return the metric Jacobian at a logical grid point.
     */
    KOKKOS_INLINE_FUNCTION
    double jacobian(const index_array_type& indices,
                    const centering_array_type& centerings) const {
        return base_type::jacobian(coordinate_tuple(indices, centerings));
    }

    /**
     * Return coordinate spacing between adjacent points of the selected centering.
     */
    KOKKOS_INLINE_FUNCTION
    double coordinate_spacing(const int dim,
                              const int index,
                              const GridCentering centering) const {
        return coordinate(dim, index + 1, centering) - coordinate(dim, index, centering);
    }

    /**
     * Return the face-to-face coordinate width of one logical cell.
     */
    KOKKOS_INLINE_FUNCTION
    double cell_width(const int dim, const int cell_index) const {
        return coordinate_spacing(dim, cell_index, GridCentering::FaceCentered);
    }

    /**
     * Return the physical distance to the next grid point along one coordinate direction.
     */
    KOKKOS_INLINE_FUNCTION
    double physical_distance(const int component,
                             const index_array_type& indices,
                             const centering_array_type& centerings) const {
        return base_type::physical_distance(
            component,
            coordinate_spacing(component, indices[component], centerings[component]),
            coordinate_tuple(indices, centerings));
    }

    /**
     * Return the physical face-to-face width of one cell along one coordinate direction.
     */
    KOKKOS_INLINE_FUNCTION
    double physical_cell_width(const int component,
                               const index_array_type& indices,
                               const centering_array_type& metric_centerings) const {
        return base_type::physical_distance(component,
                                            cell_width(component, indices[component]),
                                            coordinate_tuple(indices, metric_centerings));
    }

    /**
     * Return the logical cell index containing a physical coordinate.
     */
    KOKKOS_INLINE_FUNCTION
    int cell_index(const int dim,
                   const double x,
                   const GridDomain domain = GridDomain::PhysicalDomain) const {
        const int lower_cell = this->lower_cell_index(dim, domain);
        const int upper_cell = this->upper_cell_index(dim, domain);
        const double lower_x = lower_bound(dim, domain);
        const double upper_x = upper_bound(dim, domain);
        const bool ascending = upper_x >= lower_x;

        if ((ascending && x < lower_x) || (!ascending && x > lower_x)) {
            return lower_cell - 1;
        }
        if ((ascending && x > upper_x) || (!ascending && x < upper_x)) {
            return upper_cell + 1;
        }
        if (x == upper_x) {
            return upper_cell;
        }

        int left = lower_cell;
        int right = upper_cell;
        while (left <= right) {
            const int mid = left + (right - left) / 2;
            const double face_left = coordinate(dim, mid, GridCentering::FaceCentered);
            const double face_right = coordinate(dim, mid + 1, GridCentering::FaceCentered);
            const bool in_cell = ascending
                                     ? (x >= face_left && x < face_right)
                                     : (x <= face_left && x > face_right);

            if (in_cell) {
                return mid;
            }

            if ((ascending && x < face_left) || (!ascending && x > face_left)) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return ascending ? (x < lower_x ? lower_cell - 1 : upper_cell + 1)
                         : (x > lower_x ? lower_cell - 1 : upper_cell + 1);
    }

    /**
     * Return whether a physical coordinate lies inside the selected domain.
     */
    KOKKOS_INLINE_FUNCTION
    bool contains_coordinate(const int dim,
                             const double x,
                             const GridDomain domain = GridDomain::PhysicalDomain) const {
        const double lower_x = lower_bound(dim, domain);
        const double upper_x = upper_bound(dim, domain);
        return upper_x >= lower_x ? (x >= lower_x && x <= upper_x)
                                  : (x <= lower_x && x >= upper_x);
    }

    /**
     * Wrap a coordinate into the physical domain when the dimension is periodic.
     */
    KOKKOS_INLINE_FUNCTION
    double wrap_coordinate(const int dim, const double x) const {
        if (!this->is_periodic(dim)) {
            return x;
        }

        const double lower_x = physical_lower_bound[dim];
        const double upper_x = physical_upper_bound[dim];
        const double width = periodic_width[dim];
        if (width <= 0.0) {
            Kokkos::abort("StoredCoordinateGrid: periodic wrapping requires ascending face coordinates.");
        }

        double wrapped = x - Kokkos::floor((x - lower_x) * periodic_inv_width[dim]) * width;
        if (wrapped >= upper_x) {
            wrapped -= width;
        }
        if (wrapped < lower_x) {
            wrapped += width;
        }
        return wrapped;
    }

private:
    /**
     * Return the rectangular coordinate storage length required by the metadata.
     */
    static int max_coordinate_count(const base_type& metadata,
                                    const GridCentering centering) {
        int count = 0;
        for (int dim = 0; dim < SpaceDim; ++dim) {
            const int dim_count = metadata.total_extent(dim, centering);
            count = dim_count > count ? dim_count : count;
        }
        return count;
    }

    /**
     * Validate the face-coordinate view shape before constructor cache refreshes.
     */
    void validate_face_coordinates_for_construction() const {
        if (face_coordinates.extent(0) < static_cast<std::size_t>(SpaceDim)) {
            throw std::runtime_error(
                "StoredCoordinateGrid: face coordinate view has too few dimensions.");
        }
        if (face_coordinates.extent(1) <
            static_cast<std::size_t>(max_coordinate_count(*this, GridCentering::FaceCentered))) {
            throw std::runtime_error(
                "StoredCoordinateGrid: face coordinate view is too short.");
        }
    }

    /**
     * Derive cell-centered coordinates from adjacent face coordinates.
     */
    void refresh_cell_coordinates_from_faces_on_host() {
        auto face_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                             face_coordinates);
        auto cell_host = Kokkos::create_mirror_view(cell_coordinates);

        for (int dim = 0; dim < SpaceDim; ++dim) {
            const int count = this->total_extent(dim, GridCentering::CellCentered);
            for (int storage = 0; storage < count; ++storage) {
                cell_host(dim, storage) =
                    0.5 * (face_host(dim, storage) + face_host(dim, storage + 1));
            }
        }

        Kokkos::deep_copy(cell_coordinates, cell_host);
    }

    /**
     * Return cached lower coordinate bound for the selected domain.
     */
    KOKKOS_INLINE_FUNCTION
    double lower_bound(const int dim, const GridDomain domain) const {
        return domain == GridDomain::GhostedDomain ? ghost_lower_bound[dim]
                                                   : physical_lower_bound[dim];
    }

    /**
     * Return cached upper coordinate bound for the selected domain.
     */
    KOKKOS_INLINE_FUNCTION
    double upper_bound(const int dim, const GridDomain domain) const {
        return domain == GridDomain::GhostedDomain ? ghost_upper_bound[dim]
                                                   : physical_upper_bound[dim];
    }

    /**
     * Return the lower logical coordinate index for one centering and domain.
     */
    KOKKOS_INLINE_FUNCTION
    int lower_coordinate_index(const int dim,
                               const GridCentering centering,
                               const GridDomain domain) const {
        return centering == GridCentering::CellCentered
                   ? this->lower_cell_index(dim, domain)
                   : this->lower_face_index(dim, domain);
    }

    /**
     * Return the upper logical coordinate index for one centering and domain.
     */
    KOKKOS_INLINE_FUNCTION
    int upper_coordinate_index(const int dim,
                               const GridCentering centering,
                               const GridDomain domain) const {
        return centering == GridCentering::CellCentered
                   ? this->upper_cell_index(dim, domain)
                   : this->upper_face_index(dim, domain);
    }

    /**
     * Linearly interpolate the fractional logical index between adjacent coordinates.
     */
    KOKKOS_INLINE_FUNCTION
    double interpolate_grid_index(const int dim,
                                  const double x,
                                  const GridCentering centering,
                                  const int left_index,
                                  const int right_index) const {
        const double x_left = coordinate(dim, left_index, centering);
        const double x_right = coordinate(dim, right_index, centering);
        const double dx = x_right - x_left;
        if (dx == 0.0) {
            Kokkos::abort("StoredCoordinateGrid: duplicate coordinates are not invertible.");
        }
        return static_cast<double>(left_index) + (x - x_left) / dx;
    }

    /**
     * Convert a logical grid index to the stored coordinate index.
     */
    KOKKOS_INLINE_FUNCTION
    int storage_index(const int dim, const int logical_index) const {
        return logical_index + this->ghost_extent(dim, LowerBoundary);
    }
};
