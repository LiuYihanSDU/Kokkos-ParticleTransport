#pragma once

#include <stdexcept>

#include <Kokkos_MathematicalFunctions.hpp>

#include "Grid.hpp"

/**
 * Host-side helpers for mapped coordinate grids.
 */
namespace MappedCoordinateHostAccess {

/**
 * Build a host-callable mapping for setup and output paths.
 */
template <typename MappingType>
auto create_host_mapping(const MappingType& mapping) {
    if constexpr (requires(const MappingType& input_mapping) {
                      input_mapping.create_host_mirror();
                  }) {
        return mapping.create_host_mirror();
    } else if constexpr (requires(const MappingType& input_mapping) {
                             input_mapping.host_mapping();
                         }) {
        return mapping.host_mapping();
    } else {
        return mapping;
    }
}

/**
 * Build a host coordinate callback from a mapped-grid mapping object.
 */
template <typename MappingType>
auto create_coordinate_provider(const MappingType& mapping) {
    auto host_mapping = create_host_mapping(mapping);
    return [host_mapping](const int dim,
                          const int logical_index,
                          const GridCentering centering) {
        return host_mapping.coordinate(dim, static_cast<double>(logical_index), centering);
    };
}

/**
 * Build a host coordinate callback from a mapped coordinate grid.
 */
template <typename GridType>
auto create_coordinate_provider_from_grid(const GridType& grid) {
    return create_coordinate_provider(grid.mapping);
}

} // namespace MappedCoordinateHostAccess

/**
 * Grid whose coordinates are delegated to a device-callable mapping object.
 *
 * MappingType must be callable from device kernels. Host-side setup and output paths
 * use MappedCoordinateHostAccess::create_host_mapping. Mappings backed by device-only
 * Views should provide either create_host_mirror() or host_mapping() returning a
 * host-callable mapping with mirrored storage.
 */
template <
    int SpaceDim,
    typename MappingType,
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout,
    int VecDim = SpaceDim
>
struct MappedCoordinateGrid : public Grid<SpaceDim, DeviceType, LayoutType, VecDim> {
    using base_type = Grid<SpaceDim, DeviceType, LayoutType, VecDim>;
    using index_array_type = typename base_type::index_array_type;
    using coordinate_array_type = typename base_type::coordinate_array_type;
    using metric_array_type = typename base_type::metric_array_type;
    using centering_array_type = Kokkos::Array<GridCentering, SpaceDim>;
    using bound_array_type = Kokkos::Array<double, SpaceDim>;
    using mapping_type = MappingType;

    struct DeferBoundsCacheRefresh {};

    using base_type::h;
    using base_type::h_inv;
    using base_type::jacobian;
    using base_type::physical_distance;

    static constexpr GridCoordinateRepresentation coordinate_representation =
        GridCoordinateRepresentation::MappedCoordinates;

    mapping_type mapping;
    bound_array_type physical_lower_bound{};
    bound_array_type physical_upper_bound{};
    bound_array_type ghost_lower_bound{};
    bound_array_type ghost_upper_bound{};
    bound_array_type periodic_width{};
    bound_array_type periodic_inv_width{};

    MappedCoordinateGrid() = default;

    /**
     * Build a coordinate grid from common metadata and a mapping object.
     */
    MappedCoordinateGrid(const base_type& metadata, mapping_type input_mapping)
        : base_type(metadata), mapping(input_mapping) {
        refresh_bounds_cache_on_host();
    }

    /**
     * Build a mapped grid and defer bounds-cache refresh to a derived wrapper.
     */
    MappedCoordinateGrid(const base_type& metadata,
                         mapping_type input_mapping,
                         DeferBoundsCacheRefresh)
        : base_type(metadata), mapping(input_mapping) {}

    /**
     * Refresh coordinate bounds cached for boundary checks and periodic wrapping.
     */
    void refresh_bounds_cache_on_host() {
        const auto host_mapping = MappedCoordinateHostAccess::create_host_mapping(mapping);
        for (int dim = 0; dim < SpaceDim; ++dim) {
            physical_lower_bound[dim] =
                host_mapping.coordinate(dim, 0.0, GridCentering::FaceCentered);
            physical_upper_bound[dim] =
                host_mapping.coordinate(dim, static_cast<double>(this->extent(dim)),
                                        GridCentering::FaceCentered);
            ghost_lower_bound[dim] =
                host_mapping.coordinate(
                    dim,
                    static_cast<double>(this->lower_face_index(dim,
                                                               GridDomain::GhostedDomain)),
                    GridCentering::FaceCentered);
            ghost_upper_bound[dim] =
                host_mapping.coordinate(
                    dim,
                    static_cast<double>(this->upper_face_index(dim,
                                                               GridDomain::GhostedDomain)),
                    GridCentering::FaceCentered);
            if (physical_upper_bound[dim] <= physical_lower_bound[dim]) {
                throw std::runtime_error(
                    "MappedCoordinateGrid: face coordinates must be strictly ascending.");
            }
            periodic_width[dim] = physical_upper_bound[dim] - physical_lower_bound[dim];
            periodic_inv_width[dim] =
                periodic_width[dim] != 0.0 ? 1.0 / periodic_width[dim] : 0.0;
        }
    }

    /**
     * Return the physical coordinate delegated to the mapping object.
     */
    KOKKOS_INLINE_FUNCTION
    double coordinate(const int dim,
                      const double grid_position,
                      const GridCentering centering) const {
        return mapping.coordinate(dim, grid_position, centering);
    }

    /**
     * Return the physical coordinate delegated to the mapping object.
     */
    KOKKOS_INLINE_FUNCTION
    double coordinate(const int dim,
                      const int index,
                      const GridCentering centering) const {
        return coordinate(dim, static_cast<double>(index), centering);
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
     * Return the continuous grid coordinate delegated to the mapping object.
     */
    KOKKOS_INLINE_FUNCTION
    double grid_index(const int dim,
                      const double x,
                      const GridCentering centering) const {
        return mapping.grid_index(dim, x, centering);
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

        return static_cast<int>(
            Kokkos::floor(grid_index(dim, x, GridCentering::FaceCentered)));
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
            Kokkos::abort("MappedCoordinateGrid: periodic wrapping requires ascending face coordinates.");
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

    /**
     * Return the local cell width for a continuous cell index.
     */
    KOKKOS_INLINE_FUNCTION
    double cell_width(const int dim, const double cell_index) const {
        return coordinate(dim, cell_index + 1.0, GridCentering::FaceCentered) -
               coordinate(dim, cell_index, GridCentering::FaceCentered);
    }

    /**
     * Return coordinate spacing between adjacent points of the selected centering.
     */
    KOKKOS_INLINE_FUNCTION
    double coordinate_spacing(const int dim,
                              const double grid_position,
                              const GridCentering centering) const {
        return coordinate(dim, grid_position + 1.0, centering) -
               coordinate(dim, grid_position, centering);
    }

    /**
     * Return coordinate spacing between adjacent integer-indexed points.
     */
    KOKKOS_INLINE_FUNCTION
    double coordinate_spacing(const int dim,
                              const int index,
                              const GridCentering centering) const {
        return coordinate_spacing(dim, static_cast<double>(index), centering);
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

private:
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
};
