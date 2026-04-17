#pragma once

#include <cstddef>
#include <stdexcept>

#include <Kokkos_MathematicalFunctions.hpp>

#include "MappedCoordinateGrid.hpp"

/**
 * Analytic coordinate spacing profile used independently by each dimension.
 */
enum class AnalyticGridDiscretization : int {
    UniformSpacing = 0,
    LogarithmicSpacing = 1,
    LinearWidthSpacing = 2
};

/**
 * Device-callable analytic coordinate map stored as compact per-dimension coefficients.
 */
template <
    int SpaceDim,
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout
>
struct AnalyticCoordinateMapping {
    using memory_space = typename DeviceTraits<DeviceType>::memory_space;
    using layout_type = LayoutType;
    using discretization_view_type = Kokkos::View<int*, layout_type, memory_space>;
    using scalar_view_type = Kokkos::View<double*, layout_type, memory_space>;
    using discretization_array_type = Kokkos::Array<AnalyticGridDiscretization, SpaceDim>;
    using scalar_array_type = Kokkos::Array<double, SpaceDim>;

    discretization_view_type discretization;
    scalar_view_type start;
    scalar_view_type step;
    scalar_view_type step_growth;

    AnalyticCoordinateMapping() = default;

    /**
     * Build a uniform analytic mapping from preallocated coordinate parameters.
     */
    AnalyticCoordinateMapping(scalar_view_type input_start,
                              scalar_view_type input_step)
        : discretization("analytic_coordinate_discretization", SpaceDim),
          start(input_start),
          step(input_step),
          step_growth("analytic_coordinate_step_growth", SpaceDim) {
        Kokkos::deep_copy(discretization,
                          static_cast<int>(AnalyticGridDiscretization::UniformSpacing));
        Kokkos::deep_copy(step_growth, 0.0);
    }

    /**
     * Build an analytic mapping from preallocated device-side parameter views.
     */
    AnalyticCoordinateMapping(discretization_view_type input_discretization,
                              scalar_view_type input_start,
                              scalar_view_type input_step,
                              scalar_view_type input_step_growth)
        : discretization(input_discretization),
          start(input_start),
          step(input_step),
          step_growth(input_step_growth) {}

    /**
     * Build an analytic mapping from host-side spacing profiles and parameters.
     */
    AnalyticCoordinateMapping(const discretization_array_type& input_discretization,
                              const scalar_array_type& input_start,
                              const scalar_array_type& input_step,
                              const scalar_array_type& input_step_growth)
        : discretization("analytic_coordinate_discretization", SpaceDim),
          start("analytic_coordinate_start", SpaceDim),
          step("analytic_coordinate_step", SpaceDim),
          step_growth("analytic_coordinate_step_growth", SpaceDim) {
        assign_analytic_metadata(input_discretization, input_start, input_step,
                                 input_step_growth);
    }

    /**
     * Build an analytic mapping from host-side spacing profiles without width growth.
     */
    AnalyticCoordinateMapping(const discretization_array_type& input_discretization,
                              const scalar_array_type& input_start,
                              const scalar_array_type& input_step)
        : discretization("analytic_coordinate_discretization", SpaceDim),
          start("analytic_coordinate_start", SpaceDim),
          step("analytic_coordinate_step", SpaceDim),
          step_growth("analytic_coordinate_step_growth", SpaceDim) {
        const scalar_array_type zero_growth{};
        assign_analytic_metadata(input_discretization, input_start, input_step, zero_growth);
    }

    /**
     * Return the analytic spacing profile for one dimension.
     */
    KOKKOS_INLINE_FUNCTION
    AnalyticGridDiscretization discretization_type(const int dim) const {
        return static_cast<AnalyticGridDiscretization>(discretization(dim));
    }

    /**
     * Return the physical coordinate for a continuous grid coordinate.
     */
    KOKKOS_INLINE_FUNCTION
    double coordinate(const int dim,
                      const double grid_position,
                      const GridCentering centering) const {
        return centering == GridCentering::CellCentered
                   ? cell_coordinate(dim, grid_position)
                   : face_coordinate(dim, grid_position);
    }

    /**
     * Return the physical coordinate for an integer grid index.
     */
    KOKKOS_INLINE_FUNCTION
    double coordinate(const int dim,
                      const int index,
                      const GridCentering centering) const {
        return coordinate(dim, static_cast<double>(index), centering);
    }

    /**
     * Return the continuous grid coordinate associated with a physical coordinate.
     */
    KOKKOS_INLINE_FUNCTION
    double grid_index(const int dim,
                      const double x,
                      const GridCentering centering) const {
        return centering == GridCentering::CellCentered
                   ? cell_grid_index(dim, x)
                   : face_grid_index(dim, x);
    }

    /**
     * Build a host-callable mapping with mirrored coefficient Views.
     */
    auto create_host_mirror() const {
        using host_mapping_type =
            AnalyticCoordinateMapping<SpaceDim, HostDevice, LayoutType>;
        auto discretization_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, discretization);
        auto start_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, start);
        auto step_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, step);
        auto step_growth_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                                    step_growth);
        return host_mapping_type(discretization_host, start_host, step_host,
                                 step_growth_host);
    }

private:
    /**
     * Evaluate the selected analytic face-coordinate expression.
     */
    KOKKOS_INLINE_FUNCTION
    double face_coordinate(const int dim, const double face_index) const {
        const double x0 = start(dim);
        const double dx0 = step(dim);
        const double ddx = step_growth(dim);

        switch (discretization_type(dim)) {
            case AnalyticGridDiscretization::UniformSpacing:
                return x0 + face_index * dx0;
            case AnalyticGridDiscretization::LogarithmicSpacing:
                return Kokkos::exp(x0 + face_index * dx0);
            case AnalyticGridDiscretization::LinearWidthSpacing:
                return x0 + face_index * dx0 + 0.5 * ddx * face_index * (face_index - 1.0);
        }

        return x0;
    }

    /**
     * Evaluate the cell-center coordinate as the midpoint of neighboring faces.
     */
    KOKKOS_INLINE_FUNCTION
    double cell_coordinate(const int dim, const double cell_index) const {
        return 0.5 * (face_coordinate(dim, cell_index) +
                      face_coordinate(dim, cell_index + 1.0));
    }

    /**
     * Evaluate the inverse face-coordinate map x -> i.
     */
    KOKKOS_INLINE_FUNCTION
    double face_grid_index(const int dim, const double x) const {
        const double x0 = start(dim);
        const double dx0 = step(dim);
        const double ddx = step_growth(dim);

        switch (discretization_type(dim)) {
            case AnalyticGridDiscretization::UniformSpacing:
                return (x - x0) / dx0;
            case AnalyticGridDiscretization::LogarithmicSpacing:
                return (Kokkos::log(x) - x0) / dx0;
            case AnalyticGridDiscretization::LinearWidthSpacing:
                if (ddx == 0.0) {
                    return (x - x0) / dx0;
                }
                return (-(dx0 - 0.5 * ddx) +
                        Kokkos::sqrt((dx0 - 0.5 * ddx) * (dx0 - 0.5 * ddx) +
                                     2.0 * ddx * (x - x0))) /
                       ddx;
        }

        return 0.0;
    }

    /**
     * Evaluate the inverse cell-center coordinate map x -> i.
     */
    KOKKOS_INLINE_FUNCTION
    double cell_grid_index(const int dim, const double x) const {
        const double x0 = start(dim);
        const double dx0 = step(dim);
        const double ddx = step_growth(dim);

        switch (discretization_type(dim)) {
            case AnalyticGridDiscretization::UniformSpacing:
                return (x - x0) / dx0 - 0.5;
            case AnalyticGridDiscretization::LogarithmicSpacing: {
                const double log_center_factor = Kokkos::log(0.5 * (1.0 + Kokkos::exp(dx0)));
                return (Kokkos::log(x) - x0 - log_center_factor) / dx0;
            }
            case AnalyticGridDiscretization::LinearWidthSpacing:
                if (ddx == 0.0) {
                    return (x - x0 - 0.5 * dx0) / dx0;
                }
                return (-dx0 +
                        Kokkos::sqrt(dx0 * dx0 + 2.0 * ddx * (x - x0 - 0.5 * dx0))) /
                       ddx;
        }

        return 0.0;
    }

    /**
     * Copy host-side analytic spacing parameters into device-side Views.
     */
    void assign_analytic_metadata(const discretization_array_type& input_discretization,
                                  const scalar_array_type& input_start,
                                  const scalar_array_type& input_step,
                                  const scalar_array_type& input_step_growth) {
        auto discretization_host = Kokkos::create_mirror_view(discretization);
        auto start_host = Kokkos::create_mirror_view(start);
        auto step_host = Kokkos::create_mirror_view(step);
        auto step_growth_host = Kokkos::create_mirror_view(step_growth);

        for (int dim = 0; dim < SpaceDim; ++dim) {
            discretization_host(dim) = static_cast<int>(input_discretization[dim]);
            start_host(dim) = input_start[dim];
            step_host(dim) = input_step[dim];
            step_growth_host(dim) = input_step_growth[dim];
        }

        Kokkos::deep_copy(discretization, discretization_host);
        Kokkos::deep_copy(start, start_host);
        Kokkos::deep_copy(step, step_host);
        Kokkos::deep_copy(step_growth, step_growth_host);
    }
};

/**
 * Analytic coordinate grid implemented as a mapped grid with analytic mapping profiles.
 */
template <
    int SpaceDim,
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout,
    int VecDim = SpaceDim
>
struct AnalyticCoordinateGrid
    : public MappedCoordinateGrid<
          SpaceDim,
          AnalyticCoordinateMapping<SpaceDim, DeviceType, LayoutType>,
          DeviceType,
          LayoutType,
          VecDim> {
    using mapping_type = AnalyticCoordinateMapping<SpaceDim, DeviceType, LayoutType>;
    using mapped_base_type =
        MappedCoordinateGrid<SpaceDim, mapping_type, DeviceType, LayoutType, VecDim>;
    using base_type = Grid<SpaceDim, DeviceType, LayoutType, VecDim>;
    using memory_space = typename mapped_base_type::memory_space;
    using layout_type = typename mapped_base_type::layout_type;
    using index_array_type = typename mapped_base_type::index_array_type;
    using coordinate_array_type = typename mapped_base_type::coordinate_array_type;
    using centering_array_type = typename mapped_base_type::centering_array_type;
    using discretization_view_type = typename mapping_type::discretization_view_type;
    using scalar_view_type = typename mapping_type::scalar_view_type;
    using discretization_array_type = typename mapping_type::discretization_array_type;
    using scalar_array_type = typename mapping_type::scalar_array_type;

    using mapped_base_type::cell_index;
    using mapped_base_type::cell_width;
    using mapped_base_type::contains_coordinate;
    using mapped_base_type::coordinate;
    using mapped_base_type::coordinate_spacing;
    using mapped_base_type::coordinate_tuple;
    using mapped_base_type::grid_index;
    using mapped_base_type::h;
    using mapped_base_type::h_inv;
    using mapped_base_type::jacobian;
    using mapped_base_type::physical_cell_width;
    using mapped_base_type::physical_distance;
    using mapped_base_type::wrap_coordinate;

    static constexpr GridCoordinateRepresentation coordinate_representation =
        GridCoordinateRepresentation::AnalyticCoordinates;

    AnalyticCoordinateGrid() = default;

    /**
     * Build an analytic grid from common metadata and an analytic mapping object.
     */
    AnalyticCoordinateGrid(const base_type& metadata, mapping_type input_mapping)
        : mapped_base_type(metadata, input_mapping,
                           typename mapped_base_type::DeferBoundsCacheRefresh{}) {
        refresh_bounds_cache_on_host();
    }

    /**
     * Build an analytic grid with uniform spacing in every dimension.
     */
    AnalyticCoordinateGrid(const base_type& metadata,
                           scalar_view_type input_start,
                           scalar_view_type input_step)
        : mapped_base_type(metadata, mapping_type(input_start, input_step),
                           typename mapped_base_type::DeferBoundsCacheRefresh{}) {
        refresh_bounds_cache_on_host();
    }

    /**
     * Build an analytic grid from preallocated device-side parameter views.
     */
    AnalyticCoordinateGrid(const base_type& metadata,
                           discretization_view_type input_discretization,
                           scalar_view_type input_start,
                           scalar_view_type input_step,
                           scalar_view_type input_step_growth)
        : mapped_base_type(metadata,
                           mapping_type(input_discretization, input_start, input_step,
                                        input_step_growth),
                           typename mapped_base_type::DeferBoundsCacheRefresh{}) {
        refresh_bounds_cache_on_host();
    }

    /**
     * Build an analytic grid from host-side spacing profiles and parameters.
     */
    AnalyticCoordinateGrid(const base_type& metadata,
                           const discretization_array_type& input_discretization,
                           const scalar_array_type& input_start,
                           const scalar_array_type& input_step,
                           const scalar_array_type& input_step_growth)
        : mapped_base_type(metadata,
                           mapping_type(input_discretization, input_start, input_step,
                                        input_step_growth),
                           typename mapped_base_type::DeferBoundsCacheRefresh{}) {
        refresh_bounds_cache_on_host();
    }

    /**
     * Build an analytic grid from host-side spacing profiles without width growth.
     */
    AnalyticCoordinateGrid(const base_type& metadata,
                           const discretization_array_type& input_discretization,
                           const scalar_array_type& input_start,
                           const scalar_array_type& input_step)
        : mapped_base_type(metadata, mapping_type(input_discretization, input_start, input_step),
                           typename mapped_base_type::DeferBoundsCacheRefresh{}) {
        refresh_bounds_cache_on_host();
    }

    /**
     * Refresh coordinate bounds cached for boundary checks and periodic wrapping.
     */
    void refresh_bounds_cache_on_host() {
        validate_mapping_views_for_construction();

        auto discretization_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, this->mapping.discretization);
        auto start_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                              this->mapping.start);
        auto step_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                             this->mapping.step);
        auto step_growth_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, this->mapping.step_growth);

        for (int dim = 0; dim < SpaceDim; ++dim) {
            const AnalyticGridDiscretization type =
                static_cast<AnalyticGridDiscretization>(discretization_host(dim));
            const double x0 = start_host(dim);
            const double dx0 = step_host(dim);
            const double ddx = step_growth_host(dim);

            this->physical_lower_bound[dim] = face_coordinate_on_host(type, x0, dx0, ddx, 0.0);
            this->physical_upper_bound[dim] =
                face_coordinate_on_host(type, x0, dx0, ddx,
                                        static_cast<double>(this->extent(dim)));
            this->ghost_lower_bound[dim] =
                face_coordinate_on_host(type, x0, dx0, ddx,
                                        static_cast<double>(this->lower_face_index(
                                            dim, GridDomain::GhostedDomain)));
            this->ghost_upper_bound[dim] =
                face_coordinate_on_host(type, x0, dx0, ddx,
                                        static_cast<double>(this->upper_face_index(
                                            dim, GridDomain::GhostedDomain)));
            if (this->physical_upper_bound[dim] <= this->physical_lower_bound[dim]) {
                throw std::runtime_error(
                    "AnalyticCoordinateGrid: face coordinates must be strictly ascending.");
            }
            this->periodic_width[dim] =
                this->physical_upper_bound[dim] - this->physical_lower_bound[dim];
            this->periodic_inv_width[dim] =
                this->periodic_width[dim] != 0.0 ? 1.0 / this->periodic_width[dim] : 0.0;
        }
    }

    /**
     * Return the analytic spacing profile for one dimension.
     */
    KOKKOS_INLINE_FUNCTION
    AnalyticGridDiscretization discretization_type(const int dim) const {
        return this->mapping.discretization_type(dim);
    }

private:
    /**
     * Validate analytic coefficient view shapes before constructor cache refreshes.
     */
    void validate_mapping_views_for_construction() const {
        if (this->mapping.discretization.extent(0) < static_cast<std::size_t>(SpaceDim) ||
            this->mapping.start.extent(0) < static_cast<std::size_t>(SpaceDim) ||
            this->mapping.step.extent(0) < static_cast<std::size_t>(SpaceDim) ||
            this->mapping.step_growth.extent(0) < static_cast<std::size_t>(SpaceDim)) {
            throw std::runtime_error(
                "AnalyticCoordinateGrid: analytic coefficient views are too short.");
        }
    }

    /**
     * Evaluate the analytic face-coordinate expression on the host.
     */
    static double face_coordinate_on_host(const AnalyticGridDiscretization type,
                                          const double x0,
                                          const double dx0,
                                          const double ddx,
                                          const double face_index) {
        switch (type) {
            case AnalyticGridDiscretization::UniformSpacing:
                return x0 + face_index * dx0;
            case AnalyticGridDiscretization::LogarithmicSpacing:
                return Kokkos::exp(x0 + face_index * dx0);
            case AnalyticGridDiscretization::LinearWidthSpacing:
                return x0 + face_index * dx0 + 0.5 * ddx * face_index * (face_index - 1.0);
        }

        return x0;
    }
};

template <
    int SpaceDim,
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout,
    int VecDim = SpaceDim
>
using AnalyticUniformGrid = AnalyticCoordinateGrid<SpaceDim, DeviceType, LayoutType, VecDim>;
