#pragma once

#include <cstdint>
#include <cmath>
#include <stdexcept>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Field.hpp"
#include "ParticleSystem.hpp"
#include "RandomManager.hpp"

/**
 * Axis-aligned coordinate-space region used by particle injection kernels.
 */
template <int SpaceDim>
struct AxisAlignedParticleRegion {
    using coordinate_array_type = Kokkos::Array<double, SpaceDim>;

    coordinate_array_type lower{};
    coordinate_array_type upper{};

    /**
     * Validate region bounds on the host before launching injection kernels.
     */
    void validate() const {
        for (int dim = 0; dim < SpaceDim; ++dim) {
            if (!std::isfinite(lower[dim]) || !std::isfinite(upper[dim]) ||
                upper[dim] <= lower[dim]) {
                throw std::runtime_error(
                    "AxisAlignedParticleRegion: bounds must be finite and ascending.");
            }
        }
    }

    /**
     * Return whether one point lies inside this region.
     */
    KOKKOS_INLINE_FUNCTION
    bool contains(const coordinate_array_type& point) const {
        for (int dim = 0; dim < SpaceDim; ++dim) {
            if (point[dim] < lower[dim] || point[dim] > upper[dim]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Draw one uniformly distributed point inside this region.
     */
    template <typename GeneratorType>
    KOKKOS_INLINE_FUNCTION
    coordinate_array_type sample(GeneratorType& generator) const {
        coordinate_array_type point{};
        for (int dim = 0; dim < SpaceDim; ++dim) {
            point[dim] = generator.drand(lower[dim], upper[dim]);
        }
        return point;
    }
};

/**
 * Constant particle state assigned during injection.
 */
template <int SpaceDim>
struct ParticleInjectionKinematics {
    double momentum_magnitude{0.0};
    double mu{0.0};
    double weight{1.0};
    ParticleStatus status{ParticleStatus::Active};
};

/**
 * Return the kinetic energy implied by one injection kinematic state.
 */
template <int SpaceDim>
KOKKOS_INLINE_FUNCTION
double injection_kinetic_energy(const ParticleProperties& properties,
                                const ParticleInjectionKinematics<SpaceDim>& kinematics) {
    return properties.kinetic_energy_from_momentum_magnitude(
        kinematics.momentum_magnitude);
}

/**
 * Validate host-side particle kinematics before launching injection kernels.
 */
template <int SpaceDim>
inline void validate_particle_injection_kinematics(
    const ParticleInjectionKinematics<SpaceDim>& kinematics) {
    if (!std::isfinite(kinematics.mu) ||
        !std::isfinite(kinematics.weight) || kinematics.weight <= 0.0) {
        throw std::runtime_error(
            "ParticleInjectionKinematics: mu must be finite and weight must be positive.");
    }
    if (!std::isfinite(kinematics.momentum_magnitude) ||
        kinematics.momentum_magnitude < 0.0) {
        throw std::runtime_error(
            "ParticleInjectionKinematics: momentum magnitude must be finite and non-negative.");
    }
}

/**
 * Injection range and rejection-sampling controls.
 */
struct ParticleInjectionPlan {
    std::uint64_t first_particle_index{0};
    std::uint64_t particle_count{0};
    std::uint64_t first_particle_id{0};
    int max_attempts{64};
};

/**
 * Field-threshold comparison mode used by particle-injection predicates.
 */
enum class FieldPredicateComparison : int {
    Greater = 0,
    GreaterEqual = 1,
    Less = 2,
    LessEqual = 3,
    AbsGreater = 4,
    AbsLess = 5
};

/**
 * Particle lifecycle subset selected for recycling.
 */
enum class ParticleRecycleSelection : int {
    EscapedOnly = 0,
    InactiveOnly = 1,
    Recyclable = 2
};

/**
 * Return whether a particle status is selected for recycling.
 */
KOKKOS_INLINE_FUNCTION
bool particle_recycle_selection_matches(const int status,
                                        const ParticleRecycleSelection selection) {
    const ParticleStatus particle_status = static_cast<ParticleStatus>(status);
    switch (selection) {
        case ParticleRecycleSelection::EscapedOnly:
            return particle_status == ParticleStatus::Escaped;
        case ParticleRecycleSelection::InactiveOnly:
            return particle_status == ParticleStatus::Inactive;
        case ParticleRecycleSelection::Recyclable:
            return particle_status_is_recyclable(particle_status);
    }
    return false;
}

/**
 * Predicate that always accepts a candidate particle position.
 */
struct AlwaysAcceptParticlePredicate {
    /**
     * Return true for every candidate position.
     */
    template <typename CoordinateArrayType>
    KOKKOS_INLINE_FUNCTION
    bool operator()(const CoordinateArrayType&) const {
        return true;
    }
};

/**
 * Build a particle injection plan for a contiguous particle range.
 */
inline ParticleInjectionPlan make_particle_injection_plan(
    const std::uint64_t first_particle_index,
    const std::uint64_t particle_count,
    const int max_attempts = 64) {
    return ParticleInjectionPlan{first_particle_index,
                                 particle_count,
                                 first_particle_index,
                                 max_attempts};
}

/**
 * Build an append-style injection plan at the current particle storage tail.
 */
template <typename ParticleSystemType>
inline ParticleInjectionPlan make_particle_append_plan(
    const ParticleSystemType& particles,
    const std::uint64_t particle_count,
    const int max_attempts = 64) {
    const std::uint64_t first_index =
        static_cast<std::uint64_t>(particles.particle_count());
    return make_particle_injection_plan(first_index, particle_count, max_attempts);
}

/**
 * Build an injection region from explicit lower and upper coordinate bounds.
 */
template <int SpaceDim>
inline AxisAlignedParticleRegion<SpaceDim> make_particle_region(
    const Kokkos::Array<double, SpaceDim>& lower,
    const Kokkos::Array<double, SpaceDim>& upper) {
    AxisAlignedParticleRegion<SpaceDim> region{lower, upper};
    region.validate();
    return region;
}

/**
 * Build an injection region from a coordinate grid's public face-coordinate bounds.
 */
template <typename GridType>
inline AxisAlignedParticleRegion<GridType::space_dim> make_particle_region_from_grid(
    const GridType& grid,
    const GridDomain domain = GridDomain::PhysicalDomain) {
    static_assert(requires(const GridType& input_grid) {
                      input_grid.coordinate(0, 0, GridCentering::FaceCentered);
                  },
                  "make_particle_region_from_grid requires a coordinate grid.");
    AxisAlignedParticleRegion<GridType::space_dim> region;
    for (int dim = 0; dim < GridType::space_dim; ++dim) {
        region.lower[dim] = grid.coordinate(
            dim, grid.lower_face_index(dim, domain), GridCentering::FaceCentered);
        region.upper[dim] = grid.coordinate(
            dim, grid.upper_face_index(dim, domain), GridCentering::FaceCentered);
    }
    region.validate();
    return region;
}

/**
 * Compare a scalar field value with a configured threshold.
 */
KOKKOS_INLINE_FUNCTION
bool compare_field_predicate_value(const double value,
                                   const double threshold,
                                   const FieldPredicateComparison comparison) {
    switch (comparison) {
        case FieldPredicateComparison::Greater:
            return value > threshold;
        case FieldPredicateComparison::GreaterEqual:
            return value >= threshold;
        case FieldPredicateComparison::Less:
            return value < threshold;
        case FieldPredicateComparison::LessEqual:
            return value <= threshold;
        case FieldPredicateComparison::AbsGreater:
            return Kokkos::abs(value) > threshold;
        case FieldPredicateComparison::AbsLess:
            return Kokkos::abs(value) < threshold;
    }
    return false;
}

/**
 * Device-callable scalar Field threshold predicate for particle injection.
 */
template <typename FieldType>
struct FieldThresholdParticlePredicate {
    using field_type = FieldType;
    using grid_type = typename FieldType::grid_type;
    using coordinate_array_type = typename FieldType::coordinate_array_type;
    using index_array_type = typename FieldType::index_array_type;
    using centering_array_type = typename FieldType::centering_array_type;
    using field_accessor_type = typename FieldType::RandomAccessAccessor;

    grid_type grid;
    centering_array_type centerings{};
    field_accessor_type values;
    FieldPredicateComparison comparison{FieldPredicateComparison::GreaterEqual};
    GridDomain domain{GridDomain::PhysicalDomain};
    double threshold{0.0};
    int component{0};

    /**
     * Build a field-threshold predicate from a field and comparison settings.
     */
    FieldThresholdParticlePredicate(const FieldType& field,
                                    const double input_threshold,
                                    const FieldPredicateComparison input_comparison =
                                        FieldPredicateComparison::GreaterEqual,
                                    const int input_component = 0,
                                    const GridDomain input_domain =
                                        GridDomain::PhysicalDomain)
        : grid(field.grid),
          centerings(field.centerings),
          values(field.random_access()),
          comparison(input_comparison),
          domain(input_domain),
          threshold(input_threshold),
          component(input_component) {
        if (component < 0 || component >= FieldType::component_dim) {
            throw std::runtime_error(
                "FieldThresholdParticlePredicate: component index is out of range.");
        }
    }

    /**
     * Return true when the field value at the candidate position passes the threshold.
     */
    KOKKOS_INLINE_FUNCTION
    bool operator()(const coordinate_array_type& point) const {
        index_array_type indices{};
        for (int dim = 0; dim < FieldType::space_dim; ++dim) {
            if (!grid.contains_coordinate(dim, point[dim], domain)) {
                return false;
            }
            indices[dim] = coordinate_to_field_index(dim, point[dim]);
            if (indices[dim] < lower_index(dim) || indices[dim] > upper_index(dim)) {
                return false;
            }
        }

        return compare(values(indices, component));
    }

private:
    /**
     * Convert one coordinate to a logical field index.
     */
    KOKKOS_INLINE_FUNCTION
    int coordinate_to_field_index(const int dim, const double coordinate) const {
        if (centerings[dim] == GridCentering::CellCentered) {
            return grid.cell_index(dim, coordinate, domain);
        }

        const double logical =
            grid.grid_index(dim, coordinate, GridCentering::FaceCentered);
        return static_cast<int>(Kokkos::floor(logical + 0.5));
    }

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
     * Compare a scalar field value with the configured threshold.
     */
    KOKKOS_INLINE_FUNCTION
    bool compare(const double value) const {
        return compare_field_predicate_value(value, threshold, comparison);
    }
};

/**
 * Build a field-threshold particle-injection predicate.
 */
template <typename FieldType>
inline FieldThresholdParticlePredicate<FieldType> make_field_threshold_particle_predicate(
    const FieldType& field,
    const double threshold,
    const FieldPredicateComparison comparison = FieldPredicateComparison::GreaterEqual,
    const int component = 0,
    const GridDomain domain = GridDomain::PhysicalDomain) {
    return FieldThresholdParticlePredicate<FieldType>(field, threshold, comparison,
                                                      component, domain);
}

/**
 * Return an approximate physical volume for one logical cell.
 *
 * The estimate uses the metric Jacobian at the cell center multiplied by face-to-face
 * coordinate widths. This is a midpoint-volume rule that is portable across stored,
 * analytic, and custom mapped coordinate grids.
 */
template <typename GridType>
KOKKOS_INLINE_FUNCTION
double approximate_grid_cell_physical_volume(
    const GridType& grid,
    const typename GridType::index_array_type& cell_indices) {
    Kokkos::Array<GridCentering, GridType::space_dim> centerings{};
    for (int dim = 0; dim < GridType::space_dim; ++dim) {
        centerings[dim] = GridCentering::CellCentered;
    }

    const auto q = grid.coordinate_tuple(cell_indices, centerings);
    double volume = Kokkos::abs(grid.jacobian(q));
    for (int dim = 0; dim < GridType::space_dim; ++dim) {
        volume *= Kokkos::abs(grid.cell_width(dim, cell_indices[dim]));
    }
    return Kokkos::isfinite(volume) && volume > 0.0 ? volume : 0.0;
}

/**
 * Clamp a logical index to the selected cell domain.
 */
template <typename GridType>
KOKKOS_INLINE_FUNCTION
int clamp_grid_cell_index_for_volume(const GridType& grid,
                                     const int dim,
                                     const int index,
                                     const GridDomain domain) {
    const int lower = grid.lower_cell_index(dim, domain);
    const int upper = grid.upper_cell_index(dim, domain);
    if (index < lower) {
        return lower;
    }
    if (index > upper) {
        return upper;
    }
    return index;
}

/**
 * Convert field logical indices to the neighboring cell used for volume weighting.
 */
template <typename FieldType>
KOKKOS_INLINE_FUNCTION
typename FieldType::index_array_type field_indices_to_volume_cell_indices(
    const FieldType& field,
    const typename FieldType::index_array_type& field_indices,
    const GridDomain domain) {
    typename FieldType::index_array_type cell_indices{};
    for (int dim = 0; dim < FieldType::space_dim; ++dim) {
        cell_indices[dim] = clamp_grid_cell_index_for_volume(
            field.grid, dim, field_indices[dim], domain);
    }
    return cell_indices;
}

/**
 * Sample an index from a cumulative weight view using device-side binary search.
 */
template <typename CumulativeWeightViewType, typename SizeType, typename GeneratorType>
KOKKOS_INLINE_FUNCTION
SizeType sample_cumulative_weight_index(const CumulativeWeightViewType& cumulative_weights,
                                        const SizeType count,
                                        const double total_weight,
                                        GeneratorType& generator) {
    if (count == 0 || total_weight <= 0.0) {
        return 0;
    }

    const double target = generator.drand(0.0, total_weight);
    SizeType left = 0;
    SizeType right = count - 1;
    while (left < right) {
        const SizeType middle = left + (right - left) / 2;
        if (target < cumulative_weights(middle)) {
            right = middle;
        } else {
            left = middle + 1;
        }
    }
    return left;
}

/**
 * Device-backed list of field grid points that satisfy an injection predicate.
 */
template <typename FieldType>
struct FieldParticleCandidateSet {
    using field_type = FieldType;
    using grid_type = typename FieldType::grid_type;
    using coordinate_array_type = typename FieldType::coordinate_array_type;
    using index_array_type = typename FieldType::index_array_type;
    using size_type = typename FieldType::size_type;
    using candidate_view_type =
        Kokkos::View<size_type*, typename FieldType::layout_type,
                     typename FieldType::memory_space>;
    using cumulative_weight_view_type =
        Kokkos::View<double*, typename FieldType::layout_type,
                     typename FieldType::memory_space>;

    field_type field;
    candidate_view_type point_indices;
    cumulative_weight_view_type cumulative_weights;
    size_type candidate_count_cache{0};
    double total_weight_cache{0.0};

    FieldParticleCandidateSet() = default;

    /**
     * Build a candidate set from compacted field point indices.
     */
    FieldParticleCandidateSet(const FieldType& input_field,
                              candidate_view_type input_point_indices,
                              cumulative_weight_view_type input_cumulative_weights,
                              const size_type input_candidate_count,
                              const double input_total_weight)
        : field(input_field),
          point_indices(input_point_indices),
          cumulative_weights(input_cumulative_weights),
          candidate_count_cache(input_candidate_count),
          total_weight_cache(input_total_weight) {}

    /**
     * Return the number of valid candidate grid points.
     */
    KOKKOS_INLINE_FUNCTION
    size_type candidate_count() const {
        return candidate_count_cache;
    }

    /**
     * Return true when no field grid point satisfies the predicate.
     */
    KOKKOS_INLINE_FUNCTION
    bool empty() const {
        return candidate_count_cache == 0;
    }

    /**
     * Sample a candidate grid point by physical-volume weight.
     */
    template <typename GeneratorType>
    KOKKOS_INLINE_FUNCTION
    size_type sample_candidate_index(GeneratorType& generator) const {
        return sample_cumulative_weight_index(cumulative_weights,
                                              candidate_count_cache,
                                              total_weight_cache,
                                              generator);
    }

    /**
     * Return the coordinate of one selected candidate grid point.
     */
    KOKKOS_INLINE_FUNCTION
    coordinate_array_type coordinate(const size_type candidate_index) const {
        const index_array_type indices =
            field.logical_indices(point_indices(candidate_index));
        coordinate_array_type result{};
        for (int dim = 0; dim < FieldType::space_dim; ++dim) {
            result[dim] = field.grid.coordinate(dim, indices[dim], field.centerings[dim]);
        }
        return result;
    }
};

/**
 * Device-backed physical-cell candidate set for grid-wide injection.
 */
template <typename GridType>
struct GridParticleCandidateSet {
    using grid_type = GridType;
    using index_array_type = typename GridType::index_array_type;
    using coordinate_array_type = typename GridType::coordinate_array_type;
    using size_type = typename GridType::extent_view_type::size_type;
    using cumulative_weight_view_type =
        Kokkos::View<double*, typename GridType::layout_type,
                     typename GridType::memory_space>;

    grid_type grid;
    index_array_type lower_cell_index_cache{};
    index_array_type extents_cache{};
    cumulative_weight_view_type cumulative_weights;
    size_type candidate_count_cache{0};
    double total_weight_cache{0.0};

    GridParticleCandidateSet() = default;

    /**
     * Build a grid-cell candidate set from cached domain metadata and weights.
     */
    GridParticleCandidateSet(const grid_type& input_grid,
                             const index_array_type& input_lower_cell_index,
                             const index_array_type& input_extents,
                             cumulative_weight_view_type input_cumulative_weights,
                             const size_type input_candidate_count,
                             const double input_total_weight)
        : grid(input_grid),
          lower_cell_index_cache(input_lower_cell_index),
          extents_cache(input_extents),
          cumulative_weights(input_cumulative_weights),
          candidate_count_cache(input_candidate_count),
          total_weight_cache(input_total_weight) {}

    /**
     * Return the number of candidate physical cells.
     */
    KOKKOS_INLINE_FUNCTION
    size_type candidate_count() const {
        return candidate_count_cache;
    }

    /**
     * Return true when no cells are available.
     */
    KOKKOS_INLINE_FUNCTION
    bool empty() const {
        return candidate_count_cache == 0;
    }

    /**
     * Convert a candidate linear index into logical cell indices.
     */
    KOKKOS_INLINE_FUNCTION
    index_array_type cell_indices(size_type linear_index) const {
        index_array_type indices{};
        for (int dim = 0; dim < GridType::space_dim; ++dim) {
            const int local_index =
                static_cast<int>(linear_index % static_cast<size_type>(extents_cache[dim]));
            indices[dim] = lower_cell_index_cache[dim] + local_index;
            linear_index /= static_cast<size_type>(extents_cache[dim]);
        }
        return indices;
    }

    /**
     * Sample a candidate cell by physical-volume weight.
     */
    template <typename GeneratorType>
    KOKKOS_INLINE_FUNCTION
    size_type sample_candidate_index(GeneratorType& generator) const {
        return sample_cumulative_weight_index(cumulative_weights,
                                              candidate_count_cache,
                                              total_weight_cache,
                                              generator);
    }

    /**
     * Draw one coordinate uniformly inside a selected logical cell.
     */
    template <typename GeneratorType>
    KOKKOS_INLINE_FUNCTION
    coordinate_array_type sample_coordinate(const size_type candidate_index,
                                            GeneratorType& generator) const {
        const index_array_type indices = cell_indices(candidate_index);
        coordinate_array_type coordinate{};
        for (int dim = 0; dim < GridType::space_dim; ++dim) {
            const double lower_face =
                grid.coordinate(dim, indices[dim], GridCentering::FaceCentered);
            const double upper_face =
                grid.coordinate(dim, indices[dim] + 1, GridCentering::FaceCentered);
            coordinate[dim] = lower_face <= upper_face
                                  ? generator.drand(lower_face, upper_face)
                                  : generator.drand(upper_face, lower_face);
        }
        return coordinate;
    }
};

/**
 * Validate a field component index used by injection predicates.
 */
template <typename FieldType>
inline void validate_field_injection_component(const int component) {
    if (component < 0 || component >= FieldType::component_dim) {
        throw std::runtime_error("Field particle injection: component index is out of range.");
    }
}

/**
 * Build a compact field-grid candidate set inside an explicit coordinate region.
 */
template <typename FieldType>
inline FieldParticleCandidateSet<FieldType> make_field_particle_candidate_set_in_region(
    const FieldType& field,
    const AxisAlignedParticleRegion<FieldType::space_dim>& region,
    const double threshold,
    const FieldPredicateComparison comparison = FieldPredicateComparison::GreaterEqual,
    const int component = 0,
    const GridDomain domain = GridDomain::PhysicalDomain) {
    static_assert(requires(const typename FieldType::grid_type& input_grid) {
                      input_grid.coordinate(0, 0, GridCentering::CellCentered);
                  },
                  "Field particle candidate sets require a coordinate grid.");
    validate_field_injection_component<FieldType>(component);
    region.validate();

    using candidate_set_type = FieldParticleCandidateSet<FieldType>;
    using size_type = typename candidate_set_type::size_type;
    using candidate_view_type = typename candidate_set_type::candidate_view_type;
    using cumulative_weight_view_type =
        typename candidate_set_type::cumulative_weight_view_type;

    candidate_view_type candidate_indices("field_particle_candidate_indices",
                                          field.point_count());
    size_type candidate_count = 0;
    const auto local_field = field;
    const auto local_region = region;
    const auto local_threshold = threshold;
    const auto local_comparison = comparison;
    const auto local_component = component;
    const auto local_domain = domain;

    Kokkos::parallel_scan(
        "ParticleGenerator::make_field_particle_candidate_set_in_region",
        Kokkos::RangePolicy<typename FieldType::execution_space>(0, field.point_count()),
        KOKKOS_LAMBDA(const size_type point_index,
                      size_type& update,
                      const bool final_pass) {
            const auto indices = local_field.logical_indices(point_index);
            bool accepted = true;
            for (int dim = 0; dim < FieldType::space_dim; ++dim) {
                if (indices[dim] < local_field.lower_index(dim, local_domain) ||
                    indices[dim] > local_field.upper_index(dim, local_domain)) {
                    accepted = false;
                }
            }

            typename FieldType::coordinate_array_type coordinate{};
            if (accepted) {
                for (int dim = 0; dim < FieldType::space_dim; ++dim) {
                    coordinate[dim] = local_field.grid.coordinate(
                        dim, indices[dim], local_field.centerings[dim]);
                }
                accepted = local_region.contains(coordinate);
            }

            if (accepted) {
                accepted = compare_field_predicate_value(
                    local_field.data(point_index, local_component),
                    local_threshold,
                    local_comparison);
            }

            if (accepted) {
                if (final_pass) {
                    candidate_indices(update) = point_index;
                }
                ++update;
            }
        },
        candidate_count);
    Kokkos::fence("ParticleGenerator::make_field_particle_candidate_set_in_region");

    cumulative_weight_view_type cumulative_weights(
        "field_particle_candidate_cumulative_weights", candidate_count);
    double total_weight = 0.0;
    if (candidate_count > 0) {
        auto local_candidate_indices = candidate_indices;
        auto local_cumulative_weights = cumulative_weights;
        Kokkos::parallel_scan(
            "ParticleGenerator::field_particle_candidate_weights",
            Kokkos::RangePolicy<typename FieldType::execution_space>(
                0, candidate_count),
            KOKKOS_LAMBDA(const size_type candidate_index,
                          double& update,
                          const bool final_pass) {
                const auto field_indices =
                    local_field.logical_indices(local_candidate_indices(candidate_index));
                const auto cell_indices = field_indices_to_volume_cell_indices(
                    local_field, field_indices, local_domain);
                const double weight = approximate_grid_cell_physical_volume(
                    local_field.grid, cell_indices);
                update += weight;
                if (final_pass) {
                    local_cumulative_weights(candidate_index) = update;
                }
            },
            total_weight);
        Kokkos::fence("ParticleGenerator::field_particle_candidate_weights");
    }

    if (candidate_count > 0 &&
        (!std::isfinite(total_weight) || total_weight <= 0.0)) {
        throw std::runtime_error(
            "Field particle candidate set: total physical volume must be positive.");
    }

    return candidate_set_type(field, candidate_indices, cumulative_weights,
                              candidate_count, total_weight);
}

/**
 * Build a compact field-grid candidate set over the selected grid domain.
 */
template <typename FieldType>
inline FieldParticleCandidateSet<FieldType> make_field_particle_candidate_set(
    const FieldType& field,
    const double threshold,
    const FieldPredicateComparison comparison = FieldPredicateComparison::GreaterEqual,
    const int component = 0,
    const GridDomain domain = GridDomain::PhysicalDomain) {
    const auto region = make_particle_region_from_grid(field.grid, domain);
    return make_field_particle_candidate_set_in_region(field, region, threshold,
                                                       comparison, component, domain);
}

/**
 * Build a physical-cell candidate set over the selected coordinate-grid domain.
 */
template <typename GridType>
inline GridParticleCandidateSet<GridType> make_grid_particle_candidate_set(
    const GridType& grid,
    const GridDomain domain = GridDomain::PhysicalDomain) {
    static_assert(requires(const GridType& input_grid,
                           const typename GridType::index_array_type& input_indices) {
                      input_grid.coordinate_tuple(input_indices, GridCentering::CellCentered);
                      input_grid.cell_width(0, 0);
                  },
                  "Grid particle candidate sets require a coordinate grid.");

    using candidate_set_type = GridParticleCandidateSet<GridType>;
    using size_type = typename candidate_set_type::size_type;
    using cumulative_weight_view_type =
        typename candidate_set_type::cumulative_weight_view_type;
    using index_array_type = typename candidate_set_type::index_array_type;

    index_array_type lower_indices{};
    index_array_type extents{};
    size_type candidate_count = 1;
    for (int dim = 0; dim < GridType::space_dim; ++dim) {
        lower_indices[dim] = grid.lower_cell_index(dim, domain);
        const int upper_index = grid.upper_cell_index(dim, domain);
        extents[dim] = upper_index - lower_indices[dim] + 1;
        if (extents[dim] <= 0) {
            throw std::runtime_error(
                "Grid particle candidate set: grid domain has no cells.");
        }
        candidate_count *= static_cast<size_type>(extents[dim]);
    }

    cumulative_weight_view_type cumulative_weights(
        "grid_particle_candidate_cumulative_weights", candidate_count);
    const auto local_grid = grid;
    const auto local_lower_indices = lower_indices;
    const auto local_extents = extents;
    double total_weight = 0.0;
    Kokkos::parallel_scan(
        "ParticleGenerator::grid_particle_candidate_weights",
        Kokkos::RangePolicy<typename GridType::execution_space>(0, candidate_count),
        KOKKOS_LAMBDA(const size_type candidate_index,
                      double& update,
                      const bool final_pass) {
            size_type linear_index = candidate_index;
            index_array_type cell_indices{};
            for (int dim = 0; dim < GridType::space_dim; ++dim) {
                const int local_index =
                    static_cast<int>(linear_index %
                                     static_cast<size_type>(local_extents[dim]));
                cell_indices[dim] = local_lower_indices[dim] + local_index;
                linear_index /= static_cast<size_type>(local_extents[dim]);
            }
            const double weight =
                approximate_grid_cell_physical_volume(local_grid, cell_indices);
            update += weight;
            if (final_pass) {
                cumulative_weights(candidate_index) = update;
            }
        },
        total_weight);
    Kokkos::fence("ParticleGenerator::grid_particle_candidate_weights");

    if (!std::isfinite(total_weight) || total_weight <= 0.0) {
        throw std::runtime_error(
            "Grid particle candidate set: total physical volume must be positive.");
    }

    return candidate_set_type(grid, lower_indices, extents, cumulative_weights,
                              candidate_count, total_weight);
}

/**
 * Validate that an injection plan fits inside a particle system.
 */
template <typename ParticleSystemType>
inline void validate_particle_injection_plan(const ParticleSystemType& particles,
                                             const ParticleInjectionPlan& plan) {
    if (plan.max_attempts <= 0) {
        throw std::runtime_error("ParticleInjectionPlan: max_attempts must be positive.");
    }
    if (plan.first_particle_index >
        static_cast<std::uint64_t>(particles.particle_count())) {
        throw std::runtime_error("ParticleInjectionPlan: first index is out of range.");
    }
    if (plan.first_particle_index >
        static_cast<std::uint64_t>(particles.particle_capacity())) {
        throw std::runtime_error("ParticleInjectionPlan: first index exceeds capacity.");
    }
    if (plan.particle_count >
        static_cast<std::uint64_t>(particles.particle_capacity()) -
            plan.first_particle_index) {
        throw std::runtime_error("ParticleInjectionPlan: requested range exceeds capacity.");
    }
}

/**
 * Inject particles uniformly in a region while filtering candidates by a predicate.
 */
template <
    typename ParticleSystemType,
    typename RandomManagerType,
    typename PredicateType
>
void inject_particles_uniform_if(
    ParticleSystemType& particles,
    const RandomManagerType& random_manager,
    const AxisAlignedParticleRegion<ParticleSystemType::space_dim>& region,
    const ParticleInjectionPlan& plan,
    const ParticleInjectionKinematics<ParticleSystemType::space_dim>& kinematics,
    PredicateType predicate) {
    region.validate();
    validate_particle_injection_kinematics(kinematics);
    validate_particle_injection_plan(particles, plan);

    auto random_pool = random_manager.pool;
    auto particle_id = particles.particle_id;
    auto position = particles.position;
    auto previous_position = particles.previous_position;
    auto previous_step_position = particles.previous_step_position;
    auto momentum = particles.momentum;
    auto mu = particles.mu;
    auto particle_weight = particles.weight;
    auto initial_kinetic_energy = particles.initial_kinetic_energy;
    auto status = particles.status;
    auto split_level = particles.split_level;
    auto sort_key = particles.sort_key;
    const auto local_properties = particles.properties;
    const auto local_region = region;
    const auto local_kinematics = kinematics;
    const auto local_plan = plan;

    using size_type = typename ParticleSystemType::size_type;
    Kokkos::parallel_for(
        "ParticleGenerator::inject_particles_uniform_if",
        Kokkos::RangePolicy<typename ParticleSystemType::execution_space>(
            0, static_cast<size_type>(plan.particle_count)),
        KOKKOS_LAMBDA(const size_type local_index) {
            auto generator = random_pool.get_state();
            typename AxisAlignedParticleRegion<ParticleSystemType::space_dim>::
                coordinate_array_type candidate{};
            bool accepted = false;
            for (int attempt = 0; attempt < local_plan.max_attempts; ++attempt) {
                candidate = local_region.sample(generator);
                if (predicate(candidate)) {
                    accepted = true;
                    break;
                }
            }

            const size_type particle_index =
                static_cast<size_type>(local_plan.first_particle_index) + local_index;
            particle_id(particle_index) =
                local_plan.first_particle_id + static_cast<std::uint64_t>(local_index);
            for (int dim = 0; dim < ParticleSystemType::space_dim; ++dim) {
                position(particle_index, dim) = candidate[dim];
                previous_position(particle_index, dim) = candidate[dim];
                previous_step_position(particle_index, dim) = candidate[dim];
            }
            momentum(particle_index) = local_kinematics.momentum_magnitude;
            mu(particle_index) = local_kinematics.mu;
            particle_weight(particle_index) = local_kinematics.weight;
            initial_kinetic_energy(particle_index) =
                injection_kinetic_energy(local_properties, local_kinematics);
            status(particle_index) =
                accepted ? static_cast<int>(local_kinematics.status)
                         : static_cast<int>(ParticleStatus::Inactive);
            split_level(particle_index) = 0;
            sort_key(particle_index) = 0;
            random_pool.free_state(generator);
        });
    Kokkos::fence("ParticleGenerator::inject_particles_uniform_if");
    particles.commit_particle_range(static_cast<size_type>(plan.first_particle_index),
                                    static_cast<size_type>(plan.particle_count));
}

/**
 * Inject particles by sampling from a physical-volume-weighted field-grid candidate set.
 */
template <typename ParticleSystemType, typename RandomManagerType, typename CandidateSetType>
void inject_particles_from_field_candidates(
    ParticleSystemType& particles,
    const RandomManagerType& random_manager,
    const CandidateSetType& candidates,
    const ParticleInjectionPlan& plan,
    const ParticleInjectionKinematics<ParticleSystemType::space_dim>& kinematics =
        ParticleInjectionKinematics<ParticleSystemType::space_dim>{}) {
    static_assert(ParticleSystemType::space_dim == CandidateSetType::field_type::space_dim,
                  "ParticleSystem and field candidate set dimensions must match.");
    validate_particle_injection_kinematics(kinematics);
    validate_particle_injection_plan(particles, plan);
    if (candidates.empty()) {
        throw std::runtime_error(
            "ParticleGenerator: field-conditioned injection has no candidate grid points.");
    }

    auto random_pool = random_manager.pool;
    auto particle_id = particles.particle_id;
    auto position = particles.position;
    auto previous_position = particles.previous_position;
    auto previous_step_position = particles.previous_step_position;
    auto momentum = particles.momentum;
    auto mu = particles.mu;
    auto particle_weight = particles.weight;
    auto initial_kinetic_energy = particles.initial_kinetic_energy;
    auto status = particles.status;
    auto split_level = particles.split_level;
    auto sort_key = particles.sort_key;
    const auto local_properties = particles.properties;
    const auto local_candidates = candidates;
    const auto local_kinematics = kinematics;
    const auto local_plan = plan;

    using size_type = typename ParticleSystemType::size_type;
    Kokkos::parallel_for(
        "ParticleGenerator::inject_particles_from_field_candidates",
        Kokkos::RangePolicy<typename ParticleSystemType::execution_space>(
            0, static_cast<size_type>(plan.particle_count)),
        KOKKOS_LAMBDA(const size_type local_index) {
            auto generator = random_pool.get_state();
            const size_type candidate_index =
                local_candidates.sample_candidate_index(generator);
            const auto candidate = local_candidates.coordinate(candidate_index);

            const size_type particle_index =
                static_cast<size_type>(local_plan.first_particle_index) + local_index;
            particle_id(particle_index) =
                local_plan.first_particle_id + static_cast<std::uint64_t>(local_index);
            for (int dim = 0; dim < ParticleSystemType::space_dim; ++dim) {
                position(particle_index, dim) = candidate[dim];
                previous_position(particle_index, dim) = candidate[dim];
                previous_step_position(particle_index, dim) = candidate[dim];
            }
            momentum(particle_index) = local_kinematics.momentum_magnitude;
            mu(particle_index) = local_kinematics.mu;
            particle_weight(particle_index) = local_kinematics.weight;
            initial_kinetic_energy(particle_index) =
                injection_kinetic_energy(local_properties, local_kinematics);
            status(particle_index) = static_cast<int>(local_kinematics.status);
            split_level(particle_index) = 0;
            sort_key(particle_index) = 0;
            random_pool.free_state(generator);
        });
    Kokkos::fence("ParticleGenerator::inject_particles_from_field_candidates");
    particles.commit_particle_range(static_cast<size_type>(plan.first_particle_index),
                                    static_cast<size_type>(plan.particle_count));
}

/**
 * Inject particles by sampling physical cells from a precomputed grid candidate set.
 */
template <typename ParticleSystemType, typename RandomManagerType, typename CandidateSetType>
void inject_particles_from_grid_candidates(
    ParticleSystemType& particles,
    const RandomManagerType& random_manager,
    const CandidateSetType& candidates,
    const ParticleInjectionPlan& plan,
    const ParticleInjectionKinematics<ParticleSystemType::space_dim>& kinematics =
        ParticleInjectionKinematics<ParticleSystemType::space_dim>{}) {
    static_assert(ParticleSystemType::space_dim == CandidateSetType::grid_type::space_dim,
                  "ParticleSystem and grid candidate set dimensions must match.");
    validate_particle_injection_kinematics(kinematics);
    validate_particle_injection_plan(particles, plan);
    if (candidates.empty()) {
        throw std::runtime_error(
            "ParticleGenerator: grid injection has no candidate cells.");
    }

    auto random_pool = random_manager.pool;
    auto particle_id = particles.particle_id;
    auto position = particles.position;
    auto previous_position = particles.previous_position;
    auto previous_step_position = particles.previous_step_position;
    auto momentum = particles.momentum;
    auto mu = particles.mu;
    auto particle_weight = particles.weight;
    auto initial_kinetic_energy = particles.initial_kinetic_energy;
    auto status = particles.status;
    auto split_level = particles.split_level;
    auto sort_key = particles.sort_key;
    const auto local_properties = particles.properties;
    const auto local_candidates = candidates;
    const auto local_kinematics = kinematics;
    const auto local_plan = plan;

    using size_type = typename ParticleSystemType::size_type;
    Kokkos::parallel_for(
        "ParticleGenerator::inject_particles_from_grid_candidates",
        Kokkos::RangePolicy<typename ParticleSystemType::execution_space>(
            0, static_cast<size_type>(plan.particle_count)),
        KOKKOS_LAMBDA(const size_type local_index) {
            auto generator = random_pool.get_state();
            const size_type candidate_index =
                local_candidates.sample_candidate_index(generator);
            const auto candidate =
                local_candidates.sample_coordinate(candidate_index, generator);

            const size_type particle_index =
                static_cast<size_type>(local_plan.first_particle_index) + local_index;
            particle_id(particle_index) =
                local_plan.first_particle_id + static_cast<std::uint64_t>(local_index);
            for (int dim = 0; dim < ParticleSystemType::space_dim; ++dim) {
                position(particle_index, dim) = candidate[dim];
                previous_position(particle_index, dim) = candidate[dim];
                previous_step_position(particle_index, dim) = candidate[dim];
            }
            momentum(particle_index) = local_kinematics.momentum_magnitude;
            mu(particle_index) = local_kinematics.mu;
            particle_weight(particle_index) = local_kinematics.weight;
            initial_kinetic_energy(particle_index) =
                injection_kinetic_energy(local_properties, local_kinematics);
            status(particle_index) = static_cast<int>(local_kinematics.status);
            split_level(particle_index) = 0;
            sort_key(particle_index) = 0;
            random_pool.free_state(generator);
        });
    Kokkos::fence("ParticleGenerator::inject_particles_from_grid_candidates");
    particles.commit_particle_range(static_cast<size_type>(plan.first_particle_index),
                                    static_cast<size_type>(plan.particle_count));
}

/**
 * Inject particles uniformly in a region.
 */
template <typename ParticleSystemType, typename RandomManagerType>
void inject_particles_uniform(
    ParticleSystemType& particles,
    const RandomManagerType& random_manager,
    const AxisAlignedParticleRegion<ParticleSystemType::space_dim>& region,
    const ParticleInjectionPlan& plan,
    const ParticleInjectionKinematics<ParticleSystemType::space_dim>& kinematics =
        ParticleInjectionKinematics<ParticleSystemType::space_dim>{}) {
    inject_particles_uniform_if(particles, random_manager, region, plan, kinematics,
                                AlwaysAcceptParticlePredicate{});
}

/**
 * Inject particles across a coordinate grid's selected domain by physical-cell volume.
 */
template <typename ParticleSystemType, typename RandomManagerType, typename GridType>
void inject_particles_uniform_in_grid(
    ParticleSystemType& particles,
    const RandomManagerType& random_manager,
    const GridType& grid,
    const ParticleInjectionPlan& plan,
    const ParticleInjectionKinematics<ParticleSystemType::space_dim>& kinematics =
        ParticleInjectionKinematics<ParticleSystemType::space_dim>{},
    const GridDomain domain = GridDomain::PhysicalDomain) {
    static_assert(ParticleSystemType::space_dim == GridType::space_dim,
                  "ParticleSystem and Grid dimensions must match.");
    const auto candidates = make_grid_particle_candidate_set(grid, domain);
    inject_particles_from_grid_candidates(particles, random_manager, candidates, plan,
                                          kinematics);
}

/**
 * Inject particles in a region where a scalar field predicate is satisfied.
 */
template <typename ParticleSystemType, typename RandomManagerType, typename FieldType>
void inject_particles_uniform_where_field(
    ParticleSystemType& particles,
    const RandomManagerType& random_manager,
    const AxisAlignedParticleRegion<ParticleSystemType::space_dim>& region,
    const ParticleInjectionPlan& plan,
    const ParticleInjectionKinematics<ParticleSystemType::space_dim>& kinematics,
    const FieldType& field,
    const double threshold,
    const FieldPredicateComparison comparison = FieldPredicateComparison::GreaterEqual,
    const int component = 0,
    const GridDomain domain = GridDomain::PhysicalDomain) {
    static_assert(ParticleSystemType::space_dim == FieldType::space_dim,
                  "ParticleSystem and Field dimensions must match.");
    const auto candidates = make_field_particle_candidate_set_in_region(
        field, region, threshold, comparison, component, domain);
    inject_particles_from_field_candidates(particles, random_manager, candidates, plan,
                                           kinematics);
}

/**
 * Inject particles from all field-grid points that pass a threshold.
 */
template <typename ParticleSystemType, typename RandomManagerType, typename FieldType>
void inject_particles_uniform_where_field(
    ParticleSystemType& particles,
    const RandomManagerType& random_manager,
    const ParticleInjectionPlan& plan,
    const ParticleInjectionKinematics<ParticleSystemType::space_dim>& kinematics,
    const FieldType& field,
    const double threshold,
    const FieldPredicateComparison comparison = FieldPredicateComparison::GreaterEqual,
    const int component = 0,
    const GridDomain domain = GridDomain::PhysicalDomain) {
    static_assert(ParticleSystemType::space_dim == FieldType::space_dim,
                  "ParticleSystem and Field dimensions must match.");
    const auto candidates = make_field_particle_candidate_set(
        field, threshold, comparison, component, domain);
    inject_particles_from_field_candidates(particles, random_manager, candidates, plan,
                                           kinematics);
}

/**
 * Recycle selected particles by uniformly sampling a region with a custom predicate.
 *
 * Particle ids are kept unchanged so the storage slot remains stable.
 */
template <
    typename ParticleSystemType,
    typename RandomManagerType,
    typename PredicateType
>
void recycle_particles_uniform_if(
    ParticleSystemType& particles,
    const RandomManagerType& random_manager,
    const AxisAlignedParticleRegion<ParticleSystemType::space_dim>& region,
    const ParticleInjectionKinematics<ParticleSystemType::space_dim>& kinematics,
    PredicateType predicate,
    const ParticleRecycleSelection selection = ParticleRecycleSelection::EscapedOnly,
    const int max_attempts = 64) {
    region.validate();
    validate_particle_injection_kinematics(kinematics);
    if (max_attempts <= 0) {
        throw std::runtime_error("ParticleGenerator: recycle max_attempts must be positive.");
    }

    auto random_pool = random_manager.pool;
    auto position = particles.position;
    auto previous_position = particles.previous_position;
    auto previous_step_position = particles.previous_step_position;
    auto momentum = particles.momentum;
    auto mu = particles.mu;
    auto particle_weight = particles.weight;
    auto initial_kinetic_energy = particles.initial_kinetic_energy;
    auto status = particles.status;
    auto split_level = particles.split_level;
    auto sort_key = particles.sort_key;
    const auto local_properties = particles.properties;
    const auto local_region = region;
    const auto local_kinematics = kinematics;
    const auto local_selection = selection;
    const auto local_max_attempts = max_attempts;

    using size_type = typename ParticleSystemType::size_type;
    Kokkos::parallel_for(
        "ParticleGenerator::recycle_particles_uniform_if",
        Kokkos::RangePolicy<typename ParticleSystemType::execution_space>(
            0, particles.particle_count()),
        KOKKOS_LAMBDA(const size_type particle_index) {
            if (!particle_recycle_selection_matches(status(particle_index),
                                                    local_selection)) {
                return;
            }

            auto generator = random_pool.get_state();
            typename AxisAlignedParticleRegion<ParticleSystemType::space_dim>::
                coordinate_array_type candidate{};
            bool accepted = false;
            for (int attempt = 0; attempt < local_max_attempts; ++attempt) {
                candidate = local_region.sample(generator);
                if (predicate(candidate)) {
                    accepted = true;
                    break;
                }
            }

            for (int dim = 0; dim < ParticleSystemType::space_dim; ++dim) {
                position(particle_index, dim) = candidate[dim];
                previous_position(particle_index, dim) = candidate[dim];
                previous_step_position(particle_index, dim) = candidate[dim];
            }
            momentum(particle_index) = local_kinematics.momentum_magnitude;
            mu(particle_index) = local_kinematics.mu;
            particle_weight(particle_index) = local_kinematics.weight;
            initial_kinetic_energy(particle_index) =
                injection_kinetic_energy(local_properties, local_kinematics);
            status(particle_index) =
                accepted ? static_cast<int>(local_kinematics.status)
                         : static_cast<int>(ParticleStatus::Inactive);
            split_level(particle_index) = 0;
            sort_key(particle_index) = 0;
            random_pool.free_state(generator);
        });
    Kokkos::fence("ParticleGenerator::recycle_particles_uniform_if");
}

/**
 * Recycle selected particles by uniformly sampling a region.
 */
template <typename ParticleSystemType, typename RandomManagerType>
void recycle_particles_uniform(
    ParticleSystemType& particles,
    const RandomManagerType& random_manager,
    const AxisAlignedParticleRegion<ParticleSystemType::space_dim>& region,
    const ParticleInjectionKinematics<ParticleSystemType::space_dim>& kinematics =
        ParticleInjectionKinematics<ParticleSystemType::space_dim>{},
    const ParticleRecycleSelection selection = ParticleRecycleSelection::EscapedOnly) {
    recycle_particles_uniform_if(particles, random_manager, region, kinematics,
                                 AlwaysAcceptParticlePredicate{}, selection);
}

template <typename ParticleSystemType, typename RandomManagerType, typename CandidateSetType>
void recycle_particles_from_grid_candidates(
    ParticleSystemType& particles,
    const RandomManagerType& random_manager,
    const CandidateSetType& candidates,
    const ParticleInjectionKinematics<ParticleSystemType::space_dim>& kinematics =
        ParticleInjectionKinematics<ParticleSystemType::space_dim>{},
    ParticleRecycleSelection selection = ParticleRecycleSelection::EscapedOnly);

/**
 * Recycle selected particles by sampling a coordinate grid domain by physical-cell volume.
 */
template <typename ParticleSystemType, typename RandomManagerType, typename GridType>
void recycle_particles_uniform_in_grid(
    ParticleSystemType& particles,
    const RandomManagerType& random_manager,
    const GridType& grid,
    const ParticleInjectionKinematics<ParticleSystemType::space_dim>& kinematics =
        ParticleInjectionKinematics<ParticleSystemType::space_dim>{},
    const ParticleRecycleSelection selection = ParticleRecycleSelection::EscapedOnly,
    const GridDomain domain = GridDomain::PhysicalDomain) {
    static_assert(ParticleSystemType::space_dim == GridType::space_dim,
                  "ParticleSystem and Grid dimensions must match.");
    const auto candidates = make_grid_particle_candidate_set(grid, domain);
    recycle_particles_from_grid_candidates(particles, random_manager, candidates,
                                           kinematics, selection);
}

/**
 * Recycle selected particles by sampling physical cells from a grid candidate set.
 */
template <typename ParticleSystemType, typename RandomManagerType, typename CandidateSetType>
void recycle_particles_from_grid_candidates(
    ParticleSystemType& particles,
    const RandomManagerType& random_manager,
    const CandidateSetType& candidates,
    const ParticleInjectionKinematics<ParticleSystemType::space_dim>& kinematics,
    const ParticleRecycleSelection selection) {
    static_assert(ParticleSystemType::space_dim == CandidateSetType::grid_type::space_dim,
                  "ParticleSystem and grid candidate set dimensions must match.");
    validate_particle_injection_kinematics(kinematics);
    if (candidates.empty()) {
        throw std::runtime_error(
            "ParticleGenerator: grid recycling has no candidate cells.");
    }

    auto random_pool = random_manager.pool;
    auto position = particles.position;
    auto previous_position = particles.previous_position;
    auto previous_step_position = particles.previous_step_position;
    auto momentum = particles.momentum;
    auto mu = particles.mu;
    auto particle_weight = particles.weight;
    auto initial_kinetic_energy = particles.initial_kinetic_energy;
    auto status = particles.status;
    auto split_level = particles.split_level;
    auto sort_key = particles.sort_key;
    const auto local_properties = particles.properties;
    const auto local_candidates = candidates;
    const auto local_kinematics = kinematics;
    const auto local_selection = selection;

    using size_type = typename ParticleSystemType::size_type;
    Kokkos::parallel_for(
        "ParticleGenerator::recycle_particles_from_grid_candidates",
        Kokkos::RangePolicy<typename ParticleSystemType::execution_space>(
            0, particles.particle_count()),
        KOKKOS_LAMBDA(const size_type particle_index) {
            if (!particle_recycle_selection_matches(status(particle_index),
                                                    local_selection)) {
                return;
            }

            auto generator = random_pool.get_state();
            const size_type candidate_index =
                local_candidates.sample_candidate_index(generator);
            const auto candidate =
                local_candidates.sample_coordinate(candidate_index, generator);

            for (int dim = 0; dim < ParticleSystemType::space_dim; ++dim) {
                position(particle_index, dim) = candidate[dim];
                previous_position(particle_index, dim) = candidate[dim];
                previous_step_position(particle_index, dim) = candidate[dim];
            }
            momentum(particle_index) = local_kinematics.momentum_magnitude;
            mu(particle_index) = local_kinematics.mu;
            particle_weight(particle_index) = local_kinematics.weight;
            initial_kinetic_energy(particle_index) =
                injection_kinetic_energy(local_properties, local_kinematics);
            status(particle_index) = static_cast<int>(local_kinematics.status);
            split_level(particle_index) = 0;
            sort_key(particle_index) = 0;
            random_pool.free_state(generator);
        });
    Kokkos::fence("ParticleGenerator::recycle_particles_from_grid_candidates");
}

/**
 * Recycle selected particles by sampling a precomputed field-grid candidate set.
 */
template <typename ParticleSystemType, typename RandomManagerType, typename CandidateSetType>
void recycle_particles_from_field_candidates(
    ParticleSystemType& particles,
    const RandomManagerType& random_manager,
    const CandidateSetType& candidates,
    const ParticleInjectionKinematics<ParticleSystemType::space_dim>& kinematics =
        ParticleInjectionKinematics<ParticleSystemType::space_dim>{},
    const ParticleRecycleSelection selection = ParticleRecycleSelection::EscapedOnly) {
    static_assert(ParticleSystemType::space_dim == CandidateSetType::field_type::space_dim,
                  "ParticleSystem and field candidate set dimensions must match.");
    validate_particle_injection_kinematics(kinematics);
    if (candidates.empty()) {
        throw std::runtime_error(
            "ParticleGenerator: field-conditioned recycling has no candidate grid points.");
    }

    auto random_pool = random_manager.pool;
    auto position = particles.position;
    auto previous_position = particles.previous_position;
    auto previous_step_position = particles.previous_step_position;
    auto momentum = particles.momentum;
    auto mu = particles.mu;
    auto particle_weight = particles.weight;
    auto initial_kinetic_energy = particles.initial_kinetic_energy;
    auto status = particles.status;
    auto split_level = particles.split_level;
    auto sort_key = particles.sort_key;
    const auto local_properties = particles.properties;
    const auto local_candidates = candidates;
    const auto local_kinematics = kinematics;
    const auto local_selection = selection;

    using size_type = typename ParticleSystemType::size_type;
    Kokkos::parallel_for(
        "ParticleGenerator::recycle_particles_from_field_candidates",
        Kokkos::RangePolicy<typename ParticleSystemType::execution_space>(
            0, particles.particle_count()),
        KOKKOS_LAMBDA(const size_type particle_index) {
            if (!particle_recycle_selection_matches(status(particle_index),
                                                    local_selection)) {
                return;
            }

            auto generator = random_pool.get_state();
            const size_type candidate_index =
                local_candidates.sample_candidate_index(generator);
            const auto candidate = local_candidates.coordinate(candidate_index);

            for (int dim = 0; dim < ParticleSystemType::space_dim; ++dim) {
                position(particle_index, dim) = candidate[dim];
                previous_position(particle_index, dim) = candidate[dim];
                previous_step_position(particle_index, dim) = candidate[dim];
            }
            momentum(particle_index) = local_kinematics.momentum_magnitude;
            mu(particle_index) = local_kinematics.mu;
            particle_weight(particle_index) = local_kinematics.weight;
            initial_kinetic_energy(particle_index) =
                injection_kinetic_energy(local_properties, local_kinematics);
            status(particle_index) = static_cast<int>(local_kinematics.status);
            split_level(particle_index) = 0;
            sort_key(particle_index) = 0;
            random_pool.free_state(generator);
        });
    Kokkos::fence("ParticleGenerator::recycle_particles_from_field_candidates");
}

/**
 * Recycle selected particles from field-grid points that pass a threshold.
 */
template <typename ParticleSystemType, typename RandomManagerType, typename FieldType>
void recycle_particles_uniform_where_field(
    ParticleSystemType& particles,
    const RandomManagerType& random_manager,
    const AxisAlignedParticleRegion<ParticleSystemType::space_dim>& region,
    const ParticleInjectionKinematics<ParticleSystemType::space_dim>& kinematics,
    const FieldType& field,
    const double threshold,
    const FieldPredicateComparison comparison = FieldPredicateComparison::GreaterEqual,
    const int component = 0,
    const GridDomain domain = GridDomain::PhysicalDomain,
    const ParticleRecycleSelection selection = ParticleRecycleSelection::EscapedOnly) {
    static_assert(ParticleSystemType::space_dim == FieldType::space_dim,
                  "ParticleSystem and Field dimensions must match.");
    const auto candidates = make_field_particle_candidate_set_in_region(
        field, region, threshold, comparison, component, domain);
    recycle_particles_from_field_candidates(particles, random_manager, candidates,
                                            kinematics, selection);
}

/**
 * Recycle selected particles from all field-grid points that pass a threshold.
 */
template <typename ParticleSystemType, typename RandomManagerType, typename FieldType>
void recycle_particles_uniform_where_field(
    ParticleSystemType& particles,
    const RandomManagerType& random_manager,
    const ParticleInjectionKinematics<ParticleSystemType::space_dim>& kinematics,
    const FieldType& field,
    const double threshold,
    const FieldPredicateComparison comparison = FieldPredicateComparison::GreaterEqual,
    const int component = 0,
    const GridDomain domain = GridDomain::PhysicalDomain,
    const ParticleRecycleSelection selection = ParticleRecycleSelection::EscapedOnly) {
    static_assert(ParticleSystemType::space_dim == FieldType::space_dim,
                  "ParticleSystem and Field dimensions must match.");
    const auto candidates = make_field_particle_candidate_set(
        field, threshold, comparison, component, domain);
    recycle_particles_from_field_candidates(particles, random_manager, candidates,
                                            kinematics, selection);
}
