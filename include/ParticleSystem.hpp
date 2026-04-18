#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Grid.hpp"
#include "KokkosDevice.hpp"
#include "PhysicalConstant.hpp"
#include "UnitConverter.hpp"

/**
 * Common particle species supported by the built-in particle-parameter factory.
 */
enum class ParticleSpecies : int {
    Proton = 0,
    Electron = 1,
    Alpha = 2,
    Custom = 3
};

/**
 * Particle lifecycle state.
 */
enum class ParticleStatus : int {
    Active = 0,
    Escaped = 1,
    Inactive = 2
};

/**
 * Return whether a particle lifecycle value represents an actively transported particle.
 */
KOKKOS_INLINE_FUNCTION
constexpr bool particle_status_is_alive(const ParticleStatus status) {
    return status == ParticleStatus::Active;
}

/**
 * Return whether a raw particle lifecycle value represents an active particle.
 */
KOKKOS_INLINE_FUNCTION
constexpr bool particle_status_is_alive(const int status) {
    return particle_status_is_alive(static_cast<ParticleStatus>(status));
}

/**
 * Return whether a particle lifecycle value can be recycled into a new particle.
 */
KOKKOS_INLINE_FUNCTION
constexpr bool particle_status_is_recyclable(const ParticleStatus status) {
    return status == ParticleStatus::Escaped || status == ParticleStatus::Inactive;
}

/**
 * Return whether a raw particle lifecycle value can be recycled into a new particle.
 */
KOKKOS_INLINE_FUNCTION
constexpr bool particle_status_is_recyclable(const int status) {
    return particle_status_is_recyclable(static_cast<ParticleStatus>(status));
}

/**
 * Select whether momentum-direction connection correction should be applied.
 *
 * Scalar-momentum particle storage cannot represent a transported momentum direction.
 * The curvilinear mode is kept as a reserved API value for a future directional
 * particle model, but it is rejected by the current ParticleSystem implementation.
 */
enum class ParticleConnectionCorrectionMode : int {
    Disabled = 0,
    Curvilinear = 1
};

/**
 * Built-in particle species parameters in CGS units.
 */
struct ParticleSpeciesParametersCgs {
    ParticleSpecies species{ParticleSpecies::Custom};
    double rest_mass_cgs{1.0};
    double charge_cgs{1.0};
};

/**
 * Return built-in CGS parameters for a common particle species.
 */
KOKKOS_INLINE_FUNCTION
constexpr ParticleSpeciesParametersCgs particle_species_parameters_cgs(
    const ParticleSpecies species) {
    switch (species) {
        case ParticleSpecies::Proton:
            return ParticleSpeciesParametersCgs{
                species,
                PhysicalConstants::proton_mass_cgs,
                PhysicalConstants::elementary_charge_cgs
            };
        case ParticleSpecies::Electron:
            return ParticleSpeciesParametersCgs{
                species,
                PhysicalConstants::electron_mass_cgs,
                -PhysicalConstants::elementary_charge_cgs
            };
        case ParticleSpecies::Alpha:
            return ParticleSpeciesParametersCgs{
                species,
                6.6446573357e-24,
                2.0 * PhysicalConstants::elementary_charge_cgs
            };
        case ParticleSpecies::Custom:
            break;
    }
    return ParticleSpeciesParametersCgs{ParticleSpecies::Custom, 1.0, 1.0};
}

/**
 * Host-side species name for diagnostics and binary snapshots.
 */
inline const char* particle_species_name(const ParticleSpecies species) {
    switch (species) {
        case ParticleSpecies::Proton:
            return "proton";
        case ParticleSpecies::Electron:
            return "electron";
        case ParticleSpecies::Alpha:
            return "alpha";
        case ParticleSpecies::Custom:
            return "custom";
    }
    return "custom";
}

/**
 * Dimensionless particle parameters used by device kernels.
 */
struct ParticleProperties {
    ParticleSpecies species{ParticleSpecies::Custom};
    double rest_mass{1.0};
    double charge{1.0};
    double speed_of_light{1.0};
    double energy_scale_erg{1.0};

    /**
     * Validate that dimensionless particle parameters are physically usable.
     */
    void validate() const {
        if (!std::isfinite(rest_mass) || rest_mass <= 0.0 ||
            !std::isfinite(charge) ||
            !std::isfinite(speed_of_light) || speed_of_light <= 0.0 ||
            !std::isfinite(energy_scale_erg) || energy_scale_erg <= 0.0) {
            throw std::runtime_error(
                "ParticleProperties: rest mass and speed of light must be positive "
                "finite dimensionless values, charge must be finite, and the energy "
                "scale must be positive and finite.");
        }
    }

    /**
     * Return the relativistic gamma factor from a dimensionless momentum magnitude.
     */
    KOKKOS_INLINE_FUNCTION
    double gamma_from_momentum_magnitude(const double momentum_magnitude) const {
        const double normalized_momentum =
            momentum_magnitude / (rest_mass * speed_of_light);
        return Kokkos::sqrt(1.0 + normalized_momentum * normalized_momentum);
    }

    /**
     * Return the dimensionless speed magnitude from a dimensionless momentum magnitude.
     */
    KOKKOS_INLINE_FUNCTION
    double speed_from_momentum_magnitude(const double momentum_magnitude) const {
        const double gamma = gamma_from_momentum_magnitude(momentum_magnitude);
        return momentum_magnitude / (gamma * rest_mass);
    }

    /**
     * Return the dimensionless relativistic kinetic energy from momentum magnitude.
     */
    KOKKOS_INLINE_FUNCTION
    double kinetic_energy_from_momentum_magnitude(const double momentum_magnitude) const {
        const double gamma = gamma_from_momentum_magnitude(momentum_magnitude);
        return (gamma - 1.0) * rest_mass * speed_of_light * speed_of_light;
    }

    /**
     * Convert a dimensionless energy into electron volts.
     */
    KOKKOS_INLINE_FUNCTION
    double energy_to_electron_volt(const double dimensionless_energy) const {
        return dimensionless_energy * energy_scale_erg /
               PhysicalConstants::electron_volt_erg;
    }

    /**
     * Return the relativistic kinetic energy in electron volts.
     */
    KOKKOS_INLINE_FUNCTION
    double kinetic_energy_ev_from_momentum_magnitude(
        const double momentum_magnitude) const {
        return energy_to_electron_volt(
            kinetic_energy_from_momentum_magnitude(momentum_magnitude));
    }
};

/**
 * Build dimensionless particle properties from CGS particle parameters.
 */
inline ParticleProperties make_particle_properties_from_cgs(
    const ParticleSpecies species,
    const double rest_mass_cgs,
    const double charge_cgs,
    const Units::Nondimensionalizer& nondimensionalizer) {
    ParticleProperties properties{
        species,
        nondimensionalizer.to_dimensionless(rest_mass_cgs,
                                            Units::Dimensions::mass).raw(),
        nondimensionalizer.to_dimensionless(charge_cgs,
                                            Units::Dimensions::charge).raw(),
        nondimensionalizer.to_dimensionless(PhysicalConstants::speed_of_light).raw(),
        nondimensionalizer.units.scale(Units::Dimensions::energy)
    };
    properties.validate();
    return properties;
}

/**
 * Build dimensionless particle properties from one of the built-in species.
 */
inline ParticleProperties make_particle_properties(
    const ParticleSpecies species,
    const Units::Nondimensionalizer& nondimensionalizer) {
    const ParticleSpeciesParametersCgs cgs_parameters =
        particle_species_parameters_cgs(species);
    return make_particle_properties_from_cgs(species, cgs_parameters.rest_mass_cgs,
                                             cgs_parameters.charge_cgs,
                                             nondimensionalizer);
}

/**
 * One-particle host/device state used as a compact interface object.
 */
template <int SpaceDim>
struct ParticleState {
    using coordinate_array_type = Kokkos::Array<double, SpaceDim>;

    std::uint64_t particle_id{0};
    coordinate_array_type position{};
    double momentum_magnitude{0.0};
    double mu{0.0};
    double weight{1.0};
    double initial_kinetic_energy{0.0};
    int split_level{0};
    ParticleStatus status{ParticleStatus::Active};
};

/**
 * Host-side summary of one particle splitting pass.
 */
struct ParticleSplitStatistics {
    std::uint64_t active_count{0};
    std::uint64_t split_count{0};
    std::uint64_t skipped_capacity_count{0};
    std::uint64_t skipped_invalid_baseline_count{0};
    std::uint64_t skipped_minimum_weight_count{0};
};

/**
 * Fixed particle splitting policy used by one ParticleSystem instance.
 */
struct ParticleSplittingPolicy {
    double energy_split_ratio{2.0};
    double minimum_child_weight{0.0};

    /**
     * Validate fixed splitting controls on the host.
     */
    void validate() const {
        if (!std::isfinite(energy_split_ratio) || energy_split_ratio <= 1.0 ||
            !std::isfinite(minimum_child_weight) || minimum_child_weight < 0.0) {
            throw std::runtime_error(
                "ParticleSplittingPolicy: energy split ratio must be greater than one "
                "and minimum child weight must be finite and non-negative.");
        }
    }
};

template <
    int SpaceDim,
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout
>
struct ParticleSystemDebugger;

/**
 * Device-backed single-species particle container.
 */
template <
    int SpaceDim,
    typename DeviceType = Device,
    typename LayoutType = typename DeviceTraits<DeviceType>::array_layout
>
struct ParticleSystem {
    friend struct ParticleSystemDebugger<SpaceDim, DeviceType, LayoutType>;

    using execution_space = typename DeviceTraits<DeviceType>::execution_space;
    using memory_space = typename DeviceTraits<DeviceType>::memory_space;
    using layout_type = LayoutType;
    using size_type = typename memory_space::size_type;
    using particle_id_view_type = Kokkos::View<std::uint64_t*, layout_type, memory_space>;
    using scalar_view_type = Kokkos::View<double*, layout_type, memory_space>;
    using vector_view_type = Kokkos::View<double*[SpaceDim], layout_type, memory_space>;
    using int_view_type = Kokkos::View<int*, layout_type, memory_space>;
    using coordinate_array_type = Kokkos::Array<double, SpaceDim>;
    using particle_state_type = ParticleState<SpaceDim>;

    static constexpr int space_dim = SpaceDim;
    static_assert(SpaceDim > 0, "ParticleSystem SpaceDim must be positive.");

    ParticleProperties properties{};

private:
    ParticleSplittingPolicy splitting_policy_cache{};

public:
    ParticleConnectionCorrectionMode connection_correction_mode{
        ParticleConnectionCorrectionMode::Disabled
    };
    size_type particle_count_cache{0};
    size_type particle_capacity_cache{0};
    particle_id_view_type particle_id;
    vector_view_type position;
    vector_view_type previous_position;
    vector_view_type previous_step_position;
    scalar_view_type momentum;
    scalar_view_type mu;
    scalar_view_type weight;
    scalar_view_type initial_kinetic_energy;
    int_view_type status;
    int_view_type split_level;
    int_view_type sort_key;

    ParticleSystem() = default;

    /**
     * Build a particle system with allocated storage and dimensionless properties.
     */
    ParticleSystem(const size_type particle_count,
                   const ParticleProperties& input_properties,
                   const char* label = "particles")
        : ParticleSystem(particle_count, particle_count, input_properties,
                         ParticleSplittingPolicy{}, label) {}

    /**
     * Build a particle system with separate active count and storage capacity.
     */
    ParticleSystem(const size_type particle_count,
                   const size_type particle_capacity,
                   const ParticleProperties& input_properties,
                   const char* label = "particles")
        : ParticleSystem(particle_count, particle_capacity, input_properties,
                         ParticleSplittingPolicy{}, label) {}

    /**
     * Build a particle system with fixed splitting controls.
     */
    ParticleSystem(const size_type particle_count,
                   const size_type particle_capacity,
                   const ParticleProperties& input_properties,
                   const ParticleSplittingPolicy& input_splitting_policy,
                   const char* label = "particles")
        : properties(input_properties),
          splitting_policy_cache(input_splitting_policy),
          particle_count_cache(particle_count),
          particle_capacity_cache(particle_capacity),
          particle_id(make_label(label, "_id"), particle_capacity),
          position(make_label(label, "_position"), particle_capacity),
          previous_position(make_label(label, "_previous_position"), particle_capacity),
          previous_step_position(make_label(label, "_previous_step_position"),
                                 particle_capacity),
          momentum(make_label(label, "_momentum_magnitude"), particle_capacity),
          mu(make_label(label, "_mu"), particle_capacity),
          weight(make_label(label, "_weight"), particle_capacity),
          initial_kinetic_energy(make_label(label, "_initial_kinetic_energy"),
                                 particle_capacity),
          status(make_label(label, "_status"), particle_capacity),
          split_level(make_label(label, "_split_level"), particle_capacity),
          sort_key(make_label(label, "_sort_key"), particle_capacity) {
        properties.validate();
        splitting_policy_cache.validate();
        if (particle_capacity < particle_count) {
            throw std::runtime_error(
                "ParticleSystem: capacity must be greater than or equal to count.");
        }
        initialize_default_state();
    }

    /**
     * Return the number of stored particles.
     */
    KOKKOS_INLINE_FUNCTION
    size_type particle_count() const {
        return particle_count_cache;
    }

    /**
     * Return the allocated particle storage capacity.
     */
    KOKKOS_INLINE_FUNCTION
    size_type particle_capacity() const {
        return particle_capacity_cache;
    }

    /**
     * Return the fixed particle splitting policy for this particle system.
     */
    KOKKOS_INLINE_FUNCTION
    ParticleSplittingPolicy splitting_policy() const {
        return splitting_policy_cache;
    }

    /**
     * Return true when the container has no particles.
     */
    KOKKOS_INLINE_FUNCTION
    bool empty() const {
        return particle_count_cache == 0;
    }

    /**
     * Return the lifecycle status of one particle.
     */
    KOKKOS_INLINE_FUNCTION
    ParticleStatus particle_status(const size_type index) const {
        return static_cast<ParticleStatus>(status(index));
    }

    /**
     * Set the lifecycle status of one particle.
     */
    KOKKOS_INLINE_FUNCTION
    void set_status(const size_type index, const ParticleStatus input_status) const {
        status(index) = static_cast<int>(input_status);
    }

    /**
     * Return whether one particle is currently active.
     */
    KOKKOS_INLINE_FUNCTION
    bool is_alive(const size_type index) const {
        return particle_status_is_alive(status(index));
    }

    /**
     * Return whether one particle is available for recycling.
     */
    KOKKOS_INLINE_FUNCTION
    bool is_recyclable(const size_type index) const {
        return particle_status_is_recyclable(status(index));
    }

    /**
     * Extend the active storage range after an append-style injection.
     */
    void commit_particle_range(const size_type first_index,
                               const size_type inserted_count) {
        if (inserted_count == 0) {
            return;
        }
        if (first_index > particle_count_cache) {
            throw std::runtime_error(
                "ParticleSystem: appended particle ranges must be contiguous.");
        }
        if (inserted_count > particle_capacity_cache - first_index) {
            throw std::runtime_error(
                "ParticleSystem: appended particle range exceeds capacity.");
        }
        const size_type new_count = first_index + inserted_count;
        if (new_count > particle_count_cache) {
            particle_count_cache = new_count;
        }
    }

    /**
     * Return a compact particle state by particle index.
     */
    KOKKOS_INLINE_FUNCTION
    particle_state_type particle(const size_type index) const {
        particle_state_type state;
        state.particle_id = particle_id(index);
        for (int dim = 0; dim < SpaceDim; ++dim) {
            state.position[dim] = position(index, dim);
        }
        state.momentum_magnitude = momentum_magnitude(index);
        state.mu = mu(index);
        state.weight = weight(index);
        state.initial_kinetic_energy = initial_kinetic_energy(index);
        state.split_level = split_level(index);
        state.status = static_cast<ParticleStatus>(status(index));
        return state;
    }

    /**
     * Write a compact particle state by particle index.
     */
    KOKKOS_INLINE_FUNCTION
    void set_particle(const size_type index, const particle_state_type& state) const {
        particle_id(index) = state.particle_id;
        for (int dim = 0; dim < SpaceDim; ++dim) {
            position(index, dim) = state.position[dim];
            previous_position(index, dim) = state.position[dim];
            previous_step_position(index, dim) = state.position[dim];
        }
        momentum(index) = state.momentum_magnitude >= 0.0
                              ? state.momentum_magnitude
                              : -state.momentum_magnitude;
        mu(index) = state.mu;
        weight(index) = state.weight;
        initial_kinetic_energy(index) = state.initial_kinetic_energy;
        split_level(index) = state.split_level;
        status(index) = static_cast<int>(state.status);
    }

    /**
     * Return the squared dimensionless momentum magnitude of one particle.
     */
    KOKKOS_INLINE_FUNCTION
    double momentum_squared(const size_type index) const {
        const double particle_momentum = momentum(index);
        return particle_momentum * particle_momentum;
    }

    /**
     * Return the dimensionless momentum magnitude of one particle.
     */
    KOKKOS_INLINE_FUNCTION
    double momentum_magnitude(const size_type index) const {
        const double particle_momentum = momentum(index);
        return particle_momentum >= 0.0 ? particle_momentum : -particle_momentum;
    }

    /**
     * Return the relativistic gamma factor of one particle.
     */
    KOKKOS_INLINE_FUNCTION
    double gamma(const size_type index) const {
        return properties.gamma_from_momentum_magnitude(momentum_magnitude(index));
    }

    /**
     * Return the dimensionless speed magnitude of one particle.
     */
    KOKKOS_INLINE_FUNCTION
    double speed(const size_type index) const {
        return properties.speed_from_momentum_magnitude(momentum_magnitude(index));
    }

    /**
     * Return the dimensionless relativistic kinetic energy of one particle.
     */
    KOKKOS_INLINE_FUNCTION
    double kinetic_energy(const size_type index) const {
        return properties.kinetic_energy_from_momentum_magnitude(
            momentum_magnitude(index));
    }

    /**
     * Return the relativistic kinetic energy in electron volts.
     */
    KOKKOS_INLINE_FUNCTION
    double kinetic_energy_ev(const size_type index) const {
        return properties.energy_to_electron_volt(kinetic_energy(index));
    }

    /**
     * Return one dimensionless velocity component of one particle.
     */
    KOKKOS_INLINE_FUNCTION
    double velocity_component(const size_type index, const int dim) const {
        (void)index;
        (void)dim;
        Kokkos::abort(
            "ParticleSystem::velocity_component is undefined for scalar momentum storage.");
        return 0.0;
    }

    /**
     * Return the dimensionless velocity vector of one particle.
     */
    KOKKOS_INLINE_FUNCTION
    coordinate_array_type velocity(const size_type index) const {
        (void)index;
        coordinate_array_type result{};
        Kokkos::abort(
            "ParticleSystem::velocity is undefined for scalar momentum storage.");
        return result;
    }

    /**
     * Set whether curvilinear momentum connection correction is enabled.
     */
    void set_connection_correction_mode(const ParticleConnectionCorrectionMode mode) {
        if (mode != ParticleConnectionCorrectionMode::Disabled) {
            throw std::runtime_error(
                "ParticleSystem: curvilinear momentum correction requires directional "
                "momentum storage; the current container stores only momentum magnitude.");
        }
        connection_correction_mode = mode;
    }

    /**
     * Copy current positions into previous positions before a transport update.
     */
    void capture_previous_positions() {
        execution_space execution_instance;
        Kokkos::deep_copy(execution_instance, previous_step_position, previous_position);
        Kokkos::deep_copy(execution_instance, previous_position, position);
        execution_instance.fence("ParticleSystem::capture_previous_positions");
    }

    /**
     * Reset each particle's splitting baseline to its current kinetic energy.
     */
    void reset_split_baselines() {
        auto momenta = momentum;
        auto baselines = initial_kinetic_energy;
        auto levels = split_level;
        const auto local_properties = properties;
        const size_type count = particle_count_cache;
        Kokkos::parallel_for(
            "ParticleSystem::reset_split_baselines",
            Kokkos::RangePolicy<execution_space>(0, count),
            KOKKOS_LAMBDA(const size_type index) {
                const double particle_momentum =
                    momenta(index) >= 0.0 ? momenta(index) : -momenta(index);
                baselines(index) =
                    local_properties.kinetic_energy_from_momentum_magnitude(
                        particle_momentum);
                levels(index) = 0;
            });
        Kokkos::fence("ParticleSystem::reset_split_baselines");
    }

    /**
     * Apply coordinate-grid boundary handling to active particles.
     *
     * Periodic coordinates are wrapped back into the physical domain. Active particles
     * that leave a non-periodic boundary are marked as Escaped so injection code can
     * recycle their storage later.
     */
    template <typename GridType>
    void apply_grid_boundary_conditions(
        const GridType& grid,
        const GridDomain domain = GridDomain::PhysicalDomain) {
        static_assert(SpaceDim == GridType::space_dim,
                      "ParticleSystem and Grid dimensions must match.");
        static_assert(requires(const GridType& input_grid) {
                          input_grid.contains_coordinate(0, 0.0,
                                                         GridDomain::PhysicalDomain);
                          input_grid.wrap_coordinate(0, 0.0);
                      },
                      "ParticleSystem boundary handling requires a coordinate grid.");

        auto positions = position;
        auto previous_positions = previous_position;
        auto previous_step_positions = previous_step_position;
        auto statuses = status;
        const auto local_grid = grid;
        const size_type count = particle_count_cache;
        Kokkos::parallel_for(
            "ParticleSystem::apply_grid_boundary_conditions",
            Kokkos::RangePolicy<execution_space>(0, count),
            KOKKOS_LAMBDA(const size_type index) {
                if (!particle_status_is_alive(statuses(index))) {
                    return;
                }

                bool escaped = false;
                for (int dim = 0; dim < SpaceDim; ++dim) {
                    const double x = positions(index, dim);
                    if (local_grid.is_periodic(dim)) {
                        const double wrapped_x = local_grid.wrap_coordinate(dim, x);
                        const double periodic_shift = wrapped_x - x;
                        positions(index, dim) = wrapped_x;
                        previous_positions(index, dim) += periodic_shift;
                        previous_step_positions(index, dim) += periodic_shift;
                    } else if (!local_grid.contains_coordinate(dim, x, domain)) {
                        escaped = true;
                    }
                }

                if (escaped) {
                    statuses(index) = static_cast<int>(ParticleStatus::Escaped);
                }
            });
        Kokkos::fence("ParticleSystem::apply_grid_boundary_conditions");
    }

    /**
     * Assign sort keys using a device-callable key functor.
     */
    template <typename KeyFunctor>
    void assign_sort_keys(KeyFunctor key_functor) {
        auto keys = sort_key;
        const size_type count = particle_count_cache;
        Kokkos::parallel_for(
            "ParticleSystem::assign_sort_keys",
            Kokkos::RangePolicy<execution_space>(0, count),
            KOKKOS_LAMBDA(const size_type index) {
                keys(index) = key_functor(index);
            });
        Kokkos::fence("ParticleSystem::assign_sort_keys");
    }

    /**
     * Sort all particle arrays by the current integer sort keys.
     */
    void sort_by_key(const int minimum_key,
                     const int maximum_key,
                     const bool sort_within_bins = false) {
        if (particle_count_cache <= 1 || minimum_key == maximum_key) {
            return;
        }
        if (maximum_key < minimum_key) {
            throw std::runtime_error("ParticleSystem: invalid sort key range.");
        }
        if (particle_count_cache >
            static_cast<size_type>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "ParticleSystem: Kokkos::BinSort currently requires int-sized ranges.");
        }

        const int bin_count = maximum_key - minimum_key + 1;
        using bin_op_type = Kokkos::BinOp1D<int_view_type>;
        bin_op_type bin_op(bin_count, minimum_key, maximum_key);
        Kokkos::BinSort<int_view_type, bin_op_type> sorter(
            sort_key, 0, static_cast<int>(particle_count_cache), bin_op,
            sort_within_bins);
        sorter.create_permute_vector();
        sorter.sort(particle_id);
        sorter.sort(position);
        sorter.sort(previous_position);
        sorter.sort(previous_step_position);
        sorter.sort(momentum);
        sorter.sort(mu);
        sorter.sort(weight);
        sorter.sort(initial_kinetic_energy);
        sorter.sort(status);
        sorter.sort(split_level);
        sorter.sort(sort_key);
    }

    /**
     * Sort active particles to the front and shrink the stored range to active count.
     *
     * This is a host-side lifecycle compaction call around device reductions and
     * Kokkos::BinSort. Escaped and inactive particles remain in allocated storage after
     * the new particle_count() and may be overwritten by append-style injection.
     */
    size_type compact_active_particles() {
        if (particle_count_cache == 0) {
            return 0;
        }

        auto statuses = status;
        auto keys = sort_key;
        size_type active_count = 0;
        Kokkos::parallel_reduce(
            "ParticleSystem::compact_active_particles_keys",
            Kokkos::RangePolicy<execution_space>(0, particle_count_cache),
            KOKKOS_LAMBDA(const size_type index, size_type& update) {
                const bool active = particle_status_is_alive(statuses(index));
                keys(index) = active ? 0 : 1;
                if (active) {
                    ++update;
                }
            },
            active_count);
        Kokkos::fence("ParticleSystem::compact_active_particles_keys");

        sort_by_key(0, 1);
        particle_count_cache = active_count;
        return active_count;
    }

    /**
     * Split active particles by the fixed energy-growth policy.
     *
     * This device-side pass appends at most one child per active parent to the current
     * storage tail. The updated particle count is copied back to the host cache before
     * the function returns.
     */
    ParticleSplitStatistics split_particles_on_energy_growth() {
        ParticleSplitStatistics split_statistics;
        if (particle_count_cache == 0 || particle_count_cache >= particle_capacity_cache) {
            return split_statistics;
        }

        using counter_view_type =
            Kokkos::View<std::uint64_t*, layout_type, memory_space>;
        counter_view_type counters("particle_split_counters", 5);
        Kokkos::deep_copy(counters, static_cast<std::uint64_t>(0));

        auto ids = particle_id;
        auto positions = position;
        auto previous_positions = previous_position;
        auto previous_step_positions = previous_step_position;
        auto momenta = momentum;
        auto pitch_angle_cosines = mu;
        auto weights = weight;
        auto initial_energies = initial_kinetic_energy;
        auto statuses = status;
        auto split_levels = split_level;
        auto keys = sort_key;
        const auto local_properties = properties;
        const auto local_policy = splitting_policy_cache;
        const size_type old_count = particle_count_cache;
        const size_type capacity = particle_capacity_cache;
        const double log_split_ratio = std::log(local_policy.energy_split_ratio);

        Kokkos::parallel_for(
            "ParticleSystem::split_particles_on_energy_growth",
            Kokkos::RangePolicy<execution_space>(0, old_count),
            KOKKOS_LAMBDA(const size_type parent_index) {
                if (!particle_status_is_alive(statuses(parent_index))) {
                    return;
                }
                Kokkos::atomic_add(&counters(0), static_cast<std::uint64_t>(1));

                const double particle_momentum =
                    momenta(parent_index) >= 0.0
                        ? momenta(parent_index)
                        : -momenta(parent_index);
                const double current_energy =
                    local_properties.kinetic_energy_from_momentum_magnitude(
                        particle_momentum);
                const double baseline_energy = initial_energies(parent_index);
                if (!Kokkos::isfinite(baseline_energy) || baseline_energy <= 0.0) {
                    if (Kokkos::isfinite(current_energy) && current_energy > 0.0) {
                        initial_energies(parent_index) = current_energy;
                    }
                    Kokkos::atomic_add(&counters(3), static_cast<std::uint64_t>(1));
                    return;
                }

                const double energy_ratio = current_energy / baseline_energy;
                if (!Kokkos::isfinite(energy_ratio) ||
                    energy_ratio < local_policy.energy_split_ratio) {
                    return;
                }

                const int target_split_level = static_cast<int>(
                    Kokkos::floor(Kokkos::log(energy_ratio) / log_split_ratio));
                if (target_split_level <= split_levels(parent_index)) {
                    return;
                }

                const double child_weight = 0.5 * weights(parent_index);
                if (!Kokkos::isfinite(child_weight) || child_weight <= 0.0 ||
                    child_weight < local_policy.minimum_child_weight) {
                    Kokkos::atomic_add(&counters(4), static_cast<std::uint64_t>(1));
                    return;
                }

                const std::uint64_t append_offset = Kokkos::atomic_fetch_add(
                    &counters(1), static_cast<std::uint64_t>(1));
                const size_type child_index =
                    old_count + static_cast<size_type>(append_offset);
                if (child_index >= capacity) {
                    Kokkos::atomic_add(&counters(2), static_cast<std::uint64_t>(1));
                    return;
                }

                ids(child_index) = static_cast<std::uint64_t>(child_index);
                for (int dim = 0; dim < SpaceDim; ++dim) {
                    positions(child_index, dim) = positions(parent_index, dim);
                    previous_positions(child_index, dim) =
                        previous_positions(parent_index, dim);
                    previous_step_positions(child_index, dim) =
                        previous_step_positions(parent_index, dim);
                }
                momenta(child_index) = momenta(parent_index);
                pitch_angle_cosines(child_index) = pitch_angle_cosines(parent_index);
                weights(parent_index) = child_weight;
                weights(child_index) = child_weight;
                initial_energies(child_index) = baseline_energy;
                statuses(child_index) = static_cast<int>(ParticleStatus::Active);
                keys(child_index) = keys(parent_index);

                const int next_split_level = split_levels(parent_index) + 1;
                split_levels(parent_index) = next_split_level;
                split_levels(child_index) = next_split_level;
            });
        Kokkos::fence("ParticleSystem::split_particles_on_energy_growth");

        auto counters_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, counters);
        split_statistics.active_count = counters_host(0);
        const size_type available_capacity = particle_capacity_cache - old_count;
        const size_type appended_count = static_cast<size_type>(
            std::min<std::uint64_t>(counters_host(1), available_capacity));
        particle_count_cache = old_count + appended_count;
        split_statistics.split_count = static_cast<std::uint64_t>(appended_count);
        split_statistics.skipped_capacity_count = counters_host(2);
        split_statistics.skipped_invalid_baseline_count = counters_host(3);
        split_statistics.skipped_minimum_weight_count = counters_host(4);
        return split_statistics;
    }

    /**
     * Reject curvilinear momentum-direction correction for scalar momentum storage.
     */
    template <typename CorrectionFunctor>
    void apply_curvilinear_momentum_correction(CorrectionFunctor correction_functor) {
        (void)correction_functor;
        if (connection_correction_mode == ParticleConnectionCorrectionMode::Disabled) {
            return;
        }
        throw std::runtime_error(
            "ParticleSystem: curvilinear momentum correction requires directional "
            "momentum storage; the current container stores only momentum magnitude.");
    }

    /**
     * Save all particle data to a host-side binary snapshot.
     */
    void write_binary_snapshot(const std::string& file_path) const {
        std::ofstream stream(file_path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error(
                "ParticleSystem: failed to open binary particle output file.");
        }

        const auto id_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particle_id);
        const auto position_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, position);
        const auto previous_position_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, previous_position);
        const auto previous_step_position_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                previous_step_position);
        const auto momentum_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, momentum);
        const auto mu_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mu);
        const auto weight_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, weight);
        const auto initial_energy_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                initial_kinetic_energy);
        const auto status_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, status);
        const auto split_level_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, split_level);
        const auto sort_key_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sort_key);

        const auto magic = particle_binary_magic();
        stream.write(magic.data(), static_cast<std::streamsize>(magic.size()));
        if (!stream) {
            throw std::runtime_error("ParticleSystem: failed to write binary magic.");
        }
        write_binary_scalar(stream, particle_binary_version());
        write_binary_scalar(stream, particle_binary_endian_marker());
        write_binary_scalar(stream, static_cast<std::int32_t>(SpaceDim));
        write_binary_scalar(stream, static_cast<std::int32_t>(properties.species));
        write_binary_scalar(stream, static_cast<std::uint64_t>(particle_count_cache));
        write_binary_scalar(stream, static_cast<std::uint64_t>(particle_capacity_cache));
        write_binary_scalar(stream, properties.rest_mass);
        write_binary_scalar(stream, properties.charge);
        write_binary_scalar(stream, properties.speed_of_light);
        write_binary_scalar(stream, properties.energy_scale_erg);
        write_binary_scalar(stream, splitting_policy_cache.energy_split_ratio);
        write_binary_scalar(stream, splitting_policy_cache.minimum_child_weight);

        for (size_type index = 0; index < particle_count_cache; ++index) {
            write_binary_scalar(stream, id_host(index));
            write_binary_scalar(stream, status_host(index));
            write_binary_scalar(stream, split_level_host(index));
            write_binary_scalar(stream, sort_key_host(index));
            for (int dim = 0; dim < SpaceDim; ++dim) {
                write_binary_scalar(stream, position_host(index, dim));
            }
            for (int dim = 0; dim < SpaceDim; ++dim) {
                write_binary_scalar(stream, previous_position_host(index, dim));
            }
            for (int dim = 0; dim < SpaceDim; ++dim) {
                write_binary_scalar(stream, previous_step_position_host(index, dim));
            }
            write_binary_scalar(stream, momentum_host(index));
            write_binary_scalar(stream, mu_host(index));
            write_binary_scalar(stream, weight_host(index));
            write_binary_scalar(stream, initial_energy_host(index));
        }
    }

    /**
     * Build a particle system by reading a version-4 particle binary snapshot.
     */
    static ParticleSystem read_binary_snapshot(
        const std::string& file_path,
        const size_type minimum_capacity = 0,
        const char* label = "particles_restart") {
        std::ifstream stream(file_path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error(
                "ParticleSystem: failed to open binary particle input file.");
        }

        const auto expected_magic = particle_binary_magic();
        std::array<char, 8> magic{};
        stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
        if (!stream || magic != expected_magic) {
            throw std::runtime_error(
                "ParticleSystem: unrecognized particle binary magic.");
        }

        const auto version = read_binary_scalar<std::uint32_t>(stream);
        const auto endian_marker = read_binary_scalar<std::uint32_t>(stream);
        if (version != particle_binary_version()) {
            throw std::runtime_error(
                "ParticleSystem: only particle binary version 4 is restart-readable.");
        }
        if (endian_marker != particle_binary_endian_marker()) {
            throw std::runtime_error(
                "ParticleSystem: particle binary endian does not match this host.");
        }

        const auto file_space_dim = read_binary_scalar<std::int32_t>(stream);
        const auto file_species = read_binary_scalar<std::int32_t>(stream);
        const auto file_count = read_binary_scalar<std::uint64_t>(stream);
        const auto file_capacity = read_binary_scalar<std::uint64_t>(stream);
        if (file_space_dim != SpaceDim) {
            throw std::runtime_error(
                "ParticleSystem: snapshot dimension does not match ParticleSystem.");
        }
        if (file_species < static_cast<std::int32_t>(ParticleSpecies::Proton) ||
            file_species > static_cast<std::int32_t>(ParticleSpecies::Custom)) {
            throw std::runtime_error(
                "ParticleSystem: snapshot contains an invalid particle species.");
        }

        ParticleProperties snapshot_properties;
        snapshot_properties.species =
            static_cast<ParticleSpecies>(file_species);
        snapshot_properties.rest_mass = read_binary_scalar<double>(stream);
        snapshot_properties.charge = read_binary_scalar<double>(stream);
        snapshot_properties.speed_of_light = read_binary_scalar<double>(stream);
        snapshot_properties.energy_scale_erg = read_binary_scalar<double>(stream);
        snapshot_properties.validate();

        ParticleSplittingPolicy snapshot_policy;
        snapshot_policy.energy_split_ratio = read_binary_scalar<double>(stream);
        snapshot_policy.minimum_child_weight = read_binary_scalar<double>(stream);
        snapshot_policy.validate();

        const size_type restart_count = static_cast<size_type>(file_count);
        const size_type restart_capacity = static_cast<size_type>(
            std::max<std::uint64_t>(
                file_count,
                std::max<std::uint64_t>(
                    file_capacity,
                    static_cast<std::uint64_t>(minimum_capacity))));
        ParticleSystem particles(restart_count, restart_capacity,
                                 snapshot_properties, snapshot_policy, label);

        auto id_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                particles.particle_id);
        auto position_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                particles.position);
        auto previous_position_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                particles.previous_position);
        auto previous_step_position_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                particles.previous_step_position);
        auto momentum_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                particles.momentum);
        auto mu_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particles.mu);
        auto weight_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                particles.weight);
        auto initial_energy_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                particles.initial_kinetic_energy);
        auto status_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                particles.status);
        auto split_level_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                particles.split_level);
        auto sort_key_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                particles.sort_key);

        for (size_type index = 0; index < restart_count; ++index) {
            id_host(index) = read_binary_scalar<std::uint64_t>(stream);
            status_host(index) = read_binary_scalar<std::int32_t>(stream);
            split_level_host(index) = read_binary_scalar<std::int32_t>(stream);
            sort_key_host(index) = read_binary_scalar<std::int32_t>(stream);
            for (int dim = 0; dim < SpaceDim; ++dim) {
                position_host(index, dim) = read_binary_scalar<double>(stream);
            }
            for (int dim = 0; dim < SpaceDim; ++dim) {
                previous_position_host(index, dim) =
                    read_binary_scalar<double>(stream);
            }
            for (int dim = 0; dim < SpaceDim; ++dim) {
                previous_step_position_host(index, dim) =
                    read_binary_scalar<double>(stream);
            }
            momentum_host(index) = read_binary_scalar<double>(stream);
            mu_host(index) = read_binary_scalar<double>(stream);
            weight_host(index) = read_binary_scalar<double>(stream);
            initial_energy_host(index) = read_binary_scalar<double>(stream);
        }

        Kokkos::deep_copy(particles.particle_id, id_host);
        Kokkos::deep_copy(particles.position, position_host);
        Kokkos::deep_copy(particles.previous_position, previous_position_host);
        Kokkos::deep_copy(particles.previous_step_position,
                          previous_step_position_host);
        Kokkos::deep_copy(particles.momentum, momentum_host);
        Kokkos::deep_copy(particles.mu, mu_host);
        Kokkos::deep_copy(particles.weight, weight_host);
        Kokkos::deep_copy(particles.initial_kinetic_energy, initial_energy_host);
        Kokkos::deep_copy(particles.status, status_host);
        Kokkos::deep_copy(particles.split_level, split_level_host);
        Kokkos::deep_copy(particles.sort_key, sort_key_host);
        return particles;
    }

private:
    /**
     * Create a Kokkos label by appending a suffix.
     */
    static std::string make_label(const char* label, const char* suffix) {
        return std::string(label) + suffix;
    }

    /**
     * Initialize ids, status flags, sort keys, and numeric arrays.
     */
    void initialize_default_state() {
        auto ids = particle_id;
        auto positions = position;
        auto previous_positions = previous_position;
        auto previous_step_positions = previous_step_position;
        auto momenta = momentum;
        auto pitch_angle_cosines = mu;
        auto weights = weight;
        auto initial_energies = initial_kinetic_energy;
        auto statuses = status;
        auto split_levels = split_level;
        auto keys = sort_key;
        const size_type count = particle_count_cache;
        const size_type capacity = particle_capacity_cache;
        Kokkos::parallel_for(
            "ParticleSystem::initialize_default_state",
            Kokkos::RangePolicy<execution_space>(0, capacity),
            KOKKOS_LAMBDA(const size_type index) {
                ids(index) = static_cast<std::uint64_t>(index);
                pitch_angle_cosines(index) = 0.0;
                weights(index) = 1.0;
                initial_energies(index) = 0.0;
                statuses(index) = index < count
                                      ? static_cast<int>(ParticleStatus::Active)
                                      : static_cast<int>(ParticleStatus::Inactive);
                split_levels(index) = 0;
                keys(index) = 0;
                for (int dim = 0; dim < SpaceDim; ++dim) {
                    positions(index, dim) = 0.0;
                    previous_positions(index, dim) = 0.0;
                    previous_step_positions(index, dim) = 0.0;
                }
                momenta(index) = 0.0;
            });
        Kokkos::fence("ParticleSystem::initialize_default_state");
    }

    /**
     * Particle binary snapshot magic.
     */
    static constexpr std::array<char, 8> particle_binary_magic() {
        return {'K', 'P', 'T', 'P', 'R', 'T', '\0', '\0'};
    }

    /**
     * Particle binary snapshot version.
     */
    static constexpr std::uint32_t particle_binary_version() {
        return 4u;
    }

    /**
     * Particle binary endian marker.
     */
    static constexpr std::uint32_t particle_binary_endian_marker() {
        return 0x01020304u;
    }

    /**
     * Write a trivially copyable scalar to a particle binary stream.
     */
    template <typename T>
    static void write_binary_scalar(std::ofstream& stream, const T& value) {
        stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
        if (!stream) {
            throw std::runtime_error("ParticleSystem: failed to write binary scalar.");
        }
    }

    /**
     * Read a trivially copyable scalar from a particle binary stream.
     */
    template <typename T>
    static T read_binary_scalar(std::ifstream& stream) {
        T value{};
        stream.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!stream) {
            throw std::runtime_error("ParticleSystem: failed to read binary scalar.");
        }
        return value;
    }
};
