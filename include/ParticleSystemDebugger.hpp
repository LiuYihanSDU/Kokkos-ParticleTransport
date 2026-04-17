#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <limits>
#include <ostream>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "ParticleSystem.hpp"

/**
 * Host-side particle statistics snapshot for debug diagnostics.
 */
struct ParticleStatistics {
    std::uint64_t total_count{0};
    std::uint64_t active_count{0};
    std::uint64_t escaped_count{0};
    std::uint64_t inactive_count{0};
    double active_weight_sum{0.0};
    double mean_gamma{0.0};
    double max_gamma{0.0};
    double mean_speed{0.0};
    double max_speed{0.0};
    double active_mean_kinetic_energy_ev{0.0};
    double active_min_kinetic_energy_ev{0.0};
    double active_max_kinetic_energy_ev{0.0};
};

/**
 * Debug-only host-side diagnostics for ParticleSystem.
 *
 * This class intentionally performs host mirrors and should not be used in production
 * transport loops. In release builds with NDEBUG it throws if diagnostics are requested.
 */
template <
    int SpaceDim,
    typename DeviceType,
    typename LayoutType
>
struct ParticleSystemDebugger {
    using particle_system_type = ParticleSystem<SpaceDim, DeviceType, LayoutType>;
    using size_type = typename particle_system_type::size_type;

    /**
     * Compute host-side particle statistics for debug diagnostics.
     */
    static ParticleStatistics compute_statistics(const particle_system_type& particles) {
#ifdef NDEBUG
        (void)particles;
        throw std::runtime_error(
            "ParticleSystemDebugger::compute_statistics is available only in debug builds.");
#else
        ParticleStatistics statistics;
        statistics.total_count =
            static_cast<std::uint64_t>(particles.particle_count_cache);
        if (particles.particle_count_cache == 0) {
            return statistics;
        }

        const auto momentum_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particles.momentum);
        const auto weight_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particles.weight);
        const auto status_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, particles.status);

        double gamma_sum = 0.0;
        double speed_sum = 0.0;
        double active_weighted_energy_sum = 0.0;
        double active_min_energy_ev = std::numeric_limits<double>::infinity();
        for (size_type index = 0; index < particles.particle_count_cache; ++index) {
            const ParticleStatus particle_status =
                static_cast<ParticleStatus>(status_host(index));
            if (particle_status == ParticleStatus::Active) {
                ++statistics.active_count;
            } else if (particle_status == ParticleStatus::Escaped) {
                ++statistics.escaped_count;
            } else {
                ++statistics.inactive_count;
            }

            const double stored_momentum = momentum_host(index);
            const double p_abs =
                stored_momentum >= 0.0 ? stored_momentum : -stored_momentum;
            if (particle_status == ParticleStatus::Active) {
                const double particle_gamma =
                    particles.properties.gamma_from_momentum_magnitude(p_abs);
                const double particle_speed =
                    particles.properties.speed_from_momentum_magnitude(p_abs);
                gamma_sum += particle_gamma;
                speed_sum += particle_speed;
                statistics.max_gamma = std::max(statistics.max_gamma, particle_gamma);
                statistics.max_speed = std::max(statistics.max_speed, particle_speed);

                const double kinetic_energy =
                    particles.properties.kinetic_energy_from_momentum_magnitude(p_abs);
                const double kinetic_energy_ev =
                    particles.properties.energy_to_electron_volt(kinetic_energy);
                active_min_energy_ev =
                    std::min(active_min_energy_ev, kinetic_energy_ev);
                statistics.active_max_kinetic_energy_ev =
                    std::max(statistics.active_max_kinetic_energy_ev,
                             kinetic_energy_ev);

                const double particle_weight = weight_host(index);
                if (std::isfinite(particle_weight) && particle_weight > 0.0) {
                    statistics.active_weight_sum += particle_weight;
                    active_weighted_energy_sum +=
                        particle_weight * kinetic_energy_ev;
                }
            }
        }

        if (statistics.active_count > 0) {
            const double inverse_active_count =
                1.0 / static_cast<double>(statistics.active_count);
            statistics.mean_gamma = gamma_sum * inverse_active_count;
            statistics.mean_speed = speed_sum * inverse_active_count;
            statistics.active_min_kinetic_energy_ev = active_min_energy_ev;
        }
        if (statistics.active_weight_sum > 0.0) {
            statistics.active_mean_kinetic_energy_ev =
                active_weighted_energy_sum / statistics.active_weight_sum;
        }
        return statistics;
#endif
    }

    /**
     * Print a compact statistics report for debug runs.
     */
    static void print_statistics(const particle_system_type& particles,
                                 std::ostream& output) {
#ifdef NDEBUG
        (void)particles;
        (void)output;
        throw std::runtime_error(
            "ParticleSystemDebugger::print_statistics is available only in debug builds.");
#else
        const ParticleStatistics statistics = compute_statistics(particles);
        const std::streamsize previous_precision = output.precision();
        const std::ios::fmtflags previous_flags = output.flags();
        output << std::setprecision(17);
        output << "Particle statistics\n";
        output << "  total_count: " << statistics.total_count << '\n';
        output << "  active_count: " << statistics.active_count << '\n';
        output << "  escaped_count: " << statistics.escaped_count << '\n';
        output << "  inactive_count: " << statistics.inactive_count << '\n';
        output << "  active_weight_sum: " << statistics.active_weight_sum << '\n';
        output << "  mean_gamma: " << statistics.mean_gamma << '\n';
        output << "  max_gamma: " << statistics.max_gamma << '\n';
        output << "  mean_speed: " << statistics.mean_speed << '\n';
        output << "  max_speed: " << statistics.max_speed << '\n';
        output << "  active_mean_kinetic_energy_ev: "
               << statistics.active_mean_kinetic_energy_ev << '\n';
        output << "  active_min_kinetic_energy_ev: "
               << statistics.active_min_kinetic_energy_ev << '\n';
        output << "  active_max_kinetic_energy_ev: "
               << statistics.active_max_kinetic_energy_ev << '\n';
        output.flags(previous_flags);
        output.precision(previous_precision);
#endif
    }
};
