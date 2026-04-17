#pragma once

#include <cstdint>

#include <Kokkos_Random.hpp>

#include "KokkosDevice.hpp"

/**
 * Repository-standard random-number manager backed by Kokkos XorShift64 pools.
 */
template <typename DeviceType = Device>
struct RandomManager {
    using device_type = DeviceType;
    using execution_space = typename DeviceTraits<DeviceType>::execution_space;
    using memory_space = typename DeviceTraits<DeviceType>::memory_space;
    using random_pool_type = Kokkos::Random_XorShift64_Pool<DeviceType>;
    using generator_type = typename random_pool_type::generator_type;

    static constexpr std::uint64_t default_seed = 114514u;

    random_pool_type pool;
    std::uint64_t seed{default_seed};

    /**
     * Build a random manager from a deterministic seed.
     */
    explicit RandomManager(const std::uint64_t input_seed = default_seed)
        : pool(input_seed), seed(input_seed) {}

    /**
     * Return a thread-local generator state inside a Kokkos kernel.
     */
    KOKKOS_INLINE_FUNCTION
    generator_type get_state() const {
        return pool.get_state();
    }

    /**
     * Compatibility alias for get_state().
     */
    KOKKOS_INLINE_FUNCTION
    generator_type get_generator() const {
        return get_state();
    }

    /**
     * Return a generator state to the random pool inside a Kokkos kernel.
     */
    KOKKOS_INLINE_FUNCTION
    void free_state(generator_type& generator) const {
        pool.free_state(generator);
    }

    /**
     * Compatibility alias for free_state().
     */
    KOKKOS_INLINE_FUNCTION
    void free_generator(generator_type& generator) const {
        free_state(generator);
    }
};

/**
 * Device-callable distribution samplers built on RandomManager generator states.
 */
template <typename DeviceType = Device>
struct StochasticSampler {
    using generator_type = typename RandomManager<DeviceType>::generator_type;

    /**
     * Draw a uniform random number in [0, 1).
     */
    KOKKOS_INLINE_FUNCTION
    static double uniform01(generator_type& generator) {
        return generator.drand();
    }

    /**
     * Draw a uniform random number in [lower, upper).
     */
    KOKKOS_INLINE_FUNCTION
    static double uniform(generator_type& generator,
                          const double lower,
                          const double upper) {
        return generator.drand(lower, upper);
    }

    /**
     * Draw a standard normal random number.
     */
    KOKKOS_INLINE_FUNCTION
    static double normal(generator_type& generator) {
        return generator.normal(0.0, 1.0);
    }

    /**
     * Draw a normal random number with explicit mean and standard deviation.
     */
    KOKKOS_INLINE_FUNCTION
    static double normal(generator_type& generator,
                         const double mean,
                         const double standard_deviation) {
        return generator.normal(mean, standard_deviation);
    }

    /**
     * Draw a U[-sqrt(3), sqrt(3)] random number with unit variance.
     */
    KOKKOS_INLINE_FUNCTION
    static double uniform_sqrt3(generator_type& generator) {
        constexpr double sqrt3 = 1.7320508075688772935;
        return generator.drand(-sqrt3, sqrt3);
    }

    /**
     * Draw a random sign with equal probability.
     */
    KOKKOS_INLINE_FUNCTION
    static double random_sign(generator_type& generator) {
        return generator.drand() < 0.5 ? -1.0 : 1.0;
    }
};

