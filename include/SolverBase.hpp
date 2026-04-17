#pragma once

#include "Field.hpp"
#include "Grid.hpp"
#include "ParticleSystem.hpp"
#include "UnitConverter.hpp"

template <
    typename GridType,
    int VecDim = GridType::vec_dim,
    typename DeviceType = typename GridType::device_type
>
struct SolverBase {
    using grid_type = GridType;
    using device_type = DeviceType;
    using execution_space = typename grid_type::execution_space;
    using memory_space = typename grid_type::memory_space;
    using layout_type = typename grid_type::layout_type;
    using vector_field_type = VectorField<grid_type, VecDim, layout_type>;
    using particle_system_type =
        ParticleSystem<grid_type::space_dim, device_type, layout_type>;
    using dimensionless_value_type = Units::DimensionlessValue;

    static constexpr int space_dim = grid_type::space_dim;
    static constexpr int vec_dim = VecDim;
    static_assert(VecDim > 0, "SolverBase VecDim must be positive.");

    particle_system_type particles;
    grid_type grid;
    vector_field_type B_vec, V_body;
    dimensionless_value_type dT{1.0};
    dimensionless_value_type courant_scale{0.3};
    dimensionless_value_type Omega0{0.0};

    explicit SolverBase(grid_type input_grid,
                        vector_field_type input_B_vec,
                        vector_field_type input_V_body)
        : grid(input_grid), B_vec(input_B_vec), V_body(input_V_body) {}

    /**
     * Build solver state from an explicitly configured particle system.
     */
    SolverBase(particle_system_type input_particles,
               grid_type input_grid,
               vector_field_type input_B_vec,
               vector_field_type input_V_body)
        : particles(input_particles),
          grid(input_grid),
          B_vec(input_B_vec),
          V_body(input_V_body) {}

    /**
     * Set the already nondimensionalized solver time step.
     */
    void set_dimensionless_time_step(const dimensionless_value_type input_time_step) {
        dT = input_time_step;
    }

    /**
     * Set the already nondimensionalized Courant scale.
     */
    void set_dimensionless_courant_scale(const dimensionless_value_type input_courant_scale) {
        courant_scale = input_courant_scale;
    }

    /**
     * Set the already nondimensionalized rotation rate.
     */
    void set_dimensionless_rotation_rate(const dimensionless_value_type input_rotation_rate) {
        Omega0 = input_rotation_rate;
    }

    /**
     * Return the dimensionless solver time step as a raw scalar for kernels.
     */
    KOKKOS_INLINE_FUNCTION
    double time_step() const {
        return dT.raw();
    }

    /**
     * Return the dimensionless Courant scale as a raw scalar for kernels.
     */
    KOKKOS_INLINE_FUNCTION
    double courant_number() const {
        return courant_scale.raw();
    }

    /**
     * Return the dimensionless rotation rate as a raw scalar for kernels.
     */
    KOKKOS_INLINE_FUNCTION
    double rotation_rate() const {
        return Omega0.raw();
    }
};
