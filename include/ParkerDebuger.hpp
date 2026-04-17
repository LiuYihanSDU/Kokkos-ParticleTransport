#pragma once

#include <iomanip>
#include <iostream>
#include <ostream>

#include "ParkerSolver.hpp"
#include "UnitConverter.hpp"

/**
 * Host-side debug helper for ParkerSolver.
 */
template <typename GridType, int VecDim, typename DeviceType>
struct ParkerSolverDebugger {
    using solver_type = ParkerSolver<GridType, VecDim, DeviceType>;
    using size_type = typename solver_type::size_type;
    using debug_terms_type = typename solver_type::debug_terms_type;

    /**
     * Print basic solver grid metadata.
     */
    static void inspect_solver(const solver_type& solver,
                               std::ostream& output = std::cout) {
        output << "[Debug] Inspecting Solver Grid: ";
        for (int dim = 0; dim < GridType::space_dim; ++dim) {
            output << solver.grid.extent(dim);
            if (dim + 1 < GridType::space_dim) {
                output << ", ";
            }
        }
        output << '\n';
    }

    /**
     * Print deterministic Parker transport terms for one particle.
     */
    static void print_particle_transport_terms(
        const solver_type& solver,
        const size_type particle_index,
        const Units::UnitSystem& unit_system,
        std::ostream& output = std::cout) {
        unit_system.validate();
        const debug_terms_type terms = solver.debug_terms_for_particle(particle_index);

        const std::streamsize previous_precision = output.precision();
        const std::ios::fmtflags previous_flags = output.flags();
        output << std::setprecision(17);

        output << "[Debug] Parker particle transport terms\n";
        output << "[Debug] particle_index = " << particle_index
               << ", particle_id = " << terms.particle_id << '\n';
        print_coordinate_line(output, "x", terms.position,
                              unit_system.scale(Units::Dimensions::length),
                              "code_length", "cm");
        print_scalar_line(output, "p", terms.momentum_magnitude,
                          unit_system.scale(Units::Dimensions::momentum),
                          "code_momentum", "g cm s^-1");
        print_scalar_line(output, "gamma", terms.gamma, 1.0,
                          "dimensionless", "dimensionless");
        print_scalar_line(output, "v", terms.speed,
                          unit_system.scale(Units::Dimensions::velocity),
                          "code_velocity", "cm s^-1");
        print_vector_line(output, "B", terms.magnetic_field,
                          unit_system.scale(Units::Dimensions::magnetic_field),
                          "code_magnetic_field", "G");
        print_scalar_line(output, "|B|", terms.magnetic_field_magnitude,
                          unit_system.scale(Units::Dimensions::magnetic_field),
                          "code_magnetic_field", "G");
        print_vector_line(output, "b", terms.magnetic_direction, 1.0,
                          "dimensionless", "dimensionless");
        output << "[Debug] dX/dt = V_sw + V_d + nabla_dot_kappa\n";
        print_vector_line(output, "V_sw", terms.solar_wind_velocity,
                          unit_system.scale(Units::Dimensions::velocity),
                          "code_velocity", "cm s^-1");
        print_vector_line(output, "V_d", terms.drift_velocity,
                          unit_system.scale(Units::Dimensions::velocity),
                          "code_velocity", "cm s^-1");
        print_vector_line(output, "nabla_dot_kappa", terms.diffusion_advection,
                          unit_system.scale(Units::Dimensions::velocity),
                          "code_velocity", "cm s^-1");
        print_vector_line(output, "V_sw + V_d + nabla_dot_kappa",
                          terms.total_advection,
                          unit_system.scale(Units::Dimensions::velocity),
                          "code_velocity", "cm s^-1");
        print_coordinate_line(output, "dq/dt", terms.coordinate_rate,
                              1.0 / unit_system.scale(Units::Dimensions::time),
                              "code_coordinate/code_time", "coordinate s^-1");
        output << "[Debug] dp/dt = -(p / 3) div(V_sw)\n";
        print_scalar_line(output, "div(V_sw)", terms.solar_wind_divergence,
                          1.0 / unit_system.scale(Units::Dimensions::time),
                          "code_time^-1", "s^-1");
        print_scalar_line(output, "-(p / 3) div(V_sw)", terms.momentum_rate,
                          unit_system.scale(Units::Dimensions::momentum) /
                              unit_system.scale(Units::Dimensions::time),
                          "code_momentum/code_time", "g cm s^-2");
        print_scalar_line(output, "kappa_parallel", terms.kappa_parallel,
                          unit_system.scale(Units::Dimensions::diffusion_coefficient),
                          "code_length^2/code_time", "cm^2 s^-1");
        print_scalar_line(output, "kappa_perpendicular", terms.kappa_perpendicular,
                          unit_system.scale(Units::Dimensions::diffusion_coefficient),
                          "code_length^2/code_time", "cm^2 s^-1");
        print_coordinate_line(output, "kappa_effective", terms.kappa_effective,
                              unit_system.scale(
                                  Units::Dimensions::diffusion_coefficient),
                              "code_length^2/code_time", "cm^2 s^-1");
        print_scalar_line(output, "diffusion_dt_bound", terms.diffusion_time_step,
                          unit_system.scale(Units::Dimensions::time),
                          "code_time", "s");
        print_scalar_line(output, "advection_dt_bound", terms.advection_time_step,
                          unit_system.scale(Units::Dimensions::time),
                          "code_time", "s");
        print_scalar_line(output, "momentum_dt_bound", terms.momentum_time_step,
                          unit_system.scale(Units::Dimensions::time),
                          "code_time", "s");
        print_scalar_line(output, "local_dt_bound", terms.stable_time_step,
                          unit_system.scale(Units::Dimensions::time),
                          "code_time", "s");
        print_scalar_line(output, "courant_scaled_local_dt",
                          terms.stable_time_step * solver.courant_number(),
                          unit_system.scale(Units::Dimensions::time),
                          "code_time", "s");
        print_scalar_line(output, "solver_dt", solver.time_step(),
                          unit_system.scale(Units::Dimensions::time),
                          "code_time", "s");

        output.flags(previous_flags);
        output.precision(previous_precision);
    }

    /**
     * Compatibility name for equation-term diagnostics.
     */
    static void print_particle_equation_terms(
        const solver_type& solver,
        const size_type particle_index,
        const Units::UnitSystem& unit_system,
        std::ostream& output = std::cout) {
        print_particle_transport_terms(solver, particle_index, unit_system, output);
    }

private:
    /**
     * Print a scalar in code and dimensional units.
     */
    static void print_scalar_line(std::ostream& output,
                                  const char* name,
                                  const double code_value,
                                  const double cgs_scale,
                                  const char* code_unit,
                                  const char* cgs_unit) {
        output << "[Debug] " << name << " = " << code_value << " ["
               << code_unit << "] | " << code_value * cgs_scale << " ["
               << cgs_unit << "]\n";
    }

    /**
     * Print a VecDim vector in code and dimensional units.
     */
    template <typename ArrayType>
    static void print_vector_line(std::ostream& output,
                                  const char* name,
                                  const ArrayType& code_values,
                                  const double cgs_scale,
                                  const char* code_unit,
                                  const char* cgs_unit) {
        output << "[Debug] " << name << " = ";
        print_array(output, code_values, VecDim, 1.0);
        output << " [" << code_unit << "] | ";
        print_array(output, code_values, VecDim, cgs_scale);
        output << " [" << cgs_unit << "]\n";
    }

    /**
     * Print a SpaceDim coordinate-like array in code and dimensional units.
     */
    template <typename ArrayType>
    static void print_coordinate_line(std::ostream& output,
                                      const char* name,
                                      const ArrayType& code_values,
                                      const double cgs_scale,
                                      const char* code_unit,
                                      const char* cgs_unit) {
        output << "[Debug] " << name << " = ";
        print_array(output, code_values, GridType::space_dim, 1.0);
        output << " [" << code_unit << "] | ";
        print_array(output, code_values, GridType::space_dim, cgs_scale);
        output << " [" << cgs_unit << "]\n";
    }

    /**
     * Print a fixed-size array prefix with a scalar multiplier.
     */
    template <typename ArrayType>
    static void print_array(std::ostream& output,
                            const ArrayType& values,
                            const int count,
                            const double scale) {
        output << '(';
        for (int dim = 0; dim < count; ++dim) {
            output << values[dim] * scale;
            if (dim + 1 < count) {
                output << ", ";
            }
        }
        output << ')';
    }
};
