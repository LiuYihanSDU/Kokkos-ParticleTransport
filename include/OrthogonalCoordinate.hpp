#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

/**
 * Supported orthogonal curvilinear coordinate systems.
 */
enum class OrthogonalCoordinateSystem : int {
    Cartesian = 0,
    Polar = 1,
    Spherical = 2
};

/**
 * Geometry model for orthogonal curvilinear coordinates.
 */
template <int SpaceDim, int VecDim = SpaceDim>
struct OrthogonalGeometry {
    using coordinate_array_type = Kokkos::Array<double, SpaceDim>;
    using metric_array_type = Kokkos::Array<double, VecDim>;

    static constexpr int space_dim = SpaceDim;
    static constexpr int vec_dim = VecDim;
    static constexpr double singularity_tolerance = 1.0e-14;

    OrthogonalCoordinateSystem coordinate_system{OrthogonalCoordinateSystem::Cartesian};

    OrthogonalGeometry() = default;

    KOKKOS_INLINE_FUNCTION
    explicit OrthogonalGeometry(const OrthogonalCoordinateSystem input_coordinate_system)
        : coordinate_system(input_coordinate_system) {}

    /**
     * Return the Lame coefficient h_i at a coordinate point.
     */
    KOKKOS_INLINE_FUNCTION
    double h(const int component, const coordinate_array_type& q) const {
        switch (coordinate_system) {
            case OrthogonalCoordinateSystem::Cartesian:
                return 1.0;
            case OrthogonalCoordinateSystem::Polar:
                return polar_h(component, q);
            case OrthogonalCoordinateSystem::Spherical:
                return spherical_h(component, q);
        }

        return 1.0;
    }

    /**
     * Return the inverse Lame coefficient 1 / h_i at a coordinate point.
     */
    KOKKOS_INLINE_FUNCTION
    double h_inv(const int component, const coordinate_array_type& q) const {
        return 1.0 / h(component, q);
    }

    /**
     * Return all Lame coefficients at a coordinate point.
     */
    KOKKOS_INLINE_FUNCTION
    metric_array_type h(const coordinate_array_type& q) const {
        metric_array_type result{};
        for (int component = 0; component < VecDim; ++component) {
            result[component] = h(component, q);
        }
        return result;
    }

    /**
     * Return all inverse Lame coefficients at a coordinate point.
     */
    KOKKOS_INLINE_FUNCTION
    metric_array_type h_inv(const coordinate_array_type& q) const {
        metric_array_type result{};
        for (int component = 0; component < VecDim; ++component) {
            result[component] = h_inv(component, q);
        }
        return result;
    }

    /**
     * Return the product of active-coordinate Lame coefficients at a coordinate point.
     */
    KOKKOS_INLINE_FUNCTION
    double jacobian(const coordinate_array_type& q) const {
        double result = 1.0;
        for (int component = 0; component < SpaceDim; ++component) {
            result *= h(component, q);
        }
        return result;
    }

    /**
     * Return the physical line element h_i dq_i along one coordinate direction.
     */
    KOKKOS_INLINE_FUNCTION
    double physical_distance(const int component,
                             const double coordinate_spacing,
                             const coordinate_array_type& q) const {
        return h(component, q) * coordinate_spacing;
    }

    /**
     * Return whether a coordinate lies on a curvilinear metric singularity.
     */
    KOKKOS_INLINE_FUNCTION
    bool is_singular(const coordinate_array_type& q) const {
        if (coordinate_system == OrthogonalCoordinateSystem::Polar ||
            coordinate_system == OrthogonalCoordinateSystem::Spherical) {
            if (q[0] <= singularity_tolerance) {
                return true;
            }
        }

        if constexpr (SpaceDim > 1) {
            if (coordinate_system == OrthogonalCoordinateSystem::Spherical) {
                const double sin_theta = Kokkos::sin(q[1]);
                if (sin_theta >= -singularity_tolerance &&
                    sin_theta <= singularity_tolerance) {
                    return true;
                }
            }
        }

        return false;
    }

private:
    /**
     * Return the polar Lame coefficient for q = (r, phi[, z]).
     */
    KOKKOS_INLINE_FUNCTION
    double polar_h(const int component, const coordinate_array_type& q) const {
        return component == 1 ? q[0] : 1.0;
    }

    /**
     * Return the spherical Lame coefficient for q = (r, theta, phi).
     */
    KOKKOS_INLINE_FUNCTION
    double spherical_h(const int component, const coordinate_array_type& q) const {
        if (component == 1) {
            return q[0];
        }
        if (component == 2) {
            if constexpr (SpaceDim > 1) {
                return q[0] * Kokkos::sin(q[1]);
            }
            return 1.0;
        }
        return 1.0;
    }
};

/**
 * Backward-compatible metric alias for coordinate-space-only geometry.
 */
template <int SpaceDim>
using OrthogonalMetric = OrthogonalGeometry<SpaceDim, SpaceDim>;
