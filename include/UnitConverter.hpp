#pragma once

#include <cmath>
#include <stdexcept>

#include "KokkosDevice.hpp"

/**
 * Unit dimensions, dimensional quantities, and nondimensionalization helpers.
 */
namespace Units {

/**
 * Integer exponents of CGS base dimensions used by this codebase.
 */
struct UnitDimension {
    int length_power{0};
    int mass_power{0};
    int time_power{0};
    int charge_power{0};
    int temperature_power{0};

    KOKKOS_INLINE_FUNCTION
    constexpr bool operator==(const UnitDimension& other) const {
        return length_power == other.length_power &&
               mass_power == other.mass_power &&
               time_power == other.time_power &&
               charge_power == other.charge_power &&
               temperature_power == other.temperature_power;
    }

    KOKKOS_INLINE_FUNCTION
    constexpr bool operator!=(const UnitDimension& other) const {
        return !(*this == other);
    }
};

/**
 * Common unit dimensions used by transport solvers and coefficient models.
 */
namespace Dimensions {
inline constexpr UnitDimension dimensionless{0, 0, 0, 0, 0};
inline constexpr UnitDimension length{1, 0, 0, 0, 0};
inline constexpr UnitDimension mass{0, 1, 0, 0, 0};
inline constexpr UnitDimension time{0, 0, 1, 0, 0};
inline constexpr UnitDimension charge{0, 0, 0, 1, 0};
inline constexpr UnitDimension temperature{0, 0, 0, 0, 1};
inline constexpr UnitDimension velocity{1, 0, -1, 0, 0};
inline constexpr UnitDimension acceleration{1, 0, -2, 0, 0};
inline constexpr UnitDimension angular_frequency{0, 0, -1, 0, 0};
inline constexpr UnitDimension diffusion_coefficient{2, 0, -1, 0, 0};
inline constexpr UnitDimension number_density{-3, 0, 0, 0, 0};
inline constexpr UnitDimension momentum{1, 1, -1, 0, 0};
inline constexpr UnitDimension rigidity{2, 1, -2, -1, 0};
inline constexpr UnitDimension energy{2, 1, -2, 0, 0};
inline constexpr UnitDimension magnetic_field{1, 1, -2, -1, 0};
inline constexpr UnitDimension electric_field{1, 1, -2, -1, 0};
inline constexpr UnitDimension electric_potential{2, 1, -2, -1, 0};
} // namespace Dimensions

/**
 * Raise a positive scale by an integer exponent.
 */
KOKKOS_INLINE_FUNCTION
constexpr double integer_power(double base, int exponent) {
    double result = 1.0;
    int power = exponent >= 0 ? exponent : -exponent;
    while (power > 0) {
        if ((power & 1) != 0) {
            result *= base;
        }
        base *= base;
        power >>= 1;
    }
    return exponent >= 0 ? result : 1.0 / result;
}

/**
 * Base-unit scale system. Values are CGS units per one code unit.
 */
struct UnitSystem {
    double length_cgs{1.0};
    double mass_cgs{1.0};
    double time_cgs{1.0};
    double charge_cgs{1.0};
    double temperature_cgs{1.0};

    /**
     * Return a strict CGS unit system.
     */
    KOKKOS_INLINE_FUNCTION
    static constexpr UnitSystem cgs() {
        return UnitSystem{1.0, 1.0, 1.0, 1.0, 1.0};
    }

    /**
     * Compatibility alias for a strict CGS unit system.
     */
    KOKKOS_INLINE_FUNCTION
    static constexpr UnitSystem cgs_gaussian() {
        return cgs();
    }

    /**
     * Return the CGS scale for a derived dimension.
     */
    KOKKOS_INLINE_FUNCTION
    double scale(const UnitDimension& dimension) const {
        return integer_power(length_cgs, dimension.length_power) *
               integer_power(mass_cgs, dimension.mass_power) *
               integer_power(time_cgs, dimension.time_power) *
               integer_power(charge_cgs, dimension.charge_power) *
               integer_power(temperature_cgs, dimension.temperature_power);
    }

    /**
     * Validate that all base scales are positive and finite.
     */
    void validate() const {
        if (!std::isfinite(length_cgs) || length_cgs <= 0.0 ||
            !std::isfinite(mass_cgs) || mass_cgs <= 0.0 ||
            !std::isfinite(time_cgs) || time_cgs <= 0.0 ||
            !std::isfinite(charge_cgs) || charge_cgs <= 0.0 ||
            !std::isfinite(temperature_cgs) || temperature_cgs <= 0.0) {
            throw std::runtime_error(
                "UnitSystem: base scales must be positive finite CGS values.");
        }
    }
};

/**
 * A host/device-friendly dimensionless scalar wrapper.
 */
struct DimensionlessValue {
    double value{0.0};

    KOKKOS_INLINE_FUNCTION
    constexpr DimensionlessValue() = default;

    KOKKOS_INLINE_FUNCTION
    explicit constexpr DimensionlessValue(const double input_value)
        : value(input_value) {}

    KOKKOS_INLINE_FUNCTION
    constexpr double raw() const {
        return value;
    }

    KOKKOS_INLINE_FUNCTION
    explicit constexpr operator double() const {
        return value;
    }
};

/**
 * A scalar value stored in CGS units with compile-time dimensional exponents.
 */
template <
    int LengthPower,
    int MassPower,
    int TimePower,
    int ChargePower,
    int TemperaturePower = 0
>
struct DimensionalScalar {
    double value_cgs{0.0};

    KOKKOS_INLINE_FUNCTION
    constexpr DimensionalScalar() = default;

    KOKKOS_INLINE_FUNCTION
    explicit constexpr DimensionalScalar(const double input_value_cgs)
        : value_cgs(input_value_cgs) {}

    KOKKOS_INLINE_FUNCTION
    static constexpr UnitDimension dimension() {
        return UnitDimension{LengthPower, MassPower, TimePower, ChargePower,
                             TemperaturePower};
    }
};

using DimensionlessScalar = DimensionalScalar<0, 0, 0, 0, 0>;
using Length = DimensionalScalar<1, 0, 0, 0, 0>;
using Mass = DimensionalScalar<0, 1, 0, 0, 0>;
using Time = DimensionalScalar<0, 0, 1, 0, 0>;
using Charge = DimensionalScalar<0, 0, 0, 1, 0>;
using Temperature = DimensionalScalar<0, 0, 0, 0, 1>;
using Velocity = DimensionalScalar<1, 0, -1, 0, 0>;
using Acceleration = DimensionalScalar<1, 0, -2, 0, 0>;
using AngularFrequency = DimensionalScalar<0, 0, -1, 0, 0>;
using DiffusionCoefficient = DimensionalScalar<2, 0, -1, 0, 0>;
using NumberDensity = DimensionalScalar<-3, 0, 0, 0, 0>;
using Momentum = DimensionalScalar<1, 1, -1, 0, 0>;
using Rigidity = DimensionalScalar<2, 1, -2, -1, 0>;
using Energy = DimensionalScalar<2, 1, -2, 0, 0>;
using MagneticField = DimensionalScalar<1, 1, -2, -1, 0>;
using ElectricField = DimensionalScalar<1, 1, -2, -1, 0>;
using ElectricPotential = DimensionalScalar<2, 1, -2, -1, 0>;

/**
 * Convert CGS dimensional quantities to and from code-dimensionless quantities.
 */
struct Nondimensionalizer {
    UnitSystem units{};

    /**
     * Build a nondimensionalizer from a validated unit system.
     */
    explicit Nondimensionalizer(const UnitSystem& input_units)
        : units(input_units) {
        units.validate();
    }

    /**
     * Convert a raw CGS value with a runtime dimension to a dimensionless code value.
     */
    KOKKOS_INLINE_FUNCTION
    DimensionlessValue to_dimensionless(const double value_cgs,
                                        const UnitDimension& dimension) const {
        return DimensionlessValue(value_cgs / units.scale(dimension));
    }

    /**
     * Convert a dimensionless code value with a runtime dimension back to CGS.
     */
    KOKKOS_INLINE_FUNCTION
    double to_cgs(const DimensionlessValue value,
                  const UnitDimension& dimension) const {
        return value.value * units.scale(dimension);
    }

    /**
     * Convert a compile-time dimensional scalar to a dimensionless code value.
     */
    template <int L, int M, int T, int Q, int Temp>
    KOKKOS_INLINE_FUNCTION
    DimensionlessValue to_dimensionless(
        const DimensionalScalar<L, M, T, Q, Temp> quantity) const {
        return to_dimensionless(quantity.value_cgs,
                                DimensionalScalar<L, M, T, Q, Temp>::dimension());
    }

    /**
     * Convert a dimensionless code value back to a compile-time dimensional CGS scalar.
     */
    template <int L, int M, int T, int Q, int Temp>
    KOKKOS_INLINE_FUNCTION
    DimensionalScalar<L, M, T, Q, Temp> to_cgs(
        const DimensionlessValue value) const {
        return DimensionalScalar<L, M, T, Q, Temp>(
            to_cgs(value, DimensionalScalar<L, M, T, Q, Temp>::dimension()));
    }

    /**
     * Convert a dimensionless value in another unit system to this code unit system.
     */
    KOKKOS_INLINE_FUNCTION
    DimensionlessValue from_dimensionless_in_system(
        const DimensionlessValue source_value,
        const UnitSystem& source_units,
        const UnitDimension& dimension) const {
        return DimensionlessValue(source_value.value *
                                  source_units.scale(dimension) /
                                  units.scale(dimension));
    }

    /**
     * Convert a dimensionless value in this code unit system to another unit system.
     */
    KOKKOS_INLINE_FUNCTION
    DimensionlessValue to_dimensionless_in_system(
        const DimensionlessValue value,
        const UnitSystem& target_units,
        const UnitDimension& dimension) const {
        return DimensionlessValue(value.value * units.scale(dimension) /
                                  target_units.scale(dimension));
    }

    /**
     * Convert all components of a field from CGS units to dimensionless code units.
     */
    template <typename FieldType>
    void nondimensionalize_field_in_place(FieldType& field,
                                          const UnitDimension& dimension) const {
        const double inverse_scale = 1.0 / units.scale(dimension);
        auto values = field.data;
        const auto point_count = field.point_count();
        Kokkos::parallel_for(
            "Units::nondimensionalize_field_in_place",
            Kokkos::RangePolicy<typename FieldType::execution_space>(0, point_count),
            KOKKOS_LAMBDA(const typename FieldType::size_type point_index) {
                for (int component = 0; component < FieldType::component_dim; ++component) {
                    values(point_index, component) *= inverse_scale;
                }
            });
        Kokkos::fence("Units::nondimensionalize_field_in_place");
    }

    /**
     * Convert all components of a field from dimensionless code units back to CGS units.
     */
    template <typename FieldType>
    void dimensionalize_field_in_place(FieldType& field,
                                       const UnitDimension& dimension) const {
        const double scale = units.scale(dimension);
        auto values = field.data;
        const auto point_count = field.point_count();
        Kokkos::parallel_for(
            "Units::dimensionalize_field_in_place",
            Kokkos::RangePolicy<typename FieldType::execution_space>(0, point_count),
            KOKKOS_LAMBDA(const typename FieldType::size_type point_index) {
                for (int component = 0; component < FieldType::component_dim; ++component) {
                    values(point_index, component) *= scale;
                }
            });
        Kokkos::fence("Units::dimensionalize_field_in_place");
    }

    /**
     * Convert a field from another code unit system into this code unit system.
     */
    template <typename FieldType>
    void convert_field_from_unit_system_in_place(FieldType& field,
                                                 const UnitSystem& source_units,
                                                 const UnitDimension& dimension) const {
        const double scale_factor = source_units.scale(dimension) / units.scale(dimension);
        auto values = field.data;
        const auto point_count = field.point_count();
        Kokkos::parallel_for(
            "Units::convert_field_from_unit_system_in_place",
            Kokkos::RangePolicy<typename FieldType::execution_space>(0, point_count),
            KOKKOS_LAMBDA(const typename FieldType::size_type point_index) {
                for (int component = 0; component < FieldType::component_dim; ++component) {
                    values(point_index, component) *= scale_factor;
                }
            });
        Kokkos::fence("Units::convert_field_from_unit_system_in_place");
    }

    /**
     * Convert a field from this code unit system into another code unit system.
     */
    template <typename FieldType>
    void convert_field_to_unit_system_in_place(FieldType& field,
                                               const UnitSystem& target_units,
                                               const UnitDimension& dimension) const {
        const double scale_factor = units.scale(dimension) / target_units.scale(dimension);
        auto values = field.data;
        const auto point_count = field.point_count();
        Kokkos::parallel_for(
            "Units::convert_field_to_unit_system_in_place",
            Kokkos::RangePolicy<typename FieldType::execution_space>(0, point_count),
            KOKKOS_LAMBDA(const typename FieldType::size_type point_index) {
                for (int component = 0; component < FieldType::component_dim; ++component) {
                    values(point_index, component) *= scale_factor;
                }
            });
        Kokkos::fence("Units::convert_field_to_unit_system_in_place");
    }
};

/**
 * Build a unit system from typed base scales.
 */
KOKKOS_INLINE_FUNCTION
constexpr UnitSystem make_unit_system(const Length length,
                                      const Mass mass,
                                      const Time time,
                                      const Charge charge,
                                      const Temperature temperature =
                                          Temperature(1.0)) {
    return UnitSystem{length.value_cgs, mass.value_cgs, time.value_cgs, charge.value_cgs,
                      temperature.value_cgs};
}

/**
 * Build a unit system by specifying the derived magnetic-field scale directly.
 *
 * This is useful for importing MHD code-unit fields whose magnetic scale is defined by
 * the source code convention rather than by an explicit charge scale.
 */
KOKKOS_INLINE_FUNCTION
constexpr UnitSystem make_unit_system_from_magnetic_field_scale(
    const Length length,
    const Mass mass,
    const Time time,
    const MagneticField magnetic_field,
    const Temperature temperature = Temperature(1.0)) {
    const double charge_cgs = length.value_cgs * mass.value_cgs /
                              (time.value_cgs * time.value_cgs *
                               magnetic_field.value_cgs);
    return UnitSystem{length.value_cgs, mass.value_cgs, time.value_cgs, charge_cgs,
                      temperature.value_cgs};
}

} // namespace Units
