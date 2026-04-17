#pragma once

#include "UnitConverter.hpp"

/**
 * CGS physical constants and their symbolic metadata.
 */
namespace PhysicalConstants {

inline constexpr double pi_value = 3.141592653589793238462643383279502884;
inline constexpr double speed_of_light_cgs = 2.99792458e10;
inline constexpr double elementary_charge_cgs = 4.803204712570263e-10;
inline constexpr double boltzmann_constant_cgs = 1.380649e-16;
inline constexpr double planck_constant_cgs = 6.62607015e-27;
inline constexpr double electron_volt_erg = 1.602176634e-12;
inline constexpr double proton_mass_cgs = 1.67262192369e-24;
inline constexpr double electron_mass_cgs = 9.1093837015e-28;
inline constexpr double gaussian_vacuum_permeability_factor_cgs = 1.0;
inline constexpr double astronomical_unit_cgs = 1.495978707e13;
inline constexpr double solar_radius_cgs = 6.957e10;
inline constexpr double day_cgs = 86400.0;
inline constexpr double julian_year_cgs = 31557600.0;

inline constexpr Units::Velocity speed_of_light{speed_of_light_cgs};
inline constexpr Units::Charge elementary_charge{elementary_charge_cgs};
inline constexpr Units::DimensionalScalar<2, 1, -2, 0, -1> boltzmann_constant{
    boltzmann_constant_cgs
};
inline constexpr Units::DimensionalScalar<2, 1, -1, 0, 0> planck_constant{
    planck_constant_cgs
};
inline constexpr Units::Energy electron_volt{electron_volt_erg};
inline constexpr Units::Mass proton_mass{proton_mass_cgs};
inline constexpr Units::Mass electron_mass{electron_mass_cgs};
inline constexpr Units::DimensionlessScalar gaussian_vacuum_permeability_factor{
    gaussian_vacuum_permeability_factor_cgs
};
inline constexpr Units::Length astronomical_unit{astronomical_unit_cgs};
inline constexpr Units::Length solar_radius{solar_radius_cgs};
inline constexpr Units::Time day{day_cgs};
inline constexpr Units::Time julian_year{julian_year_cgs};

/**
 * Physical constant identifiers used for diagnostics and configuration.
 */
enum class ConstantSymbol : int {
    Pi = 0,
    SpeedOfLight = 1,
    ElementaryCharge = 2,
    BoltzmannConstant = 3,
    PlanckConstant = 4,
    ElectronVolt = 5,
    ProtonMass = 6,
    ElectronMass = 7,
    GaussianVacuumPermeabilityFactor = 8,
    AstronomicalUnit = 9,
    SolarRadius = 10,
    Day = 11,
    JulianYear = 12
};

/**
 * Host-side physical constant metadata for diagnostics and configuration.
 */
struct ConstantInfo {
    ConstantSymbol id{ConstantSymbol::Pi};
    const char* ascii_symbol{"pi"};
    const char* latex_symbol{"\\pi"};
    const char* name{"pi"};
    Units::UnitDimension dimension{Units::Dimensions::dimensionless};
    double value_cgs{pi_value};
};

/**
 * Return physical constant metadata by symbolic identifier.
 */
constexpr ConstantInfo constant_info(const ConstantSymbol symbol) {
    switch (symbol) {
        case ConstantSymbol::Pi:
            return ConstantInfo{symbol, "pi", "\\pi", "pi",
                                Units::Dimensions::dimensionless, pi_value};
        case ConstantSymbol::SpeedOfLight:
            return ConstantInfo{symbol, "c", "c", "speed of light",
                                Units::Dimensions::velocity, speed_of_light_cgs};
        case ConstantSymbol::ElementaryCharge:
            return ConstantInfo{symbol, "e", "e", "elementary charge",
                                Units::Dimensions::charge, elementary_charge_cgs};
        case ConstantSymbol::BoltzmannConstant:
            return ConstantInfo{symbol, "k_B", "k_B", "Boltzmann constant",
                                Units::UnitDimension{2, 1, -2, 0, -1},
                                boltzmann_constant_cgs};
        case ConstantSymbol::PlanckConstant:
            return ConstantInfo{symbol, "h", "h", "Planck constant",
                                Units::UnitDimension{2, 1, -1, 0, 0},
                                planck_constant_cgs};
        case ConstantSymbol::ElectronVolt:
            return ConstantInfo{symbol, "eV", "\\mathrm{eV}", "electron volt",
                                Units::Dimensions::energy, electron_volt_erg};
        case ConstantSymbol::ProtonMass:
            return ConstantInfo{symbol, "m_p", "m_p", "proton mass",
                                Units::Dimensions::mass, proton_mass_cgs};
        case ConstantSymbol::ElectronMass:
            return ConstantInfo{symbol, "m_e", "m_e", "electron mass",
                                Units::Dimensions::mass, electron_mass_cgs};
        case ConstantSymbol::GaussianVacuumPermeabilityFactor:
            return ConstantInfo{symbol, "mu_0_factor", "\\mu_0^{\\mathrm{CGS}}",
                                "Gaussian vacuum permeability factor",
                                Units::Dimensions::dimensionless,
                                gaussian_vacuum_permeability_factor_cgs};
        case ConstantSymbol::AstronomicalUnit:
            return ConstantInfo{symbol, "au", "\\mathrm{au}", "astronomical unit",
                                Units::Dimensions::length, astronomical_unit_cgs};
        case ConstantSymbol::SolarRadius:
            return ConstantInfo{symbol, "R_sun", "R_\\odot", "nominal solar radius",
                                Units::Dimensions::length, solar_radius_cgs};
        case ConstantSymbol::Day:
            return ConstantInfo{symbol, "day", "\\mathrm{day}", "day",
                                Units::Dimensions::time, day_cgs};
        case ConstantSymbol::JulianYear:
            return ConstantInfo{symbol, "yr", "\\mathrm{yr}", "Julian year",
                                Units::Dimensions::time, julian_year_cgs};
    }
    return ConstantInfo{};
}

/**
 * Build a common heliospheric unit system.
 *
 * Defaults are length = 1 au, time = 1 day, mass = proton mass, and
 * charge = elementary charge.
 */
KOKKOS_INLINE_FUNCTION
constexpr Units::UnitSystem heliospheric_unit_system(
    const Units::Length length_scale = astronomical_unit,
    const Units::Time time_scale = day,
    const Units::Mass mass_scale = proton_mass,
    const Units::Charge charge_scale = elementary_charge,
    const Units::Temperature temperature_scale = Units::Temperature(1.0)) {
    return Units::make_unit_system(length_scale, mass_scale, time_scale, charge_scale,
                                   temperature_scale);
}

} // namespace PhysicalConstants
