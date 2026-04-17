#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Field.hpp"
#include "ParticleSystem.hpp"
#include "TurbulenceProperties.hpp"

/**
 * Legacy heliospheric diffusion-coefficient model from the project README.
 */
namespace LegencyModel {

inline constexpr double pi = 3.141592653589793238462643383279502884;

/**
 * Formula used for perpendicular diffusion.
 */
enum class PerpendicularDiffusionModelKind : int {
    ReadmeFormula = 0,
    ConstantParallelRatio = 1
};

/**
 * Numerical and safety parameters for the legacy diffusion model.
 */
struct DiffusionModelParameters {
    PerpendicularDiffusionModelKind perpendicular_model{
        PerpendicularDiffusionModelKind::ReadmeFormula
    };
    double parallel_prefactor{1.622};
    double perpendicular_prefactor{0.198};
    double perpendicular_parallel_ratio{0.0};
    double minimum_magnetic_field_magnitude{0.0};
    double minimum_solar_wind_speed{0.0};
    double minimum_sigma2{0.0};

    /**
     * Return a parameter bundle using kappa_perp = ratio * kappa_parallel.
     */
    static DiffusionModelParameters with_constant_perpendicular_parallel_ratio(
        const double ratio) {
        DiffusionModelParameters parameters;
        parameters.set_constant_perpendicular_parallel_ratio(ratio);
        return parameters;
    }

    /**
     * Return an updated parameter bundle using kappa_perp = ratio * kappa_parallel.
     */
    static DiffusionModelParameters with_constant_perpendicular_parallel_ratio(
        const double ratio,
        DiffusionModelParameters parameters) {
        parameters.set_constant_perpendicular_parallel_ratio(ratio);
        return parameters;
    }

    /**
     * Select kappa_perp = ratio * kappa_parallel for the perpendicular branch.
     */
    void set_constant_perpendicular_parallel_ratio(const double ratio) {
        perpendicular_model = PerpendicularDiffusionModelKind::ConstantParallelRatio;
        perpendicular_parallel_ratio = ratio;
        validate();
    }

    /**
     * Restore the README perpendicular formula branch.
     */
    void use_readme_perpendicular_formula() {
        perpendicular_model = PerpendicularDiffusionModelKind::ReadmeFormula;
        validate();
    }

    /**
     * Return true when kappa_perp is a constant fraction of kappa_parallel.
     */
    KOKKOS_INLINE_FUNCTION
    bool uses_constant_perpendicular_parallel_ratio() const {
        return perpendicular_model ==
               PerpendicularDiffusionModelKind::ConstantParallelRatio;
    }

    /**
     * Validate host-side model controls.
     */
    void validate() const {
        if (!std::isfinite(parallel_prefactor) || parallel_prefactor <= 0.0 ||
            !std::isfinite(perpendicular_prefactor) || perpendicular_prefactor <= 0.0 ||
            !std::isfinite(perpendicular_parallel_ratio) ||
            perpendicular_parallel_ratio < 0.0 ||
            !std::isfinite(minimum_magnetic_field_magnitude) ||
            minimum_magnetic_field_magnitude < 0.0 ||
            !std::isfinite(minimum_solar_wind_speed) ||
            minimum_solar_wind_speed < 0.0 ||
            !std::isfinite(minimum_sigma2) || minimum_sigma2 < 0.0) {
            throw std::runtime_error(
                "LegencyModel::DiffusionModelParameters: coefficients and floors "
                "must be finite positive or non-negative values.");
        }
        if (uses_constant_perpendicular_parallel_ratio() &&
            perpendicular_parallel_ratio <= 0.0) {
            throw std::runtime_error(
                "LegencyModel::DiffusionModelParameters: constant perpendicular "
                "ratio must be positive when selected.");
        }
    }
};

/**
 * Reference gyrofrequency split used to separate species, field strength, and gamma.
 */
struct GyrofrequencyReference {
    double magnetic_field_reference{1.0};
    double omega_reference{1.0};

    /**
     * Build a reference from dimensionless particle properties and B_ref.
     */
    static GyrofrequencyReference from_particle_properties(
        const ParticleProperties& properties,
        const double input_magnetic_field_reference) {
        properties.validate();
        if (!std::isfinite(input_magnetic_field_reference) ||
            input_magnetic_field_reference <= 0.0) {
            throw std::runtime_error(
                "LegencyModel::GyrofrequencyReference: B_ref must be positive.");
        }

        const double charge_magnitude =
            properties.charge >= 0.0 ? properties.charge : -properties.charge;
        const double omega_reference =
            charge_magnitude * input_magnetic_field_reference /
            (properties.rest_mass * properties.speed_of_light);
        if (!std::isfinite(omega_reference) || omega_reference <= 0.0) {
            throw std::runtime_error(
                "LegencyModel::GyrofrequencyReference: omega_ref must be positive.");
        }

        return GyrofrequencyReference{input_magnetic_field_reference,
                                      omega_reference};
    }

    /**
     * Return the nonrelativistic local gyrofrequency for a magnetic-field magnitude.
     */
    KOKKOS_INLINE_FUNCTION
    double local_nonrelativistic_omega(const double magnetic_field_magnitude) const {
        return omega_reference * magnetic_field_magnitude / magnetic_field_reference;
    }

    /**
     * Return the relativistic local gyrofrequency for a magnetic-field magnitude.
     */
    KOKKOS_INLINE_FUNCTION
    double local_relativistic_omega(const double magnetic_field_magnitude,
                                    const double gamma) const {
        if (gamma <= 0.0) {
            Kokkos::abort("LegencyModel: relativistic gyrofrequency requires gamma > 0.");
        }
        return local_nonrelativistic_omega(magnetic_field_magnitude) / gamma;
    }
};

/**
 * Diffusion coefficients before applying particle-gamma factors.
 */
struct GammaIndependentDiffusionCoefficients {
    double kappa_parallel_gamma_one{0.0};
    double kappa_perpendicular_gamma_one{0.0};
    double omega_nonrelativistic{0.0};
    double magnetic_field_magnitude{0.0};
    double solar_wind_speed{0.0};
    PerpendicularDiffusionModelKind perpendicular_model{
        PerpendicularDiffusionModelKind::ReadmeFormula
    };
};

/**
 * Particle-specific diffusion coefficients at one location.
 */
struct DiffusionCoefficients {
    double kappa_parallel{0.0};
    double kappa_perpendicular{0.0};
};

/**
 * Numerical controls for focused-transport pitch-angle scattering.
 */
struct PitchAngleScatteringParameters {
    double spectral_index{5.0 / 3.0};
    double resonance_regularization_h0{1.0e-6};
    double minimum_magnetic_field_magnitude{0.0};
    double minimum_sigma2{0.0};
    double minimum_correlation_length{0.0};
    double minimum_speed{0.0};

    /**
     * Validate host-side pitch-angle scattering controls.
     */
    void validate() const {
        if (!std::isfinite(spectral_index) || spectral_index <= 1.0 ||
            !std::isfinite(resonance_regularization_h0) ||
            resonance_regularization_h0 <= 0.0 ||
            !std::isfinite(minimum_magnetic_field_magnitude) ||
            minimum_magnetic_field_magnitude < 0.0 ||
            !std::isfinite(minimum_sigma2) || minimum_sigma2 < 0.0 ||
            !std::isfinite(minimum_correlation_length) ||
            minimum_correlation_length < 0.0 ||
            !std::isfinite(minimum_speed) || minimum_speed < 0.0) {
            throw std::runtime_error(
                "LegencyModel::PitchAngleScatteringParameters: coefficients "
                "and floors must be finite, with spectral_index > 1 and "
                "resonance_regularization_h0 > 0.");
        }
    }
};

/**
 * Local focused-transport pitch-angle scattering coefficients.
 */
struct PitchAngleScatteringCoefficients {
    double D_mumu{0.0};
    double dD_mumu_dmu{0.0};
    double omega{0.0};
    double xi{0.0};
    double regularized_abs_mu{0.0};
};

/**
 * Grid fields containing the gamma-independent diffusion coefficients.
 */
template <typename GridType>
struct GammaIndependentDiffusionFieldSet {
    using grid_type = GridType;
    using scalar_field_type = ScalarField<grid_type>;

    PerpendicularDiffusionModelKind perpendicular_model{
        PerpendicularDiffusionModelKind::ReadmeFormula
    };
    scalar_field_type kappa_parallel_gamma_one;
    scalar_field_type kappa_perpendicular_gamma_one;

    GammaIndependentDiffusionFieldSet() = default;

    /**
     * Allocate coefficient fields on a grid.
     */
    GammaIndependentDiffusionFieldSet(const grid_type& grid,
                                      const char* label = "legacy_diffusion",
                                      const PerpendicularDiffusionModelKind model_kind =
                                          PerpendicularDiffusionModelKind::ReadmeFormula)
        : perpendicular_model(model_kind) {
        const std::string prefix(label);
        kappa_parallel_gamma_one =
            scalar_field_type(grid, (prefix + "_kappa_parallel_gamma_one").c_str());
        kappa_perpendicular_gamma_one =
            scalar_field_type(grid,
                              (prefix + "_kappa_perpendicular_gamma_one").c_str());
    }
};

/**
 * Return a device-safe absolute value.
 */
KOKKOS_INLINE_FUNCTION
double absolute_value(const double value) {
    return value >= 0.0 ? value : -value;
}

/**
 * Return the vector magnitude at one field point.
 */
template <typename VectorFieldType>
KOKKOS_INLINE_FUNCTION
double vector_magnitude_at(const VectorFieldType& vector_field,
                           const typename VectorFieldType::index_array_type& indices) {
    double magnitude_squared = 0.0;
    for (int component = 0; component < VectorFieldType::component_dim; ++component) {
        const double value = vector_field(indices, component);
        magnitude_squared += value * value;
    }
    return Kokkos::sqrt(magnitude_squared);
}

/**
 * Return the nonrelativistic local gyrofrequency |q| |B| / (m c).
 */
KOKKOS_INLINE_FUNCTION
double nonrelativistic_gyrofrequency(const ParticleProperties& properties,
                                     const double magnetic_field_magnitude) {
    if (magnetic_field_magnitude <= 0.0) {
        Kokkos::abort("LegencyModel: gyrofrequency requires |B| > 0.");
    }
    return absolute_value(properties.charge) * magnetic_field_magnitude /
           (properties.rest_mass * properties.speed_of_light);
}

/**
 * Return the relativistic local gyrofrequency |q| |B| / (gamma m c).
 */
KOKKOS_INLINE_FUNCTION
double relativistic_gyrofrequency(const ParticleProperties& properties,
                                  const double magnetic_field_magnitude,
                                  const double gamma) {
    if (gamma <= 0.0) {
        Kokkos::abort("LegencyModel: gyrofrequency requires gamma > 0.");
    }
    return nonrelativistic_gyrofrequency(properties, magnetic_field_magnitude) /
           gamma;
}

/**
 * Apply a positive lower floor when configured.
 */
KOKKOS_INLINE_FUNCTION
double apply_positive_floor(const double value, const double floor_value) {
    if (floor_value <= 0.0) {
        return value;
    }
    return value < floor_value ? floor_value : value;
}

/**
 * Return a device-safe pitch-angle sign.
 */
KOKKOS_INLINE_FUNCTION
double pitch_angle_sign(const double mu) {
    if (mu > 0.0) {
        return 1.0;
    }
    if (mu < 0.0) {
        return -1.0;
    }
    return 0.0;
}

/**
 * Return |mu| + h0 for resonant pitch-angle denominators.
 */
KOKKOS_INLINE_FUNCTION
double regularized_abs_pitch_angle_mu(
    const double mu,
    const PitchAngleScatteringParameters& parameters =
        PitchAngleScatteringParameters{}) {
    return absolute_value(mu) + parameters.resonance_regularization_h0;
}

/**
 * Return kappa_parallel for gamma = 1.
 */
KOKKOS_INLINE_FUNCTION
double parallel_diffusion_gamma_one(
    const double solar_wind_speed,
    const double correlation_length,
    const double omega_nonrelativistic,
    const double sigma2,
    const DiffusionModelParameters& parameters = DiffusionModelParameters{}) {
    if (solar_wind_speed <= 0.0 || correlation_length <= 0.0 ||
        omega_nonrelativistic <= 0.0 || sigma2 <= 0.0) {
        Kokkos::abort(
            "LegencyModel: kappa_parallel requires positive Vsw, Lc, omega, and sigma2.");
    }

    return parameters.parallel_prefactor *
           Kokkos::pow(solar_wind_speed, 4.0 / 3.0) *
           Kokkos::pow(correlation_length, 2.0 / 3.0) /
           (Kokkos::pow(omega_nonrelativistic, 1.0 / 3.0) * sigma2);
}

/**
 * Return kappa_perpendicular for gamma = 1.
 */
KOKKOS_INLINE_FUNCTION
double perpendicular_diffusion_gamma_one(
    const double solar_wind_speed,
    const double l2d,
    const double sigma2_2d,
    const double kappa_parallel_gamma_one,
    const DiffusionModelParameters& parameters = DiffusionModelParameters{}) {
    if (solar_wind_speed <= 0.0 || l2d <= 0.0 || sigma2_2d <= 0.0 ||
        kappa_parallel_gamma_one <= 0.0) {
        Kokkos::abort(
            "LegencyModel: kappa_perpendicular requires positive Vsw, L2D, "
            "sigma2_2D, and kappa_parallel.");
    }

    return (solar_wind_speed / 3.0) *
           Kokkos::pow(3.0 * kappa_parallel_gamma_one / solar_wind_speed,
                       1.0 / 3.0) *
           Kokkos::pow(parameters.perpendicular_prefactor * sigma2_2d * l2d,
                       2.0 / 3.0);
}

/**
 * Return kappa_perpendicular = ratio * kappa_parallel for gamma = 1.
 */
KOKKOS_INLINE_FUNCTION
double perpendicular_diffusion_constant_ratio_gamma_one(
    const double kappa_parallel_gamma_one,
    const double ratio) {
    if (kappa_parallel_gamma_one <= 0.0 || ratio <= 0.0) {
        Kokkos::abort(
            "LegencyModel: constant-ratio kappa_perpendicular requires positive "
            "kappa_parallel and ratio.");
    }
    return ratio * kappa_parallel_gamma_one;
}

/**
 * Return kappa_perpendicular for gamma = 1 using the selected branch.
 */
KOKKOS_INLINE_FUNCTION
double perpendicular_diffusion_gamma_one(
    const double solar_wind_speed,
    const double l2d,
    const double sigma2_2d,
    const double kappa_parallel_gamma_one,
    const DiffusionModelParameters& parameters,
    const PerpendicularDiffusionModelKind model_kind) {
    if (model_kind == PerpendicularDiffusionModelKind::ConstantParallelRatio) {
        return perpendicular_diffusion_constant_ratio_gamma_one(
            kappa_parallel_gamma_one, parameters.perpendicular_parallel_ratio);
    }

    return perpendicular_diffusion_gamma_one(solar_wind_speed, l2d, sigma2_2d,
                                            kappa_parallel_gamma_one, parameters);
}

/**
 * Evaluate both gamma-independent diffusion coefficients at one location.
 */
KOKKOS_INLINE_FUNCTION
GammaIndependentDiffusionCoefficients evaluate_gamma_independent_coefficients(
    const ParticleProperties& properties,
    const double magnetic_field_magnitude,
    const double solar_wind_speed,
    const TurbulencePropertyState& turbulence,
    const DiffusionModelParameters& parameters = DiffusionModelParameters{}) {
    const double safe_b =
        apply_positive_floor(magnetic_field_magnitude,
                             parameters.minimum_magnetic_field_magnitude);
    const double safe_vsw =
        apply_positive_floor(solar_wind_speed,
                             parameters.minimum_solar_wind_speed);
    const double safe_sigma2 =
        apply_positive_floor(turbulence.sigma2, parameters.minimum_sigma2);
    const double omega_nonrel =
        nonrelativistic_gyrofrequency(properties, safe_b);

    GammaIndependentDiffusionCoefficients coefficients;
    coefficients.omega_nonrelativistic = omega_nonrel;
    coefficients.magnetic_field_magnitude = safe_b;
    coefficients.solar_wind_speed = safe_vsw;
    coefficients.kappa_parallel_gamma_one =
        parallel_diffusion_gamma_one(safe_vsw, turbulence.lc, omega_nonrel,
                                     safe_sigma2, parameters);
    coefficients.perpendicular_model = parameters.perpendicular_model;
    coefficients.kappa_perpendicular_gamma_one =
        perpendicular_diffusion_gamma_one(
            safe_vsw, turbulence.l2d, safe_sigma2,
            coefficients.kappa_parallel_gamma_one, parameters,
            parameters.perpendicular_model);
    return coefficients;
}

/**
 * Return the gamma factor for the selected perpendicular-diffusion branch.
 */
KOKKOS_INLINE_FUNCTION
double perpendicular_gamma_factor(const double gamma,
                                  const PerpendicularDiffusionModelKind model_kind) {
    if (gamma <= 0.0) {
        Kokkos::abort("LegencyModel: diffusion gamma correction requires gamma > 0.");
    }
    return model_kind == PerpendicularDiffusionModelKind::ConstantParallelRatio
               ? Kokkos::pow(gamma, 1.0 / 3.0)
               : Kokkos::pow(gamma, 1.0 / 9.0);
}

/**
 * Apply particle-gamma factors to precomputed gamma-independent scalar values.
 */
KOKKOS_INLINE_FUNCTION
DiffusionCoefficients apply_particle_gamma(
    const double kappa_parallel_gamma_one,
    const double kappa_perpendicular_gamma_one,
    const double gamma,
    const PerpendicularDiffusionModelKind perpendicular_model) {
    if (gamma <= 0.0) {
        Kokkos::abort("LegencyModel: diffusion gamma correction requires gamma > 0.");
    }

    DiffusionCoefficients result;
    result.kappa_parallel =
        kappa_parallel_gamma_one * Kokkos::pow(gamma, 1.0 / 3.0);
    result.kappa_perpendicular =
        kappa_perpendicular_gamma_one *
        perpendicular_gamma_factor(gamma, perpendicular_model);
    return result;
}

/**
 * Apply particle-gamma factors using the selected model parameters.
 */
KOKKOS_INLINE_FUNCTION
DiffusionCoefficients apply_particle_gamma(
    const double kappa_parallel_gamma_one,
    const double kappa_perpendicular_gamma_one,
    const double gamma,
    const DiffusionModelParameters& parameters) {
    return apply_particle_gamma(kappa_parallel_gamma_one,
                                kappa_perpendicular_gamma_one,
                                gamma, parameters.perpendicular_model);
}

/**
 * Apply particle-gamma factors to precomputed gamma-independent coefficients.
 */
KOKKOS_INLINE_FUNCTION
DiffusionCoefficients apply_particle_gamma(
    const GammaIndependentDiffusionCoefficients& coefficients,
    const double gamma) {
    return apply_particle_gamma(coefficients.kappa_parallel_gamma_one,
                                coefficients.kappa_perpendicular_gamma_one,
                                gamma, coefficients.perpendicular_model);
}

/**
 * Return the slab-spectrum normalization used by the README D_mumu formula.
 */
KOKKOS_INLINE_FUNCTION
double pitch_angle_spectrum_normalization(
    const double spectral_index) {
    if (spectral_index <= 1.0) {
        Kokkos::abort(
            "LegencyModel: pitch-angle spectral index must be greater than one.");
    }
    const double sine_value = Kokkos::sin(pi / spectral_index);
    if (sine_value == 0.0) {
        Kokkos::abort(
            "LegencyModel: invalid pitch-angle spectrum normalization.");
    }
    return spectral_index * sine_value / pi;
}

/**
 * Return D_mumu from the README focused-transport scattering formula.
 */
KOKKOS_INLINE_FUNCTION
double pitch_angle_diffusion_coefficient(
    const double mu,
    const double omega,
    const double speed,
    const double sigma2,
    const double correlation_length,
    const PitchAngleScatteringParameters& parameters =
        PitchAngleScatteringParameters{}) {
    if (omega <= 0.0 || speed <= 0.0 || sigma2 <= 0.0 ||
        correlation_length <= 0.0) {
        Kokkos::abort(
            "LegencyModel: D_mumu requires positive omega, speed, sigma2, and Lc.");
    }

    const double one_minus_mu2 = 1.0 - mu * mu;
    if (one_minus_mu2 <= 0.0) {
        return 0.0;
    }

    const double regularized_abs_mu =
        regularized_abs_pitch_angle_mu(mu, parameters);
    const double gamma_k = parameters.spectral_index;
    const double xi = omega * correlation_length / speed;
    const double abs_mu_gamma = Kokkos::pow(regularized_abs_mu, gamma_k);
    const double xi_gamma = Kokkos::pow(xi, gamma_k);
    const double denominator = abs_mu_gamma + xi_gamma;
    if (denominator <= 0.0) {
        Kokkos::abort("LegencyModel: D_mumu denominator must be positive.");
    }

    const double prefactor =
        0.25 * pi * omega * sigma2 *
        pitch_angle_spectrum_normalization(gamma_k);
    return prefactor * one_minus_mu2 *
           xi * Kokkos::pow(regularized_abs_mu, gamma_k - 1.0) /
           denominator;
}

/**
 * Return dD_mumu / dmu using the compact README derivative form.
 */
KOKKOS_INLINE_FUNCTION
double pitch_angle_diffusion_derivative(
    const double mu,
    const double D_mumu,
    const double omega,
    const double speed,
    const double correlation_length,
    const PitchAngleScatteringParameters& parameters =
        PitchAngleScatteringParameters{}) {
    if (D_mumu == 0.0) {
        return 0.0;
    }
    if (omega <= 0.0 || speed <= 0.0 || correlation_length <= 0.0) {
        Kokkos::abort(
            "LegencyModel: dD_mumu/dmu requires positive omega, speed, and Lc.");
    }

    const double sign_mu = pitch_angle_sign(mu);
    if (sign_mu == 0.0) {
        return 0.0;
    }

    const double one_minus_mu2 = 1.0 - mu * mu;
    if (one_minus_mu2 <= 0.0) {
        return 0.0;
    }

    const double regularized_abs_mu =
        regularized_abs_pitch_angle_mu(mu, parameters);
    const double gamma_k = parameters.spectral_index;
    const double xi = omega * correlation_length / speed;
    const double abs_mu_gamma = Kokkos::pow(regularized_abs_mu, gamma_k);
    const double xi_gamma = Kokkos::pow(xi, gamma_k);
    const double denominator = abs_mu_gamma + xi_gamma;
    if (denominator <= 0.0) {
        Kokkos::abort(
            "LegencyModel: dD_mumu/dmu denominator must be positive.");
    }

    const double shape_derivative =
        sign_mu *
        ((gamma_k - 1.0) / regularized_abs_mu -
         gamma_k * Kokkos::pow(regularized_abs_mu, gamma_k - 1.0) /
             denominator);
    const double endpoint_derivative = -2.0 * mu / one_minus_mu2;
    return D_mumu * (endpoint_derivative + shape_derivative);
}

/**
 * Evaluate D_mumu and dD_mumu / dmu at one particle state.
 */
KOKKOS_INLINE_FUNCTION
PitchAngleScatteringCoefficients evaluate_pitch_angle_scattering(
    const ParticleProperties& properties,
    const double magnetic_field_magnitude,
    const double speed,
    const double gamma,
    const double mu,
    const double sigma2,
    const double correlation_length,
    const PitchAngleScatteringParameters& parameters =
        PitchAngleScatteringParameters{}) {
    const double safe_b =
        apply_positive_floor(magnetic_field_magnitude,
                             parameters.minimum_magnetic_field_magnitude);
    const double safe_sigma2 =
        apply_positive_floor(sigma2, parameters.minimum_sigma2);
    const double safe_correlation_length =
        apply_positive_floor(correlation_length,
                             parameters.minimum_correlation_length);
    const double safe_speed = apply_positive_floor(speed, parameters.minimum_speed);
    if (safe_b <= 0.0 || safe_sigma2 <= 0.0 ||
        safe_correlation_length <= 0.0 || safe_speed <= 0.0) {
        Kokkos::abort(
            "LegencyModel: pitch-angle scattering requires positive local inputs.");
    }

    PitchAngleScatteringCoefficients coefficients;
    coefficients.omega =
        relativistic_gyrofrequency(properties, safe_b, gamma);
    coefficients.xi =
        coefficients.omega * safe_correlation_length / safe_speed;
    coefficients.regularized_abs_mu =
        regularized_abs_pitch_angle_mu(mu, parameters);
    coefficients.D_mumu = pitch_angle_diffusion_coefficient(
        mu, coefficients.omega, safe_speed, safe_sigma2,
        safe_correlation_length, parameters);
    coefficients.dD_mumu_dmu = pitch_angle_diffusion_derivative(
        mu, coefficients.D_mumu, coefficients.omega, safe_speed,
        safe_correlation_length, parameters);
    return coefficients;
}

/**
 * Return one component of kappa_ij from parallel/perpendicular scalars and b_i b_j.
 */
KOKKOS_INLINE_FUNCTION
double diffusion_tensor_component(const double kappa_parallel,
                                  const double kappa_perpendicular,
                                  const double magnetic_direction_i,
                                  const double magnetic_direction_j,
                                  const int i,
                                  const int j) {
    const double isotropic_part = i == j ? kappa_perpendicular : 0.0;
    return isotropic_part +
           (kappa_parallel - kappa_perpendicular) *
           magnetic_direction_i * magnetic_direction_j;
}

/**
 * Return one diffusion-tensor component directly from a magnetic-field vector.
 */
template <typename VectorFieldType>
KOKKOS_INLINE_FUNCTION
double diffusion_tensor_component_from_field(
    const VectorFieldType& magnetic_field,
    const typename VectorFieldType::index_array_type& indices,
    const double kappa_parallel,
    const double kappa_perpendicular,
    const int i,
    const int j,
    const double minimum_magnetic_field_magnitude = 0.0) {
    if (i < 0 || i >= VectorFieldType::component_dim ||
        j < 0 || j >= VectorFieldType::component_dim) {
        Kokkos::abort("LegencyModel: diffusion tensor component is out of range.");
    }

    const double magnetic_field_magnitude =
        apply_positive_floor(vector_magnitude_at(magnetic_field, indices),
                             minimum_magnetic_field_magnitude);
    if (magnetic_field_magnitude <= 0.0) {
        Kokkos::abort(
            "LegencyModel: diffusion tensor requires a nonzero magnetic field.");
    }

    const double inverse_magnetic_field_magnitude =
        1.0 / magnetic_field_magnitude;
    return diffusion_tensor_component(
        kappa_parallel, kappa_perpendicular,
        magnetic_field(indices, i) * inverse_magnetic_field_magnitude,
        magnetic_field(indices, j) * inverse_magnetic_field_magnitude,
        i, j);
}

/**
 * Return whether logical field indices lie in a selected domain.
 */
template <typename FieldType>
KOKKOS_INLINE_FUNCTION
bool indices_inside_domain(const FieldType& field,
                           const typename FieldType::index_array_type& indices,
                           const GridDomain domain) {
    for (int dim = 0; dim < FieldType::space_dim; ++dim) {
        if (indices[dim] < field.lower_index(dim, domain) ||
            indices[dim] > field.upper_index(dim, domain)) {
            return false;
        }
    }
    return true;
}

/**
 * Validate that fields share the same cell-centered storage domain.
 */
template <typename ReferenceFieldType, typename CheckedFieldType>
void validate_same_cell_centered_storage(
    const ReferenceFieldType& reference_field,
    const CheckedFieldType& checked_field,
    const char* operation_name) {
    static_assert(std::is_same_v<typename ReferenceFieldType::grid_type,
                                 typename CheckedFieldType::grid_type>,
                  "LegencyModel diffusion fields require the same grid type.");

    if (reference_field.point_count() != checked_field.point_count()) {
        throw std::runtime_error(
            std::string("LegencyModel: point-count mismatch in ") +
            operation_name + ".");
    }

    for (int dim = 0; dim < ReferenceFieldType::space_dim; ++dim) {
        if (reference_field.centering(dim) != GridCentering::CellCentered ||
            checked_field.centering(dim) != GridCentering::CellCentered ||
            reference_field.lower_index(dim, GridDomain::GhostedDomain) !=
                checked_field.lower_index(dim, GridDomain::GhostedDomain) ||
            reference_field.upper_index(dim, GridDomain::GhostedDomain) !=
                checked_field.upper_index(dim, GridDomain::GhostedDomain)) {
            throw std::runtime_error(
                std::string("LegencyModel: storage-domain mismatch in ") +
                operation_name + ".");
        }
    }
}

/**
 * Fill gamma-independent diffusion fields from B, Vsw, and turbulence properties.
 */
template <
    typename MagneticFieldType,
    typename SolarWindVelocityFieldType,
    typename TurbulencePropertiesType,
    typename ParallelFieldType,
    typename PerpendicularFieldType
>
void fill_gamma_independent_diffusion_fields(
    const MagneticFieldType& magnetic_field,
    const SolarWindVelocityFieldType& solar_wind_velocity,
    const TurbulencePropertiesType& turbulence_properties,
    const ParticleProperties& particle_properties,
    ParallelFieldType& kappa_parallel_gamma_one,
    PerpendicularFieldType& kappa_perpendicular_gamma_one,
    const DiffusionModelParameters& parameters = DiffusionModelParameters{},
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    static_assert(MagneticFieldType::component_dim > 0,
                  "LegencyModel requires a vector magnetic field.");
    static_assert(SolarWindVelocityFieldType::component_dim > 0,
                  "LegencyModel requires a vector solar-wind field.");
    static_assert(ParallelFieldType::component_dim == 1 &&
                      PerpendicularFieldType::component_dim == 1,
                  "LegencyModel diffusion coefficient outputs must be scalar fields.");
    static_assert(std::is_same_v<typename MagneticFieldType::grid_type,
                                 typename TurbulencePropertiesType::grid_type>,
                  "LegencyModel turbulence properties must use the same grid type.");

    parameters.validate();
    particle_properties.validate();
    validate_same_cell_centered_storage(magnetic_field, solar_wind_velocity,
                                        "fill_gamma_independent_diffusion_fields");
    validate_same_cell_centered_storage(magnetic_field, kappa_parallel_gamma_one,
                                        "fill_gamma_independent_diffusion_fields");
    validate_same_cell_centered_storage(magnetic_field, kappa_perpendicular_gamma_one,
                                        "fill_gamma_independent_diffusion_fields");

    const MagneticFieldType local_magnetic_field = magnetic_field;
    const SolarWindVelocityFieldType local_solar_wind_velocity =
        solar_wind_velocity;
    const auto turbulence = turbulence_properties.accessor();
    const auto local_particle_properties = particle_properties;
    const auto local_parameters = parameters;
    const ParallelFieldType local_parallel = kappa_parallel_gamma_one;
    auto parallel_data = kappa_parallel_gamma_one.data;
    auto perpendicular_data = kappa_perpendicular_gamma_one.data;
    const auto count = kappa_parallel_gamma_one.point_count();

    Kokkos::parallel_for(
        "LegencyModel::fill_gamma_independent_diffusion_fields",
        Kokkos::RangePolicy<typename ParallelFieldType::execution_space>(0, count),
        KOKKOS_LAMBDA(const typename ParallelFieldType::size_type point_index) {
            const auto indices = local_parallel.logical_indices(point_index);
            if (!indices_inside_domain(local_parallel, indices,
                                       GridDomain::PhysicalDomain)) {
                return;
            }

            const double magnetic_field_magnitude =
                vector_magnitude_at(local_magnetic_field, indices);
            const double solar_wind_speed =
                vector_magnitude_at(local_solar_wind_velocity, indices);
            const TurbulencePropertyState turbulence_state = turbulence(indices);
            const GammaIndependentDiffusionCoefficients coefficients =
                evaluate_gamma_independent_coefficients(
                    local_particle_properties, magnetic_field_magnitude,
                    solar_wind_speed, turbulence_state, local_parameters);

            parallel_data(point_index, 0) =
                coefficients.kappa_parallel_gamma_one;
            perpendicular_data(point_index, 0) =
                coefficients.kappa_perpendicular_gamma_one;
        });
    Kokkos::fence("LegencyModel::fill_gamma_independent_diffusion_fields");

    kappa_parallel_gamma_one.fill_ghost_cells(ghost_fill_mode);
    kappa_perpendicular_gamma_one.fill_ghost_cells(ghost_fill_mode);
}

/**
 * Allocate and fill gamma-independent diffusion fields.
 */
template <
    typename MagneticFieldType,
    typename SolarWindVelocityFieldType,
    typename TurbulencePropertiesType
>
auto make_gamma_independent_diffusion_fields(
    const MagneticFieldType& magnetic_field,
    const SolarWindVelocityFieldType& solar_wind_velocity,
    const TurbulencePropertiesType& turbulence_properties,
    const ParticleProperties& particle_properties,
    const char* label = "legacy_diffusion",
    const DiffusionModelParameters& parameters = DiffusionModelParameters{},
    const FieldGhostFillMode ghost_fill_mode =
        FieldGhostFillMode::PeriodicOrClamped) {
    using grid_type = typename MagneticFieldType::grid_type;
    GammaIndependentDiffusionFieldSet<grid_type> fields(
        magnetic_field.grid, label, parameters.perpendicular_model);
    fill_gamma_independent_diffusion_fields(
        magnetic_field, solar_wind_velocity, turbulence_properties,
        particle_properties, fields.kappa_parallel_gamma_one,
        fields.kappa_perpendicular_gamma_one, parameters, ghost_fill_mode);
    return fields;
}

} // namespace LegencyModel
