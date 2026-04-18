#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "Grid.hpp"
#include "KokkosDevice.hpp"

/**
 * Transport model that produced the particle input used by emission post-processing.
 */
enum class EmissionTransportModel : int {
    Parker = 0,
    Focused = 1,
    Unknown = 2
};

/**
 * Source kind used to provide magnetic and flow fields for emission synthesis.
 */
enum class EmissionFieldSourceKind : int {
    FileBacked = 0,
    Analytic = 1
};

/**
 * Source kind used to provide thermal/background plasma parameters.
 */
enum class EmissionBackgroundSourceKind : int {
    Constant = 0,
    Map = 1,
    MhdConverted = 2,
    Fitted = 3,
    Analytic = 4
};

/**
 * Source kind used to provide nonthermal particle distributions for emission synthesis.
 */
enum class EmissionParticleSourceKind : int {
    ParkerSnapshot = 0,
    FocusedSnapshot = 1,
    ReducedDistribution = 2,
    Analytic = 3
};

/**
 * Availability state of pitch-angle information for a local emission source.
 */
enum class EmissionPitchAngleMode : int {
    Unavailable = 0,
    Isotropic = 1,
    PitchAngleCosine = 2
};

/**
 * ROI selector kind used by emission analysis products.
 */
enum class EmissionRoiKind : int {
    Pixel = 0,
    Rectangle = 1,
    BeamCentered = 2,
    Mask = 3
};

/**
 * Return whether one transport model provides physically meaningful stored `mu`.
 */
KOKKOS_INLINE_FUNCTION
constexpr bool emission_transport_has_physical_mu(
    const EmissionTransportModel transport_model) {
    return transport_model == EmissionTransportModel::Focused;
}

/**
 * Return the default pitch-angle mode implied by a transport model.
 */
KOKKOS_INLINE_FUNCTION
constexpr EmissionPitchAngleMode emission_default_pitch_angle_mode(
    const EmissionTransportModel transport_model) {
    return emission_transport_has_physical_mu(transport_model)
               ? EmissionPitchAngleMode::PitchAngleCosine
               : EmissionPitchAngleMode::Unavailable;
}

/**
 * Return whether a pitch-angle mode carries usable `mu` information.
 */
KOKKOS_INLINE_FUNCTION
constexpr bool emission_pitch_angle_mode_has_mu(
    const EmissionPitchAngleMode pitch_angle_mode) {
    return pitch_angle_mode == EmissionPitchAngleMode::PitchAngleCosine;
}

/**
 * Unit metadata carried by emission products and provider outputs.
 */
struct EmissionUnitMetadata {
    std::string coordinate_unit{"code_length"};
    std::string magnetic_field_unit{"code_magnetic_field"};
    std::string velocity_unit{"code_velocity"};
    std::string number_density_unit{"cm^-3"};
    std::string temperature_unit{"K"};
    std::string specific_intensity_unit{"erg s^-1 cm^-2 Hz^-1 sr^-1"};
    std::string brightness_temperature_unit{"K"};
};

/**
 * Ordered frequency grid used by one emission synthesis or analysis product.
 */
struct EmissionFrequencyGrid {
    std::vector<double> frequency_hz;

    /**
     * Validate positivity, finiteness, and strict monotonic increase.
     */
    void validate() const {
        if (frequency_hz.empty()) {
            throw std::runtime_error(
                "EmissionFrequencyGrid: frequency grid must not be empty.");
        }
        for (std::size_t index = 0; index < frequency_hz.size(); ++index) {
            const double frequency = frequency_hz[index];
            if (!std::isfinite(frequency) || frequency <= 0.0) {
                throw std::runtime_error(
                    "EmissionFrequencyGrid: frequency values must be positive and finite.");
            }
            if (index > 0 && !(frequency > frequency_hz[index - 1])) {
                throw std::runtime_error(
                    "EmissionFrequencyGrid: frequency grid must be strictly increasing.");
            }
        }
    }
};

/**
 * Observer configuration used by one emission synthesis run.
 */
template <int SpaceDim>
struct EmissionObserver {
    using coordinate_array_type = Kokkos::Array<double, SpaceDim>;

    coordinate_array_type line_of_sight_direction{};
    double distance_from_origin{1.0};

    /**
     * Validate and normalize the LOS direction on the host.
     */
    void validate_and_normalize() {
        double magnitude_squared = 0.0;
        for (int dim = 0; dim < SpaceDim; ++dim) {
            const double value = line_of_sight_direction[dim];
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "EmissionObserver: line-of-sight components must be finite.");
            }
            magnitude_squared += value * value;
        }
        if (!std::isfinite(distance_from_origin) || distance_from_origin <= 0.0) {
            throw std::runtime_error(
                "EmissionObserver: observer distance must be positive and finite.");
        }
        if (!(magnitude_squared > 0.0) || !std::isfinite(magnitude_squared)) {
            throw std::runtime_error(
                "EmissionObserver: line-of-sight vector must have non-zero magnitude.");
        }

        const double inverse_magnitude = 1.0 / std::sqrt(magnitude_squared);
        for (int dim = 0; dim < SpaceDim; ++dim) {
            line_of_sight_direction[dim] *= inverse_magnitude;
        }
    }
};

/**
 * One local magnetic and flow-field sample passed into emission closure.
 */
template <int SpaceDim>
struct EmissionFieldSample {
    using coordinate_array_type = Kokkos::Array<double, SpaceDim>;

    coordinate_array_type coordinate{};
    coordinate_array_type magnetic_field{};
    coordinate_array_type flow_velocity{};

    /**
     * Validate that all sampled values are finite.
     */
    void validate() const {
        for (int dim = 0; dim < SpaceDim; ++dim) {
            if (!std::isfinite(coordinate[dim]) ||
                !std::isfinite(magnetic_field[dim]) ||
                !std::isfinite(flow_velocity[dim])) {
                throw std::runtime_error(
                    "EmissionFieldSample: coordinate, magnetic field, and flow values "
                    "must be finite.");
            }
        }
    }
};

/**
 * One local thermal/background plasma sample used by the microwave backend.
 */
template <int SpaceDim>
struct EmissionBackgroundSample {
    using coordinate_array_type = Kokkos::Array<double, SpaceDim>;

    coordinate_array_type coordinate{};
    double thermal_electron_density{0.0};
    double electron_temperature{0.0};
    double line_of_sight_depth{0.0};
    double background_specific_intensity{0.0};
    double background_brightness_temperature{0.0};

    /**
     * Validate positivity and finiteness of local background parameters.
     */
    void validate() const {
        for (int dim = 0; dim < SpaceDim; ++dim) {
            if (!std::isfinite(coordinate[dim])) {
                throw std::runtime_error(
                    "EmissionBackgroundSample: coordinate values must be finite.");
            }
        }
        if (!std::isfinite(thermal_electron_density) || thermal_electron_density < 0.0 ||
            !std::isfinite(electron_temperature) || electron_temperature < 0.0 ||
            !std::isfinite(line_of_sight_depth) || line_of_sight_depth < 0.0 ||
            !std::isfinite(background_specific_intensity) ||
            background_specific_intensity < 0.0 ||
            !std::isfinite(background_brightness_temperature) ||
            background_brightness_temperature < 0.0) {
            throw std::runtime_error(
                "EmissionBackgroundSample: background quantities must be finite and "
                "non-negative.");
        }
    }
};

/**
 * Reduced nonthermal particle description used by the first emission closure stage.
 */
template <int SpaceDim>
struct EmissionParticleMoments {
    using coordinate_array_type = Kokkos::Array<double, SpaceDim>;

    coordinate_array_type coordinate{};
    double nonthermal_electron_density{0.0};
    double minimum_energy_ev{0.0};
    double maximum_energy_ev{0.0};
    double power_law_index{4.0};
    double mean_pitch_angle_cosine{0.0};
    double pitch_angle_cosine_variance{0.0};
    double particle_weight_sum{0.0};
    EmissionPitchAngleMode pitch_angle_mode{EmissionPitchAngleMode::Unavailable};

    /**
     * Validate finiteness and ordering of reduced particle moments.
     */
    void validate() const {
        for (int dim = 0; dim < SpaceDim; ++dim) {
            if (!std::isfinite(coordinate[dim])) {
                throw std::runtime_error(
                    "EmissionParticleMoments: coordinate values must be finite.");
            }
        }
        if (!std::isfinite(nonthermal_electron_density) ||
            nonthermal_electron_density < 0.0 ||
            !std::isfinite(minimum_energy_ev) || minimum_energy_ev < 0.0 ||
            !std::isfinite(maximum_energy_ev) || maximum_energy_ev < minimum_energy_ev ||
            !std::isfinite(power_law_index) || power_law_index <= 0.0 ||
            !std::isfinite(mean_pitch_angle_cosine) ||
            !std::isfinite(pitch_angle_cosine_variance) ||
            pitch_angle_cosine_variance < 0.0 ||
            !std::isfinite(particle_weight_sum) || particle_weight_sum < 0.0) {
            throw std::runtime_error(
                "EmissionParticleMoments: particle moments must be finite, energies "
                "must be ordered, and densities/weights must be non-negative.");
        }
        if (pitch_angle_mode != EmissionPitchAngleMode::Unavailable &&
            pitch_angle_mode != EmissionPitchAngleMode::Isotropic &&
            pitch_angle_mode != EmissionPitchAngleMode::PitchAngleCosine) {
            throw std::runtime_error(
                "EmissionParticleMoments: unsupported pitch-angle mode.");
        }
    }

    /**
     * Apply the transport-model rule for usable pitch-angle information.
     */
    void apply_transport_model_policy(const EmissionTransportModel transport_model) {
        if (!emission_transport_has_physical_mu(transport_model)) {
            pitch_angle_mode = EmissionPitchAngleMode::Unavailable;
            mean_pitch_angle_cosine = 0.0;
            pitch_angle_cosine_variance = 0.0;
        }
    }
};

/**
 * One fully closed local microwave source passed to the emission backend.
 */
template <int SpaceDim>
struct EmissionSourceCell {
    EmissionFieldSample<SpaceDim> field;
    EmissionBackgroundSample<SpaceDim> background;
    EmissionParticleMoments<SpaceDim> particles;
    double viewing_angle_radian{0.0};

    /**
     * Validate one source-cell closure before backend evaluation.
     */
    void validate() const {
        field.validate();
        background.validate();
        particles.validate();
        const double pi = std::acos(-1.0);
        if (!std::isfinite(viewing_angle_radian) || viewing_angle_radian < 0.0 ||
            viewing_angle_radian > pi) {
            throw std::runtime_error(
                "EmissionSourceCell: viewing angle must be finite and lie in [0, pi].");
        }
    }
};
