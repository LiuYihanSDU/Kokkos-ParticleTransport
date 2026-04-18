#pragma once

#include <functional>
#include <utility>

#include "EmissionTypes.hpp"

/**
 * Host-side field provider interface used by emission post-processing setup.
 */
template <int SpaceDim>
struct EmissionFieldProvider {
    using coordinate_array_type = Kokkos::Array<double, SpaceDim>;

    virtual ~EmissionFieldProvider() = default;

    /**
     * Return the backing source kind used by the provider.
     */
    virtual EmissionFieldSourceKind source_kind() const = 0;

    /**
     * Return the coordinate system expected by the provider.
     */
    virtual OrthogonalCoordinateSystem coordinate_system() const = 0;

    /**
     * Sample magnetic and flow fields at one coordinate.
     */
    virtual EmissionFieldSample<SpaceDim> sample(
        const coordinate_array_type& coordinate) const = 0;
};

/**
 * Host-side background provider interface used by emission closure.
 */
template <int SpaceDim>
struct EmissionBackgroundProvider {
    using coordinate_array_type = Kokkos::Array<double, SpaceDim>;

    virtual ~EmissionBackgroundProvider() = default;

    /**
     * Return the background-parameter source kind.
     */
    virtual EmissionBackgroundSourceKind source_kind() const = 0;

    /**
     * Sample thermal/background quantities at one coordinate.
     */
    virtual EmissionBackgroundSample<SpaceDim> sample(
        const coordinate_array_type& coordinate) const = 0;
};

/**
 * Host-side reduced particle provider interface used by emission closure.
 */
template <int SpaceDim>
struct EmissionParticleDistributionProvider {
    using coordinate_array_type = Kokkos::Array<double, SpaceDim>;

    virtual ~EmissionParticleDistributionProvider() = default;

    /**
     * Return the particle source kind used by this provider.
     */
    virtual EmissionParticleSourceKind source_kind() const = 0;

    /**
     * Return the transport model that produced the particle information.
     */
    virtual EmissionTransportModel transport_model() const = 0;

    /**
     * Return the provider pitch-angle interpretation after transport-model rules.
     */
    virtual EmissionPitchAngleMode pitch_angle_mode() const {
        return emission_default_pitch_angle_mode(transport_model());
    }

    /**
     * Sample reduced nonthermal moments at one coordinate.
     */
    virtual EmissionParticleMoments<SpaceDim> sample(
        const coordinate_array_type& coordinate) const = 0;
};

/**
 * Field provider backed by analytic magnetic and flow functions.
 */
template <
    int SpaceDim,
    typename MagneticFieldFunctor,
    typename VelocityFieldFunctor
>
struct AnalyticEmissionFieldProvider final : public EmissionFieldProvider<SpaceDim> {
    using coordinate_array_type = typename EmissionFieldProvider<SpaceDim>::coordinate_array_type;

    MagneticFieldFunctor magnetic_field_functor;
    VelocityFieldFunctor velocity_field_functor;
    OrthogonalCoordinateSystem coordinate_system_cache{
        OrthogonalCoordinateSystem::Cartesian
    };

    /**
     * Construct an analytic provider from magnetic and flow functors.
     */
    AnalyticEmissionFieldProvider(
        MagneticFieldFunctor input_magnetic_field_functor,
        VelocityFieldFunctor input_velocity_field_functor,
        const OrthogonalCoordinateSystem input_coordinate_system =
            OrthogonalCoordinateSystem::Cartesian)
        : magnetic_field_functor(std::move(input_magnetic_field_functor)),
          velocity_field_functor(std::move(input_velocity_field_functor)),
          coordinate_system_cache(input_coordinate_system) {}

    EmissionFieldSourceKind source_kind() const override {
        return EmissionFieldSourceKind::Analytic;
    }

    OrthogonalCoordinateSystem coordinate_system() const override {
        return coordinate_system_cache;
    }

    EmissionFieldSample<SpaceDim> sample(
        const coordinate_array_type& coordinate) const override {
        EmissionFieldSample<SpaceDim> field_sample{};
        field_sample.coordinate = coordinate;
        field_sample.magnetic_field = magnetic_field_functor(coordinate);
        field_sample.flow_velocity = velocity_field_functor(coordinate);
        field_sample.validate();
        return field_sample;
    }
};

/**
 * Background provider backed by one constant thermal/background state.
 */
template <int SpaceDim>
struct ConstantEmissionBackgroundProvider final
    : public EmissionBackgroundProvider<SpaceDim> {
    using coordinate_array_type = typename EmissionBackgroundProvider<SpaceDim>::coordinate_array_type;

    EmissionBackgroundSample<SpaceDim> constant_sample{};
    EmissionBackgroundSourceKind source_kind_cache{
        EmissionBackgroundSourceKind::Constant
    };

    /**
     * Build a constant background provider from one reference sample.
     */
    explicit ConstantEmissionBackgroundProvider(
        const EmissionBackgroundSample<SpaceDim>& input_sample,
        const EmissionBackgroundSourceKind input_source_kind =
            EmissionBackgroundSourceKind::Constant)
        : constant_sample(input_sample),
          source_kind_cache(input_source_kind) {
        constant_sample.validate();
    }

    EmissionBackgroundSourceKind source_kind() const override {
        return source_kind_cache;
    }

    EmissionBackgroundSample<SpaceDim> sample(
        const coordinate_array_type& coordinate) const override {
        EmissionBackgroundSample<SpaceDim> result = constant_sample;
        result.coordinate = coordinate;
        result.validate();
        return result;
    }
};

/**
 * Background provider backed by one callable host-side function.
 */
template <int SpaceDim, typename SampleFunctor>
struct FunctionalEmissionBackgroundProvider final
    : public EmissionBackgroundProvider<SpaceDim> {
    using coordinate_array_type = typename EmissionBackgroundProvider<SpaceDim>::coordinate_array_type;

    SampleFunctor sample_functor;
    EmissionBackgroundSourceKind source_kind_cache{
        EmissionBackgroundSourceKind::Analytic
    };

    /**
     * Build a functional background provider from one callable sample function.
     */
    explicit FunctionalEmissionBackgroundProvider(
        SampleFunctor input_sample_functor,
        const EmissionBackgroundSourceKind input_source_kind =
            EmissionBackgroundSourceKind::Analytic)
        : sample_functor(std::move(input_sample_functor)),
          source_kind_cache(input_source_kind) {}

    EmissionBackgroundSourceKind source_kind() const override {
        return source_kind_cache;
    }

    EmissionBackgroundSample<SpaceDim> sample(
        const coordinate_array_type& coordinate) const override {
        EmissionBackgroundSample<SpaceDim> result = sample_functor(coordinate);
        result.coordinate = coordinate;
        result.validate();
        return result;
    }
};

/**
 * Particle provider backed by one host-side reduced-moments function.
 *
 * The provider enforces the repository rule that Parker transport outputs do not
 * carry usable `mu` information for emission reconstruction.
 */
template <int SpaceDim, typename SampleFunctor>
struct FunctionalEmissionParticleProvider final
    : public EmissionParticleDistributionProvider<SpaceDim> {
    using coordinate_array_type =
        typename EmissionParticleDistributionProvider<SpaceDim>::coordinate_array_type;

    SampleFunctor sample_functor;
    EmissionParticleSourceKind source_kind_cache{
        EmissionParticleSourceKind::ReducedDistribution
    };
    EmissionTransportModel transport_model_cache{EmissionTransportModel::Unknown};
    EmissionPitchAngleMode requested_pitch_angle_mode{
        EmissionPitchAngleMode::Unavailable
    };

    /**
     * Build a functional particle provider with explicit transport-model metadata.
     */
    FunctionalEmissionParticleProvider(
        SampleFunctor input_sample_functor,
        const EmissionTransportModel input_transport_model,
        const EmissionParticleSourceKind input_source_kind =
            EmissionParticleSourceKind::ReducedDistribution,
        const EmissionPitchAngleMode input_pitch_angle_mode =
            EmissionPitchAngleMode::Unavailable)
        : sample_functor(std::move(input_sample_functor)),
          source_kind_cache(input_source_kind),
          transport_model_cache(input_transport_model),
          requested_pitch_angle_mode(input_pitch_angle_mode) {}

    EmissionParticleSourceKind source_kind() const override {
        return source_kind_cache;
    }

    EmissionTransportModel transport_model() const override {
        return transport_model_cache;
    }

    EmissionPitchAngleMode pitch_angle_mode() const override {
        if (!emission_transport_has_physical_mu(transport_model_cache)) {
            return EmissionPitchAngleMode::Unavailable;
        }
        return requested_pitch_angle_mode;
    }

    EmissionParticleMoments<SpaceDim> sample(
        const coordinate_array_type& coordinate) const override {
        EmissionParticleMoments<SpaceDim> result = sample_functor(coordinate);
        result.coordinate = coordinate;
        result.pitch_angle_mode = pitch_angle_mode();
        result.apply_transport_model_policy(transport_model_cache);
        result.validate();
        return result;
    }
};
