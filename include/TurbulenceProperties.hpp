#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "Field.hpp"
#include "PhysicalConstant.hpp"
#include "UnitConverter.hpp"

/**
 * Radial model used to initialize turbulence properties.
 */
enum class TurbulencePropertyModelKind : int {
    Constant = 0,
    HeliosphericRadial = 1
};

/**
 * How the effective correlation length Lc is initialized.
 */
enum class TurbulenceCorrelationLengthModel : int {
    SlabLength = 0,
    Constant = 1
};

/**
 * Storage policy for turbulence properties.
 */
enum class TurbulencePropertyStorageMode : int {
    Auto = 0,
    ConstantOnly = 1,
    Field = 2
};

/**
 * Turbulence-property values and radial derivatives at one location.
 */
struct TurbulencePropertyState {
    double sigma2{0.0};
    double l2d{0.0};
    double lslab{0.0};
    double lc{0.0};
    double d_sigma2_dr{0.0};
    double d_l2d_dr{0.0};
    double d_lslab_dr{0.0};
    double d_lc_dr{0.0};
};

/**
 * Device-callable turbulence-property model.
 *
 * All lengths are stored in code units. The heliospheric radial model interprets the
 * input radius as a code-unit length and internally converts it to astronomical units.
 */
struct TurbulencePropertyModel {
    TurbulencePropertyModelKind kind{TurbulencePropertyModelKind::Constant};
    TurbulenceCorrelationLengthModel lc_model{
        TurbulenceCorrelationLengthModel::SlabLength
    };
    double au_in_code_units{1.0};
    double constant_sigma2{0.5};
    double constant_l2d{1.0};
    double constant_lslab{1.0};
    double constant_lc{1.0};
    double heliospheric_l2d_at_one_au_in_au{0.0074};
    double heliospheric_l2d_power{1.1};
    double heliospheric_lslab_to_l2d{3.9};

    /**
     * Build a constant model from already nondimensionalized code-unit lengths.
     */
    static TurbulencePropertyModel constant_code_units(const double sigma2,
                                                       const double l2d,
                                                       const double lslab,
                                                       const double lc = -1.0) {
        TurbulencePropertyModel model;
        model.kind = TurbulencePropertyModelKind::Constant;
        model.lc_model = lc > 0.0 ? TurbulenceCorrelationLengthModel::Constant
                                  : TurbulenceCorrelationLengthModel::SlabLength;
        model.constant_sigma2 = sigma2;
        model.constant_l2d = l2d;
        model.constant_lslab = lslab;
        model.constant_lc = lc > 0.0 ? lc : lslab;
        model.validate();
        return model;
    }

    /**
     * Build a constant model from AU lengths and a target nondimensionalizer.
     */
    static TurbulencePropertyModel constant_au(
        const double sigma2,
        const double l2d_au,
        const double lslab_au,
        const Units::Nondimensionalizer& nondimensionalizer,
        const double lc_au = -1.0) {
        const double au_code = astronomical_unit_in_code_units(nondimensionalizer);
        return constant_code_units(sigma2, l2d_au * au_code, lslab_au * au_code,
                                   lc_au > 0.0 ? lc_au * au_code : -1.0);
    }

    /**
     * Build a constant model from CGS lengths and a target nondimensionalizer.
     */
    static TurbulencePropertyModel constant_cgs(
        const double sigma2,
        const Units::Length l2d,
        const Units::Length lslab,
        const Units::Nondimensionalizer& nondimensionalizer,
        const Units::Length lc = Units::Length(-1.0)) {
        return constant_code_units(
            sigma2,
            nondimensionalizer.to_dimensionless(l2d).raw(),
            nondimensionalizer.to_dimensionless(lslab).raw(),
            lc.value_cgs > 0.0 ? nondimensionalizer.to_dimensionless(lc).raw() : -1.0);
    }

    /**
     * Build the heliospheric radial turbulence-property preset.
     */
    static TurbulencePropertyModel heliospheric_radial(
        const Units::Nondimensionalizer& nondimensionalizer,
        const TurbulenceCorrelationLengthModel input_lc_model =
            TurbulenceCorrelationLengthModel::SlabLength,
        const double constant_lc_au = -1.0) {
        TurbulencePropertyModel model;
        model.kind = TurbulencePropertyModelKind::HeliosphericRadial;
        model.lc_model = input_lc_model;
        model.au_in_code_units = astronomical_unit_in_code_units(nondimensionalizer);
        if (input_lc_model == TurbulenceCorrelationLengthModel::Constant) {
            if (constant_lc_au <= 0.0) {
                throw std::runtime_error(
                    "TurbulencePropertyModel: constant heliospheric Lc must be positive.");
            }
            model.constant_lc = constant_lc_au * model.au_in_code_units;
        }
        model.validate();
        return model;
    }

    /**
     * Return true when the model is spatially constant.
     */
    KOKKOS_INLINE_FUNCTION
    bool is_constant() const {
        return kind == TurbulencePropertyModelKind::Constant;
    }

    /**
     * Evaluate turbulence properties at a code-unit radial distance.
     */
    KOKKOS_INLINE_FUNCTION
    TurbulencePropertyState evaluate_from_radius(const double radius_code) const {
        switch (kind) {
            case TurbulencePropertyModelKind::Constant:
                return evaluate_constant();
            case TurbulencePropertyModelKind::HeliosphericRadial:
                return evaluate_heliospheric_radial(radius_code);
        }
        return evaluate_constant();
    }

    /**
     * Validate model coefficients on the host before field initialization.
     */
    void validate() const {
        if (!std::isfinite(au_in_code_units) || au_in_code_units <= 0.0 ||
            !std::isfinite(constant_sigma2) || constant_sigma2 <= 0.0 ||
            !std::isfinite(constant_l2d) || constant_l2d <= 0.0 ||
            !std::isfinite(constant_lslab) || constant_lslab <= 0.0 ||
            !std::isfinite(constant_lc) || constant_lc <= 0.0 ||
            !std::isfinite(heliospheric_l2d_at_one_au_in_au) ||
            heliospheric_l2d_at_one_au_in_au <= 0.0 ||
            !std::isfinite(heliospheric_l2d_power) ||
            heliospheric_l2d_power <= 0.0 ||
            !std::isfinite(heliospheric_lslab_to_l2d) ||
            heliospheric_lslab_to_l2d <= 0.0) {
            throw std::runtime_error(
                "TurbulencePropertyModel: coefficients must be positive finite values.");
        }
    }

private:
    /**
     * Return one astronomical unit in code length units.
     */
    static double astronomical_unit_in_code_units(
        const Units::Nondimensionalizer& nondimensionalizer) {
        return nondimensionalizer.to_dimensionless(
            PhysicalConstants::astronomical_unit).raw();
    }

    /**
     * Evaluate the constant turbulence-property model.
     */
    KOKKOS_INLINE_FUNCTION
    TurbulencePropertyState evaluate_constant() const {
        TurbulencePropertyState state;
        state.sigma2 = constant_sigma2;
        state.l2d = constant_l2d;
        state.lslab = constant_lslab;
        state.lc = lc_model == TurbulenceCorrelationLengthModel::SlabLength
                       ? constant_lslab
                       : constant_lc;
        return state;
    }

    /**
     * Evaluate the heliospheric radial turbulence-property preset.
     */
    KOKKOS_INLINE_FUNCTION
    TurbulencePropertyState evaluate_heliospheric_radial(
        const double radius_code) const {
        TurbulencePropertyState state;
        const double radius_au = radius_code > 0.0 ? radius_code / au_in_code_units : 0.0;
        if (radius_au <= 0.0) {
            state.lc = lc_model == TurbulenceCorrelationLengthModel::Constant
                           ? constant_lc
                           : 0.0;
            return state;
        }

        evaluate_heliospheric_sigma2(radius_au, state);
        state.l2d = heliospheric_l2d_at_one_au_in_au * au_in_code_units *
                    Kokkos::pow(radius_au, heliospheric_l2d_power);
        state.d_l2d_dr = heliospheric_l2d_at_one_au_in_au *
                         heliospheric_l2d_power *
                         Kokkos::pow(radius_au, heliospheric_l2d_power - 1.0);
        state.lslab = heliospheric_lslab_to_l2d * state.l2d;
        state.d_lslab_dr = heliospheric_lslab_to_l2d * state.d_l2d_dr;
        if (lc_model == TurbulenceCorrelationLengthModel::Constant) {
            state.lc = constant_lc;
            state.d_lc_dr = 0.0;
        } else {
            state.lc = state.lslab;
            state.d_lc_dr = state.d_lslab_dr;
        }
        return state;
    }

    /**
     * Evaluate the piecewise heliospheric sigma^2 profile and radial derivative.
     */
    KOKKOS_INLINE_FUNCTION
    void evaluate_heliospheric_sigma2(const double radius_au,
                                      TurbulencePropertyState& state) const {
        if (radius_au < 0.5) {
            state.sigma2 = 0.1 * Kokkos::pow(radius_au / 0.1, 0.5);
            state.d_sigma2_dr =
                (0.1 * 0.5 / 0.1 * Kokkos::pow(radius_au / 0.1, -0.5)) /
                au_in_code_units;
        } else if (radius_au < 2.0) {
            state.sigma2 = 0.15 * Kokkos::pow(radius_au / 0.1, 0.25);
            state.d_sigma2_dr =
                (0.15 * 0.25 / 0.1 * Kokkos::pow(radius_au / 0.1, -0.75)) /
                au_in_code_units;
        } else {
            state.sigma2 = 0.32;
            state.d_sigma2_dr = 0.0;
        }
    }
};

/**
 * Return the radial distance represented by an orthogonal-coordinate tuple.
 */
template <typename GridType>
KOKKOS_INLINE_FUNCTION
double turbulence_radial_distance(
    const GridType& grid,
    const typename GridType::coordinate_array_type& q) {
    switch (grid.coordinate_system()) {
        case OrthogonalCoordinateSystem::Cartesian: {
            double radius_squared = 0.0;
            for (int dim = 0; dim < GridType::space_dim; ++dim) {
                radius_squared += q[dim] * q[dim];
            }
            return Kokkos::sqrt(radius_squared);
        }
        case OrthogonalCoordinateSystem::Polar:
        case OrthogonalCoordinateSystem::Spherical:
            return q[0];
    }
    return q[0];
}

/**
 * Grid-bound turbulence-property storage and initialization helper.
 */
template <typename GridType>
struct TurbulenceProperties {
    using grid_type = GridType;
    using scalar_field_type = ScalarField<grid_type>;
    using index_array_type = typename grid_type::index_array_type;
    using coordinate_array_type = typename grid_type::coordinate_array_type;
    using size_type = typename scalar_field_type::size_type;
    using execution_space = typename grid_type::execution_space;

    /**
     * Device-side accessor for coefficient kernels.
     */
    struct Accessor {
        TurbulencePropertyStorageMode storage_mode{
            TurbulencePropertyStorageMode::ConstantOnly
        };
        TurbulencePropertyState constant_state{};
        typename scalar_field_type::RandomAccessAccessor sigma2;
        typename scalar_field_type::RandomAccessAccessor l2d;
        typename scalar_field_type::RandomAccessAccessor lslab;
        typename scalar_field_type::RandomAccessAccessor lc;
        typename scalar_field_type::RandomAccessAccessor d_sigma2_dr;
        typename scalar_field_type::RandomAccessAccessor d_l2d_dr;
        typename scalar_field_type::RandomAccessAccessor d_lslab_dr;
        typename scalar_field_type::RandomAccessAccessor d_lc_dr;

        /**
         * Return turbulence properties at one logical grid point.
         */
        KOKKOS_INLINE_FUNCTION
        TurbulencePropertyState operator()(const index_array_type& indices) const {
            if (storage_mode == TurbulencePropertyStorageMode::ConstantOnly) {
                return constant_state;
            }

            TurbulencePropertyState state;
            state.sigma2 = sigma2(indices);
            state.l2d = l2d(indices);
            state.lslab = lslab(indices);
            state.lc = lc(indices);
            state.d_sigma2_dr = d_sigma2_dr(indices);
            state.d_l2d_dr = d_l2d_dr(indices);
            state.d_lslab_dr = d_lslab_dr(indices);
            state.d_lc_dr = d_lc_dr(indices);
            return state;
        }
    };

    grid_type grid;
    TurbulencePropertyModel model{};
    TurbulencePropertyStorageMode storage_mode{
        TurbulencePropertyStorageMode::ConstantOnly
    };
    TurbulencePropertyState constant_state{};
    scalar_field_type sigma2;
    scalar_field_type l2d;
    scalar_field_type lslab;
    scalar_field_type lc;
    scalar_field_type d_sigma2_dr;
    scalar_field_type d_l2d_dr;
    scalar_field_type d_lslab_dr;
    scalar_field_type d_lc_dr;

    TurbulenceProperties() = default;

    /**
     * Build turbulence properties over a coordinate grid.
     */
    TurbulenceProperties(
        const grid_type& input_grid,
        const TurbulencePropertyModel& input_model,
        const TurbulencePropertyStorageMode requested_storage =
            TurbulencePropertyStorageMode::Auto,
        const char* label = "turbulence")
        : grid(input_grid), model(input_model) {
        static_assert(requires(const grid_type& local_grid,
                               const index_array_type& local_indices) {
                          local_grid.coordinate_tuple(local_indices,
                                                      GridCentering::CellCentered);
                      },
                      "TurbulenceProperties requires a coordinate grid.");
        model.validate();
        constant_state = model.evaluate_from_radius(model.au_in_code_units);
        storage_mode = resolve_storage_mode(requested_storage, model);
        if (storage_mode == TurbulencePropertyStorageMode::Field) {
            allocate_fields(label);
            initialize_fields();
        }
    }

    /**
     * Return true when properties are stored as grid fields.
     */
    bool stores_fields() const {
        return storage_mode == TurbulencePropertyStorageMode::Field;
    }

    /**
     * Return a device-side accessor for coefficient kernels.
     */
    Accessor accessor() const {
        return Accessor{
            storage_mode,
            constant_state,
            sigma2.random_access(),
            l2d.random_access(),
            lslab.random_access(),
            lc.random_access(),
            d_sigma2_dr.random_access(),
            d_l2d_dr.random_access(),
            d_lslab_dr.random_access(),
            d_lc_dr.random_access()
        };
    }

private:
    /**
     * Resolve automatic storage decisions.
     */
    static TurbulencePropertyStorageMode resolve_storage_mode(
        const TurbulencePropertyStorageMode requested_storage,
        const TurbulencePropertyModel& input_model) {
        if (requested_storage == TurbulencePropertyStorageMode::Auto) {
            return input_model.is_constant() ? TurbulencePropertyStorageMode::ConstantOnly
                                             : TurbulencePropertyStorageMode::Field;
        }
        if (requested_storage == TurbulencePropertyStorageMode::ConstantOnly &&
            !input_model.is_constant()) {
            throw std::runtime_error(
                "TurbulenceProperties: non-constant models require field storage.");
        }
        return requested_storage;
    }

    /**
     * Allocate scalar fields with stable labels.
     */
    void allocate_fields(const char* label) {
        const std::string prefix(label);
        sigma2 = scalar_field_type(grid, (prefix + "_sigma2").c_str());
        l2d = scalar_field_type(grid, (prefix + "_l2d").c_str());
        lslab = scalar_field_type(grid, (prefix + "_lslab").c_str());
        lc = scalar_field_type(grid, (prefix + "_lc").c_str());
        d_sigma2_dr = scalar_field_type(grid, (prefix + "_d_sigma2_dr").c_str());
        d_l2d_dr = scalar_field_type(grid, (prefix + "_d_l2d_dr").c_str());
        d_lslab_dr = scalar_field_type(grid, (prefix + "_d_lslab_dr").c_str());
        d_lc_dr = scalar_field_type(grid, (prefix + "_d_lc_dr").c_str());
    }

public:
    /*
     * CUDA extended lambdas require this field-initialization kernel helper to be public.
     */

    /**
     * Fill turbulence-property fields from the configured model.
     */
    void initialize_fields() {
        const auto local_grid = grid;
        const auto local_model = model;
        auto sigma2_data = sigma2.data;
        auto l2d_data = l2d.data;
        auto lslab_data = lslab.data;
        auto lc_data = lc.data;
        auto d_sigma2_dr_data = d_sigma2_dr.data;
        auto d_l2d_dr_data = d_l2d_dr.data;
        auto d_lslab_dr_data = d_lslab_dr.data;
        auto d_lc_dr_data = d_lc_dr.data;
        const scalar_field_type local_sigma2 = sigma2;
        Kokkos::parallel_for(
            "TurbulenceProperties::initialize_fields",
            Kokkos::RangePolicy<execution_space>(0, sigma2.point_count()),
            KOKKOS_LAMBDA(const size_type point_index) {
                const index_array_type indices = local_sigma2.logical_indices(point_index);
                const coordinate_array_type q =
                    local_grid.coordinate_tuple(indices, GridCentering::CellCentered);
                const double radius = turbulence_radial_distance(local_grid, q);
                const TurbulencePropertyState state =
                    local_model.evaluate_from_radius(radius);
                sigma2_data(point_index, 0) = state.sigma2;
                l2d_data(point_index, 0) = state.l2d;
                lslab_data(point_index, 0) = state.lslab;
                lc_data(point_index, 0) = state.lc;
                d_sigma2_dr_data(point_index, 0) = state.d_sigma2_dr;
                d_l2d_dr_data(point_index, 0) = state.d_l2d_dr;
                d_lslab_dr_data(point_index, 0) = state.d_lslab_dr;
                d_lc_dr_data(point_index, 0) = state.d_lc_dr;
            });
        Kokkos::fence("TurbulenceProperties::initialize_fields");
    }
};
