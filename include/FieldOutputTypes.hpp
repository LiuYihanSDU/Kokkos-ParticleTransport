#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/**
 * Host-side field output helpers and serializable snapshot types.
 */
namespace FieldOutput {

/**
 * Host-side unit metadata carried by serializable field output snapshots.
 */
struct FieldOutputMetadata {
    std::string coordinate_unit{"code_length"};
    std::string value_unit{"code_value"};
    double coordinate_scale_cgs{1.0};
    double value_scale_cgs{1.0};
};

/**
 * Serializable field snapshot prepared for visualization output.
 */
struct FieldOutputSnapshot {
    std::string field_name{"field"};
    std::string coordinate_unit{"code_length"};
    std::string value_unit{"code_value"};
    std::int32_t space_dim{0};
    std::int32_t component_dim{0};
    std::int32_t output_component_dim{0};
    std::int32_t coordinate_system{0};
    std::int32_t domain{0};
    double coordinate_scale_cgs{1.0};
    double value_scale_cgs{1.0};
    std::array<std::int32_t, 3> dimensions{1, 1, 1};
    std::array<std::int32_t, 3> lower_indices{0, 0, 0};
    std::array<std::int32_t, 3> upper_indices{0, 0, 0};
    std::array<std::int32_t, 3> centerings{0, 0, 0};
    std::vector<double> points;
    std::vector<double> values;

    /**
     * Return the number of structured points in this snapshot.
     */
    [[nodiscard]] std::size_t point_count() const {
        return static_cast<std::size_t>(dimensions[0]) *
               static_cast<std::size_t>(dimensions[1]) *
               static_cast<std::size_t>(dimensions[2]);
    }

    /**
     * Return true when the output payload should be interpreted as a vector field.
     */
    [[nodiscard]] bool is_vector() const {
        return output_component_dim > 1;
    }
};

} // namespace FieldOutput
