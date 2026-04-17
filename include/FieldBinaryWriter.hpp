#pragma once

#include <string>

#include "FieldBinaryIO.hpp"
#include "FieldOutputCommon.hpp"

/**
 * Field binary writer entry points.
 */
namespace FieldOutput {

/**
 * Write a field to the repository custom binary snapshot format.
 */
template <typename FieldType>
void write_field_binary(const FieldType& field,
                        const std::string& file_path,
                        const std::string& field_name = "field",
                        const GridDomain domain = GridDomain::PhysicalDomain,
                        const FieldOutputMetadata& metadata = FieldOutputMetadata{}) {
    auto snapshot = make_field_output_snapshot(field, field_name, domain);
    snapshot.coordinate_unit = metadata.coordinate_unit;
    snapshot.value_unit = metadata.value_unit;
    snapshot.coordinate_scale_cgs = metadata.coordinate_scale_cgs;
    snapshot.value_scale_cgs = metadata.value_scale_cgs;
    write_field_binary_snapshot(snapshot, file_path);
}

} // namespace FieldOutput
