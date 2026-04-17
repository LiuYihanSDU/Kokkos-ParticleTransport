#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "FieldOutputTypes.hpp"

/**
 * Binary serialization helpers for FieldOutputSnapshot.
 */
namespace FieldOutput {

inline constexpr std::array<char, 8> field_binary_magic{
    'K', 'P', 'T', 'F', 'L', 'D', '1', '\0'
};
inline constexpr std::uint32_t field_binary_version = 2;
inline constexpr std::uint32_t field_binary_min_supported_version = 1;
inline constexpr std::uint32_t field_binary_endian_marker = 0x01020304u;

/**
 * Write one trivially copyable scalar to a binary stream.
 */
template <typename T>
void write_binary_scalar(std::ofstream& stream, const T value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!stream) {
        throw std::runtime_error("FieldBinaryIO: failed to write binary scalar.");
    }
}

/**
 * Read one trivially copyable scalar from a binary stream.
 */
template <typename T>
T read_binary_scalar(std::ifstream& stream) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream) {
        throw std::runtime_error("FieldBinaryIO: failed to read binary scalar.");
    }
    return value;
}

/**
 * Write a double vector to a binary stream.
 */
inline void write_double_vector(std::ofstream& stream, const std::vector<double>& values) {
    if (!values.empty()) {
        stream.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(double)));
        if (!stream) {
            throw std::runtime_error("FieldBinaryIO: failed to write binary payload.");
        }
    }
}

/**
 * Read a fixed-size double vector from a binary stream.
 */
inline std::vector<double> read_double_vector(std::ifstream& stream, const std::size_t count) {
    std::vector<double> values(count);
    if (!values.empty()) {
        stream.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(double)));
        if (!stream) {
            throw std::runtime_error("FieldBinaryIO: failed to read binary payload.");
        }
    }
    return values;
}

/**
 * Write a string payload to a binary stream.
 */
inline void write_binary_string(std::ofstream& stream, const std::string& value) {
    if (!value.empty()) {
        stream.write(value.data(), static_cast<std::streamsize>(value.size()));
        if (!stream) {
            throw std::runtime_error("FieldBinaryIO: failed to write string payload.");
        }
    }
}

/**
 * Read a fixed-size string payload from a binary stream.
 */
inline std::string read_binary_string(std::ifstream& stream, const std::uint64_t size) {
    if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("FieldBinaryIO: binary string is too large.");
    }
    std::string value;
    value.resize(static_cast<std::size_t>(size));
    if (size > 0) {
        stream.read(value.data(), static_cast<std::streamsize>(size));
        if (!stream) {
            throw std::runtime_error("FieldBinaryIO: failed to read string payload.");
        }
    }
    return value;
}

/**
 * Validate snapshot dimensions before writing.
 */
inline void validate_field_output_snapshot(const FieldOutputSnapshot& snapshot) {
    if (snapshot.space_dim < 1 || snapshot.space_dim > 3) {
        throw std::runtime_error("FieldBinaryIO: snapshot SpaceDim must be between 1 and 3.");
    }
    if (snapshot.output_component_dim != 1 && snapshot.output_component_dim != 3) {
        throw std::runtime_error(
            "FieldBinaryIO: output component count must be either 1 or 3.");
    }
    if (!std::isfinite(snapshot.coordinate_scale_cgs) ||
        snapshot.coordinate_scale_cgs <= 0.0 ||
        !std::isfinite(snapshot.value_scale_cgs) ||
        snapshot.value_scale_cgs <= 0.0) {
        throw std::runtime_error("FieldBinaryIO: snapshot CGS scales must be positive.");
    }
    for (const auto dimension : snapshot.dimensions) {
        if (dimension <= 0) {
            throw std::runtime_error("FieldBinaryIO: snapshot dimensions must be positive.");
        }
    }

    const std::size_t point_count = snapshot.point_count();
    if (snapshot.points.size() != point_count * 3) {
        throw std::runtime_error("FieldBinaryIO: point payload size does not match metadata.");
    }
    if (snapshot.values.size() !=
        point_count * static_cast<std::size_t>(snapshot.output_component_dim)) {
        throw std::runtime_error("FieldBinaryIO: value payload size does not match metadata.");
    }
}

/**
 * Write a serializable field snapshot to a custom binary file.
 */
inline void write_field_binary_snapshot(const FieldOutputSnapshot& snapshot,
                                        const std::string& file_path) {
    validate_field_output_snapshot(snapshot);

    std::ofstream stream(file_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("FieldBinaryIO: failed to open binary output file.");
    }

    stream.write(field_binary_magic.data(),
                 static_cast<std::streamsize>(field_binary_magic.size()));
    if (!stream) {
        throw std::runtime_error("FieldBinaryIO: failed to write binary magic.");
    }

    write_binary_scalar(stream, field_binary_version);
    write_binary_scalar(stream, field_binary_endian_marker);
    write_binary_scalar(stream, snapshot.space_dim);
    write_binary_scalar(stream, snapshot.component_dim);
    write_binary_scalar(stream, snapshot.output_component_dim);
    write_binary_scalar(stream, snapshot.coordinate_system);
    write_binary_scalar(stream, snapshot.domain);
    for (const auto value : snapshot.dimensions) {
        write_binary_scalar(stream, value);
    }
    for (const auto value : snapshot.lower_indices) {
        write_binary_scalar(stream, value);
    }
    for (const auto value : snapshot.upper_indices) {
        write_binary_scalar(stream, value);
    }
    for (const auto value : snapshot.centerings) {
        write_binary_scalar(stream, value);
    }

    const auto point_count = static_cast<std::uint64_t>(snapshot.point_count());
    const auto name_size = static_cast<std::uint64_t>(snapshot.field_name.size());
    const auto coordinate_unit_size =
        static_cast<std::uint64_t>(snapshot.coordinate_unit.size());
    const auto value_unit_size = static_cast<std::uint64_t>(snapshot.value_unit.size());
    write_binary_scalar(stream, point_count);
    write_binary_scalar(stream, name_size);
    write_binary_scalar(stream, coordinate_unit_size);
    write_binary_scalar(stream, value_unit_size);
    write_binary_scalar(stream, snapshot.coordinate_scale_cgs);
    write_binary_scalar(stream, snapshot.value_scale_cgs);
    write_binary_string(stream, snapshot.field_name);
    write_binary_string(stream, snapshot.coordinate_unit);
    write_binary_string(stream, snapshot.value_unit);

    write_double_vector(stream, snapshot.points);
    write_double_vector(stream, snapshot.values);
}

/**
 * Read a serializable field snapshot from a custom binary file.
 */
inline FieldOutputSnapshot read_field_binary_snapshot(const std::string& file_path) {
    std::ifstream stream(file_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("FieldBinaryIO: failed to open binary input file.");
    }

    std::array<char, 8> magic{};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!stream || magic != field_binary_magic) {
        throw std::runtime_error("FieldBinaryIO: unrecognized field binary file.");
    }

    const auto version = read_binary_scalar<std::uint32_t>(stream);
    if (version < field_binary_min_supported_version || version > field_binary_version) {
        throw std::runtime_error("FieldBinaryIO: unsupported field binary version.");
    }
    const auto endian_marker = read_binary_scalar<std::uint32_t>(stream);
    if (endian_marker != field_binary_endian_marker) {
        throw std::runtime_error("FieldBinaryIO: unsupported binary endian marker.");
    }

    FieldOutputSnapshot snapshot;
    snapshot.space_dim = read_binary_scalar<std::int32_t>(stream);
    snapshot.component_dim = read_binary_scalar<std::int32_t>(stream);
    snapshot.output_component_dim = read_binary_scalar<std::int32_t>(stream);
    snapshot.coordinate_system = read_binary_scalar<std::int32_t>(stream);
    snapshot.domain = read_binary_scalar<std::int32_t>(stream);
    for (auto& value : snapshot.dimensions) {
        value = read_binary_scalar<std::int32_t>(stream);
    }
    for (auto& value : snapshot.lower_indices) {
        value = read_binary_scalar<std::int32_t>(stream);
    }
    for (auto& value : snapshot.upper_indices) {
        value = read_binary_scalar<std::int32_t>(stream);
    }
    for (auto& value : snapshot.centerings) {
        value = read_binary_scalar<std::int32_t>(stream);
    }

    const auto point_count = read_binary_scalar<std::uint64_t>(stream);
    const auto name_size = read_binary_scalar<std::uint64_t>(stream);
    std::uint64_t coordinate_unit_size = 0;
    std::uint64_t value_unit_size = 0;
    if (version >= 2) {
        coordinate_unit_size = read_binary_scalar<std::uint64_t>(stream);
        value_unit_size = read_binary_scalar<std::uint64_t>(stream);
        snapshot.coordinate_scale_cgs = read_binary_scalar<double>(stream);
        snapshot.value_scale_cgs = read_binary_scalar<double>(stream);
    }
    if (snapshot.output_component_dim != 1 && snapshot.output_component_dim != 3) {
        throw std::runtime_error(
            "FieldBinaryIO: unsupported binary output component count.");
    }
    for (const auto dimension : snapshot.dimensions) {
        if (dimension <= 0) {
            throw std::runtime_error("FieldBinaryIO: binary dimensions must be positive.");
        }
    }
    if (point_count >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / 3)) {
        throw std::runtime_error("FieldBinaryIO: binary point count is too large.");
    }
    if (name_size >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("FieldBinaryIO: binary field name is too large.");
    }
    if (coordinate_unit_size >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        value_unit_size >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("FieldBinaryIO: binary unit label is too large.");
    }

    snapshot.field_name = read_binary_string(stream, name_size);
    if (version >= 2) {
        snapshot.coordinate_unit = read_binary_string(stream, coordinate_unit_size);
        snapshot.value_unit = read_binary_string(stream, value_unit_size);
    }

    if (snapshot.point_count() != static_cast<std::size_t>(point_count)) {
        throw std::runtime_error("FieldBinaryIO: binary point count does not match dimensions.");
    }

    snapshot.points = read_double_vector(stream, static_cast<std::size_t>(point_count) * 3);
    snapshot.values = read_double_vector(
        stream,
        static_cast<std::size_t>(point_count) *
            static_cast<std::size_t>(snapshot.output_component_dim));
    validate_field_output_snapshot(snapshot);
    return snapshot;
}

} // namespace FieldOutput
