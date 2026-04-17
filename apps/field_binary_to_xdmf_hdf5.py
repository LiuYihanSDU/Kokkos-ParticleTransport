#!/usr/bin/env python3
"""Convert KokkosParticleTransport field binary snapshots to HDF5 plus XDMF."""

from __future__ import annotations

import argparse
import html
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


FIELD_BINARY_MAGIC = b"KPTFLD1\0"
FIELD_BINARY_VERSION = 2
FIELD_BINARY_MIN_SUPPORTED_VERSION = 1
FIELD_BINARY_ENDIAN_MARKER = 0x01020304


@dataclass
class FieldOutputSnapshot:
    """Host-side field snapshot loaded from the repository binary format."""

    field_name: str
    coordinate_unit: str
    value_unit: str
    space_dim: int
    component_dim: int
    output_component_dim: int
    coordinate_system: int
    domain: int
    coordinate_scale_cgs: float
    value_scale_cgs: float
    dimensions: tuple[int, int, int]
    lower_indices: tuple[int, int, int]
    upper_indices: tuple[int, int, int]
    centerings: tuple[int, int, int]
    points: object
    values: object

    @property
    def point_count(self) -> int:
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

    @property
    def is_vector(self) -> bool:
        return self.output_component_dim > 1


class BinaryReader:
    """Small binary reader with explicit endian handling."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = path.open("rb")
        self.endian = "<"

    def close(self) -> None:
        self.file.close()

    def read_exact(self, size: int) -> bytes:
        data = self.file.read(size)
        if len(data) != size:
            raise RuntimeError("unexpected end of binary snapshot")
        return data

    def read_scalar(self, fmt: str):
        return struct.unpack(self.endian + fmt, self.read_exact(struct.calcsize(fmt)))[0]

    def read_string(self, size: int) -> str:
        return self.read_exact(size).decode("utf-8")


def detect_endian(version_bytes: bytes, marker_bytes: bytes) -> tuple[str, int]:
    """Return file endian and decoded binary version."""
    marker_little = struct.unpack("<I", marker_bytes)[0]
    if marker_little == FIELD_BINARY_ENDIAN_MARKER:
        return "<", struct.unpack("<I", version_bytes)[0]
    if marker_little == 0x04030201:
        return ">", struct.unpack(">I", version_bytes)[0]
    raise RuntimeError("unsupported binary endian marker")


def read_field_binary_snapshot(path: Path, np_module) -> FieldOutputSnapshot:
    """Read a v1 or v2 KokkosParticleTransport binary field snapshot."""
    reader = BinaryReader(path)
    try:
        if reader.read_exact(len(FIELD_BINARY_MAGIC)) != FIELD_BINARY_MAGIC:
            raise RuntimeError("unrecognized field binary magic")

        version_bytes = reader.read_exact(4)
        marker_bytes = reader.read_exact(4)
        reader.endian, version = detect_endian(version_bytes, marker_bytes)
        if not (FIELD_BINARY_MIN_SUPPORTED_VERSION <= version <= FIELD_BINARY_VERSION):
            raise RuntimeError(f"unsupported field binary version: {version}")

        space_dim = reader.read_scalar("i")
        component_dim = reader.read_scalar("i")
        output_component_dim = reader.read_scalar("i")
        coordinate_system = reader.read_scalar("i")
        domain = reader.read_scalar("i")
        dimensions = tuple(reader.read_scalar("i") for _ in range(3))
        lower_indices = tuple(reader.read_scalar("i") for _ in range(3))
        upper_indices = tuple(reader.read_scalar("i") for _ in range(3))
        centerings = tuple(reader.read_scalar("i") for _ in range(3))
        point_count = reader.read_scalar("Q")
        field_name_size = reader.read_scalar("Q")

        coordinate_unit = "code_length"
        value_unit = "code_value"
        coordinate_scale_cgs = 1.0
        value_scale_cgs = 1.0
        coordinate_unit_size = 0
        value_unit_size = 0
        if version >= 2:
            coordinate_unit_size = reader.read_scalar("Q")
            value_unit_size = reader.read_scalar("Q")
            coordinate_scale_cgs = reader.read_scalar("d")
            value_scale_cgs = reader.read_scalar("d")

        field_name = reader.read_string(field_name_size)
        if version >= 2:
            coordinate_unit = reader.read_string(coordinate_unit_size)
            value_unit = reader.read_string(value_unit_size)

        validate_metadata(space_dim, output_component_dim, dimensions,
                          point_count, coordinate_scale_cgs, value_scale_cgs)

        point_values = point_count * 3
        field_values = point_count * output_component_dim
        points = np_module.frombuffer(reader.read_exact(point_values * 8),
                                      dtype=np_module.dtype(reader.endian + "f8")).copy()
        values = np_module.frombuffer(reader.read_exact(field_values * 8),
                                      dtype=np_module.dtype(reader.endian + "f8")).copy()
        return FieldOutputSnapshot(
            field_name=field_name,
            coordinate_unit=coordinate_unit,
            value_unit=value_unit,
            space_dim=space_dim,
            component_dim=component_dim,
            output_component_dim=output_component_dim,
            coordinate_system=coordinate_system,
            domain=domain,
            coordinate_scale_cgs=coordinate_scale_cgs,
            value_scale_cgs=value_scale_cgs,
            dimensions=dimensions,
            lower_indices=lower_indices,
            upper_indices=upper_indices,
            centerings=centerings,
            points=points,
            values=values,
        )
    finally:
        reader.close()


def validate_metadata(space_dim: int, output_component_dim: int,
                      dimensions: tuple[int, int, int], point_count: int,
                      coordinate_scale_cgs: float, value_scale_cgs: float) -> None:
    """Validate snapshot metadata before allocating payload arrays."""
    if space_dim < 1 or space_dim > 3:
        raise RuntimeError("snapshot SpaceDim must be between 1 and 3")
    if output_component_dim not in (1, 3):
        raise RuntimeError("output component count must be either 1 or 3")
    if any(dimension <= 0 for dimension in dimensions):
        raise RuntimeError("snapshot dimensions must be positive")
    expected_point_count = dimensions[0] * dimensions[1] * dimensions[2]
    if expected_point_count != point_count:
        raise RuntimeError("binary point count does not match dimensions")
    if not math.isfinite(coordinate_scale_cgs) or coordinate_scale_cgs <= 0.0:
        raise RuntimeError("coordinate_scale_cgs must be positive and finite")
    if not math.isfinite(value_scale_cgs) or value_scale_cgs <= 0.0:
        raise RuntimeError("value_scale_cgs must be positive and finite")


def parse_positive_scale(value: str) -> float:
    """Parse a positive finite scale factor."""
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise argparse.ArgumentTypeError("scale must be positive and finite")
    return scale


def dimensionalize_snapshot(snapshot: FieldOutputSnapshot,
                            coordinate_scale_override: float | None,
                            value_scale_override: float | None,
                            coordinate_unit_override: str | None,
                            value_unit_override: str | None,
                            field_name_override: str | None,
                            keep_code_units: bool) -> FieldOutputSnapshot:
    """Apply metadata-derived or explicit CGS scale factors to host arrays."""
    coordinate_scale = 1.0 if keep_code_units else snapshot.coordinate_scale_cgs
    value_scale = 1.0 if keep_code_units else snapshot.value_scale_cgs
    if coordinate_scale_override is not None:
        coordinate_scale = coordinate_scale_override
    if value_scale_override is not None:
        value_scale = value_scale_override

    snapshot.points *= coordinate_scale
    snapshot.values *= value_scale
    snapshot.coordinate_scale_cgs = coordinate_scale
    snapshot.value_scale_cgs = value_scale
    if field_name_override is not None:
        snapshot.field_name = field_name_override
    if coordinate_unit_override is not None:
        snapshot.coordinate_unit = coordinate_unit_override
    elif not keep_code_units and coordinate_scale != 1.0:
        snapshot.coordinate_unit = "cm"
    if value_unit_override is not None:
        snapshot.value_unit = value_unit_override
    elif not keep_code_units and value_scale != 1.0:
        snapshot.value_unit = "cgs_value"
    if keep_code_units and coordinate_unit_override is None:
        snapshot.coordinate_unit = "code_length"
    if keep_code_units and value_unit_override is None:
        snapshot.value_unit = "code_value"
    return snapshot


def xdmf_point_dimensions(snapshot: FieldOutputSnapshot) -> str:
    return f"{snapshot.dimensions[2]} {snapshot.dimensions[1]} {snapshot.dimensions[0]} 3"


def xdmf_value_dimensions(snapshot: FieldOutputSnapshot) -> str:
    base = f"{snapshot.dimensions[2]} {snapshot.dimensions[1]} {snapshot.dimensions[0]}"
    if snapshot.is_vector:
        return f"{base} {snapshot.output_component_dim}"
    return base


def write_hdf5_xdmf(snapshot: FieldOutputSnapshot, hdf5_path: Path, xdmf_path: Path,
                    h5py_module, np_module) -> None:
    """Write the loaded snapshot as ParaView-readable HDF5 plus XDMF."""
    nx, ny, nz = snapshot.dimensions
    points = np_module.asarray(snapshot.points, dtype=np_module.float64).reshape((nz, ny, nx, 3))
    if snapshot.is_vector:
        values = np_module.asarray(snapshot.values, dtype=np_module.float64).reshape(
            (nz, ny, nx, snapshot.output_component_dim))
    else:
        values = np_module.asarray(snapshot.values, dtype=np_module.float64).reshape((nz, ny, nx))

    with h5py_module.File(hdf5_path, "w") as h5_file:
        h5_file.create_dataset("points", data=points)
        h5_file.create_dataset("values", data=values)
        h5_file.create_dataset("dimensions",
                               data=np_module.asarray(snapshot.dimensions, dtype=np_module.int32))
        h5_file.create_dataset("lower_indices",
                               data=np_module.asarray(snapshot.lower_indices, dtype=np_module.int32))
        h5_file.create_dataset("upper_indices",
                               data=np_module.asarray(snapshot.upper_indices, dtype=np_module.int32))
        h5_file.create_dataset("centerings",
                               data=np_module.asarray(snapshot.centerings, dtype=np_module.int32))
        h5_file.create_dataset("coordinate_scale_cgs",
                               data=np_module.asarray([snapshot.coordinate_scale_cgs],
                                                      dtype=np_module.float64))
        h5_file.create_dataset("value_scale_cgs",
                               data=np_module.asarray([snapshot.value_scale_cgs],
                                                      dtype=np_module.float64))
        h5_file.attrs["field_name"] = snapshot.field_name
        h5_file.attrs["coordinate_unit"] = snapshot.coordinate_unit
        h5_file.attrs["value_unit"] = snapshot.value_unit

    write_xdmf(snapshot, hdf5_path, xdmf_path)


def write_xdmf(snapshot: FieldOutputSnapshot, hdf5_path: Path, xdmf_path: Path) -> None:
    """Write an XDMF sidecar referencing the HDF5 heavy-data file."""
    escaped_name = html.escape(snapshot.field_name, quote=True)
    escaped_hdf5 = html.escape(str(hdf5_path), quote=True)
    escaped_coordinate_unit = html.escape(snapshot.coordinate_unit, quote=True)
    escaped_value_unit = html.escape(snapshot.value_unit, quote=True)
    topology_dimensions = f"{snapshot.dimensions[2]} {snapshot.dimensions[1]} {snapshot.dimensions[0]}"
    attribute_type = "Vector" if snapshot.is_vector else "Scalar"
    xdmf_path.write_text(
        f"""<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="{escaped_name}" GridType="Uniform">
      <Information Name="CoordinateUnit" Value="{escaped_coordinate_unit}"/>
      <Information Name="ValueUnit" Value="{escaped_value_unit}"/>
      <Information Name="CoordinateScaleCgs" Value="{snapshot.coordinate_scale_cgs}"/>
      <Information Name="ValueScaleCgs" Value="{snapshot.value_scale_cgs}"/>
      <Topology TopologyType="3DSMesh" Dimensions="{topology_dimensions}"/>
      <Geometry GeometryType="XYZ">
        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="{xdmf_point_dimensions(snapshot)}">{escaped_hdf5}:/points</DataItem>
      </Geometry>
      <Attribute Name="{escaped_name}" AttributeType="{attribute_type}" Center="Node">
        <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="{xdmf_value_dimensions(snapshot)}">{escaped_hdf5}:/values</DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
""",
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse converter command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a KokkosParticleTransport field binary snapshot to HDF5 plus XDMF."
    )
    parser.add_argument("input", type=Path, help="Input .kptbin field snapshot.")
    parser.add_argument("hdf5", type=Path, help="Output .h5 file.")
    parser.add_argument("xdmf", type=Path, help="Output .xdmf sidecar file.")
    parser.add_argument("--coordinate-scale-cgs", type=parse_positive_scale,
                        help="Override the coordinate CGS scale stored in the binary file.")
    parser.add_argument("--value-scale-cgs", type=parse_positive_scale,
                        help="Override the field-value CGS scale stored in the binary file.")
    parser.add_argument("--coordinate-unit",
                        help="Override the coordinate unit label written to XDMF/HDF5 metadata.")
    parser.add_argument("--value-unit",
                        help="Override the field-value unit label written to XDMF/HDF5 metadata.")
    parser.add_argument("--field-name",
                        help="Override the field name stored in the binary snapshot.")
    parser.add_argument("--keep-code-units", action="store_true",
                        help="Write raw code-unit payloads instead of applying CGS scale metadata.")
    return parser.parse_args()


def load_hdf5_modules():
    """Load Python HDF5 dependencies only when conversion is actually requested."""
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Python HDF5 conversion requires h5py and numpy. "
                           "Install them with `python3 -m pip install h5py numpy`.") from exc
    return h5py, np


def main() -> int:
    args = parse_arguments()
    try:
        h5py_module, np_module = load_hdf5_modules()
        snapshot = read_field_binary_snapshot(args.input, np_module)
        dimensionalize_snapshot(
            snapshot,
            args.coordinate_scale_cgs,
            args.value_scale_cgs,
            args.coordinate_unit,
            args.value_unit,
            args.field_name,
            args.keep_code_units,
        )
        write_hdf5_xdmf(snapshot, args.hdf5, args.xdmf, h5py_module, np_module)
    except Exception as exc:
        print(f"field_binary_to_xdmf_hdf5.py failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
