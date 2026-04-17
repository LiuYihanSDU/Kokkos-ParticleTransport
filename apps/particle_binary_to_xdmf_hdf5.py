#!/usr/bin/env python3
"""Convert KokkosParticleTransport particle binary snapshots to HDF5 plus XDMF."""

from __future__ import annotations

import argparse
import html
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


PARTICLE_BINARY_MAGIC = b"KPTPRT\0\0"
LEGACY_PARTICLE_BINARY_MAGIC_V2 = b"KPTPRT1\0"
PARTICLE_BINARY_VERSION = 4
PARTICLE_BINARY_MIN_SUPPORTED_VERSION = 2
PARTICLE_BINARY_ENDIAN_MARKER = 0x01020304
ELECTRON_VOLT_ERG = 1.602176634e-12
PARTICLE_STATUS_ACTIVE = 0
PARTICLE_STATUS_ESCAPED = 1
PARTICLE_STATUS_INACTIVE = 2


@dataclass
class ParticleSnapshot:
    """Host-side particle snapshot loaded from the repository binary format."""

    version: int
    space_dim: int
    species: int
    particle_count: int
    particle_capacity: int
    rest_mass: float
    charge: float
    speed_of_light: float
    energy_scale_erg: float
    energy_split_ratio: float
    minimum_child_weight: float
    particle_id: object
    status: object
    split_level: object
    sort_key: object
    position: object
    previous_position: object
    previous_step_position: object
    momentum: object
    momentum_magnitude: object
    momentum_direction_available: bool
    mu: object
    weight: object
    initial_kinetic_energy: object
    initial_kinetic_energy_ev: object
    gamma: object
    speed: object
    kinetic_energy: object
    kinetic_energy_ev: object
    is_active: object
    is_escaped: object
    is_inactive: object
    position_unit: str = "code_length"
    momentum_unit: str = "code_momentum"


@dataclass(frozen=True)
class ParticleScalarAttributeSpec:
    """XDMF metadata for one scalar particle attribute."""

    name: str
    number_type: str = "Float"
    precision: int = 8


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
            raise RuntimeError("unexpected end of particle binary snapshot")
        return data

    def read_scalar(self, fmt: str):
        return struct.unpack(self.endian + fmt, self.read_exact(struct.calcsize(fmt)))[0]


def detect_endian(version_bytes: bytes, marker_bytes: bytes) -> tuple[str, int]:
    """Return file endian and decoded binary version."""
    marker_little = struct.unpack("<I", marker_bytes)[0]
    if marker_little == PARTICLE_BINARY_ENDIAN_MARKER:
        return "<", struct.unpack("<I", version_bytes)[0]
    if marker_little == 0x04030201:
        return ">", struct.unpack(">I", version_bytes)[0]
    raise RuntimeError("unsupported particle binary endian marker")


def validate_metadata(space_dim: int,
                      particle_count: int,
                      particle_capacity: int,
                      rest_mass: float,
                      speed_of_light: float,
                      energy_scale_erg: float) -> None:
    """Validate snapshot metadata before allocating payload arrays."""
    if space_dim < 1 or space_dim > 3:
        raise RuntimeError("particle snapshot SpaceDim must be between 1 and 3")
    if particle_count < 0:
        raise RuntimeError("particle count must be non-negative")
    if particle_capacity < particle_count:
        raise RuntimeError("particle capacity must be greater than or equal to count")
    if not math.isfinite(rest_mass) or rest_mass <= 0.0:
        raise RuntimeError("particle rest mass must be positive and finite")
    if not math.isfinite(speed_of_light) or speed_of_light <= 0.0:
        raise RuntimeError("particle speed of light must be positive and finite")
    if not math.isfinite(energy_scale_erg) or energy_scale_erg <= 0.0:
        raise RuntimeError("particle energy scale must be positive and finite")


def read_particle_binary_snapshot(path: Path, np_module) -> ParticleSnapshot:
    """Read a v2, v3, or v4 KokkosParticleTransport particle binary snapshot."""
    reader = BinaryReader(path)
    try:
        magic = reader.read_exact(8)
        if magic not in (PARTICLE_BINARY_MAGIC, LEGACY_PARTICLE_BINARY_MAGIC_V2):
            raise RuntimeError("unrecognized particle binary magic")

        version_bytes = reader.read_exact(4)
        marker_bytes = reader.read_exact(4)
        reader.endian, version = detect_endian(version_bytes, marker_bytes)
        if not (PARTICLE_BINARY_MIN_SUPPORTED_VERSION <= version <= PARTICLE_BINARY_VERSION):
            raise RuntimeError(f"unsupported particle binary version: {version}")
        if magic == LEGACY_PARTICLE_BINARY_MAGIC_V2 and version != 2:
            raise RuntimeError("legacy particle magic is only valid for version 2")
        if magic == PARTICLE_BINARY_MAGIC and version < 3:
            raise RuntimeError("version 2 particle files must use legacy particle magic")

        space_dim = reader.read_scalar("i")
        species = reader.read_scalar("i")
        particle_count = reader.read_scalar("Q")
        particle_capacity = particle_count
        if version >= 3:
            particle_capacity = reader.read_scalar("Q")
        rest_mass = reader.read_scalar("d")
        charge = reader.read_scalar("d")
        speed_of_light = reader.read_scalar("d")
        energy_scale_erg = reader.read_scalar("d")
        energy_split_ratio = 2.0
        minimum_child_weight = 0.0
        if version >= 3:
            energy_split_ratio = reader.read_scalar("d")
            minimum_child_weight = reader.read_scalar("d")

        validate_metadata(space_dim, particle_count, particle_capacity, rest_mass,
                          speed_of_light, energy_scale_erg)

        particle_id = np_module.empty(particle_count, dtype=np_module.uint64)
        status = np_module.empty(particle_count, dtype=np_module.int32)
        split_level = np_module.empty(particle_count, dtype=np_module.int32)
        sort_key = np_module.empty(particle_count, dtype=np_module.int32)
        position = np_module.zeros((particle_count, 3), dtype=np_module.float64)
        previous_position = np_module.zeros((particle_count, 3), dtype=np_module.float64)
        previous_step_position = np_module.zeros((particle_count, 3), dtype=np_module.float64)
        momentum = np_module.zeros((particle_count, 3), dtype=np_module.float64)
        momentum_magnitude = np_module.empty(particle_count, dtype=np_module.float64)
        mu = np_module.empty(particle_count, dtype=np_module.float64)
        weight = np_module.empty(particle_count, dtype=np_module.float64)
        initial_kinetic_energy = np_module.empty(particle_count, dtype=np_module.float64)

        for index in range(particle_count):
            particle_id[index] = reader.read_scalar("Q")
            status[index] = reader.read_scalar("i")
            split_level[index] = reader.read_scalar("i")
            sort_key[index] = reader.read_scalar("i")
            for dim in range(space_dim):
                position[index, dim] = reader.read_scalar("d")
            for dim in range(space_dim):
                previous_position[index, dim] = reader.read_scalar("d")
            if version >= 3:
                for dim in range(space_dim):
                    previous_step_position[index, dim] = reader.read_scalar("d")
            else:
                previous_step_position[index, :] = previous_position[index, :]
            if version >= 4:
                momentum_magnitude[index] = reader.read_scalar("d")
            else:
                for dim in range(space_dim):
                    momentum[index, dim] = reader.read_scalar("d")
            mu[index] = reader.read_scalar("d")
            weight[index] = reader.read_scalar("d")
            initial_kinetic_energy[index] = reader.read_scalar("d")

        momentum_direction_available = version < 4
        if momentum_direction_available:
            momentum_magnitude = compute_momentum_magnitude(momentum, np_module)
        gamma, speed, kinetic_energy, kinetic_energy_ev = compute_kinematics_from_magnitude(
            momentum_magnitude, rest_mass, speed_of_light, energy_scale_erg, np_module)
        initial_kinetic_energy_ev = initial_kinetic_energy * energy_scale_erg / ELECTRON_VOLT_ERG
        is_active, is_escaped, is_inactive = compute_status_flags(status, np_module)

        return ParticleSnapshot(
            version=version,
            space_dim=space_dim,
            species=species,
            particle_count=particle_count,
            particle_capacity=particle_capacity,
            rest_mass=rest_mass,
            charge=charge,
            speed_of_light=speed_of_light,
            energy_scale_erg=energy_scale_erg,
            energy_split_ratio=energy_split_ratio,
            minimum_child_weight=minimum_child_weight,
            particle_id=particle_id,
            status=status,
            split_level=split_level,
            sort_key=sort_key,
            position=position,
            previous_position=previous_position,
            previous_step_position=previous_step_position,
            momentum=momentum,
            momentum_magnitude=momentum_magnitude,
            momentum_direction_available=momentum_direction_available,
            mu=mu,
            weight=weight,
            initial_kinetic_energy=initial_kinetic_energy,
            initial_kinetic_energy_ev=initial_kinetic_energy_ev,
            gamma=gamma,
            speed=speed,
            kinetic_energy=kinetic_energy,
            kinetic_energy_ev=kinetic_energy_ev,
            is_active=is_active,
            is_escaped=is_escaped,
            is_inactive=is_inactive,
        )
    finally:
        reader.close()


def compute_momentum_magnitude(momentum, np_module):
    """Compute momentum magnitude from legacy vector momentum snapshots."""
    return np_module.linalg.norm(momentum, axis=1)


def compute_kinematics_from_magnitude(momentum_magnitude, rest_mass: float,
                                      speed_of_light: float,
                                      energy_scale_erg: float, np_module):
    """Compute relativistic diagnostic quantities in code units and eV."""
    normalized_momentum = momentum_magnitude / (rest_mass * speed_of_light)
    gamma = np_module.sqrt(1.0 + normalized_momentum * normalized_momentum)
    speed = momentum_magnitude / (gamma * rest_mass)
    kinetic_energy = (gamma - 1.0) * rest_mass * speed_of_light * speed_of_light
    kinetic_energy_ev = kinetic_energy * energy_scale_erg / ELECTRON_VOLT_ERG
    return gamma, speed, kinetic_energy, kinetic_energy_ev


def compute_status_flags(status, np_module):
    """Build integer status masks for ParaView coloring and threshold filters."""
    is_active = (status == PARTICLE_STATUS_ACTIVE).astype(np_module.int32)
    is_escaped = (status == PARTICLE_STATUS_ESCAPED).astype(np_module.int32)
    is_inactive = (status == PARTICLE_STATUS_INACTIVE).astype(np_module.int32)
    return is_active, is_escaped, is_inactive


def parse_positive_scale(value: str) -> float:
    """Parse a positive finite scale factor."""
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise argparse.ArgumentTypeError("scale must be positive and finite")
    return scale


def apply_output_scales(snapshot: ParticleSnapshot,
                        position_scale: float,
                        position_unit: str,
                        momentum_scale: float,
                        momentum_unit: str) -> ParticleSnapshot:
    """Apply visualization-only scale factors to coordinates and momentum arrays."""
    snapshot.position *= position_scale
    snapshot.previous_position *= position_scale
    snapshot.previous_step_position *= position_scale
    snapshot.momentum *= momentum_scale
    snapshot.momentum_magnitude *= momentum_scale
    snapshot.position_unit = position_unit
    snapshot.momentum_unit = momentum_unit
    return snapshot


def create_dataset(h5_file, name: str, data, unit: str | None = None,
                   description: str | None = None):
    """Create an HDF5 dataset and attach lightweight visualization metadata."""
    dataset = h5_file.create_dataset(name, data=data)
    if unit is not None:
        dataset.attrs["unit"] = unit
    if description is not None:
        dataset.attrs["description"] = description
    return dataset


def write_hdf5_xdmf(snapshot: ParticleSnapshot, hdf5_path: Path, xdmf_path: Path,
                    h5py_module, np_module) -> None:
    """Write the loaded particle snapshot as ParaView-readable HDF5 plus XDMF."""
    with h5py_module.File(hdf5_path, "w") as h5_file:
        create_dataset(h5_file, "position",
                       np_module.asarray(snapshot.position, dtype=np_module.float64),
                       snapshot.position_unit, "Particle position used as XDMF geometry.")
        create_dataset(h5_file, "previous_position",
                       np_module.asarray(snapshot.previous_position, dtype=np_module.float64),
                       snapshot.position_unit, "Particle position at the previous step.")
        create_dataset(h5_file, "previous_step_position",
                       np_module.asarray(snapshot.previous_step_position, dtype=np_module.float64),
                       snapshot.position_unit,
                       "Particle position before the last completed update.")
        create_dataset(h5_file, "momentum",
                       np_module.asarray(snapshot.momentum, dtype=np_module.float64),
                       snapshot.momentum_unit,
                       "Legacy particle momentum vector; zero when the snapshot stores only magnitude.")
        create_dataset(h5_file, "momentum_magnitude", snapshot.momentum_magnitude,
                       snapshot.momentum_unit, "Stored particle momentum magnitude.")
        create_dataset(h5_file, "particle_id", snapshot.particle_id, None,
                       "Stable particle identifier.")
        create_dataset(h5_file, "status", snapshot.status, None,
                       "Lifecycle status: 0 active, 1 escaped, 2 inactive.")
        create_dataset(h5_file, "split_level", snapshot.split_level, None,
                       "Current particle splitting level.")
        create_dataset(h5_file, "sort_key", snapshot.sort_key, None,
                       "Last particle sorting key.")
        create_dataset(h5_file, "mu", snapshot.mu, "dimensionless",
                       "Pitch-angle cosine used by focused-transport runs.")
        create_dataset(h5_file, "weight", snapshot.weight, "dimensionless",
                       "Statistical particle weight.")
        create_dataset(h5_file, "initial_kinetic_energy",
                       snapshot.initial_kinetic_energy, "code_energy",
                       "Baseline kinetic energy in code units.")
        create_dataset(h5_file, "initial_kinetic_energy_ev",
                       snapshot.initial_kinetic_energy_ev, "eV",
                       "Baseline kinetic energy in electron volts.")
        create_dataset(h5_file, "gamma", snapshot.gamma, "dimensionless",
                       "Relativistic Lorentz factor computed from momentum.")
        create_dataset(h5_file, "speed", snapshot.speed, "code_velocity",
                       "Particle speed computed from relativistic momentum.")
        create_dataset(h5_file, "kinetic_energy", snapshot.kinetic_energy,
                       "code_energy", "Current kinetic energy in code units.")
        create_dataset(h5_file, "kinetic_energy_ev", snapshot.kinetic_energy_ev,
                       "eV", "Current kinetic energy in electron volts.")
        create_dataset(h5_file, "is_active", snapshot.is_active, None,
                       "Integer mask equal to 1 for active particles.")
        create_dataset(h5_file, "is_escaped", snapshot.is_escaped, None,
                       "Integer mask equal to 1 for escaped particles.")
        create_dataset(h5_file, "is_inactive", snapshot.is_inactive, None,
                       "Integer mask equal to 1 for inactive particles.")
        h5_file.attrs["version"] = snapshot.version
        h5_file.attrs["space_dim"] = snapshot.space_dim
        h5_file.attrs["species"] = snapshot.species
        h5_file.attrs["species_name"] = species_name(snapshot.species)
        h5_file.attrs["particle_count"] = snapshot.particle_count
        h5_file.attrs["particle_capacity"] = snapshot.particle_capacity
        h5_file.attrs["rest_mass"] = snapshot.rest_mass
        h5_file.attrs["charge"] = snapshot.charge
        h5_file.attrs["speed_of_light"] = snapshot.speed_of_light
        h5_file.attrs["energy_scale_erg"] = snapshot.energy_scale_erg
        h5_file.attrs["energy_split_ratio"] = snapshot.energy_split_ratio
        h5_file.attrs["minimum_child_weight"] = snapshot.minimum_child_weight
        h5_file.attrs["position_unit"] = snapshot.position_unit
        h5_file.attrs["momentum_unit"] = snapshot.momentum_unit
        h5_file.attrs["momentum_direction_available"] = snapshot.momentum_direction_available

    write_xdmf(snapshot, hdf5_path, xdmf_path)


def species_name(species: int) -> str:
    """Return the host-side species name matching ParticleSpecies."""
    if species == 0:
        return "proton"
    if species == 1:
        return "electron"
    if species == 2:
        return "alpha"
    return "custom"


def data_item(hdf5_path: Path, dataset: str, dimensions: str,
              number_type: str = "Float", precision: int = 8) -> str:
    """Return one XDMF HDF5 DataItem."""
    escaped_hdf5 = html.escape(str(hdf5_path), quote=True)
    escaped_dataset = html.escape(dataset, quote=True)
    return (
        f'<DataItem Format="HDF" NumberType="{number_type}" Precision="{precision}" '
        f'Dimensions="{dimensions}">{escaped_hdf5}:/{escaped_dataset}</DataItem>'
    )


PARTICLE_VECTOR_ATTRIBUTES = (
    "momentum",
    "previous_position",
    "previous_step_position",
)

PARTICLE_SCALAR_ATTRIBUTES = (
    ParticleScalarAttributeSpec("particle_id", "UInt", 8),
    ParticleScalarAttributeSpec("status", "Int", 4),
    ParticleScalarAttributeSpec("is_active", "Int", 4),
    ParticleScalarAttributeSpec("is_escaped", "Int", 4),
    ParticleScalarAttributeSpec("is_inactive", "Int", 4),
    ParticleScalarAttributeSpec("split_level", "Int", 4),
    ParticleScalarAttributeSpec("sort_key", "Int", 4),
    ParticleScalarAttributeSpec("mu", "Float", 8),
    ParticleScalarAttributeSpec("weight", "Float", 8),
    ParticleScalarAttributeSpec("momentum_magnitude", "Float", 8),
    ParticleScalarAttributeSpec("gamma", "Float", 8),
    ParticleScalarAttributeSpec("speed", "Float", 8),
    ParticleScalarAttributeSpec("kinetic_energy", "Float", 8),
    ParticleScalarAttributeSpec("kinetic_energy_ev", "Float", 8),
    ParticleScalarAttributeSpec("initial_kinetic_energy", "Float", 8),
    ParticleScalarAttributeSpec("initial_kinetic_energy_ev", "Float", 8),
)


def write_scalar_attribute_lines(hdf5_path: Path, name: str, number_type: str,
                                 precision: int, count: int) -> str:
    """Return XDMF lines for one scalar point attribute."""
    escaped_name = html.escape(name, quote=True)
    return (
        f'      <Attribute Name="{escaped_name}" AttributeType="Scalar" Center="Node">\n'
        f"        {data_item(hdf5_path, name, str(count), number_type, precision)}\n"
        f"      </Attribute>\n"
    )


def write_vector_attribute_lines(hdf5_path: Path, name: str, count: int) -> str:
    """Return XDMF lines for one vector point attribute."""
    escaped_name = html.escape(name, quote=True)
    return (
        f'      <Attribute Name="{escaped_name}" AttributeType="Vector" Center="Node">\n'
        f"        {data_item(hdf5_path, name, f'{count} 3')}\n"
        f"      </Attribute>\n"
    )


def write_xdmf(snapshot: ParticleSnapshot, hdf5_path: Path, xdmf_path: Path) -> None:
    """Write an XDMF Polyvertex sidecar referencing the HDF5 heavy-data file."""
    count = snapshot.particle_count
    escaped_species = html.escape(species_name(snapshot.species), quote=True)
    escaped_position_unit = html.escape(snapshot.position_unit, quote=True)
    escaped_momentum_unit = html.escape(snapshot.momentum_unit, quote=True)

    attributes = ""
    for attribute_name in PARTICLE_VECTOR_ATTRIBUTES:
        attributes += write_vector_attribute_lines(hdf5_path, attribute_name, count)
    for attribute in PARTICLE_SCALAR_ATTRIBUTES:
        attributes += write_scalar_attribute_lines(
            hdf5_path, attribute.name, attribute.number_type, attribute.precision, count)

    xdmf_path.write_text(
        f"""<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="particles" GridType="Uniform">
      <Information Name="Species" Value="{escaped_species}"/>
      <Information Name="PositionUnit" Value="{escaped_position_unit}"/>
      <Information Name="MomentumUnit" Value="{escaped_momentum_unit}"/>
      <Information Name="SpaceDim" Value="{snapshot.space_dim}"/>
      <Information Name="ParticleCount" Value="{snapshot.particle_count}"/>
      <Information Name="ParticleCapacity" Value="{snapshot.particle_capacity}"/>
      <Information Name="EnergyScaleErg" Value="{snapshot.energy_scale_erg}"/>
      <Information Name="MomentumDirectionAvailable" Value="{int(snapshot.momentum_direction_available)}"/>
      <Topology TopologyType="Polyvertex" NumberOfElements="{count}"/>
      <Geometry GeometryType="XYZ">
        {data_item(hdf5_path, "position", f"{count} 3")}
      </Geometry>
{attributes}    </Grid>
  </Domain>
</Xdmf>
""",
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse converter command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a KokkosParticleTransport particle binary snapshot to HDF5 plus XDMF."
    )
    parser.add_argument("input", type=Path, help="Input particle binary snapshot.")
    parser.add_argument("hdf5", type=Path, help="Output .h5 file.")
    parser.add_argument("xdmf", type=Path, help="Output .xdmf sidecar file.")
    parser.add_argument("--position-scale", type=parse_positive_scale, default=1.0,
                        help="Visualization scale applied to particle coordinates.")
    parser.add_argument("--position-unit", default="code_length",
                        help="Coordinate unit label written to HDF5/XDMF metadata.")
    parser.add_argument("--momentum-scale", type=parse_positive_scale, default=1.0,
                        help="Visualization scale applied to particle momentum magnitudes.")
    parser.add_argument("--momentum-unit", default="code_momentum",
                        help="Momentum unit label written to HDF5/XDMF metadata.")
    return parser.parse_args()


def load_hdf5_modules():
    """Load Python HDF5 dependencies only when conversion is actually requested."""
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Python particle HDF5 conversion requires h5py and numpy. "
                           "Install them with `python3 -m pip install h5py numpy`.") from exc
    return h5py, np


def main() -> int:
    args = parse_arguments()
    try:
        h5py_module, np_module = load_hdf5_modules()
        snapshot = read_particle_binary_snapshot(args.input, np_module)
        apply_output_scales(snapshot, args.position_scale, args.position_unit,
                            args.momentum_scale, args.momentum_unit)
        write_hdf5_xdmf(snapshot, args.hdf5, args.xdmf, h5py_module, np_module)
    except Exception as exc:
        print(f"particle_binary_to_xdmf_hdf5.py failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
