#!/usr/bin/env python3
"""Convert 2D Athena++ snapshots into compact solver and emission inputs."""

from __future__ import annotations

import argparse
import json
import os
import struct
from glob import glob
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - runtime dependency check
    np = None

try:
    import h5py
except ImportError:  # pragma: no cover - runtime dependency check
    h5py = None

try:
    from scipy.interpolate import RegularGridInterpolator
except ImportError:  # pragma: no cover - runtime dependency check
    RegularGridInterpolator = None

def require_conversion_dependencies():
    if np is None:
        raise RuntimeError("athena2bin.py requires numpy to convert Athena++ files.")
    if h5py is None:
        raise RuntimeError("athena2bin.py requires h5py to convert Athena++ files.")

def decode_hdf5_names(values):
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]

def dataset_variable_names(file_handle):
    if not all(name in file_handle.attrs
               for name in ("DatasetNames", "NumVariables", "VariableNames")):
        raise RuntimeError(
            "Athena file does not expose DatasetNames, NumVariables, and "
            "VariableNames attributes; velocity components cannot be mapped safely."
        )

    dataset_names = decode_hdf5_names(file_handle.attrs["DatasetNames"])
    variable_names = decode_hdf5_names(file_handle.attrs["VariableNames"])
    variable_counts = [int(value) for value in file_handle.attrs["NumVariables"]]
    mapping = {}
    offset = 0
    for dataset_name, variable_count in zip(dataset_names, variable_counts):
        mapping[dataset_name] = variable_names[offset:offset + variable_count]
        offset += variable_count
    return mapping

def find_variable_index(names, candidates, dataset_name):
    for candidate in candidates:
        if candidate in names:
            return names.index(candidate)
    raise RuntimeError(
        f"Could not find any of {candidates} in Athena dataset {dataset_name}. "
        f"Available variables: {names}"
    )

def find_optional_variable_index(names, candidates):
    for candidate in candidates:
        if candidate in names:
            return names.index(candidate)
    return None

def dataset_layout(dataset, variable_count, block_count, dataset_name):
    shape = dataset.shape
    if len(shape) != 5:
        raise RuntimeError(
            f"Athena dataset {dataset_name} must be rank 5, got shape {shape}."
        )
    if shape[0] == variable_count and shape[1] == block_count:
        return True, shape[2:]
    if shape[0] == block_count and shape[1] == variable_count:
        return False, shape[2:]
    raise RuntimeError(
        f"Cannot infer variable/block layout for Athena dataset {dataset_name} "
        f"with shape {shape}, variable_count={variable_count}, block_count={block_count}."
    )

def bilinear_resample(data, x_raw, y_raw, x_new, y_new):
    x_index = np.searchsorted(x_raw, x_new, side="right") - 1
    y_index = np.searchsorted(y_raw, y_new, side="right") - 1
    x_index = np.clip(x_index, 0, len(x_raw) - 2)
    y_index = np.clip(y_index, 0, len(y_raw) - 2)

    x0 = x_raw[x_index]
    x1 = x_raw[x_index + 1]
    y0 = y_raw[y_index]
    y1 = y_raw[y_index + 1]
    tx = (x_new - x0) / np.maximum(x1 - x0, 1.0e-300)
    ty = (y_new - y0) / np.maximum(y1 - y0, 1.0e-300)

    lower_x = x_index[None, :]
    upper_x = (x_index + 1)[None, :]
    lower_y = y_index[:, None]
    upper_y = (y_index + 1)[:, None]
    tx2 = tx[None, :]
    ty2 = ty[:, None]

    f00 = data[lower_y, lower_x]
    f10 = data[lower_y, upper_x]
    f01 = data[upper_y, lower_x]
    f11 = data[upper_y, upper_x]
    return (
        (1.0 - tx2) * (1.0 - ty2) * f00
        + tx2 * (1.0 - ty2) * f10
        + (1.0 - tx2) * ty2 * f01
        + tx2 * ty2 * f11
    ).astype(np.float64)

def normalized_reference_map(values, reference_value, normalization):
    array = np.asarray(values, dtype=np.float64)
    if normalization == "none":
        return array * reference_value
    if normalization == "mean":
        finite_positive = array[np.isfinite(array) & (array > 0.0)]
        denominator = float(np.mean(finite_positive)) if finite_positive.size > 0 else 1.0
    elif normalization == "max":
        finite_positive = array[np.isfinite(array) & (array > 0.0)]
        denominator = float(np.max(finite_positive)) if finite_positive.size > 0 else 1.0
    else:
        raise ValueError(f"unsupported background normalization: {normalization}")
    if denominator <= 0.0 or not np.isfinite(denominator):
        denominator = 1.0
    return array / denominator * reference_value

def write_emission_background(
    output_file,
    input_file,
    frame,
    x_edges,
    y_edges,
    rho_code,
    pressure_code,
    temperature_proxy_code,
    reference_density_cm3,
    reference_temperature_mk,
    normalization,
    athena_attrs,
):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ny = len(y_edges) - 1
    nx = len(x_edges) - 1
    if rho_code is None:
        rho_code = np.ones((ny, nx), dtype=np.float64)
    if temperature_proxy_code is None:
        temperature_proxy_code = np.ones((ny, nx), dtype=np.float64)

    thermal_density_cm3 = normalized_reference_map(
        rho_code, reference_density_cm3, normalization
    )
    temperature_mk = normalized_reference_map(
        temperature_proxy_code, reference_temperature_mk, normalization
    )

    with h5py.File(output_path, "w") as handle:
        handle.attrs["source_file"] = str(input_file)
        handle.attrs["frame"] = int(frame)
        handle.attrs["background_normalization"] = normalization
        handle.attrs["reference_density_cm3"] = float(reference_density_cm3)
        handle.attrs["reference_temperature_mk"] = float(reference_temperature_mk)
        handle.attrs["schema"] = "kpt_emission_background_v1"
        for key, value in athena_attrs.items():
            handle.attrs[key] = value
        handle.create_dataset("x_edges", data=np.asarray(x_edges, dtype=np.float64))
        handle.create_dataset("y_edges", data=np.asarray(y_edges, dtype=np.float64))
        handle.create_dataset("rho_code", data=np.asarray(rho_code, dtype=np.float64))
        if pressure_code is not None:
            handle.create_dataset(
                "pressure_code", data=np.asarray(pressure_code, dtype=np.float64)
            )
        handle.create_dataset(
            "temperature_proxy_code",
            data=np.asarray(temperature_proxy_code, dtype=np.float64),
        )
        density_dataset = handle.create_dataset(
            "thermal_density_cm3", data=thermal_density_cm3
        )
        density_dataset.attrs["unit"] = "cm^-3"
        temperature_dataset = handle.create_dataset("temperature_mk", data=temperature_mk)
        temperature_dataset.attrs["unit"] = "MK"

def export_athena_final(input_file, output_file, target_shape,
                        emission_background_file=None,
                        frame=0,
                        reference_density_cm3=5.0e9,
                        reference_temperature_mk=8.0,
                        background_normalization="mean"):
    require_conversion_dependencies()
    with h5py.File(input_file, 'r') as f:
        x_min, x_max = f.attrs['RootGridX1'][:2]
        y_min, y_max = f.attrs['RootGridX2'][:2]

        print(f"\n>>> Processing file: {os.path.basename(input_file)}")
        print(f"    Physical range: X[{x_min:.2e}, {x_max:.2e}], "
              f"Y[{y_min:.2e}, {y_max:.2e}]")

        locs = f['LogicalLocations'][:]
        block_count = len(locs)
        names_by_dataset = dataset_variable_names(f)
        b_names = names_by_dataset.get("B", ["Bcc1", "Bcc2", "Bcc3"])
        prim_names = names_by_dataset.get("prim")
        if prim_names is None:
            raise RuntimeError(
                f"Athena file {input_file} has no prim dataset metadata."
            )

        bx_index = find_variable_index(b_names, ["Bcc1", "B1", "B_x"], "B")
        by_index = find_variable_index(b_names, ["Bcc2", "B2", "B_y"], "B")
        bz_index = find_variable_index(b_names, ["Bcc3", "B3", "B_z"], "B")
        vx_index = find_variable_index(prim_names, ["vel1", "v1", "V1"], "prim")
        vy_index = find_variable_index(prim_names, ["vel2", "v2", "V2"], "prim")
        vz_index = find_variable_index(prim_names, ["vel3", "v3", "V3"], "prim")
        rho_index = find_optional_variable_index(
            prim_names, ["rho", "dens", "density", "d"]
        )
        pressure_index = find_optional_variable_index(
            prim_names, ["press", "pressure", "prs", "pgas", "p"]
        )

        b_var_first, block_shape = dataset_layout(f["B"], len(b_names),
                                                  block_count, "B")
        prim_var_first, prim_block_shape = dataset_layout(f["prim"], len(prim_names),
                                                          block_count, "prim")
        if tuple(block_shape) != tuple(prim_block_shape):
            raise RuntimeError(
                "Athena B and prim datasets use different block shapes."
            )
        nz_b, ny_b, nx_b = block_shape
        if nz_b != 1:
            raise RuntimeError(
                f"athena2bin currently supports 2D Athena files with nz=1, got nz={nz_b}."
            )
        max_is = (locs[:, 0].max() + 1)
        max_js = (locs[:, 1].max() + 1)
        nx_global = max_is * nx_b
        ny_global = max_js * ny_b

        def stitch_component(dataset_name, var_index):
            data_blocks = f[dataset_name]
            names = names_by_dataset[dataset_name]
            var_first, _ = dataset_layout(data_blocks, len(names), block_count,
                                          dataset_name)
            full_data = np.zeros((ny_global, nx_global))
            for b in range(len(locs)):
                i_off = locs[b, 0] * nx_b
                j_off = locs[b, 1] * ny_b
                if var_first:
                    block_data = data_blocks[var_index, b, 0, :, :]
                else:
                    block_data = data_blocks[b, var_index, 0, :, :]
                full_data[j_off:j_off+ny_b, i_off:i_off+nx_b] = block_data
            return full_data

        bx = stitch_component('B', bx_index)
        by = stitch_component('B', by_index)
        bz = stitch_component('B', bz_index)
        vx = stitch_component('prim', vx_index)
        vy = stitch_component('prim', vy_index)
        vz = stitch_component('prim', vz_index)
        rho = stitch_component('prim', rho_index) if rho_index is not None else None
        pressure = (
            stitch_component('prim', pressure_index)
            if pressure_index is not None
            else None
        )

        xi_edges = np.linspace(x_min, x_max, target_shape[0] + 1).astype(np.float64)
        yi_edges = np.linspace(y_min, y_max, target_shape[1] + 1).astype(np.float64)

        xi_centers = 0.5 * (xi_edges[:-1] + xi_edges[1:])
        yi_centers = 0.5 * (yi_edges[:-1] + yi_edges[1:])

        raw_dx = (x_max - x_min) / nx_global
        raw_dy = (y_max - y_min) / ny_global
        x_raw = x_min + (np.arange(nx_global) + 0.5) * raw_dx
        y_raw = y_min + (np.arange(ny_global) + 0.5) * raw_dy

        def resample(data):
            if RegularGridInterpolator is not None:
                fn = RegularGridInterpolator((y_raw, x_raw), data, method='cubic', bounds_error=False, fill_value=None)
                mesh_y, mesh_x = np.meshgrid(yi_centers, xi_centers, indexing='ij')
                return fn(np.stack([mesh_y, mesh_x], axis=-1)).astype(np.float64)
            return bilinear_resample(data, x_raw, y_raw, xi_centers, yi_centers)

        print(f"    Resampling to {target_shape[0]}x{target_shape[1]}...")
        if output_file is not None:
            bx_new_2d = resample(bx)
            by_new_2d = resample(by)
            bz_new_2d = resample(bz)
            vx_new_2d = resample(vx)
            vy_new_2d = resample(vy)
            vz_new_2d = resample(vz)
            bx_new = bx_new_2d.flatten()
            by_new = by_new_2d.flatten()
            bz_new = bz_new_2d.flatten()
            vx_new = vx_new_2d.flatten()
            vy_new = vy_new_2d.flatten()
            vz_new = vz_new_2d.flatten()

        rho_new = resample(rho) if rho is not None else None
        pressure_new = resample(pressure) if pressure is not None else None
        temperature_proxy_new = None
        if rho_new is not None and pressure_new is not None:
            temperature_proxy_new = np.zeros_like(rho_new)
            valid = np.isfinite(rho_new) & np.isfinite(pressure_new) & (rho_new > 0.0)
            temperature_proxy_new[valid] = pressure_new[valid] / rho_new[valid]

        athena_attrs = {
            "RootGridX1": json.dumps([float(x) for x in f.attrs["RootGridX1"]]),
            "RootGridX2": json.dumps([float(x) for x in f.attrs["RootGridX2"]]),
            "DatasetNames": json.dumps(dataset_variable_names(f)),
        }
        for attr_name in ("Time", "NumCycles"):
            if attr_name in f.attrs:
                value = f.attrs[attr_name]
                if np.isscalar(value):
                    athena_attrs[attr_name] = float(value)

    if output_file is not None:
        with open(output_file, 'wb') as f_out:
            f_out.write(struct.pack('ii', target_shape[0], target_shape[1]))

            xi_edges.tofile(f_out)
            yi_edges.tofile(f_out)

            bx_new.tofile(f_out)
            by_new.tofile(f_out)
            bz_new.tofile(f_out)

            vx_new.tofile(f_out)
            vy_new.tofile(f_out)
            vz_new.tofile(f_out)

        print(f"    --- Wrote {output_file} ({os.path.getsize(output_file)/1024**2:.2f} MB)")

    if emission_background_file is not None:
        write_emission_background(
            emission_background_file,
            input_file,
            frame,
            xi_edges,
            yi_edges,
            rho_new,
            pressure_new,
            temperature_proxy_new,
            reference_density_cm3,
            reference_temperature_mk,
            background_normalization,
            athena_attrs,
        )
        print(f"    --- Wrote emission background {emission_background_file}")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert 2D Athena++ .athdf files to compact Kokkos field binaries."
    )
    parser.add_argument(
        "input_files",
        nargs="*",
        help="Input Athena++ .athdf files. If omitted, --input-glob is used.",
    )
    parser.add_argument(
        "--input-glob",
        default="/home/liuyh/data/Athena++/athena_reconnection_test/reconnection.prim.*.athdf",
        help="Glob used when no positional input files are supplied.",
    )
    parser.add_argument("--output-dir", default="./field")
    parser.add_argument(
        "--skip-compact-output",
        action="store_true",
        help="Only write optional emission background files, not compact field binaries.",
    )
    parser.add_argument("--target-nx", type=int, default=1024)
    parser.add_argument("--target-ny", type=int, default=1024)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--emission-background-dir",
        default=None,
        help="Optional directory for per-frame emission-background HDF5 files.",
    )
    parser.add_argument(
        "--reference-density-cm3",
        type=float,
        default=5.0e9,
        help="Reference thermal density used to scale rho maps for emission.",
    )
    parser.add_argument(
        "--reference-temperature-mk",
        type=float,
        default=8.0,
        help="Reference temperature used to scale pressure/rho maps for emission.",
    )
    parser.add_argument(
        "--background-normalization",
        choices=("mean", "max", "none"),
        default="mean",
        help="How raw Athena background maps are normalized before reference scaling.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    import_files = list(args.input_files) if args.input_files else glob(args.input_glob)
    import_files.sort()
    if args.end_frame is None:
        args.end_frame = args.start_frame + len(import_files) - 1
    if args.start_frame < 0 or args.end_frame < args.start_frame:
        raise SystemExit("invalid --start-frame/--end-frame range")
    if args.skip_compact_output and args.emission_background_dir is None:
        raise SystemExit("--skip-compact-output requires --emission-background-dir")
    if args.input_files:
        selected_files = import_files
        output_frames = list(range(args.start_frame, args.start_frame + len(selected_files)))
    else:
        if args.end_frame >= len(import_files):
            raise SystemExit(
                f"--end-frame {args.end_frame} exceeds available input count {len(import_files)}"
            )
        selected_files = import_files[args.start_frame:args.end_frame + 1]
        output_frames = list(range(args.start_frame, args.end_frame + 1))

    target_shape = (args.target_nx, args.target_ny)
    output_dir = args.output_dir
    if not args.skip_compact_output and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.emission_background_dir is not None:
        os.makedirs(args.emission_background_dir, exist_ok=True)

    for frame, f_in in zip(output_frames, selected_files):
        output_file = None
        if not args.skip_compact_output:
            output_file = os.path.join(output_dir, "field{:0>5d}.bin".format(frame))
        emission_background_file = None
        if args.emission_background_dir is not None:
            emission_background_file = os.path.join(
                args.emission_background_dir,
                "emission_background_{:0>5d}.h5".format(frame),
            )
        export_athena_final(
            f_in,
            output_file,
            target_shape,
            emission_background_file=emission_background_file,
            frame=frame,
            reference_density_cm3=args.reference_density_cm3,
            reference_temperature_mk=args.reference_temperature_mk,
            background_normalization=args.background_normalization,
        )
