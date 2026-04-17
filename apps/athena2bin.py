import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import struct
import os
from glob import glob

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

def export_athena_final(input_file, output_file, target_shape):
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

        xi_edges = np.linspace(x_min, x_max, target_shape[0] + 1).astype(np.float64)
        yi_edges = np.linspace(y_min, y_max, target_shape[1] + 1).astype(np.float64)

        xi_centers = 0.5 * (xi_edges[:-1] + xi_edges[1:])
        yi_centers = 0.5 * (yi_edges[:-1] + yi_edges[1:])

        raw_dx = (x_max - x_min) / nx_global
        raw_dy = (y_max - y_min) / ny_global
        x_raw = x_min + (np.arange(nx_global) + 0.5) * raw_dx
        y_raw = y_min + (np.arange(ny_global) + 0.5) * raw_dy

        def resample(data):
            fn = RegularGridInterpolator((y_raw, x_raw), data, method='cubic', bounds_error=False, fill_value=None)
            mesh_y, mesh_x = np.meshgrid(yi_centers, xi_centers, indexing='ij')
            return fn(np.stack([mesh_y, mesh_x], axis=-1)).astype(np.float64)

        print(f"    Resampling to {target_shape[0]}x{target_shape[1]} with z components...")
        bx_new = resample(bx).flatten()
        by_new = resample(by).flatten()
        bz_new = resample(bz).flatten()
        vx_new = resample(vx).flatten()
        vy_new = resample(vy).flatten()
        vz_new = resample(vz).flatten()

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


if __name__ == "__main__":
    import_files = glob("/home/liuyh/data/Athena++/athena_reconnection_test/reconnection.prim.*.athdf")
    import_files.sort()

    target_shape = (1024, 1024)
    output_dir = "./field/"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    for (i, f_in) in enumerate(import_files):
        output_file = os.path.join(output_dir, "field{:0>5d}.bin".format(i))
        export_athena_final(f_in, output_file, target_shape)
