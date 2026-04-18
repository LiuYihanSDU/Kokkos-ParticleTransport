from pygsfit_cp.pygsfit_cp_src import *
import os
import pkg_resources
from pygsfit_cp.pygsfit_cp_utils import *
import pygsfit_cp.results_convertor as rc


def main():
    # init multiple
    package_path = pkg_resources.resource_filename('pygsfit_cp', '')
    out_dir = '' # add your output path here
    # out_dir = None # will use default: ~/pygsfit_cp_output
    inp_fov = [[880, -340], [1000, -210]]  # please use the value you tested in the previous cells
    mapcubes_dir = package_path + '/demo/'  # Replace with your own data folder or fits file or .zip, .tar, .tar.gz, .tar.bz2 files
    eovsa_map_cubes = process_input_path(mapcubes_dir)
    if not eovsa_map_cubes:
        print("The input list is empty. Please check your mapcubes_dir.")
        sys.exit(1)  # Exit with a non-zero status to indicate an error
    cur_fixed_mask = get_fixed_fov_mask(eovsa_map_cubes[0], inp_fov)
    map_files_by_frame = []
    fitting_files_by_frame = []
    for emi, ceomap in enumerate(eovsa_map_cubes):
        gs = pygsfit_cp_class(filename=ceomap, out_dir=out_dir)
        gs.pix_mask = cur_fixed_mask

        # to_script tag
        gs.ninput = np.array([6, 0, 1, 30, 1, 1],
                             dtype='int32')  # Nparms, Angular_mode,Npix, Nfreq(will be replaced) fitting mode, stokes
        gs.rinput = np.array([0.17, 1e-6, 1.0, 4.0, 8.0, 0.015], dtype='float64')  # real_input
        gs.initial_parguess = np.array([  # *Value, *lower boundary * upper boundary
            [10.0, 0.0001, 2000.0],  # n_nth;    1d7 cm^-3
            [1.0, 0.01, 10.0],  # B;    1d2G
            [60.0, 22.0, 87.0],  # theta;    deg
            [10.0, 0.01, 600.0],  # n_th;    1d9 cm^-3
            [3.5, 1.6, 10.0],  # Delta;    No
            [5.0, 0.1, 10.0],  # E_max;    MeV
            [10.0, 1.5, 60.0],  # T_e;    MK
        ], dtype=np.float64)
        # gs.update_flux_threshold_mask(threshold=8.0) #summed flux density > 8sfu for current pixel

        # to_script tag
        gs.rms_factor = 4
        gs.update_rms()
        gs.integrated_threshold_sfu = 10.0
        gs.update_flux_threshold_mask()
        gs.background_xyrange = [[0.45, 0.75], [0.55, 0.85]]  #

        # to_script tag
        gs.start_freq = 4.e9
        gs.end_freq = 14.e9

        # to_script tag
        # gs.update_flux_threshold_mask()
        cur_save_file = gs.do_fit(mode='batch')

        cur_map_file = rc.create_params_map(cur_save_file)
        map_files_by_frame.append(cur_map_file)
        fitting_files_by_frame.append(cur_save_file)

    fitting_res_file = os.path.join(gs.out_dir, 'fitting_results.h5')
    combine_hdf5_files(file_list=fitting_files_by_frame,
                       output_file=fitting_res_file)  # please make sure the original files would NOT be deleted!
    print('All the fitting results are save to ', fitting_res_file)
    print('The map files are: ')
    print(map_files_by_frame)
    rh_pro = os.path.join(package_path, 'IDL_utils', 'read_hdf5.pro')
    pp_pro = os.path.join(package_path, 'IDL_utils', 'process_params_maps.pro')
    mp_pro = os.path.join(gs.out_dir, 'make_params_maps.pro')
    rc.create_idl_pro_file(rh_pro, pp_pro, map_files_by_frame, mp_pro)
    print('Fitting and priliminary processed are finished, please run {} in IDL to create the sav file for gsviewer')


if __name__ == '__main__':
    main()
