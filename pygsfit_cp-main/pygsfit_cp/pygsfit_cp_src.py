import copy
import ctypes
import os
import platform
import subprocess

import astropy.units as u
import dask
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import threading
from dask.distributed import Client
import json



from pygsfit_cp import pygsfit_cp_utils as ut
from pygsfit_cp import ndfits
import h5py


class pygsfit_cp_class:
    def __init__(self, filename=None, out_dir=None, fit_fov=None):
        self.filename = filename if filename is not None else os.path.join(os.path.dirname(__file__),
                                                                           'demo/eovsa_allbd_demo_0.fits')
        #self.filename = filename
        self.out_dir = out_dir if out_dir else os.path.join(os.path.expanduser('~'), 'pygsfit_cp_output')
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.libpath = self.get_lib()
        self.margin_pix = 5
        self.meta, self.tb_data = ndfits.read(self.filename)
        cx1 = self.meta['header']['CRVAL1'] - self.meta['header']['CDELT1']*  self.meta['header']['CRPIX1']*0.5
        cx2 = self.meta['header']['CRVAL1'] + self.meta['header']['CDELT1']*  self.meta['header']['CRPIX1']*0.5
        cy1 = self.meta['header']['CRVAL2'] - self.meta['header']['CDELT2'] * self.meta['header']['CRPIX2'] * 0.5
        cy2 = self.meta['header']['CRVAL2'] + self.meta['header']['CDELT2'] * self.meta['header']['CRPIX2'] * 0.5
        self.fit_fov = fit_fov if fit_fov is not None else [[cx1,cy1],[cx2,cy2]] # in format of fov [[x1, y1],[x2,y2]]
        self.start_freq = self.meta['ref_cfreqs'][0]  # Hz
        self.end_freq = self.meta['ref_cfreqs'][-1]  # Hz
        self.freq_ghz = self.meta['ref_cfreqs'] / 1e9
        self.integrated_threshold_sfu = 1  # sfu
        self.rms_factor = 1  # Uncertainty_rms = RMS * rms_factor
        self.background_xyrange = [[0.15, 0.75], [0.25,
                                                  0.85]]  # percentage of the start and end range in X and Y, should be within [0,1]
        self.pix_mask=None
        self.ninput = np.array([7, 0, 1, 30, 1, 1],
                               dtype='int32')  # Nparms, Angular_mode,Npix, Nfreq(will be replaced) fitting mode, stokes
        self.rinput = np.array([0.17, 1e-6, 1.0, 4.0, 8.0, 0.015], dtype='float64')  # real_input
        self.initial_parguess = np.array([
            [10.0, 0.0001, 2000.0],  # n_nth;    1d7 cm^-3
            [4.0, 0.01, 30.0],  # B;    1d2G
            [60.0, 22.0, 87.0],  # theta;    deg
            [10.0, 0.01, 600.0],  # n_th;    1d9 cm^-3
            [4.5, 1.6, 10.0],  # Delta;    No
            [5.0, 0.1, 10.0],  # E_max;    MeV
            [5.0, 1.5, 60.0],  # T_e;    MK
        ], dtype=np.float64)
        if self.meta['header']['BTYPE'] == 'Brightness Temperature' or 'K' in self.meta['header']['BUNIT']:
            self.get_3d_fluxdata_array()
        self.plot_mode = True
        self.update_flux_threshold_mask()
        self.update_rms()

    def get_lib(self):
        if platform.system() == 'Linux':
            linux_dir = os.path.join(os.path.dirname(__file__), 'linux/')
            libpath = os.path.join(linux_dir, 'fit_Spectrum_Kl.so')
        elif platform.system() == 'Darwin':
            if platform.machine() == 'x86':
                unix_dir = os.path.join(os.path.dirname(__file__), 'unix/x86/')
                libpath = os.path.join(unix_dir, 'fit_Spectrum_Kl.so')
            if platform.machine() == 'arm64':
                unix_dir = os.path.join(os.path.dirname(__file__), 'unix/arm64/')
                libpath = os.path.join(unix_dir, 'fit_Spectrum_Kl.so')
        elif platform.system() == 'Windows':
            libpath = os.path.join(os.path.dirname(__file__), 'win/gs_fit_1D.dll')

        if not os.path.exists(libpath):
            makefile = os.path.join(unix_dir, 'makefile')
            if makefile:
                #cwd = os.path.dirname(__file__)
                os.chdir(unix_dir)
                # subprocess.run(['rm', '*.o'])
                subprocess.run(['make', 'clean'])
                subprocess.run(['make'])
                if not os.path.exists(libpath):
                    raise Exception('The attempt to make a shareable library failed, not able to go on....')
        return libpath

    def get_3d_fluxdata_array(self):
        # ndfits file usually in Tb, if not, skip this step
        flx_data = np.zeros_like(self.tb_data)
        for freq_idx, cfreq in enumerate(self.meta['ref_cfreqs']):
            flx_data[0, freq_idx, :, :] = ut.sfu2tb_2d(cfreq, self.tb_data[0, freq_idx, :, :],
                                                       area=self.meta['header']['CDELT1'] * self.meta['header'][
                                                           'CDELT2'] * u.arcsec ** 2, reverse=True)
        self.flux_data = flx_data[0, :, :, :]

    def do_fit(self, mode='batch', inp_coord=None, world=True):
        """

        :param mode: 'single' or 'batch'
        :param inp_coord: [930, -260]  (world, when world==True) or [124,200] (pixel, when world == False), always in x, y order.
        :param world:
        :return:
        """
        self.update_flux_threshold_mask()
        start_freq_idx = np.argmin(np.abs(self.meta['ref_cfreqs'] - self.start_freq))
        end_freq_idx = np.argmin(np.abs(self.meta['ref_cfreqs'] - self.end_freq))+1
        self.ffghz = np.array(self.meta['ref_cfreqs'][start_freq_idx:end_freq_idx] / 1.e9, dtype='float64')
        self.ninput[3] = len(self.ffghz)
        spec_in_list = []
        if mode == 'single':
            if inp_coord is not None:
                if world:
                    cur_pix = ut.world2pix(inp_coord, self.meta['refmap'])
                    inp_coord = [int(cur_pix[0].value), int(cur_pix[1].value)]
                self.coordinates = [inp_coord[::-1]]
            else:
                print('The first pixel of the selected ROI will be fitted')
        else:
            self.plot_mode=False
        for coord_idx, coord in enumerate(self.coordinates):
            spec_in = np.zeros((1, end_freq_idx - start_freq_idx, 4), dtype='float64', order='F')
            spec_in[0, :, 0] = self.flux_data[start_freq_idx:end_freq_idx, coord[0], coord[1]]
            spec_in[0, :, 2] = self.rms[start_freq_idx:end_freq_idx] + self.rms_factor * self.rms[
                                                                                         start_freq_idx:end_freq_idx] / self.ffghz
            spec_in_list.append(spec_in)
        print('Current fitting range is {0} - {1} Hz'.format(self.start_freq, self.end_freq))
        print('Long inputs are: ', self.ninput)
        print('Real inputs are: ', self.rinput)
        print('Params(ranges) are: ', self.initial_parguess)
        print('{0} pixels to be fitted'.format(len(self.coordinates)))
        cfmode = "plotting (single pixel)" if self.plot_mode else "batch"
        print(f"will be fitted in {cfmode} mode")
        #user_response = input("Do you want to continue? Enter 'y' for Yes or 'n' for No: ").strip().lower()
        #while True:
        #if user_response == 'y':
        #print("Continuing...")
        repeated_row = np.array([5.0, 0.2, 20.0], dtype=np.float64)
        repeated_rows = np.tile(repeated_row, (8, 1))
        final_parguess = np.asfortranarray(np.vstack((self.initial_parguess, repeated_rows)))
        if mode == 'single':
            fit_res = pyWrapper_Fit_Spectrum_Kl(self.libpath, self.ninput, self.rinput, final_parguess,
                                                self.ffghz, spec_in_list[0])
            plot_fitting_res(fit_res, self.ffghz, spec_in_list[0])
            return 1
        # elif mode == 'batch':
        #     tasks = []
        #     for cidx, cur_spec_in in enumerate(spec_in_list):
        #         extra_info = {'filename': self.filename, 'start_freq_idx': start_freq_idx,
        #                       'end_freq_idx': end_freq_idx,
        #                       'parguess': final_parguess, 'ninput': self.ninput, 'rinput': self.rinput,
        #                       'coord': list(self.coordinates[cidx]),
        #                       'spec_in': spec_in_list[cidx], 'freq_fitted': self.ffghz, 'task_idx': cidx,
        #                       'out_dir': self.out_dir, 'rms_range': self.background_xyrange, 'data_saved_range': self.data_saved_range,
        #                       'world_center': self.data_saved_world_center}
        #
        #         # Number of parallel tasks
        #         cspec_in = copy.deepcopy((spec_in_list[cidx]))
        #
        #         # tasks.append(dask.delayed(pyWrapper_Fit_Spectrum_Kl)(self.libpath, self.ninput, self.rinput,
        #         #                                                      final_parguess, self.ffghz,
        #         #                                                      spec_in_list[cidx], info=extra_info))
        #         tasks.append(dask.delayed(pyWrapper_Fit_Spectrum_Kl)(copy.deepcopy(self.libpath), copy.deepcopy(self.ninput), copy.deepcopy(self.rinput), copy.deepcopy(final_parguess),
        #                              copy.deepcopy(self.ffghz), cspec_in, info=copy.deepcopy(extra_info)))
        elif mode == 'batch':
            client = Client(processes=True, n_workers=4)
            tasks = []
            for cidx, cur_spec_in in enumerate(spec_in_list):
                extra_info = {'filename': self.filename, 'start_freq_idx': start_freq_idx,
                              'end_freq_idx': end_freq_idx,
                              'parguess': final_parguess, 'ninput': self.ninput, 'rinput': self.rinput,
                              'coord': list(self.coordinates[cidx]),
                              'spec_in': spec_in_list[cidx], 'freq_fitted': self.ffghz, 'task_idx': cidx,
                              'out_dir': self.out_dir, 'rms_range': self.background_xyrange,
                              'data_saved_range': self.data_saved_range,
                              'world_center': self.data_saved_world_center}

                cspec_in = copy.deepcopy((spec_in_list[cidx]))

                task = dask.delayed(pyWrapper_Fit_Spectrum_Kl)(copy.deepcopy(self.libpath),
                                                               copy.deepcopy(self.ninput),
                                                               copy.deepcopy(self.rinput),
                                                               copy.deepcopy(final_parguess),
                                                               copy.deepcopy(self.ffghz), cspec_in,
                                                               info=copy.deepcopy(extra_info))
                tasks.append(task)

            # Compute tasks using dask
            out_filenames = dask.compute(*tasks, scheduler='processes')
            #out_filenames = dask.compute(*tasks)
            final_out_fname = os.path.join(self.out_dir,os.path.basename(self.filename).replace('fits', 'h5'))
            merge_task = dask.delayed(ut.combine_hdf5_files)(out_filenames, final_out_fname, True)
            merge_task.compute()
            return final_out_fname
        else:
            print('Mode can only be single or batch')

        #break  # Break out of the loop to continue execution within the function
            # elif user_response == 'n':
            #     print("Exiting function...")
            #     return  # Exit the current function
            # else:
            #     print("Invalid input. Please enter 'y' for Yes or 'n' for No.")

    def set_integrated_threshold_sfu(self, value):
        self.integrated_threshold_sfu = float(value)
        self.update_flux_threshold_mask()

    def update_flux_threshold_mask(self):
        """
            if self.pix_mask is provided, fit_fov would not be used
            if self.integrated_threshold_sfu<=0.0, would not be used
        """
        if self.pix_mask is None:
            self.fov_mask = ut.create_fov_mask(self.meta['refmap'], self.fit_fov)
        else:
            self.fov_mask = self.pix_mask
        fov_y, fov_x = np.where(~self.fov_mask)
        self.fov_coordinates = list(zip(fov_y, fov_x))
        self.final_mask = self.fov_mask
        if self.integrated_threshold_sfu>=0.0:
            if not hasattr(self, 'rms'):
                self.update_rms()
            # creat data mask in X and Y plane with provided threshold(in sfu)
            # squeezed_flxdata = (data - rms[:, np.newaxis, np.newaxis]) * freq_ghz[:, np.newaxis, np.newaxis]
            squeezed_flxdata = self.flux_data - self.rms[:, np.newaxis, np.newaxis]
            summed_over_freq = squeezed_flxdata.sum(axis=0)
            masked_data = np.ma.masked_where(summed_over_freq <= self.integrated_threshold_sfu, summed_over_freq)
            self.threshold_mask = masked_data.mask
            if np.all(self.threshold_mask):
                print('The threshold is too large, none of the pixels is selected. Please try again')
                return
            self.final_mask = np.logical_or(self.fov_mask, self.threshold_mask)
        y, x = np.where(~self.final_mask)
        self.coordinates = list(zip(y, x))
        y_values = [coord[0] for coord in self.fov_coordinates]
        x_values = [coord[1] for coord in self.fov_coordinates]
        self.min_y = max(min(y_values)-self.margin_pix, 0)
        self.max_y = min(max(y_values)+self.margin_pix, self.meta['refmap'].data.shape[0])
        self.min_x = max(min(x_values)-self.margin_pix, 0)
        self.max_x = min(max(x_values)+self.margin_pix, self.meta['refmap'].data.shape[1])
        self.data_saved_range = [[self.min_y,self.max_y], [self.min_x, self.max_x]]
        world_center = self.meta['refmap'].pixel_to_world(int((self.min_x+self.max_x)/2)*u.pix,int((self.min_y+self.max_y)/2)*u.pix )
        self.data_saved_world_center = [world_center.Tx.value, world_center.Ty.value]

    def update_rms(self, xyrange=None):
        """

        :param xyrange: xyrange: percentage of the start and end range in X and Y, should be within [0,1]
        """
        if xyrange is not None:
            self.background_xyrange = xyrange
        [[xstart, ystart], [xend, yend]] = self.background_xyrange
        height, width = self.flux_data[0].shape
        self.rms = np.zeros((len(self.freq_ghz)), dtype=np.float64)
        for i in range(len(self.freq_ghz)):
            self.rms[i] = np.std(self.flux_data[i,
                                 int(height * ystart):int(height * yend),
                                 int(width * xstart):int(width * xend)]) * self.rms_factor

    def plot_threshold_mask_rms(self, tar_freq_ghz=None):
        """
        Show the current mask and background region used to calculate the RMS.
        :param tar_freq_ghz: The freq to be displayed in GHz
        """
        if tar_freq_ghz is not None:
            target_idx = np.argmin(np.abs(self.meta['ref_cfreqs'] - tar_freq_ghz * 1.e9))
        else:
            target_idx = np.argmin(np.abs(self.meta['ref_cfreqs'] - self.start_freq))

        # plot the eovsa image at selected freq in GHz
        extent = ut.extent_convertor(self.meta['header'])
        img = plt.imshow(self.flux_data[target_idx, :, :], origin='lower', extent=extent, cmap='jet')
        img_m = plt.imshow(~self.final_mask, origin='lower', extent=extent, cmap='viridis',
                           alpha=0.5)  # plot the mask based on the threshold

        # Plot a box showing the region where RMS is calculated
        box_percentage = self.background_xyrange
        x_range = extent[1] - extent[0]  # Total x range in solar coordinates
        y_range = extent[3] - extent[2]  # Total y range in solar coordinates
        box_origin_solar = (extent[0] + box_percentage[0][0] * x_range, extent[2] + box_percentage[0][1] * y_range)
        box_width_solar = (box_percentage[1][0] - box_percentage[0][0]) * x_range
        box_height_solar = (box_percentage[1][1] - box_percentage[0][1]) * y_range
        plt.gca().add_patch(
            patches.Rectangle(box_origin_solar, box_width_solar, box_height_solar, linewidth=1, edgecolor='r',
                              facecolor='none'))
        plt.gca().add_patch(
            patches.Rectangle((self.fit_fov[0][0], self.fit_fov[0][1]), self.fit_fov[1][0]-self.fit_fov[0][0], self.fit_fov[1][1]-self.fit_fov[0][1], linewidth=1, edgecolor='g',
                              facecolor='none'))

        plt.gca().set_xlabel('Solar X \n [arcsec]')
        plt.gca().set_ylabel('Solar Y \n [arcsec]')
        plt.title('{0} pixels are selected'.format(np.sum(~self.final_mask)))
        rms_position = (box_origin_solar[0] + 0.02 * box_width_solar, box_origin_solar[1] + 0.02 * box_height_solar)
        fov_position = (self.fit_fov[0][0], self.fit_fov[0][1]-10)
        plt.text(*rms_position, 'Background\n Region', color='r', fontsize=9, va='bottom')
        plt.text(*fov_position, 'FOV', color='g', fontsize=9, va='bottom')
        cbar = plt.colorbar(img)
        cbar.set_label('Flux Density [sfu]')

        # img.set_extent(extent)
        plt.show()

    def documentation(self, inp_name):
        """
        Calling convention of each inp arr
        :param inp_name:only 'ninput', 'rinput', and 'initial_parguess' are supported
        """
        docs_dir = os.path.join(os.path.dirname(__file__), 'docs/')
        long_path = os.path.join(docs_dir, 'Long_input.txt')
        real_path = os.path.join(docs_dir, 'Real_input.txt')
        parm_path = os.path.join(docs_dir, 'Parms_input.txt')
        file_paths = {'ninput': long_path, 'rinput': real_path,
                      'initial_parguess': parm_path}
        with open(file_paths[inp_name], 'r') as file:
            content = file.read()
            print("=============")
            print(content)
            print("=============")


    def save_notebook_cells(self, notebook_path, pyfile_path):
        """
        Save code cells from a Jupyter notebook to a Python script file in a main() format.

        :param notebook_path: Path to the input Jupyter notebook file.
        :param pyfile_path: Path to the output Python script file.
        """
        to_script_tag = 'to_script tag'
        init_tag = '#init multiple\n'

        # Load the notebook
        with open(notebook_path, 'r') as nbfile:
            notebook = json.load(nbfile)

        # Open the output Python file
        with open(pyfile_path, 'w') as pyfile:
            #Write the imports
            code_block_import = """
                from pygsfit_cp.pygsfit_cp_src import *
                import os
                import pkg_resources
                from pygsfit_cp.pygsfit_cp_utils import *
                import pygsfit_cp.results_convertor as rc
            """
            indented_source = '\n'.join(line.lstrip() for line in code_block_import.splitlines())
            pyfile.write(indented_source + '\n\n')
            # Write the main function header
            pyfile.write('def main():\n')

            # If multiple_frames is True, find and write the '#init multiple' block first
            #if multiple_frames:
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    if init_tag in cell['source']:
                        # Add the '#init multiple' block indented for the main() function
                        indented_source = ''.join('    ' + line for line in cell['source'])
                        pyfile.write(indented_source + '\n\n')
                        break
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    cell_source = ''.join(cell['source'])
                    if to_script_tag in cell_source:
                        indented_source = ''.join('        ' + line for line in cell['source'])
                        pyfile.write(indented_source + '\n\n')
            code_block_1 = """
            cur_map_file = rc.create_params_map(cur_save_file)
            map_files_by_frame.append(cur_map_file)
            fitting_files_by_frame.append(cur_save_file)
            """
            indented_source = '\n'.join('        ' + line.lstrip() for line in code_block_1.splitlines())
            pyfile.write(indented_source + '\n\n')
            code_block_2 = """
            fitting_res_file = os.path.join(gs.out_dir,'fitting_results.h5')
            combine_hdf5_files(file_list=fitting_files_by_frame,output_file=fitting_res_file) #please make sure the original files would NOT be deleted!
            print('All the fitting results are save to ', fitting_res_file)
            print('The map files are: ' )
            print(map_files_by_frame)
            rh_pro = os.path.join(package_path,'IDL_utils','read_hdf5.pro')
            pp_pro = os.path.join(package_path,'IDL_utils','process_params_maps.pro')
            mp_pro = os.path.join(gs.out_dir,'make_params_maps.pro')
            rc.create_idl_pro_file(rh_pro, pp_pro, map_files_by_frame, mp_pro)
            print('Fitting and priliminary processed are finished, please run {} in IDL to create the sav file for gsviewer')
            """
            indented_source = '\n'.join('    ' + line.lstrip() for line in code_block_2.splitlines())
            pyfile.write(indented_source + '\n\n')

            # Write the main execution block
            pyfile.write("\n\nif __name__ == '__main__':\n")
            pyfile.write("    main()\n")

        print(pyfile_path, 'is saved.')


lock = threading.Lock()
def pyWrapper_Fit_Spectrum_Kl(cur_libpath, ninput, rinput, parguess, freq, spec_in, info=None):
    """
    A python wrapper to call  Dr.Fleishman's Fortran code: fit_Spectrum_Kl.for/fit_Spectrum_Kl.so, all the input should be
    numpy array with dtype='float64', order='F', txt files can be created by calling get_Table().
    :param ninput: np.array([7, 0, 1, 30, 1, 1], dtype='int32'), 30 here will be replace by n_freq later. See Long_input.txt
    :param rinput: see real_input.txt, for example: np.array([0.17, 1e-6, 1.0, 4.0, 8.0, 0.015], dtype='float64')
    :param parguess: Input parameters ([guess, min, max]*15), see Parms_input.txt
    :param freq: freqs in GHz, example:    freq = np.array([3.42, 3.92, 4.42,.......], dtype='float64', order='F')
    :param spec_in: spectrum/uncertainty to be fitted, (1, n_freq, 4), spectrum:[0,:,0], uncertainty:[0,:,2]
    :param info: extra info to be save, if provided, fitting results will be saved in to a hdf5.
    :return:fitted spectrum, parameters and corresponding uncerntainties.
    """
    # orgnize the input args
    n_freq = len(freq)
    ninput[3] = np.int32(n_freq)
    # create the output holder
    spec_out = np.zeros((1, n_freq, 2), dtype='float64', order='F')
    aparms = np.zeros((1, 8), dtype='float64', order='F')
    eparms = np.zeros((1, 8), dtype='float64', order='F')

    # prepare the pointer of the i/o array
    inp_arrays = [ninput, rinput, parguess, freq, spec_in, aparms, eparms, spec_out]
    ct_pointers = [arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for arr in inp_arrays]
    ninput_ct, rinput_ct, parguess_ct, freq_ct, spec_in_ct, aparms_ct, eparms_ct, spec_out_ct = ct_pointers
    argv = (ctypes.POINTER(ctypes.c_double) * 8)(ninput_ct, rinput_ct, parguess_ct, freq_ct, spec_in_ct, aparms_ct,
                                                 eparms_ct, spec_out_ct)
    with lock:
        if platform.system() == 'Linux' or platform.system() == 'Darwin':
            libc_mw = ctypes.CDLL(cur_libpath)
            mwfunc = libc_mw.get_mw_fit_
        if platform.system() == 'Windows':
            libc_mw = ctypes.WinDLL(cur_libpath)
            mwfunc = libc_mw.get_mw_fit
        res = mwfunc(ctypes.c_longlong(8), argv)
    if info is not None:
        out_fname = os.path.join(info['out_dir'], 'task_{0:0=4d}.hdf5'.format(info['task_idx']))
        #info_serialized = json.dumps(info)
        with h5py.File(out_fname, 'w') as f:
            # Create datasets within the HDF5 file
            f.create_dataset('spec_out', data=spec_out)
            f.create_dataset('aparms', data=aparms)
            f.create_dataset('eparms', data=eparms)
            info_group = f.create_group('info')
            for key, value in info.items():
                #if isinstance(value, (int, float, str, list, np.ndarray)):
                if isinstance(value, list):
                    value = np.array(value)
                info_group.create_dataset(key, data=value)
                #else:
                #    raise TypeError(f"Unsupported data type for key {key}: {type(value)}")
            #f.create_dataset('info', data=info)
        # with h5py.File('yourfile.h5', 'r') as f:
        #     info = json.loads(f['info'][()].decode())
        #with open(out_fname, 'wb') as f:
        #    pickle.dump((spec_out, aparms, eparms, info), f)
        # pickle.dump((spec_out, aparms, eparms, info), open(out_fname, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        return out_fname
    else:
        return (spec_out, aparms, eparms, info)


def plot_fitting_res(fit_res, freq, spec_fitted):
    plt.plot(freq, fit_res[0][0, :, 0] + fit_res[0][0, :, 1], label='model')
    plt.errorbar(freq, spec_fitted[0, :, 0], yerr=spec_fitted[0, :, 2], linestyle='None', marker='*', label='data')
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Flux Density [sfu]')
    ax.set_xlabel('Freq [Hz]')
    ax.legend()
    # print('fitting result are: ', fit_res[1])
    plt.show()
