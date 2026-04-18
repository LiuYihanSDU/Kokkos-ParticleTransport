#== A python convertor allows user to convert the the fitting reslults in the bach mode to a hdf5 file which can be
#== converted to a sav file that can be opened by the the gsfitviewer

from scipy.io import readsav
import h5py
from pygsfit_cp import ndfits
from datetime import datetime
import numpy as np
import os


def read_sample():
    demo_sav = '/Users/walterwei/ssw/packages/gsfit/demo/demo_eovsa_gsfitmaps.sav'
    demo = readsav(demo_sav,python_dict=True)
    print('')


def create_template_group(f, cmeta, info):
    #cmeta,tb_data = ndfits.read('/Users/walterwei/Downloads/20220511/slf_final_XX_t19_allbd.fits')
    #eparms = 1.0
    #with h5py.File(tmp_fname, 'w') as f:
    template_group = f.create_group('TEMPLATE')
    template_group.create_dataset('B0', data=0.0)#????
    num_xaxis = info['data_saved_range'][1][1]-info['data_saved_range'][1][0]
    num_yaxis = info['data_saved_range'][0][1]-info['data_saved_range'][0][0]
    #template_group.create_dataset('DATA', data=np.zeros(cmeta['refmap'].meta['naxis1'], cmeta['refmap'].meta['naxis2']))
    template_group.create_dataset('DATA', data=np.zeros(num_yaxis, num_xaxis))
    template_group.create_dataset('DATAUNITS', '')
    template_group.create_dataset('DUR', data=cmeta['refmap'].meta['exptime'])
    template_group.create_dataset('DX', data=cmeta['refmap'].meta['cdelt1'])
    template_group.create_dataset('DY', data=cmeta['refmap'].meta['cdelt2'])
    template_group.create_dataset('FREQ', data=cmeta['refmap'].meta['crval3']/1.e9)
    template_group.create_dataset('FREQUNIT', data='GHz')
    template_group.create_dataset('ID', '')
    template_group.create_dataset('L0', data=0.0)#????
    template_group.create_dataset('RMS', data=0.0) #?????
    template_group.create_dataset('RMSUNITS', data='sfu')
    template_group.create_dataset('RMSXRAN', data=[info['rms_range'][0][0], info['rms_range'][1][0]])
    template_group.create_dataset('RMSYRAN', data=[info['rms_range'][0][1], info['rms_range'][1][1]])
    template_group.create_dataset('ROLL_ANGLE', data=0.00) #????
    template_group.create_dataset('ROLL_CENTER', data=[0, 0])
    template_group.create_dataset('RSUN', data=cmeta['refmap'].meta['rsun_obs'])
    template_group.create_dataset('SCALEFAC', data=0.90299976) #????
    template_group.create_dataset('SNR', data=100.0) #????
    template_group.create_dataset('SOHO', data=0)#????
    template_group.create_dataset('STOKES', data=cmeta['pol_names'][0])
    parsed_date = datetime.strptime(cmeta['refmap'].meta['date-obs'], '%Y-%m-%dT%H:%M:%S.%f')
    template_group.create_dataset('TIME', data=parsed_date.strftime('%d-%b-%Y %H:%M:%S.%f')[:-3])
    template_group.create_dataset('XC', data=info['world_center'][0])
    template_group.create_dataset('XUNITS', data=cmeta['refmap'].meta['cunit1'])
    template_group.create_dataset('YC', data=info['world_center'][1])
    template_group.create_dataset('YUNITS', data=cmeta['refmap'].meta['cunit2'])

def copy_and_modify_group(src_group, dst_group, new_params_dict):
    for name, dataset in src_group.items():
        data = dataset[()]
        if name in new_params_dict:
            data = new_params_dict[name]
        dst_group.create_dataset(name, data=data)

def create_params_map(save_file, map_file=None):
    """
    convert the output of the saving files into parameter maps, usage can be found in
    Tutorial.ipynb
    @param save_file: the output of the spectral fitting (h5 file) :
    @param map_file:  the output
    """
    base, ext = os.path.splitext(save_file)
    if map_file is None:
        map_file = f"{base}_params_map{ext}"
    print('map file will be saved to ', map_file)
    cinfo = read_info(save_file)
    cmeta, tb_data = ndfits.read(cinfo['filename'])
    res_array, peak_flux, peak_freq,wb, wb_err, wnth, wnth_err , flux_data, err_data, fit_data = read_fitting_result(save_file, cmeta, tb_data, cinfo)
    sig_array = [peak_flux, peak_freq,wb, wb_err, wnth, wnth_err]
    sig_array = [carray[cinfo['data_saved_range'][0][0]:cinfo['data_saved_range'][0][1], cinfo['data_saved_range'][1][0]:cinfo['data_saved_range'][1][1]] for carray in sig_array]
    res_array = [carray[cinfo['data_saved_range'][0][0]:cinfo['data_saved_range'][0][1], cinfo['data_saved_range'][1][0]:cinfo['data_saved_range'][1][1]] for carray in res_array]
    peak_flux, peak_freq, wb, wb_err, wnth, wnth_err = sig_array
    dat_array = [flux_data, err_data, fit_data]
    dat_array = [carray[:,:,cinfo['data_saved_range'][0][0]:cinfo['data_saved_range'][0][1],
                 cinfo['data_saved_range'][1][0]:cinfo['data_saved_range'][1][1]] for carray in dat_array]
    flux_data, err_data, fit_data = dat_array
    #res_array = cur_array[:, cinfo['data_saved_range'][0][0]:cinfo['data_saved_range'][0][1], cinfo['data_saved_range'][1][0]:cinfo['data_saved_range'][1][1]]
    params_dict = {
        'B': {'DATAUNITS': 'G', 'ID': 'B', 'DATA':res_array[1]},
        'CHISQR': {'DATAUNITS': '', 'ID': 'CHISQR', 'DATA':res_array[7]},
        'DELTA': {'DATAUNITS': '', 'ID': 'Delta', 'DATA':res_array[4]},
        'ERRB': {'DATAUNITS': 'G', 'ID': 'errB', 'DATA':res_array[9]},
        'ERRDELTA': {'DATAUNITS': '', 'ID': 'errDelta', 'DATA':res_array[12]},
        'ERRE_MAX': {'DATAUNITS': 'MeV', 'ID': 'errE_max', 'DATA':res_array[13]},
        'ERRN_NTH': {'DATAUNITS': 'cm^-3', 'ID': 'errn_nth', 'DATA':res_array[8]},
        'ERRN_TH': {'DATAUNITS': 'cm^-3', 'ID': 'errn_th', 'DATA':res_array[11]},
        'ERRPEAKFLUX': {'DATAUNITS': 'sfu', 'ID': 'errpeakflux', 'DATA':np.zeros_like(peak_flux)}, #?????
        'ERRPEAKFREQ': {'DATAUNITS': 'GHz', 'ID': 'errpeakfreq', 'DATA':np.zeros_like(peak_freq)}, #?????
        'ERRTHETA': {'DATAUNITS': 'deg', 'ID': 'errtheta', 'DATA':res_array[10]},
        'ERRT_E': {'DATAUNITS': 'MK', 'ID': 'errT_e', 'DATA':res_array[14]},
        'ERRWB': {'DATAUNITS': 'erg', 'ID': 'errWB', 'DATA':wb_err},
        'ERRWNTH': {'DATAUNITS': 'erg', 'ID': 'errWnth', 'DATA':wnth_err}, #????? #keV
        'E_MAX': {'DATAUNITS': 'MeV', 'ID': 'E_max', 'DATA':res_array[5]},
        'N_NTH': {'DATAUNITS': 'cm^-3', 'ID': 'n_nth', 'DATA':res_array[0]},
        'N_TH': {'DATAUNITS': 'cm^-3', 'ID': 'n_th', 'DATA':res_array[3]},
        'PEAKFLUX': {'DATAUNITS': 'sfu', 'ID': 'peakflux', 'DATA':peak_flux},
        'PEAKFREQ': {'DATAUNITS': 'GHz', 'ID': 'errpeakfreq', 'DATA':peak_freq},
        'RESIDUAL': {'DATAUNITS': 'sfu', 'ID': 'Residual', 'DATA':np.zeros_like(res_array[7])}, #?????????
        'THETA': {'DATAUNITS': 'deg', 'ID': 'theta', 'DATA': res_array[2]},
        'T_E': {'DATAUNITS': 'MK', 'ID': 'T_e', 'DATA': res_array[6]},
        'WB': {'DATAUNITS': 'erg', 'ID': 'WB', 'DATA': wb},
        'WNTH': {'DATAUNITS': 'erg', 'ID': 'Wnth', 'DATA': wnth},  # ????? #keV

    }
    with h5py.File(map_file, 'w') as f:
        maps_group = f.create_group('MAPS')
        create_template_group(f, cmeta, cinfo)
        template_group = f['TEMPLATE']

        for cparm in params_dict.keys():
            small_group = maps_group.create_group(cparm)

            new_params = {
                'ID': params_dict[cparm]['ID'],
                'DATAUNITS': params_dict[cparm]['DATAUNITS'],
                'DATA': params_dict[cparm]['DATA']
            }
            copy_and_modify_group(template_group, small_group, new_params)

        freq_dependent_terms = ['DATAMAPS', 'ERRMAPS', 'FITMAPS']
        for cki, ckey in enumerate(freq_dependent_terms):
            maps_group.create_group(ckey)
        for cfi, cfreq in enumerate(cmeta['ref_cfreqs']/1.e9):
            spw_group = maps_group['DATAMAPS'].create_group('spw_{0:0=2d}'.format(cfi))
            fdt_params = {'DATAUNITS': 'sfu', 'ID': 'Measured Flux @ {} GHz'.format(cfreq), 'DATA':flux_data[0,cfi,:,:], 'FREQ': cfreq}
            copy_and_modify_group(template_group, spw_group, fdt_params)

            spw_group = maps_group['ERRMAPS'].create_group('spw_{0:0=2d}'.format(cfi))
            fdt_params = {'DATAUNITS': 'sfu', 'ID': 'Applied Fit Error @ {} GHz'.format(cfreq), 'DATA':err_data[0,cfi,:,:], 'FREQ': cfreq}
            copy_and_modify_group(template_group, spw_group, fdt_params)

            spw_group = maps_group['FITMAPS'].create_group('spw_{0:0=2d}'.format(cfi))
            fdt_params = {'DATAUNITS': 'sfu', 'ID': 'Fit Flux @ {} GHz'.format(cfreq), 'DATA':fit_data[0,cfi,:,:], 'FREQ': cfreq}
            copy_and_modify_group(template_group, spw_group, fdt_params)
        del f['TEMPLATE']
    print('results are saved to ', map_file)
    return map_file

def read_fitting_result(save_file, cmeta, tb_data, cinfo):
    with h5py.File(save_file, 'r') as f:
        f_arrays = [np.zeros((cmeta['refmap'].meta['naxis1'], cmeta['refmap'].meta['naxis2'])) for _ in range(16)]
        peak_flux = np.zeros((cmeta['refmap'].meta['naxis1'], cmeta['refmap'].meta['naxis2']))
        peak_freq = np.zeros((cmeta['refmap'].meta['naxis1'], cmeta['refmap'].meta['naxis2']))
        wb = np.zeros((cmeta['refmap'].meta['naxis1'], cmeta['refmap'].meta['naxis2']))
        wb_err = np.zeros((cmeta['refmap'].meta['naxis1'], cmeta['refmap'].meta['naxis2']))
        wnth = np.zeros((cmeta['refmap'].meta['naxis1'], cmeta['refmap'].meta['naxis2']))
        wnth_err = np.zeros((cmeta['refmap'].meta['naxis1'], cmeta['refmap'].meta['naxis2']))
        vol_pix = cinfo['rinput'][3]*cinfo['rinput'][4]*(7.27e7**3) #arcsec^3 area*depth
        flux_data =np.zeros_like(tb_data)
        err_data = np.zeros_like(tb_data)
        fit_data = np.zeros_like(tb_data)
        # for spwi in range(tb_data.shape[1]):
        #     flux_data[0,spwi,:,:] = sfu2tb_2d(cinfo['freq'][spwi]*1.e9,tb_data[0,spwi,:,:],area=cmeta['refmap'].meta['cdelt1']*cmeta['refmap'].meta['cdelt2']* u.arcsec ** 2,reverse=True)
        task_names = [name for name in f.keys() if name.startswith('task_')]
        for task_name in task_names:
            group_cont = read_group(f[task_name])
            # if task_name == task_names[0]:
            #     cinfo = group_cont['info']

            aparms = group_cont['aparms'][0]
            eparms = group_cont['eparms'][0]

            coord = group_cont['info']['coord']
            y, x = coord
            for i, value in enumerate(group_cont['info']['spec_in'][0, :, 0]):
                flux_data[0,i+cinfo['start_freq_idx'], y, x] = value
                err_data[0,i+cinfo['start_freq_idx'],  y, x] = group_cont['info']['spec_in'][0, i, 2]
                fit_data[0,i+cinfo['start_freq_idx'],  y, x] = group_cont['spec_out'][0, i, 0]+group_cont['spec_out'][0, i, 1]
            for i, value in enumerate(aparms):
                f_arrays[i][y, x] = value
            for i, value in enumerate(eparms, start=len(aparms)):
                f_arrays[i][y, x] = value

            peak_flux[y,x] = np.max(flux_data[0,:,y, x])
            peak_freq[y,x] = cmeta['ref_cfreqs'][np.argmax(flux_data[0,:,y, x])]/1.e9
            wb[y,x] = vol_pix*aparms[1]**2/(8.0*np.pi)
            wb_err[y,x] = vol_pix*2*aparms[1]*eparms[1]/(8.0*np.pi)
            wnth[y,x] = vol_pix*1.6e-9*cinfo['rinput'][5]*1.e3*(aparms[4]-1)/(aparms[4]-2)*aparms[0]#? #convert emin from MeV to keV?
            wnth_err[y,x] = vol_pix*1.6e-9*cinfo['rinput'][5]*1.e3*(eparms[4]**2*eparms[0]**2/(aparms[4]-2)**4+((aparms[4]-1)/(aparms[4]-2)*eparms[0])**2)**0.5 #convert emin from MeV to keV
    return (f_arrays, peak_flux, peak_flux, wb, wb_err, wnth, wnth_err, flux_data, err_data, fit_data)

def read_info(save_file):
    with h5py.File(save_file, 'r') as f:
        task_names = [name for name in f.keys() if name.startswith('task_')]
        group_cont = read_group(f[task_names[0]])
        cinfo = group_cont['info']
    return cinfo

def read_group(group):
    content = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            content[key] = item[()]
        elif isinstance(item, h5py.Group):
            content[key] = read_group(item)
    return content

def create_idl_pro_file(path1, path2, file_list, output_pro_file):
    """
    Creates an IDL .pro file with the provided paths and file list.

    Parameters:
    path1 (str): The first file path to include in the .pro file.
    path2 (str): The second file path to include in the .pro file.
    file_list (str): The variable name or content to be used in the process_files call.
    output_pro_file (str): The output .pro file name to write.
    """
    # Construct the content of the .pro file
    content = f"""@{path1}
    @{path2}
    file_list = {file_list}
    process_files, file_list
    ;process_files, file_list, '/path/to/your_own_sav_file.sav'
    end
    """

    # Write the content to the specified output .pro file
    with open(output_pro_file, 'w') as file:
        file.write(content)

    print(f"{output_pro_file} has been created successfully.")


def main():
    import pygsfit_cp_utils as ut
    res_list = ut.makelist(tdir='/Users/walterwei/Downloads/20220511/gsfit_test/pix_t/', keyword1='allbd.h5')
    for i, cres in enumerate(res_list):
       create_params_map(cres)




if __name__ == "__main__":
    main()