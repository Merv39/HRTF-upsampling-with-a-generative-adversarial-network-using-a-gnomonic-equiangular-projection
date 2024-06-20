import pickle
import os
import glob
import numpy as np
import torch
import shutil
import re
from pathlib import Path
import pyfar as pf

from audioprocessing.audio_processing import reverberate_hrtf
from preprocessing.utils import hrtf_to_hrir, trim_hrir, calc_hrtf
'''
def run_temporal_window_baseline(config, temporal_window_output_path, subject_file=None):

    if subject_file is None:
        valid_data_paths = glob.glob('%s/%s_*' % (config.valid_hrtf_merge_dir, config.dataset))
        valid_data_file_names = ['/' + os.path.basename(x) for x in valid_data_paths]
    else:
        valid_data_file_names = ['/' + subject_file]

    # Clear/Create directory
    shutil.rmtree(Path(temporal_window_output_path), ignore_errors=True)
    Path(temporal_window_output_path).mkdir(parents=True, exist_ok=True)

    projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
    with open(projection_filename, "rb") as f:
        (cube_coords, sphere_coords, euclidean_sphere_triangles, euclidean_sphere_coeffs) = pickle.load(f)

    for file_name in valid_data_file_names:
        with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
            hr_hrtf = pickle.load(f)

        # permute to shape that reverberate_hrtf expects
        lr_hrtf = torch.permute(
            reverberate_hrtf(torch.permute(hr_hrtf, (3, 0, 1, 2)), config.wetdry_ratio)
            , (1, 2, 3, 0)
        )
        #TODO: CHANGE SAVED HRTF TO WINDOWED HRTF
        # print(lr_hrtf.shape)
        # gated_hrtf = noisegate_hrtf(lr_hrtf, threshold=0.1)
        # # gated_hrtf = torch.ones(5, 16, 16, 256) # this should have a high LSD error
        # # gated_hrtf = hr_hrtf #this should have 0 LSD error

        # # check if the tensor was actually processed
        # lr_hrtf = torch.permute(lr_hrtf, (1, 2, 3, 0)) #permute to original shape
        # print("Tensors same:", torch.equal(gated_hrtf, lr_hrtf))
        # print("Tensors same:", torch.equal(gated_hrtf, hr_hrtf))

        baseline_hrtf = lr_hrtf #THIS SHOULD HAVE HIGH LSD ERROR

        with open(temporal_window_output_path + file_name, "wb") as file:
            pickle.dump(baseline_hrtf, file)                
            #Generates a full HRTF with interpolated data (Just dump a full HRTF)

        print('Created temporal window baseline %s' % file_name.replace('/', ''))

    return cube_coords, sphere_coords
'''
from model.util import spectral_distortion_metric

def run_temporal_window_baseline(config, temporal_window_output_path, cube, sphere, subject_file=None):

    if subject_file is None:
        valid_data_paths = glob.glob('%s/%s_*' % (config.valid_hrtf_merge_dir, config.dataset))
        valid_data_file_names = ['/' + os.path.basename(x) for x in valid_data_paths]
    else:
        valid_data_file_names = ['/' + subject_file]

    # Clear/Create directory
    shutil.rmtree(Path(temporal_window_output_path), ignore_errors=True)
    Path(temporal_window_output_path).mkdir(parents=True, exist_ok=True)

    for file_name in valid_data_file_names:
        with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
            hr_hrtf = pickle.load(f)

        # make a corrupted version of the hrtf
        lr_hrtf = torch.permute(reverberate_hrtf(torch.permute(hr_hrtf, (3, 0, 1, 2))),(1, 2, 3, 0))

        hr_hrtf_left = hr_hrtf[:, :, :, :config.nbins_hrtf]
        hr_hrtf_right = hr_hrtf[:, :, :, config.nbins_hrtf:]

        lr_hrtf_left = lr_hrtf[:, :, :, :config.nbins_hrtf]
        lr_hrtf_right = lr_hrtf[:, :, :, config.nbins_hrtf:]
        #TODO: temporal window (Truncate in the time domain)
        '''
        A hrtf is measured with speakers around a person which play a sine sweep.
        Microphones are placed in each ear to measure the change in frequencies arriving into the ear.
        In real life, the trucation time would be determined by calculating the distance from the speaker to the ear.
        '''
        # Convert the HRTFs into HRIRs
        lr_hrir, _ , _ = hrtf_to_hrir(lr_hrtf, config, cube, sphere)

        # Apply truncation in the time domain
        #lr_hrir = trim_hrir(lr_hrir, ..., ...)

        # Apply FFT to return to the frequency domain
        lr_hrtf = calc_hrtf(config, lr_hrir)
        print("modified:", lr_hrtf.shape)

        # modify left and right hrtf individually then merge
        temporal_window_hr_merged = torch.tensor(np.concatenate((lr_hrtf_left, lr_hrtf_right), axis=3))
        print("goal:", temporal_window_hr_merged.shape)

        with open(temporal_window_output_path + file_name, "wb") as file:
            pickle.dump(temporal_window_hr_merged, file)

        print('Created temporal window baseline %s' % file_name.replace('/', ''))

    return