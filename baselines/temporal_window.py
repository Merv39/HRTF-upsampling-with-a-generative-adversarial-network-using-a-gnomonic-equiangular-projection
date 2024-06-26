import pickle
import os
import glob
import numpy as np
import torch
import shutil
import re
from pathlib import Path
import pyfar as pf

from audioprocessing.audio_processing import reverberate_hrtf, apply_to_hrir_points, apply_to_hrtf_points
from preprocessing.utils import hrtf_to_hrir, trim_hrir, calc_hrtf

from model.util import spectral_distortion_metric

def truncate_array(array:np.ndarray, n:int)-> np.ndarray:
    array[n:] = 0.0
    return array

def run_temporal_window_baseline(config, temporal_window_output_path, subject_file=None):

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

        #TODO: temporal window (Truncate in the time domain)
        '''
        A hrtf is measured with speakers around a person which play a sine sweep.
        Microphones are placed in each ear to measure the change in frequencies arriving into the ear.
        In real life, the trucation time would be determined by calculating the distance from the speaker to the ear.
        '''
        # Apply truncation in the time domain: cut off the ends of np.ndarray (the total length is config.NBINS_HRTF * 2)
        temporal_window_hr_merged = apply_to_hrir_points(lr_hrtf, truncate_array, 150)

        with open(temporal_window_output_path + file_name, "wb") as file:
            pickle.dump(temporal_window_hr_merged, file)

        print('Created temporal window baseline %s' % file_name.replace('/', ''))

    return