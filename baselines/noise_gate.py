import pickle
import os
import glob
import numpy as np
import torch
import shutil
from pathlib import Path

from audioprocessing.audio_processing import reverberate_hrtf

def noisegate_hrtf(hrtf:torch.Tensor, threshold):
    #Set values in tensor to 0 (primitive noise gate without attack/release)
    #TODO: CREATE A TENSOR WHICH HOLDS THE NOISEGATED HRTF TO BE SAVED
    gated_hrtf = hrtf.clone()
    # gated_hrtf[gated_hrtf <= threshold] = 0.0
    gated_hrtf = torch.nn.Threshold(threshold, 0.0)(gated_hrtf)
    return gated_hrtf

def run_noisegate_baseline(config, noisegate_output_path, subject_file=None):

    if subject_file is None:
        valid_data_paths = glob.glob('%s/%s_*' % (config.valid_hrtf_merge_dir, config.dataset))
        valid_data_file_names = ['/' + os.path.basename(x) for x in valid_data_paths]
    else:
        valid_data_file_names = ['/' + subject_file]

    # Clear/Create directory
    shutil.rmtree(Path(noisegate_output_path), ignore_errors=True)
    Path(noisegate_output_path).mkdir(parents=True, exist_ok=True)

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
        #TODO: CHANGE SAVED HRTF TO NOISEGATED HRTF
        print(lr_hrtf.shape)
        gated_hrtf = noisegate_hrtf(lr_hrtf, threshold=0.1)
        # gated_hrtf = torch.ones(5, 16, 16, 256) # this should have a high LSD error
        # gated_hrtf = hr_hrtf #this should have 0 LSD error

        # check if the tensor was actually processed
        lr_hrtf = torch.permute(lr_hrtf, (1, 2, 3, 0)) #permute to original shape
        print("Tensors same:", torch.equal(gated_hrtf, lr_hrtf))
        print("Tensors same:", torch.equal(gated_hrtf, hr_hrtf))

        with open(noisegate_output_path + file_name, "wb") as file:
            pickle.dump(gated_hrtf, file)                
            #Generates a full HRTF with interpolated data (Just dump a full HRTF)

        print('Created noisegate baseline %s' % file_name.replace('/', ''))

    return cube_coords, sphere_coords