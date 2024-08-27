import pickle
import os
import glob
import torch
import shutil
from pathlib import Path

from model.dataset import modify_hrtf
from audioprocessing.audio_processing import hrtf_to_wav

def run_passthrough_baseline(config, output_path, subject_file=None, name="passthrough"):

    if subject_file is None:
        valid_data_paths = glob.glob('%s/%s_*' % (config.valid_hrtf_merge_dir, config.dataset))
        valid_data_file_names = ['/' + os.path.basename(x) for x in valid_data_paths]
    else:
        valid_data_file_names = ['/' + subject_file]

    # Clear/Create directory
    shutil.rmtree(Path(output_path), ignore_errors=True)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for file_name in valid_data_file_names:
        with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
            hr_hrtf = pickle.load(f)
        # hrtf_to_wav(hr_hrtf)

        # make a corrupted version of the hrtf
        lr_hrtf = torch.permute(modify_hrtf(torch.permute(hr_hrtf, (3, 0, 1, 2))),(1, 2, 3, 0))
        # print(hr_hrtf.shape)
        # print(lr_hrtf.shape)
        # hrtf_to_wav(lr_hrtf)

        with open(output_path + file_name, "wb") as file:
            pickle.dump(lr_hrtf, file)

        print(f'Created {name} baseline %s' % file_name.replace('/', ''))

    return