import argparse
import os
import pickle
import torch
import numpy as np
import importlib

from config import Config
from model.train import train
from model.test import test
from model.util import load_dataset
from preprocessing.cubed_sphere import CubedSphere
from preprocessing.utils import interpolate_fft, generate_euclidean_cube, convert_to_sofa, \
     merge_files, gen_sofa_preprocess, get_hrtf_from_ds, clear_create_directories
from model import util
from baselines.barycentric_interpolation import run_barycentric_interpolation
from baselines.hrtf_selection import run_hrtf_selection
from evaluation.evaluation import run_lsd_evaluation, run_localisation_evaluation
from hrtfdata.full import SONICOM

from audioprocessing.audio_processing import modify_sofa
from baselines.noise_gate import run_noisegate_baseline
from baselines.temporal_window import run_temporal_window_baseline

PI_4 = np.pi / 4

# Random seed to maintain reproducible results
torch.manual_seed(0)
np.random.seed(0)


def main(config, mode):
    # Initialise Config object
    data_dir = config.raw_hrtf_dir / config.dataset.upper()
    print(os.getcwd())
    print(config.dataset)

    imp = importlib.import_module('hrtfdata.full')
    print(imp)
    load_function = getattr(imp, config.dataset.upper())

    # loads the HRTFDataset Object into ds and projects it onto a cubed sphere
    if mode == 'generate_projection':
        # Must be run in this mode once per dataset, finds barycentric coordinates for each point in the cubed sphere
        # No need to load the entire dataset in this case
        ds = SONICOM(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'left', 'domain': 'time'}}, subject_ids='first')
        # need to use protected member to get this data, no getters
        cs = CubedSphere(mask=ds[0]['features'].mask, row_angles=ds.row_angles, column_angles=ds.column_angles)
        generate_euclidean_cube(config, cs.get_sphere_coords(), edge_len=config.hrtf_size)

    # injects reverb into the sofa files and preprocesses them to the training format
    elif mode == 'generate_corruption':
        #copy the existing SOFA files, corrupt them and save them to a new location
        modify_sofa(data_dir, config.corrupted_sofa_dir)


    elif mode == 'preprocess':
        # Interpolates data to find HRIRs on cubed sphere, then FFT to obtain HRTF, finally splits data into train and
        # val sets and saves processed data
        ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'both', 'domain': 'time'}})
        cs = CubedSphere(mask=ds[0]['features'].mask, row_angles=ds.row_angles, column_angles=ds.column_angles)

        # need to use protected member to get this data, no getters
        projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
        with open(projection_filename, "rb") as file:
            cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

        # Clear/Create directories
        clear_create_directories(config)

        # Split data into train and test sets
        train_size = int(len(set(ds.subject_ids)) * config.train_samples_ratio)
        train_sample = np.random.choice(list(set(ds.subject_ids)), train_size, replace=False)

        # collect all train_hrtfs to get mean and sd
        train_hrtfs = torch.empty(size=(2 * train_size, 5, config.hrtf_size, config.hrtf_size, config.nbins_hrtf))
        j = 0
        for i in range(len(ds)):
            if i % 10 == 0:
                print(f"HRTF {i} out of {len(ds)} ({round(100 * i / len(ds))}%)")

            # Verification that HRTF is valid
            if np.isnan(ds[i]['features']).any():
                print(f'HRTF (Subject ID: {i}) contains nan values')
                continue

            features = ds[i]['features'].data.reshape(*ds[i]['features'].shape[:-2], -1)
            clean_hrtf = interpolate_fft(config, cs, features, sphere, sphere_triangles, sphere_coeffs,
                                             cube, fs_original=ds.hrir_samplerate, edge_len=config.hrtf_size)
            hrtf_original, phase_original, sphere_original = get_hrtf_from_ds(config, ds, i)

            # save cleaned hrtfdata
            if ds.subject_ids[i] in train_sample:
                projected_dir = config.train_hrtf_dir
                projected_dir_original = config.train_original_hrtf_dir
                train_hrtfs[j] = clean_hrtf
                j += 1
            else:
                projected_dir = config.valid_hrtf_dir
                projected_dir_original = config.valid_original_hrtf_dir

            subject_id = str(ds.subject_ids[i])
            side = ds.sides[i]
            print("Creating HR Train and Valid")
            with open('%s/%s_mag_%s%s.pickle' % (projected_dir, config.dataset, subject_id, side), "wb") as file:
                pickle.dump(clean_hrtf, file)

            print("Creating Original Train and Valid")
            with open('%s/%s_mag_%s%s.pickle' % (projected_dir_original, config.dataset, subject_id, side), "wb") as file:
                pickle.dump(hrtf_original, file)

            with open('%s/%s_phase_%s%s.pickle' % (projected_dir_original, config.dataset, subject_id, side), "wb") as file:
                pickle.dump(phase_original, file)

        if config.merge_flag:
            print("Merging...")
            merge_files(config)

        if config.gen_sofa_flag:
            gen_sofa_preprocess(config, cube, sphere, sphere_original)

        # save dataset mean and standard deviation for each channel, across all HRTFs in the training data
        mean = torch.mean(train_hrtfs, [0, 1, 2, 3])
        std = torch.std(train_hrtfs, [0, 1, 2, 3])
        min_hrtf = torch.min(train_hrtfs)
        max_hrtf = torch.max(train_hrtfs)
        mean_std_filename = config.mean_std_filename
        with open(mean_std_filename, "wb") as file:
            pickle.dump((mean, std, min_hrtf, max_hrtf), file)

    elif mode == 'train':
        print("Begin Training")
        # Trains the GANs, according to the parameters specified in Config
        train_prefetcher, _ = load_dataset(config, mean=None, std=None)
        print("Loaded all datasets successfully.")

        util.initialise_folders(config, overwrite=True)
        train(config, train_prefetcher)

    elif mode == 'test':
        _, test_prefetcher = load_dataset(config, mean=None, std=None)
        print("Loaded all datasets successfully.")

        test(config, test_prefetcher)

        run_lsd_evaluation(config, config.valid_path)
        run_localisation_evaluation(config, config.valid_path)

    elif mode == 'barycentric_baseline':
        barycentric_data_folder = f'/barycentric_interpolated_data_{config.upscale_factor}'
        barycentric_output_path = config.barycentric_hrtf_dir + barycentric_data_folder
        cube, sphere = run_barycentric_interpolation(config, barycentric_output_path)

        if config.gen_sofa_flag:
            convert_to_sofa(barycentric_output_path, config, cube, sphere)
            print('Created barycentric baseline sofa files')

        config.path = config.barycentric_hrtf_dir

        file_ext = f'lsd_errors_barycentric_interpolated_data_{config.upscale_factor}.pickle'
        run_lsd_evaluation(config, barycentric_output_path, file_ext)

        file_ext = f'loc_errors_barycentric_interpolated_data_{config.upscale_factor}.pickle'
        run_localisation_evaluation(config, barycentric_output_path, file_ext)

    elif mode == 'temporal_window_baseline':
        #TODO: implement a baseline where the time of the impulse is truncated to a certain window before audio reflections occur

        # Get Coordinated from projection file
        projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
        with open(projection_filename, "rb") as f:
            (cube, sphere, _, _) = pickle.load(f)

        temporal_window_data_folder = f'/temporal_window_interpolated_data_{config.upscale_factor}'
        temporal_window_output_path = config.temporal_window_hrtf_dir + temporal_window_data_folder
        run_temporal_window_baseline(config, temporal_window_output_path, cube, sphere)

        if config.gen_sofa_flag:
            convert_to_sofa(temporal_window_output_path, config, cube, sphere)
            print('Created temporal window baseline sofa files')

        config.path = config.temporal_window_hrtf_dir

        file_ext = f'lsd_errors_temporal_window_interpolated_data_{config.upscale_factor}.pickle'
        run_lsd_evaluation(config, temporal_window_output_path, file_ext)

        file_ext = f'loc_errors_temporal_window_interpolated_data_{config.upscale_factor}.pickle'
        run_localisation_evaluation(config, temporal_window_output_path, file_ext)

    elif mode == 'noise_gate_baseline':
        #TODO: implement a noise gate
        noisegate_data_folder = f'/noisegate_interpolated_data_{config.upscale_factor}'
        noisegate_output_path = config.noisegate_hrtf_dir + noisegate_data_folder
        cube, sphere = run_noisegate_baseline(config, noisegate_output_path)

        if config.gen_sofa_flag:
            convert_to_sofa(noisegate_output_path, config, cube, sphere)
            print('Created noise gate baseline sofa files')

        config.path = config.noisegate_hrtf_dir

        file_ext = f'lsd_errors_noisegate_interpolated_data_{config.upscale_factor}.pickle'
        run_lsd_evaluation(config, noisegate_output_path, file_ext)

        file_ext = f'loc_errors_noisegate_interpolated_data_{config.upscale_factor}.pickle'
        run_localisation_evaluation(config, noisegate_output_path, file_ext)

    elif mode == 'noise_suppression':
        #TODO: implement a noise gate
        pass

    elif mode == 'hrtf_selection_baseline':

        run_hrtf_selection(config, config.hrtf_selection_dir)

        projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
        with open(projection_filename, "rb") as f:
            (cube, sphere, _, _) = pickle.load(f)

        if config.gen_sofa_flag:
            convert_to_sofa(config.hrtf_selection_dir, config, cube, sphere)
            print('Created barycentric baseline sofa files')

        config.path = config.hrtf_selection_dir

        file_ext = f'lsd_errors_hrtf_selection_minimum_data.pickle'
        run_lsd_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='minimum')
        file_ext = f'loc_errors_hrtf_selection_minimum_data.pickle'
        run_localisation_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='minimum')

        file_ext = f'lsd_errors_hrtf_selection_maximum_data.pickle'
        run_lsd_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='maximum')
        file_ext = f'loc_errors_hrtf_selection_maximum_data.pickle'
        run_localisation_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='maximum')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("-t", "--tag")
    parser.add_argument("-c", "--hpc")
    args = parser.parse_args()

    if args.hpc == "True":
        hpc = True
    elif args.hpc == "False":
        hpc = False
    else:
        raise RuntimeError("Please enter 'True' or 'False' for the hpc tag (-c/--hpc)")

    if args.tag:
        tag = args.tag
    else:
        tag = None

    config = Config(tag, using_hpc=hpc)
    main(config, args.mode)
