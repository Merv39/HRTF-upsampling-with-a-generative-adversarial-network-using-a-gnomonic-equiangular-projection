import json
from pathlib import Path

# CONSTANTS FOR EASY ACCESS
MERGE_FLAG = True
GEN_SOFA_FLAG = True
NBINS_HRTF = 128
HRTF_SIZE = 16
UPSCALE_FACTOR = 1
TRAIN_SAMPLES_RATIO = 0.8
HRIR_SAMPLERATE = 48000.0

WETDRY_RATIO = 0.5

class Config:
    """Config class

    Set using HPC to true in order to use appropriate paths for HPC
    """

    def __init__(self, tag, using_hpc, dataset=None, existing_model_tag=None, data_dir=None):

        # overwrite settings with arguments provided
        self.tag = tag if tag is not None else 'pub-prep-upscale-sonicom-sonicom-synthetic-tl-2'
        self.dataset = dataset if dataset is not None else 'Sonicom'
        self.data_dir = data_dir if data_dir is not None else '/data/' + self.dataset

        if existing_model_tag is not None:
            self.start_with_existing_model = True
        else:
            self.start_with_existing_model = False

        self.existing_model_tag = existing_model_tag if existing_model_tag is not None else None

        # Data processing parameters
        self.merge_flag = MERGE_FLAG
        self.gen_sofa_flag = GEN_SOFA_FLAG
        self.nbins_hrtf = NBINS_HRTF  # make this a power of 2
        self.hrtf_size = HRTF_SIZE
        self.upscale_factor = UPSCALE_FACTOR  # can only take values: 2, 4 ,8, 16
        self.train_samples_ratio = TRAIN_SAMPLES_RATIO
        self.hrir_samplerate = HRIR_SAMPLERATE

        # Reverb Parameters
        self.wetdry_ratio = WETDRY_RATIO

        # Data dirs
        FOLDER_VERSION_NAME = "HRTF-GANs-30May24-Reverberationadversarial-network-using-a-gnomonic-equiangular-projection"
        if using_hpc:
            # HPC data dirs -- CHANGE THE PATH WHEN TESTING DIFFERENT CODE
            # self.data_dirs_path = '/rds/general/user/mgw23/home/HRTF-GANs-27Sep22-prep-for-publication' \
            #                       'adversarial-network-using-a-gnomonic-equiangular-projection'
            self.data_dirs_path = '/rds/general/user/mgw23/home/'+FOLDER_VERSION_NAME
            self.raw_hrtf_dir = Path('/rds/general/project/sonicom/live/HRTF Datasets')
            self.amt_dir = '/rds/general/user/mgw23/AMT/amt_code'
        else:
            # local data dirs
            self.data_dirs_path = 'Z:/home/'+FOLDER_VERSION_NAME
            self.raw_hrtf_dir = Path('Z:/projects/sonicom/live/HRTF Datasets')
            self.amt_dir = 'Z:/home/AMT/amt_code'

        self.runs_folder = '/runs-hpc'
        self.path = f'{self.data_dirs_path}{self.runs_folder}/{self.tag}'
        self.existing_model_path = f'{self.data_dirs_path}{self.runs_folder}/{self.existing_model_tag}'

        self.valid_path = f'{self.data_dirs_path}{self.runs_folder}/{self.tag}/valid'
        self.model_path = f'{self.data_dirs_path}{self.runs_folder}/{self.tag}'

        self.projection_dir = f'{self.data_dirs_path}/projection_coordinates'
        self.baseline_dir = '/baseline_results/' + self.dataset
        
        self.corrupted_sofa_dir =               self.data_dirs_path + self.data_dir + '/corrupted'

        self.train_hrtf_dir =                   self.data_dirs_path + self.data_dir + '/hr/train'
        self.valid_hrtf_dir =                   self.data_dirs_path + self.data_dir + '/hr/valid'
        self.train_original_hrtf_dir =          self.data_dirs_path + self.data_dir + '/original/train'
        self.valid_original_hrtf_dir =          self.data_dirs_path + self.data_dir + '/original/valid'

        self.train_hrtf_merge_dir =             self.data_dirs_path + self.data_dir + '/hr_merge/train'
        self.valid_hrtf_merge_dir =             self.data_dirs_path + self.data_dir + '/hr_merge/valid'
        self.train_original_hrtf_merge_dir =    self.data_dirs_path + self.data_dir + '/merge_original/train'
        self.valid_original_hrtf_merge_dir =    self.data_dirs_path + self.data_dir + '/merge_original/valid'

        self.mean_std_filename =                self.data_dirs_path + self.data_dir + '/mean_std_' + self.dataset
        self.barycentric_hrtf_dir =             self.data_dirs_path + self.baseline_dir + '/barycentric/valid'
        self.hrtf_selection_dir =               self.data_dirs_path + self.baseline_dir + '/hrtf_selection/valid'
        self.noisegate_hrtf_dir =               self.data_dirs_path + self.baseline_dir + '/noise_gate/valid'
        self.temporal_window_hrtf_dir =         self.data_dirs_path + self.baseline_dir + '/temporal_window/valid'

        # Training hyperparams
        self.batch_size = 1
        self.num_workers = 1
        self.num_epochs = 300  # was originally 250
        self.lr_gen = 0.0002
        self.lr_dis = 0.0000015
        # how often to train the generator
        self.critic_iters = 4

        # Loss function weight
        self.content_weight = 0.01
        self.adversarial_weight = 0.01

        # betas for Adam optimizer
        self.beta1 = 0.9
        self.beta2 = 0.999

        self.ngpu = 1
        if self.ngpu > 0:
            self.device_name = "cuda:0"
        else:
            self.device_name = 'cpu'

    def save(self):
        j = {}
        for k, v in self.__dict__.items():
            j[k] = v
        with open(f'{self.path}/config.json', 'w') as f:
            json.dump(list(j), f)

    def load(self):
        with open(f'{self.path}/config.json', 'r') as f:
            j = json.load(f)
            for k, v in j.items():
                setattr(self, k, v)

    def get_train_params(self):
        return self.batch_size, self.beta1, self.beta2, self.num_epochs, self.lr_gen, self.lr_dis, self.critic_iters
