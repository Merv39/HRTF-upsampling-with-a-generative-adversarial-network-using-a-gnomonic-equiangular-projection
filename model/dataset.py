import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
import config
import biquad
from scipy.signal import iirfilter, sosfilt

from audioprocessing.audio_processing import reverberate_hrtf
from audioprocessing.audio_processing import apply_to_hrtf_points, apply_to_hrir_points, hz_to_bin, bin_to_hz

TYPE = None
TRUNCATE = None
FILTERTYPE = None
CUTOFF_FREQ = None
CUTOFF_FREQ2 = None
FILTERGAIN = None

def load_settings(args):
    global TYPE, TRUNCATE, FILTERTYPE, CUTOFF_FREQ, CUTOFF_FREQ2, FILTERGAIN
    import sys
    if 'config' in sys.modules:
        del sys.modules['config']
    
    import config #re-import in case type has changed
    TYPE = args.type if args.type else config.TYPE
    TRUNCATE = True if args.truncate else config.TRUNCATE
    FILTERTYPE = args.filtertype if args.filtertype else config.FILTERTYPE
    CUTOFF_FREQ = float(args.cutfreq) if args.cutfreq else config.CUTOFF_FREQ
    CUTOFF_FREQ2 = float(args.cutfreq2) if args.cutfreq2 else config.CUTOFF_FREQ2
    FILTERGAIN = float(args.gain) if args.gain else config.FILTERGAIN

    print("settings:")
    print("TYPE:", TYPE)
    print("TRUNCATE:", TRUNCATE)
    print("FILTERTYPE:", FILTERTYPE)
    print("CUTOFF_FREQ:", CUTOFF_FREQ)
    print("CUTOFF_FREQ2:", CUTOFF_FREQ2)
    print("FILTERGAIN:", FILTERGAIN)

def modify_hrtf(*args):
    '''
    This function to selects how the HRTF should be changed
    '''
    if TYPE == "downsample":
        return downsample_hrtf(*args)
    if TYPE == "filter":
        return filter_hrtf(*args)
    if TYPE == "none":
        return args[0] #this should be the hrtf
    else:
        return reverberate_hrtf(*args, truncate=TRUNCATE)
    
def filter_array(array:np.ndarray, cutoff=0, type="lowpass", filterclass="frequency", gain=12.0)->np.ndarray:
    '''Takes in frequency domain array, and applys filter
    
    Cutoff the number of frequency bins'''
    if filterclass == "frequency":
        #applied on hrtf
        freq_mask = np.ones_like(array)
        scaling_factor = 10 ** (-gain / 20.0)
        cutoff = hz_to_bin(cutoff)

        if type == "lowpass" or type == "highcut":
            freq_mask[cutoff:] = 0.0
        elif type == "highpass" or type == "lowcut":
            freq_mask[:cutoff] = 0.0
        elif type == "highshelf":
            freq_mask[cutoff:] = scaling_factor
        elif type == "lowshelf":
            freq_mask[:cutoff] = scaling_factor

        array = array * freq_mask
        array[array == 0.0] = config.EPSILON
        return array
    else:
        #applied on hrir
        order = 2
        nyquist = config.HRIR_SAMPLERATE / 2
        norm_cutoff = cutoff / nyquist
        if type == "lowpass" or type == "highcut":
            sos = iirfilter(
            N=order,
            Wn=norm_cutoff,
            btype='lowpass',
            analog=False,
            ftype='butter',
            output='sos'
        )
        elif type == "highpass" or type == "lowcut":
            sos = iirfilter(
            N=order,
            Wn=norm_cutoff,
            btype='highpass',
            analog=False,
            ftype='butter',
            output='sos'
        )
        elif type == "highshelf":
            f = biquad.highshelf(sr=config.HRIR_SAMPLERATE, f=cutoff, g=gain)
            return f(array)
        elif type == "lowshelf":
            f = biquad.lowshelf(sr=config.HRIR_SAMPLERATE, f=cutoff, g=gain)
            return f(array)
        elif type == "bandpass":
            sos = iirfilter(
            N=order,
            Wn=[norm_cutoff, CUTOFF_FREQ2 / nyquist],
            btype='bandpass',
            analog=False,
            ftype='butter',
            output='sos'
        )
        elif type == "bandstop":
            sos = iirfilter(
            N=order,
            Wn=[norm_cutoff, CUTOFF_FREQ2 / nyquist],
            btype='bandstop',
            analog=False,
            ftype='butter',
            output='sos'
        )
        return sosfilt(sos, array)

def filter_hrtf(hr_hrtf:torch.Tensor):
    cutoff = CUTOFF_FREQ
    # print(f'cutoff bin:', cutoff)
    lr_hrtf = hr_hrtf.permute(1,2,3,0).clone() # (PANELS, X, Y, CHANNELS)

    # Frequency Domain Filter
    # lr_hrtf = apply_to_hrtf_points(lr_hrtf, False, filter_array, cutoff, "highshelf", filterclass="frequency")

    # Time Domain Filter
    lr_hrtf = apply_to_hrir_points(lr_hrtf, False, filter_array, cutoff, FILTERTYPE, filterclass="IIR", gain=FILTERGAIN)

    lr_hrtf = lr_hrtf.permute(3,0,1,2) # (CHANNELS, PANELS, X, Y)
    # print("Reverb Tensors same:", torch.equal(hr_hrtf, lr_hrtf))
    return lr_hrtf

# based on https://github.com/Lornatang/SRGAN-PyTorch/blob/7292452634137d8f5d4478e44727ec1166a89125/dataset.py
def downsample_hrtf(hr_hrtf, hrtf_size=None, upscale_factor=None):
    import config
    hrtf_size = hrtf_size if hrtf_size != None else config.HRTF_SIZE
    upscale_factor = upscale_factor if upscale_factor != None else config.UPSCALE_FACTOR
    # downsample hrtf
    if upscale_factor == hrtf_size:
        mid_pos = int(hrtf_size / 2)
        lr_hrtf = hr_hrtf[:, :, mid_pos, mid_pos, None, None]
    else:
        lr_hrtf = torch.nn.functional.interpolate(hr_hrtf, scale_factor=1 / upscale_factor)

    return lr_hrtf

class TrainValidHRTFDataset(Dataset):
    """Define training/valid dataset loading methods.
    Args:
        hrtf_dir (str): Train/Valid dataset address.
        hrtf_size (int): High resolution hrtf size.
        upscale_factor (int): hrtf up scale factor.
        transform (callable): A function/transform that takes in an HRTF and returns a transformed version.
    """

    def __init__(self, hrtf_dir: str, hrtf_size: int, upscale_factor: int, transform=None, run_validation =True) -> None:
        super(TrainValidHRTFDataset, self).__init__()
        # Get all hrtf file names in folder
        self.hrtf_file_names = [os.path.join(hrtf_dir, hrtf_file_name) for hrtf_file_name in os.listdir(hrtf_dir)
                                if os.path.isfile(os.path.join(hrtf_dir, hrtf_file_name))]

        if run_validation:
            valid_hrtf_file_names = []
            for hrtf_file_name in self.hrtf_file_names:
                file = open(hrtf_file_name, 'rb')
                hrtf = pickle.load(file)
                if not np.isnan(np.sum(hrtf.cpu().data.numpy())):
                    valid_hrtf_file_names.append(hrtf_file_name)
            self.hrtf_file_names = valid_hrtf_file_names

        # Specify the high-resolution hrtf size, with equal length and width
        self.hrtf_size = hrtf_size
        # How many times the high-resolution hrtf is the low-resolution hrtf
        self.upscale_factor = upscale_factor
        # transform to be applied (preprocessing directly to the clean data)
        self.transform = transform

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of hrtf data
        with open(self.hrtf_file_names[batch_index], "rb") as file:
            hrtf = pickle.load(file)

        # hrtf processing operations
        # hr = high-res, lr = low-res
        if self.transform is not None:
            # If using a transform, treat panels as batch dim such that dims are (panels, channels, X, Y)
            hr_hrtf = torch.permute(hrtf, (0, 3, 1, 2))
            # Then, transform hr_hrtf to normalize and swap panel/channel dims to get channels first
            hr_hrtf = torch.permute(self.transform(hr_hrtf), (1, 0, 2, 3))
        else:
            # If no transform, go directly to (channels, panels, X, Y)
            hr_hrtf = torch.permute(hrtf, (3, 0, 1, 2))

        # downsample hrtf
        # lr_hrtf = downsample_hrtf(hr_hrtf, self.hrtf_size, self.upscale_factor)
        lr_hrtf = modify_hrtf(hr_hrtf)
        if torch.equal(lr_hrtf, hr_hrtf):
            print("HRTF UNCHANGED")
        
        return {"lr": lr_hrtf, "hr": hr_hrtf, "filename": self.hrtf_file_names[batch_index]}

    def __len__(self) -> int:
        return len(self.hrtf_file_names)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        print("Cuda Prefetcher")
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        print("Preloading")
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        # print("ToDevice")
        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
