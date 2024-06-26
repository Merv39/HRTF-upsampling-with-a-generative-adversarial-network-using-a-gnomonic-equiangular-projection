import sofar as sf
import pyfar as pf
import os
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import librosa
from scipy.signal import hilbert
import scipy

# so that it can import as if from project root directory
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))
# for p in sys.path:
#     print( p )

import config

# from audioprocessing.visualiser import Visualiser # If running from main.py
# from visualiser import Visualiser # If running this script

ORIGINAL_SOFA_PATH = r"C:\Users\March\Desktop\Msc AI\MSc Project\Sonicom_1.sofa"
MODIFIED_SOFA_PATH = os.path.join('audioprocessing', 'modified_sofa_file.sofa')
VERBOSE = False

def debug(*str):
    if VERBOSE:
        print(*str)

def wetdry_tensor(wet_signal: torch.Tensor, dry_signal: torch.Tensor, ratio: float) -> torch.Tensor:
    # Mix the wet and dry signals
    mixed_signal = ((1 - ratio) * dry_signal) + (ratio * wet_signal)
    return mixed_signal

def wetdry(wet_signal: pf.Signal, dry_signal: pf.Signal, ratio: float) -> pf.Signal:
    # Convert signals to numpy array
    sample_rate = dry_signal.sampling_rate
    wet_signal = wet_signal.time
    dry_signal = dry_signal.time
    # Pad both signals to same length if neccessary
    if len(dry_signal) < len(wet_signal):
        dry_signal = np.pad(dry_signal, (0, len(wet_signal)))
    
    # Mix the wet and dry signals
    mixed_signal = ((1 - ratio) * dry_signal) + (ratio * wet_signal)
    return pf.Signal(mixed_signal, sample_rate)

def read_sofa(sofa_path):
    # Read SOFA file
    sofa_data = sf.read_sofa(sofa_path)
    sample_rate = sofa_data.Data_SamplingRate
    audio_data = sofa_data.Data_IR
    debug(audio_data.shape)
    return

def modify_sofa(ORIGINAL_SOFA_PATH=ORIGINAL_SOFA_PATH, MODIFIED_SOFA_PATH=MODIFIED_SOFA_PATH, WETDRY=0.5):
    # Read SOFA file
    sofa_data = sf.read_sofa(ORIGINAL_SOFA_PATH)
    sample_rate = sofa_data.Data_SamplingRate

    # Extract audio data: numpy array of shape M x R x N (Measurements x Recievers x Samples)
    # (864, 2, 128)
    audio_data = sofa_data.Data_IR
    modified_audio_data = audio_data

    # Apply audio effect(s)
    debug(audio_data.shape)
    debug(len(audio_data))
    
    # Apply Noise
    # mean = 0; std_dev = 1; magnitude = 0.005
    # noise = magnitude * np.random.normal(mean, std_dev, audio_data.shape)
    # modified_audio_data += noise

    # Apply Reverb (takes a couple mins)
    HRIR_signal = pf.Signal(sofa_data.Data_IR, sample_rate)
    reverb_signal = pf.io.read_audio("audioprocessing/IMP-classroom.wav")
    convolved_audio = pf.dsp.convolve(HRIR_signal, reverb_signal, mode='full')
    convolved_audio = convolved_audio[:,:,:128]     #cut off excess time
    modified_audio_data = wetdry(convolved_audio, HRIR_signal, WETDRY).time

    # Write modified audio to SOFA file
    sofa_data.Data_IR = modified_audio_data
    sf.write_sofa(MODIFIED_SOFA_PATH, sofa_data)

    # Check if SOFA file succesfully modified
    debug("success" if not sf.equals(
            sf.read_sofa(ORIGINAL_SOFA_PATH),
            sf.read_sofa(MODIFIED_SOFA_PATH)
        ) else "failed")

def modify_pickle(filepath:str):
    # Read a batch of hrtf data (pickle from hr_merge is a batch of hrtf data)
    with open(filepath, "rb") as file:
        hrtf = pickle.load(file)

    # hrtf processing operations
    # If no transform, go directly to (channels, panels, X, Y)
    original_hrtf = torch.permute(hrtf, (3, 0, 1, 2))   #reorders tensor dimensions
    print(original_hrtf.shape)
    print(original_hrtf[1][0][1][1])

    # modify hrtf
    modified_hrtf = reverberate_hrtf(original_hrtf, wetdry)

    print(modified_hrtf.shape)
    
    return {"lr": modified_hrtf, "hr": original_hrtf, "filename": filepath}

def plot_fft(signal:pf.Signal, label="Impulse") -> None:
    ax = pf.plot.freq(signal, log_prefix=10, label=label)
    ax.set_title("'power'")
    ax.set_ylim(-80, 10)
    ax.legend(loc='upper left')
    plt.show()

# Converts frequency bins from sample rate to powers of 2
def frequency_bin_mapping(signal:pf.Signal, target_bins=256):
    fs = signal.sampling_rate
    signal = signal.time[0]
    debug(signal.shape) #(sampling_rate, )

    # Perform FFT
    fft_result = np.fft.fft(signal)

    return frequency_bin_mapping_freq_domain(fft_result, fs, target_bins=target_bins)

def frequency_bin_mapping_freq_domain(freq_signal:np.ndarray, fs, target_bins=256):
    # Calculate frequency bins
    n = len(freq_signal)
    freqs = np.fft.fftfreq(n, d=1/fs)

    # Only take the positive part of the spectrum
    positive_freqs = freqs[:n//2]
    positive_fft_result = freq_signal[:n//2]

    # Map frequency bins to powers of two
    power_of_two_bins = 2 ** np.arange(int(np.log2(n//2)))
    power_of_two_freqs = []

    # Find the closest frequency bin for each power of two
    for p in power_of_two_bins:
        closest_index = (np.abs(positive_freqs - p)).argmin()
        # power_of_two_freqs.append((positive_freqs[closest_index], positive_fft_result[closest_index]))
        power_of_two_freqs.append(positive_fft_result[closest_index])

    # # Print frequency bins and corresponding FFT values
    # for f, fft_val in power_of_two_freqs:
    #     print(f"Frequency: {f:.2f} Hz (Closest to Power of Two), FFT Value: {fft_val:.2f}")

    # Pad array up to target number of bins
    power_of_two_freqs = np.array(power_of_two_freqs)
    power_of_two_freqs = np.pad(power_of_two_freqs, (0, target_bins-len(power_of_two_freqs)), mode='constant')
    return power_of_two_freqs

def minimum_phase_ifft(hrtf:np.ndarray)->np.ndarray:
    '''takes in mono hrtf point, and returns mono hrir point'''
    hrtf[hrtf == 0.0] = 1.0e-08
    phase = np.imag(-hilbert(np.log(np.abs(hrtf))))

    # hrir = scipy.fft.irfft(np.concatenate((np.array([0]), np.abs(hrtf))) * np.exp(1j * phase))
    hrir = scipy.fft.irfft((np.abs(hrtf) * np.exp(1j * phase)))

    return hrir

# modified from https://dsp.stackexchange.com/a/40821
# if the target number of bins is more than half, don't half
def goertzel_algorithm_time(signal_time:np.ndarray, fs, target_bins=256, plot=False) -> tuple[np.ndarray, np.ndarray]:
    '''takes signal in time domain returns the shortened signal in frequency and time domain
    '''
    L = len(signal_time) #length in the time domain

    # Calculate full FFT for reference
    signal_freq = np.fft.fft(signal_time)
    f1 = np.linspace(0, fs, L, endpoint=False)

    # Calculate every 2nd sample of FFT
    # Perform the aliasing operation in time domain
    mid_index = L // 2
    if mid_index < target_bins:
        first_split = signal_time[:target_bins]
        second_split = signal_time[target_bins:]
        second_split = np.pad(second_split, pad_width=(0, target_bins-len(second_split)), mode='constant', constant_values=0.0)
        signal_time2 = first_split + second_split
    elif L % 2 == 0:
        signal_time2 = signal_time[:mid_index] + signal_time[mid_index:]
    else:
        signal_time2 = signal_time[:mid_index] + signal_time[mid_index+1:]
    signal_freq2 = np.fft.fft(signal_time2)
    f2 = np.linspace(0, fs, L//2, endpoint=False)

    if plot:
        plt.plot(f2, abs(signal_freq2), 'go-')
        plt.plot(f1, abs(signal_freq), 'rx-')

        plt.xlim((0, fs/2))
        plt.show()
    return signal_time2, signal_freq2

def goertzel_algorithm_freq(signal_freq:np.ndarray, fs, target_bins=256, phase=False) -> np.ndarray:
    '''Input: frequency domain signal
    Returns: shortened frequency domain signal'''
    #inverse FFT to time domain
    if phase:
        signal_time = np.fft.ifft(signal_freq)
    else:
        signal_time = minimum_phase_ifft(signal_freq)
        # signal_time = np.fft.irfft(signal_freq)

    #while frequency bins is not the desired number, keep repeating
    while len(signal_freq) > target_bins:
        signal_time, signal_freq = goertzel_algorithm_time(signal_time, fs, target_bins)
    
    return np.abs(signal_freq)

def goertzel_algorithm_time_to_freq(signal_time:np.ndarray, fs, target_bins=256) -> np.ndarray:
    '''Input: time domain signal
    Returns: shortened frequency domain signal'''
    #inverse FFT to time domain
    signal_freq = np.fft.fft(signal_time)

    #while frequency bins is not the desired number, keep repeating
    while len(signal_freq) > target_bins:
        signal_time, signal_freq = goertzel_algorithm_time(signal_time, fs, target_bins)
    
    return np.abs(signal_freq)

def normalise_signal(hrtf: torch.Tensor)-> torch.Tensor:
    '''Prevents clipping'''
    # find the highest value
    highest_val = torch.max(torch.abs(hrtf))
    debug(highest_val)
    # scale down the entire tensor by that amount
    normalised_hrtf = hrtf / highest_val
    return normalised_hrtf

def apply_to_hrtf_points(hrtf:torch.Tensor, func:callable, *args, **kwargs)-> torch.Tensor:
    '''Takes in HRTF (no phase) of shape [5, 16, 16, 256] [PANELS, X, Y, CHANNELS] and applys a function to each point in the frequency domain'''
    dims = hrtf.shape
    PANELS = dims[0]; X = dims[1]; Y = dims[2]; CHANNELS = dims[3]

    modified_hrtf = hrtf.clone()  #original tensor size; values are replaced anyway
    for x in range(X):
        for y in range(Y):
            for panels in range(PANELS):
                hrtf_point = hrtf[panels][x][y]
                hrtf_point_left = hrtf_point[:config.NBINS_HRTF].numpy()
                hrtf_point_right = hrtf_point[config.NBINS_HRTF:].numpy()
                modified_signal_left = func(hrtf_point_left, *args, **kwargs)
                modified_signal_right = func(hrtf_point_right, *args, **kwargs)
                
                modified_signal_left = torch.from_numpy(
                    #modified_signal[:256]
                    goertzel_algorithm_freq(modified_signal_left, fs=config.HRIR_SAMPLERATE, target_bins=config.NBINS_HRTF)
                    #frequency_bin_mapping_freq_domain(modified_signal, fs=config.HRIR_SAMPLERATE)
                )
                modified_signal_right = torch.from_numpy(
                    #modified_signal[:256]
                    goertzel_algorithm_freq(modified_signal_right, fs=config.HRIR_SAMPLERATE, target_bins=config.NBINS_HRTF)
                    #frequency_bin_mapping_freq_domain(modified_signal, fs=config.HRIR_SAMPLERATE)
                )

                modified_signal = torch.concatenate([modified_signal_left, modified_signal_right])
                # modified_hrtf[panels][x][y] = np.abs(modified_signal)
                modified_hrtf[panels][x][y] = normalise_signal(np.abs(modified_signal))
    return modified_hrtf

def apply_to_hrir_points(hrtf:torch.Tensor, func:callable, *args, **kwargs)-> torch.Tensor:
    '''Takes in HRTF (no phase) of shape [5, 16, 16, 256] [PANELS, X, Y, CHANNELS] and applys a function to each point in the time domain'''
    dims = hrtf.shape
    PANELS = dims[0]; X = dims[1]; Y = dims[2]; CHANNELS = dims[3]

    modified_hrtf = hrtf.clone()  #original tensor size; values are replaced anyway

    for x in range(X):
        for y in range(Y):
            for panels in range(PANELS):
                hrtf_point = hrtf[panels][x][y]
                hrtf_point_left = hrtf_point[:config.NBINS_HRTF].numpy()
                hrtf_point_right = hrtf_point[config.NBINS_HRTF:].numpy()

                hrir_point_left = minimum_phase_ifft(hrtf_point_left) #inverse fft with magnitude, no phase
                hrir_point_right = minimum_phase_ifft(hrtf_point_right) #inverse fft with magnitude, no phase
                hrir_point_left = func(hrir_point_left, *args)
                hrir_point_right = func(hrir_point_right, *args)

                modified_signal_left = torch.from_numpy(
                    goertzel_algorithm_time_to_freq(hrir_point_left, fs=config.HRIR_SAMPLERATE, target_bins=config.NBINS_HRTF)
                )
                
                modified_signal_right = torch.from_numpy(
                    goertzel_algorithm_time_to_freq(hrir_point_right, fs=config.HRIR_SAMPLERATE, target_bins=config.NBINS_HRTF)
                )

                modified_signal = torch.concatenate([modified_signal_left, modified_signal_right])
                # modified_hrtf[panels][x][y] = np.abs(modified_signal)
                modified_hrtf[panels][x][y] = normalise_signal(np.abs(modified_signal))

    return modified_hrtf

def reverberate_hrtf(hr_hrtf:torch.Tensor, wetdry=0.5, truncate=True):
    """ Apply reverb to hrtf. Expects hrtf of shape [256, 5, 16, 16] (CHANNELS, PANELS, X, Y)
    Returns hrtf of shape [256, 5, 16, 16]
    """
    # Read and Convert Impulse Response to the Frequency Domain
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, 'IMP-classroom.wav')

    # Convert the reverb to the correct sample rate and number of frequency bins for convolution
    reverb_audio = librosa.load(filepath, sr=config.HRIR_SAMPLERATE, mono=True)[0]
    reverb_audio_freq = np.fft.fft(reverb_audio)
    reverb_signal_freq = goertzel_algorithm_freq(reverb_audio_freq, config.HRIR_SAMPLERATE, target_bins=config.NBINS_HRTF, phase=True)

    lr_hrtf = hr_hrtf.permute(1,2,3,0).clone() # (PANELS, X, Y, CHANNELS)
    reverb_hrtf = apply_to_hrtf_points(lr_hrtf, np.convolve, reverb_signal_freq, mode='full')
    lr_hrtf = wetdry_tensor(reverb_hrtf, lr_hrtf, wetdry)
    lr_hrtf = lr_hrtf.permute(3,0,1,2) # (CHANNELS, PANELS, X, Y)
    # print("Reverb Tensors same:", torch.equal(hr_hrtf, lr_hrtf))
    return lr_hrtf

def check_signal(data:np.ndarray, input_domain:str, *str):
    """ checks the shape of the signal, in both time and frequency domain
    """
    signal = None
    sampling_rate = 44115
    signal = pf.Signal(data, sampling_rate, domain=input_domain)
    debug(str, "Frequency shape:", signal.freq.shape)
    debug(str, "Time shape:", signal.time.shape)

from scipy.io import wavfile
import pyroomacoustics
def calculate_RT60(filepath):
    # load time domain signal x and sampling frequency
    fs, x = wavfile.read(filepath)
    x = x[:, 0].astype(np.float64)
    print(x)
    print(x.shape, fs)
    # x = x[:len(x)//8]

    # returns the rt60 in milliseconds?
    rt60val = pyroomacoustics.experimental.rt60.measure_rt60(x, fs=fs, decay_db=60, energy_thres=0.95, plot=True, rt60_tgt=None)
    plt.show()
    print(rt60val)
#  --testing--
# modify_sofa()
# Visualiser(MODIFIED_SOFA_PATH)
#Visualiser(r"c:\Users\March\Desktop\Msc AI\MSc Project\Sonicom_1.sofa")
# read_sofa(r"c:\Users\March\Desktop\Msc AI\MSc Project\Sonicom_1.sofa")
# read_sofa(r"c:\Users\March\Desktop\Msc AI\MSc Project\Reverberated_Sonicom_1.sofa")
# current_dir = os.path.dirname(os.path.abspath(__file__))
# filepath = os.path.join(current_dir, 'IMP-classroom.wav')
# calculate_RT60(filepath)
# modify_pickle("audioprocessing/Sonicom_mag_1.pickle") #from hr_merge