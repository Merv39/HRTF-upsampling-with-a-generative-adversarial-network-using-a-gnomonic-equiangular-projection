import sofar as sf
import pyfar as pf
import os
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt

from audioprocessing.visualiser import Visualiser

ORIGINAL_SOFA_PATH = r"C:\Users\March\Desktop\Msc AI\MSc Project\Sonicom_1.sofa"
MODIFIED_SOFA_PATH = os.path.join('audioprocessing', 'modified_sofa_file.sofa')
VERBOSE = True

def debug(*str):
    if VERBOSE:
        print(*str)

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
    print(signal.shape) #(sampling_rate, )

    # Perform FFT
    fft_result = np.fft.fft(signal)

    # Calculate frequency bins
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/fs)

    # Only take the positive part of the spectrum
    positive_freqs = freqs[:n//2]
    positive_fft_result = fft_result[:n//2]

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


# Reverb in training
def reverberate_hrtf(hr_hrtf:torch.Tensor, wetdry=0.5):
    # Read and Convert Impulse Response to the Frequency Domain
    reverb_signal = pf.io.read_audio("audioprocessing/IMP-classroom.wav")

    # Convert pyfar signal to pytorch tensor
    #TODO: Apply FFT with the power spectrum for each channel to represent a power of 2
    reverb_signal.fft_norm = 'power'

    # plot_fft(reverb_signal, "Reverb Impulse")
    debug(reverb_signal.freq.shape)
    debug(reverb_signal.freq[0][500]) #complex number
    debug(abs(reverb_signal.freq[0][500]))
    power_reverb_signal = torch.from_numpy(frequency_bin_mapping(reverb_signal))


    dims = hr_hrtf.shape
    FREQUENCY_BINS = dims[0]; PANELS = dims[1]; X = dims[2]; Y = dims[3]
    debug("hrtf shape", hr_hrtf.shape)
    debug(hr_hrtf[0][0][0][0])
    
    # Apply convolution in frequency domain
    # Convolve for each point on each panel
    lr_hrtf = hr_hrtf.permute(1,2,3,0).clone() # (PANELS, X, Y, CHANNELS)
    for x in range(X):
        for y in range(Y):
            for panels in range(PANELS):
                convolved_signal = np.convolve(lr_hrtf[panels][x][y], power_reverb_signal, mode='full')
                # Truncate the result to match the size of the hrtf signal
                convolved_signal = torch.from_numpy(convolved_signal[:FREQUENCY_BINS])
                debug(convolved_signal.shape)
                lr_hrtf[panels][x][y] = convolved_signal
    lr_hrtf = lr_hrtf.permute(3,0,1,2) # (CHANNELS, PANELS, X, Y)
    debug("Tensors same:", torch.equal(hr_hrtf, lr_hrtf))
    return lr_hrtf

#testing
# modify_sofa()
# Visualiser(MODIFIED_SOFA_PATH)
modify_pickle("audioprocessing/Sonicom_mag_1.pickle") #from hr_merge

# for every sofa file in the dataset, apply audio effect