import sofar as sf
import pyfar as pf
import os
import numpy as np
from visualiser import Visualiser

ORIGINAL_SOFA_PATH = r"C:\Users\March\Desktop\Msc AI\MSc Project\Sonicom_1.sofa"
MODIFIED_SOFA_PATH = os.path.join('audioprocessing', 'modified_sofa_file.sofa')
VERBOSE = True

def debug(str):
    if VERBOSE:
        print(str)

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

def modify_sofa():
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
    modified_audio_data = wetdry(convolved_audio, HRIR_signal, 0.5).time

    # Write modified audio to SOFA file
    sofa_data.Data_IR = modified_audio_data
    sf.write_sofa(MODIFIED_SOFA_PATH, sofa_data)

    # Check if SOFA file succesfully modified
    debug("success" if not sf.equals(
            sf.read_sofa(ORIGINAL_SOFA_PATH),
            sf.read_sofa(MODIFIED_SOFA_PATH)
        ) else "failed")

#testing
modify_sofa()
Visualiser(MODIFIED_SOFA_PATH)

# for every sofa file in the dataset, apply audio effect