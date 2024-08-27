from spatialaudiometrics import load_data as ld
from spatialaudiometrics import hrtf_metrics as hf
from spatialaudiometrics import visualisation as vis
from pathlib import Path
import matplotlib.pyplot as plt
import sofa
import numpy as np
import os, glob
import pickle
from typing import Tuple, List
from model.dataset import noisy_array
from plot import *
from audioprocessing.audio_processing import *

# LSD CONSTANTS

LOGS_FOLDER = r"Z:\home\HRTF-GANs-30May24-Reverberationadversarial-network-using-a-gnomonic-equiangular-projection\Logs"
LOWPASS_FOLDER = LOGS_FOLDER + "\Lowpass"
LOWPASS_PASSTHROUGH_FOLDER = r"Z:\home\HRTF-upsampling-with-a-generative-adversarial-network-using-a-gnomonic-equiangular-projection\analysis-lowpasses"

HRTF_SELECTION_LOG = LOGS_FOLDER+"\hrtfselection.pbs.o9888610"
IMPULSE_BASELINE_LOG = LOGS_FOLDER+"\impulsebaseline.pbs.o9855501"

#logs holds a tuple of (filename, label)
LOGS = []
LOGS2 = []

LOWPASS_NAMES = ("0.5k", "1k", "2k", "4k", "6k", "8k", "10k", "12k")
LOWPASS_LOGS = LOWPASS_NAMES
LABELS = []

for filename in LOWPASS_LOGS:
    LOGS.append((os.path.join(LOWPASS_FOLDER, filename+".txt"), filename+" GAN"))

for filename in LOWPASS_LOGS:
    LOGS2.append((os.path.join(LOWPASS_PASSTHROUGH_FOLDER, "lowpass"+filename+".txt"), filename))

# LOCALISATION CONSTANTS
LOCALISATION_FOLDER = LOGS_FOLDER + "\Localisation"

# FUNCTIONS
def collect_LSD_error(filename: str) -> []:
    LSD_Error = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("LSD Error of subject") and ":" in line:
                LSD_Error.append(float(line.split(":")[1][:-3])) #get the LSD Error without dB/n at the end
    return LSD_Error

def collect_HRTF_Selection_LSD_error(filename: str) -> []:
    LSD_Error = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("Average for ") and ":" in line:
                LSD_Error.append(float(line.split(":")[1]))
    return LSD_Error

def plotLSD():
    magnitudes = []
    for file, label in LOGS:
        magnitudes.append(collect_LSD_error(file))
        LABELS.append(label)

    if True:
        #HRTF Selection
        magnitudes.append(collect_HRTF_Selection_LSD_error(HRTF_SELECTION_LOG))
        LABELS.append("HRTF Selection")

        #Impulse Baseline
        magnitudes.append(collect_LSD_error(IMPULSE_BASELINE_LOG))
        LABELS.append("Impulse GAN")

    passthrough_magnitudes = []
    for file, label in LOGS2:
        passthrough_magnitudes.append(collect_LSD_error(file))

    length = len(magnitudes)
    x = np.linspace(start=1, stop=length, num=length)
    y = []
    y_err = []

    for i in range(length): #for each category, calculate the mean and std
        y.append(np.mean(magnitudes[i]))
        y_err.append(np.std(magnitudes[i]))

    length2 = len(passthrough_magnitudes)
    x2 = np.linspace(start=1, stop=length, num=length2)
    y2 = []
    y2_err = []

    for i in range(length2): #for each category, calculate the mean and std
        y2.append(np.mean(passthrough_magnitudes[i]))
        y2_err.append(np.std(passthrough_magnitudes[i]))

    # # Plotting the error bar graph
    plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=5, capthick=2, elinewidth=1, label="GAN")
    # plt.errorbar(x2, y2, yerr=y_err, fmt='o', capsize=5, capthick=2, elinewidth=1, label="Passthrough")


    # Adding labels and title
    plt.xticks(x, labels=LABELS, rotation=45, ha='right')
    # plt.xlabel('Low-pass Frequency (Hz)')
    plt.xlabel('Method')

    plt.ylabel('LSD Error (dB)')
    plt.title('LSD Error Comparison (lower is better)')
    # plt.legend()

    # Display the plot
    plt.savefig('plot.png', bbox_inches='tight')
    plt.show()

def collect_Localisation_errors(filename:str) -> Tuple[List[float], List[float], List[float]]: #returns 3 variables
    #Read the pickle files
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    #Sort into ACC, RMS, Quadrant
    #One array for each
    Acc_Errors = []
    RMS_Errors = []
    Quadrant_Errors = []

    #for each SOFA file/Target HRTF
    for i in range(len(data)):
        Acc_Errors.append(data[i][1])
        RMS_Errors.append(data[i][2])
        Quadrant_Errors.append(data[i][3])
    
    return Acc_Errors, RMS_Errors, Quadrant_Errors

def plot_metric(magnitudes, ylabel='Magnitude of Eval Metric', absolute = True, label=None):
    y = []
    y_err = []

    if absolute:
        magnitudes = np.abs(magnitudes)

    for i in range(len(magnitudes)): #for each category, calculate the mean and std
        y.append(np.mean(magnitudes[i]))
        y_err.append(np.std(magnitudes[i]))
    
    x = np.linspace(start=1, stop=len(y), num=len(y))

    #Plot ACC graph
    plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=5, capthick=2, elinewidth=1, label=label)

    # Adding labels and title
    plt.xticks(x, labels=LOWPASS_NAMES, rotation=45, ha='right')
    plt.xlabel('Low-pass Frequency (Hz)')

    plt.ylabel(ylabel)
    plt.title('Error Comparison (lower is better)')
    plt.legend()

    # Display the plot
    plt.savefig(ylabel+'.png', bbox_inches='tight')

def collect_Localisation_from_log(filename:str) -> Tuple[List[float], List[float], List[float]]: #returns 3 variables
    Acc_Errors = []
    RMS_Errors = []
    Quadrant_Errors = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("pol_acc1") and ":" in line:
                Acc_Errors.append(float(line.split(":")[1]))
            
            if line.startswith("pol_rms1") and ":" in line:
                RMS_Errors.append(float(line.split(":")[1]))
            
            if line.startswith("querr1") and ":" in line:
                Quadrant_Errors.append(float(line.split(":")[1]))
            
    return Acc_Errors, RMS_Errors, Quadrant_Errors

def plotLocalisation():
    magnitudes_Acc=[]
    magnitudes_RMS=[]
    magnitudes_Quad=[]

    #for each datapoint, get the localisation errors

    # Get a list of all files, then sort them alphabetically
    for name in LOWPASS_NAMES:
        print(name)
        filepath = os.path.join(LOCALISATION_FOLDER, "GAN", name + ".pickle")
        Acc_Errors, RMS_Errors, Quadrant_Errors = collect_Localisation_errors(filepath)

        magnitudes_Acc.append(Acc_Errors)
        magnitudes_RMS.append(RMS_Errors)
        magnitudes_Quad.append(Quadrant_Errors)

    magnitudes_Acc2=[]
    magnitudes_RMS2=[]
    magnitudes_Quad2=[]

    for name in LOWPASS_NAMES:
        print(name)
        filepath = os.path.join(LOCALISATION_FOLDER, "Crossover", name + ".pickle")
        Acc_Errors, RMS_Errors, Quadrant_Errors = collect_Localisation_errors(filepath)

        magnitudes_Acc2.append(Acc_Errors)
        magnitudes_RMS2.append(RMS_Errors)
        magnitudes_Quad2.append(Quadrant_Errors)
    
    magnitudes_Acc3=[]
    magnitudes_RMS3=[]
    magnitudes_Quad3=[]

    for name in LOWPASS_NAMES:
        print(name)
        filepath = os.path.join(LOCALISATION_FOLDER, "Passthrough", name + ".txt")
        Acc_Errors, RMS_Errors, Quadrant_Errors = collect_Localisation_from_log(filepath)

        magnitudes_Acc3.append(Acc_Errors)
        magnitudes_RMS3.append(RMS_Errors)
        magnitudes_Quad3.append(Quadrant_Errors)
    
    plot_metric(magnitudes_Acc, ylabel="Absolute Value of Accuracy", label="GAN")
    plot_metric(magnitudes_Acc2, ylabel="Absolute Value of Accuracy", label="Crossover")
    plot_metric(magnitudes_Acc3, ylabel="Absolute Value of Accuracy", label="Passthrough")
    plt.show()
    
    plot_metric(magnitudes_RMS, ylabel="Absolute Value of RMS", label="GAN")
    plot_metric(magnitudes_RMS2, ylabel="Absolute Value of RMS", label="Crossover")
    plot_metric(magnitudes_RMS3, ylabel="Absolute Value of RMS", label="Passthrough")
    plt.show()

    plot_metric(magnitudes_Quad, ylabel="Absolute Value of Quadrant Errors", label="GAN")
    plot_metric(magnitudes_Quad2, ylabel="Absolute Value of Quadrant Errors", label="Crossover")
    plot_metric(magnitudes_Quad3, ylabel="Absolute Value of Quadrant Errors", label="Passthrough")
    plt.show()


# plotLSD()
# plotLocalisation()



#make an empty array of size 256
# array = np.empty(256)
# noise = noisy_array(array) #applied on time domain
# # plot_impulse_response(noise)
# noise_freq = magnitude_fft(noise)


# frequencies = np.fft.fftfreq(256, 1/config.HRIR_SAMPLERATE)
# x_labels = frequencies

# step = 10
# indices = np.arange(1, len(noise_freq) + 1)  # Indices starting from 1 to avoid log(0)

# # Set custom x-axis labels at the chosen indices
# plt.xticks(ticks=indices[::step], labels=[x_labels[i] for i in indices[::step] - 1], rotation=45)

# # Set the x-axis to a logarithmic scale
# plt.xscale('log')
# plt.plot(decibels(noise_freq))
# plt.savefig("Noise")
# plt.show()
