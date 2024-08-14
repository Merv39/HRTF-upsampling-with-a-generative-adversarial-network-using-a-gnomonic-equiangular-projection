import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfreqz, iirfilter
import biquad

# Function to plot frequency response
def plot_frequency_response(sos, fs, label):
    w, h = sosfreqz(sos, worN=2000, fs=fs)
    plt.plot(w, 20 * np.log10(np.abs(h)), label=label)


# Sampling frequency
fs = 48000  # 48 kHz

plt.figure(figsize=(15, 10))

# Low-pass at 10 kHz
sos = butter(2, 10000, btype='low', fs=fs, output='sos')
plot_frequency_response(sos, fs, 'Low-pass at 10 kHz')

# Low-pass at 6 kHz
sos = butter(2, 6000, btype='low', fs=fs, output='sos')
plot_frequency_response(sos, fs, 'Low-pass at 6 kHz')

# High-pass at 2 kHz
sos = butter(2, 2000, btype='high', fs=fs, output='sos')
plot_frequency_response(sos, fs, 'High-pass at 2 kHz')

# High-pass at 1 kHz
sos = butter(2, 1000, btype='high', fs=fs, output='sos')
plot_frequency_response(sos, fs, 'High-pass at 1 kHz')

# Low-Shelf of +12 dB at 5 kHz
f = biquad.lowshelf(sr=fs, f=5000, g=12)
impulse_time = np.zeros(fs)
impulse_time[0] = 1.0
filtered_time = f(impulse_time)
filtered_freq = np.fft.fft(filtered_time)
plt.plot(20 * np.log10(np.abs(filtered_freq[:fs//2])), label="Low-Shelf -12 dB at 5 kHz")

# # Low-Shelf of -12 dB at 5 kHz
f = biquad.lowshelf(sr=fs, f=5000, g=-12)
impulse_time = np.zeros(fs)
impulse_time[0] = 1.0
filtered_time = f(impulse_time)
filtered_freq = np.fft.fft(filtered_time)
plt.plot(20 * np.log10(np.abs(filtered_freq[:fs//2])), label="Low-Shelf +12 dB at 5 kHz")

# High-Shelf of +12 dB at 5 kHz
f = biquad.highshelf(sr=fs, f=5000, g=12)
impulse_time = np.zeros(fs)
impulse_time[0] = 1.0
filtered_time = f(impulse_time)
filtered_freq = np.fft.fft(filtered_time)
plt.plot(20 * np.log10(np.abs(filtered_freq[:fs//2])), label="High-Shelf +12 dB at 5 kHz")

# High-Shelf of -12 dB at 5 kHz
f = biquad.highshelf(sr=fs, f=5000, g=-12)
impulse_time = np.zeros(fs)
impulse_time[0] = 1.0
filtered_time = f(impulse_time)
filtered_freq = np.fft.fft(filtered_time)
plt.plot(20 * np.log10(np.abs(filtered_freq[:fs//2])), label="High-Shelf -12 dB at 5 kHz")

# Band-pass from 1 to 10 kHz
sos = butter(2, [1000, 10000], btype='band', fs=fs, output='sos')
plot_frequency_response(sos, fs, 'Band-pass from 1 to 10 kHz')

# Band-pass from 2 to 6 kHz
sos = butter(2, [2000, 6000], btype='band', fs=fs, output='sos')
plot_frequency_response(sos, fs, 'Band-pass from 2 to 6 kHz')

# Band-stop from 5 to 10 kHz
sos = butter(2, [5000, 10000], btype='bandstop', fs=fs, output='sos')
plot_frequency_response(sos, fs, 'Band-stop from 5 to 10 kHz')

# Band-stop from 10 to 15 kHz
sos = butter(2, [10000, 15000], btype='bandstop', fs=fs, output='sos')
plot_frequency_response(sos, fs, 'Band-stop from 10 to 15 kHz')

# Finalizing the plot
plt.title('Frequency Responses of Various Filters (2nd Order IIR Filters)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.ylim([-50, 20])

# plt.xscale('log')
plt.xlim(20, 20000)  # Optional: Set x-axis limits

plt.legend()
plt.grid(True)
plt.show()
