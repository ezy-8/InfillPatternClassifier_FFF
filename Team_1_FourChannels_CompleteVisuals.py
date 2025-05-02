#%% 1. libraries and dataset
import pandas as pd
import matplotlib.pyplot as plt

filePath = 'Team_1_SixChannels'

date = 20250502
pattern = 'concentric'
sampling = '5Hz'
run = '1'

# p - print bed accelerometer; n - nozzle; s - sound sensor (Right, Left)
df = pd.read_csv(filePath + f'/{date}_{pattern}_{sampling}_{run}.csv',
                        names=['Time', 'Xp', 'Yp', 'Zp', 'Yn', 'SoundR', 'SoundL'])
dfNew = df['Time'].str.split(',', expand=True)
dfNew = dfNew.apply(pd.to_numeric, errors='coerce')

# define channels and eliminate bad rows (the first 40 seconds where no printing occurs)
dropRowValue = 30
dfNew = dfNew[dropRowValue:]

# replace anomalies at yP (which is the only channel that acts up)
minIndex = dfNew[2].idxmin()
dfNew.at[minIndex, 2] = dfNew[2][minIndex - 1]  # replace with the previous value

time = dfNew[0]  # time in seconds
xP, yP, zP = dfNew[1], dfNew[2], dfNew[3]
yN = dfNew[4]
sR, sL = dfNew[5], dfNew[6]

print(f'Loaded all data, dropped first {dropRowValue} rows (or equivalent to almost {dropRowValue} seconds at {sampling})')

#%% 3. Visualization
# acoustic channels
figureOne, axes = plt.subplots(2, 1, figsize=(20, 20))

acousticTitles = [f'Sound Sensor Right {pattern}', f'Sound Sensor Left {pattern}']
allSounds = [sR, sL]

for i, ax in enumerate(axes.flat):
    ax.set_title(acousticTitles[i])
    ax.plot(allSounds[i]) 
    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude (volts)')
plt.tight_layout()

# save results
figureOne.savefig('3 Figures' + f'/Acoustic Channels for {date}_{pattern}_{sampling}_{run}.png')

#%% vibration channels of print bed accelerometer
figureTwo, axes = plt.subplots(4, 1, figsize=(20, 20))

vibrationTitles = ['Print Bed Movement [X]', 'Print Bed Movement [Y]', 
                   'Print Bed Movement [Z]', 'Nozzle Movement [Y]']

allVibrations = [xP, yP, zP, yN]

for i, ax in enumerate(axes.flat):
    ax.set_title(vibrationTitles[i])
    ax.plot(allVibrations[i]) 
    ax.set_xlabel('Samples')
    ax.set_ylabel('Acceleration (m/s^2)')
plt.tight_layout()

# save results
figureTwo.savefig('3 Figures' + f'/Accelerometer Channels for {date}_{pattern}_{sampling}_{run}.png')

print(len(xP), len(yP), len(zP), len(yN), len(sR), len(sL))
# %% Fourier transform
import numpy as np

# Compute the FFT
signal = xP  # Choose one of the signals to analyze

fft_signal = np.fft.fft(signal)
sample_rate = 1
frequencies = np.fft.fftfreq(signal.size, d=1/sample_rate)

# Plot the frequency spectrum
plt.plot(frequencies[:signal.size//2], np.abs(fft_signal)[:signal.size//2])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title(f"Frequency Spectrum - xP")
plt.grid(True)
plt.show()

# %%
# Compute the FFT
signal = yP  # Choose one of the signals to analyze

fft_signal = np.fft.fft(signal)
sample_rate = 1
frequencies = np.fft.fftfreq(signal.size, d=1/sample_rate)

# Plot the frequency spectrum
plt.plot(frequencies[:signal.size//2], np.abs(fft_signal)[:signal.size//2])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Frequency Spectrum - yP")
plt.grid(True)
plt.show()
# %%
# Compute the FFT
signal = yN  # Choose one of the signals to analyze

fft_signal = np.fft.fft(signal)
sample_rate = 1
frequencies = np.fft.fftfreq(signal.size, d=1/sample_rate)

# Plot the frequency spectrum
plt.plot(frequencies[:signal.size//2], np.abs(fft_signal)[:signal.size//2])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Frequency Spectrum - yN")
plt.grid(True)
plt.show()
# %%
