#%% 1. libraries and dataset
import pandas as pd
import matplotlib.pyplot as plt

filePath = 'Team_1_SixChannels'

date = 20250430
pattern = 'triangle'
sampling = '1Hz6Ch'
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
from scipy.fft import rfft, rfftfreq
fourierTransform = rfft(sR)  # FFT of the right sound sensor
sampling_rate = 1  # Hz, adjust as needed
frequencies = rfftfreq(len(sR), d=1/sampling_rate)

amplitude = 2 * np.abs(fourierTransform) / len(sR)

plt.plot(frequencies, amplitude)
plt.title('Frequency Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.show()
# %%
