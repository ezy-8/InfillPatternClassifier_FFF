#%% 1. libraries and dataset
import pandas as pd
import matplotlib.pyplot as plt

filePath = 'Team_1_FourChannels'

date = 20250502
pattern = 'concentric'
channels = '4'
sampling = '5Hz'
run = '1'

# p - print bed accelerometer; n - nozzle; s - sound sensor (Right, Left)
df = pd.read_csv(filePath + f'/{date}_{pattern}_{sampling}_{channels}_{run}.csv', 
                 names=['Time', 'Yp', 'Yn', 'SoundR', 'SoundL'])
dfNew = df['Time'].str.split(',', expand=True)
dfNew = dfNew.apply(pd.to_numeric, errors='coerce')

# remove first row anomaly at yP (which is the only channel that acts up when data collection starts)
badRow = 1
dfNew = dfNew[badRow:]

#define channels and eliminate bad rows (the first 30 seconds where no printing occurs, but 5Hz so 150 rows)
dropRowValue = 150
dfNew = dfNew[dropRowValue:]

time = dfNew[0]  # time in seconds
yP, yN = dfNew[1], dfNew[2]
sR, sL = dfNew[3], dfNew[4]

print(f'Loaded all data, dropped first {dropRowValue} rows (or equivalent to almost 30 seconds at {sampling})')

# %% Data Restructuring (sample, window size, and features)
import numpy as np

totalSamples = []

for i in range(dropRowValue+1, len(df)):
    totalSamples.append(np.array([yP.loc[i], yN.loc[i], sR.loc[i], sL.loc[i]]))
    
totalSamples = np.array(totalSamples)
print(f'Total samples: {totalSamples.shape}')

# %% visualize windows
i, j = 0, 500  # i and j are the start and end of the window
import matplotlib.pyplot as plt
plt.figure(0)
plt.plot(totalSamples[:, 0][i:j], label='Y nozzle (X)')
plt.figure(1)
plt.plot(totalSamples[:, 1][i:j], label='Y print bed')
plt.figure(2)
plt.plot(totalSamples[:, 2][i:j], label='Sound Right')
plt.figure(3)
plt.plot(totalSamples[:, 3][i:j], label='Sound Left')
plt.legend()

#%% More visualization in different domains
# Select a signal to transform (e.g., yP - Y nozzle)
signal = totalSamples[:, 0][i:j]

sampling_rate = 5  # 5 Hz as per your data

# Compute FFT
fft_values = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), d=1/sampling_rate)  # d=1/sampling_rate is the time step

# Take the magnitude of the FFT
magnitude = np.abs(fft_values)

# Plot the FFT
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])  # Only positive frequencies
plt.title('Frequency Domain Representation (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()
plt.show()

# %% Wavelet Spectrogram
import pywt
import numpy as np
import matplotlib.pyplot as plt

# Define the wavelet and scales
wavelet = 'cmor'  # Complex Morlet wavelet
scales = np.arange(1, 128)  # Range of scales for the wavelet transform

signal = totalSamples[:, 0][i:j]

# Perform Continuous Wavelet Transform (CWT)
coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1/sampling_rate)

# Plot the wavelet spectrogram
plt.figure(figsize=(12, 6))
plt.imshow(np.abs(coefficients), extent=[0, len(signal), frequencies[-1], frequencies[0]],
           aspect='auto', cmap='viridis')
plt.colorbar(label='Magnitude')
plt.title('Wavelet Spectrogram (CWT)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()

#%%
import scipy.stats as stats
from scipy.fft import fft, fftfreq

FEATURES = ['MIN','MAX','MEAN','RMS','VAR','STD','POWER','PEAK','P2P','CREST FACTOR','SKEW','KURTOSIS',
            'MAX_f','SUM_f','MEAN_f','VAR_f','PEAK_f','SKEW_f','KURTOSIS_f']

def features_extraction(df):

    Min=[];Max=[];Mean=[];Rms=[];Var=[];Std=[];Power=[];Peak=[];Skew=[];Kurtosis=[];P2p=[];CrestFactor=[];
    FormFactor=[]; PulseIndicator=[];
    Max_f=[];Sum_f=[];Mean_f=[];Var_f=[];Peak_f=[];Skew_f=[];Kurtosis_f=[]

    X = df
    ## TIME DOMAIN ##

    Min.append(np.min(X))
    Max.append(np.max(X))
    Mean.append(np.mean(X))
    Rms.append(np.sqrt(np.mean(X**2)))
    Var.append(np.var(X))
    Std.append(np.std(X))
    Power.append(np.mean(X**2))
    Peak.append(np.max(np.abs(X)))
    P2p.append(np.ptp(X))
    CrestFactor.append(np.max(np.abs(X))/np.sqrt(np.mean(X**2)))
    Skew.append(stats.skew(X))
    Kurtosis.append(stats.kurtosis(X))
    FormFactor.append(np.sqrt(np.mean(X**2))/np.mean(X))
    PulseIndicator.append(np.max(np.abs(X))/np.mean(X))

    ## FREQ DOMAIN ##
    ft = fft(X)
    S = np.abs(ft**2)/len(df)
    Max_f.append(np.max(S))
    Sum_f.append(np.sum(S))
    Mean_f.append(np.mean(S))
    Var_f.append(np.var(S))
    Peak_f.append(np.max(np.abs(S)))
    Skew_f.append(stats.skew(X))
    Kurtosis_f.append(stats.kurtosis(X))
    
    #Create dataframe from features
    df_features = pd.DataFrame(index = [FEATURES],
                               data = [Min,Max,Mean,Rms,Var,Std,Power,Peak,P2p,CrestFactor,Skew,Kurtosis,
                                       Max_f,Sum_f,Mean_f,Var_f,Peak_f,Skew_f,Kurtosis_f])
    return df_features

FEATURES = features_extraction(signal)
print(FEATURES)

#%%
X = totalSamples[:, 0][i:j]
ft = fft(X)
S = np.abs(ft**2)/len(X)

# %% FIXED (unadaptive) windows, adjust later and do not make adaptive
steps = 100
ch = 4

if len(totalSamples) % steps != 0:
    print('Data length is not divisible by window size.')
    # Truncate the data to make it divisible by the window size
    removal = len(totalSamples) % steps
    # Remove the last few samples
    totalSamplesNew = totalSamples[:-removal]
    print(f'Removed {removal} samples to make the data length divisible by {steps}.')
    # Reshape into 3d array
    totalSamplesNew = totalSamplesNew.reshape(-1, steps, ch)
    print(totalSamplesNew.shape)
else:
    print('Data length is divisible by window size.')
    totalSamplesNew = totalSamples.reshape(-1, steps, ch)
    print(totalSamplesNew.shape)

# %% save preprocessed data
np.save('4 Machine Learning' + f'/{date}_{pattern}_{sampling}_{channels}_{run}_PreprocessedWith{steps}Windows.npy', totalSamplesNew)

print(steps)
# %%
