#%% Libraries and dataset
import pandas as pd
import matplotlib.pyplot as plt

filePath = 'Team_1_FourChannels'

date = 20250502
pattern = 'triangle'
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

# %% Restructure data (sample, features)
import numpy as np

totalSamples = []

for i in range(dropRowValue+1, len(df)):
    totalSamples.append(np.array([yP.loc[i], yN.loc[i], sR.loc[i], sL.loc[i]]))
    
totalSamples = np.array(totalSamples)
print(f'Total samples: {totalSamples.shape}')

# %% Visualize windows
import matplotlib.pyplot as plt

i, j = 0, 5  # i and j are the start and end of the window

signal_0 = totalSamples[:, 0][i:j]
signal_1 = totalSamples[:, 1][i:j]
signal_2 = totalSamples[:, 2][i:j]
signal_3 = totalSamples[:, 3][i:j]

plt.figure(0)
plt.plot(signal_0, label='Y nozzle (X)', color='r')
plt.legend()
plt.figure(1)
plt.plot(signal_1, label='Y print bed', color='orange')
plt.legend()
plt.figure(2)
plt.plot(signal_2, label='Sound Right', color='green')
plt.legend()
plt.figure(3)
plt.plot(signal_3, label='Sound Left', color='blue')
plt.legend()

#%% Restructure data (sample, window size, features)
steps = 5
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

#%% TIME DOMAIN
import scipy.stats as stats

totalSamplesFlat = totalSamplesNew.reshape(len(totalSamplesNew), -1)

def timeDomainFeatures(signal):
    Min, Max, Mean, RMS, Var, Std, Power, Peak, Skew, Kurtosis = [], [], [], [], [], [], [], [], [], []
    P2P, CrestFactor, FormFactor, PulseIndicator = [], [], [], []
    
    Min.append(np.min(signal))
    Max.append(np.max(signal))
    Mean.append(np.mean(signal))
    RMS.append(np.sqrt(np.mean(signal**2)))
    Var.append(np.var(signal))
    Std.append(np.std(signal))
    Power.append(np.mean(signal**2))
    Peak.append(np.max(np.abs(signal)))
    P2P.append(np.ptp(signal))
    CrestFactor.append(np.max(np.abs(signal))/np.sqrt(np.mean(signal**2)))
    Skew.append(stats.skew(signal))
    Kurtosis.append(stats.kurtosis(signal))
    FormFactor.append(np.sqrt(np.mean(signal**2))/np.mean(signal))
    PulseIndicator.append(np.max(np.abs(signal))/np.mean(signal))
    
    return np.array([Min, Max, Mean, RMS, Var, Std, Power, Peak, Skew, Kurtosis, 
                     P2P, CrestFactor, FormFactor, PulseIndicator])
    
tdf = timeDomainFeatures(totalSamplesFlat)
print(tdf.shape)

#%% Preprocessed features in the time domain
tdf = np.concatenate([signal_0_tdf, signal_1_tdf, signal_2_tdf, signal_3_tdf], axis=1)
print(tdf.shape)

np.save('4 Machine Learning' + f'/{date}_{pattern}_{sampling}_{channels}_{run}_TimeDomainFeatures.npy', tdf)

#%% FREQUENCY DOMAIN
'''
from scipy.fft import fft, fftfreq

ft = fft(X)
S = np.abs(ft**2)/len(df)

Max_f, Sum_f, Mean_f, Var_f, Peak_f, Skew_f, Kurtosis_f = [], [], [], [], [], [], []

Max_f.append(np.max(S))
Sum_f.append(np.sum(S))
Mean_f.append(np.mean(S))
Var_f.append(np.var(S))
Peak_f.append(np.max(np.abs(S)))
Skew_f.append(stats.skew(X))
Kurtosis_f.append(stats.kurtosis(X))

# save preprocessed data
np.save('4 Machine Learning' + f'/{date}_{pattern}_{sampling}_{channels}_{run}_PreprocessedWith{steps}Windows.npy', totalSamplesNew)

print(steps)'''
