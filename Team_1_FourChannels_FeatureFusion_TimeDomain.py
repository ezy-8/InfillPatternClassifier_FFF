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

i, j = 0, 50  # i and j are the start and end of the window

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
def extract_time_domain_features(signal_window):
    features = {
        'mean': np.mean(signal_window),
        'std': np.std(signal_window),
        'rms': np.sqrt(np.mean(signal_window**2)),
        'min': np.min(signal_window),
        'max': np.max(signal_window),
        'variance': np.var(signal_window),
        'peak_to_peak': np.ptp(signal_window),  # max-min
    }
    return features

signal = totalSamples[:, 0]
window_size = 200
step_size = 100

features_list = []

for start in range(0, len(signal) - window_size + 1, step_size):
    window = signal[start:start+window_size]
    features = extract_time_domain_features(window)
    features['window_start'] = start
    features_list.append(features)

df_features = pd.DataFrame(features_list)
print(df_features.head())

#%% Preprocessed features in the time domain
np.save('4 Machine Learning' + f'/{date}_{pattern}_{sampling}_{channels}_{run}_TimeDomainFeatures.npy', tdf)

#%% FREQUENCY DOMAIN
'''
totalSamplesFlat = totalSamplesNew.reshape(len(totalSamplesNew), -1)

tdf = []

for i in range(totalSamplesFlat.shape[0]):
    for j in range(totalSamplesFlat.shape[1]):
        Min = np.min(totalSamplesFlat[i][:, j])
        Max = np.max(totalSamplesFlat[i][:, j])
        Mean = np.mean(totalSamplesFlat[i][:, j])
        RMS = np.sqrt(np.mean(totalSamplesFlat[i][:, j]**2))
        Var = np.var(totalSamplesFlat[i][:, j])
        Std = np.std(totalSamplesFlat[i][:, j])
        Power = np.mean(totalSamplesFlat[i][:, j]**2)
        Peak = np.max(np.abs(totalSamplesFlat[i][:, j]))
        P2P = np.ptp(totalSamplesFlat[i][:, j])
        CrestFactor = np.max(np.abs(totalSamplesFlat[i][:, j]))/np.sqrt(np.mean(totalSamplesFlat[i][:, j]**2))
        FormFactor = np.sqrt(np.mean(totalSamplesFlat[i][:, j]**2))/np.mean(totalSamplesFlat[i][:, j])
        PulseIndicator = np.max(np.abs(totalSamplesFlat[i][:, j]))/np.mean(totalSamplesFlat[i][:, j])
        tdf.append([Min, Max, Mean, RMS, Var, Std, Power, Peak, P2P, 
                    CrestFactor, FormFactor, PulseIndicator])
                    
tdf = np.array(tdf)
print(tdf.shape)

#Skew = stats.skew(totalSamplesFlat)
#Kurtosis = stats.kurtosis(totalSamplesFlat)
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
