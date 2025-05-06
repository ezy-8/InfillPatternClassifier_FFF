#%% Libraries and dataset
import pandas as pd
import matplotlib.pyplot as plt

filePath = 'Team_1_FourChannels'

date = 20250502
pattern = 'concentric'
channels = '4'
sampling = '5Hz'
run = '1'
steps = 100

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

# %% Restructure data (channel, all dimensions)
import numpy as np

printBed = []
nozzle = []
soundRight = []
soundLeft = []

for i in range(dropRowValue+1, len(df)):
    printBed.append(yP.loc[i])
    nozzle.append(yN.loc[i])
    soundRight.append(sR.loc[i])
    soundLeft.append(sL.loc[i])

printBed = np.array(printBed)
nozzle = np.array(nozzle)
soundRight = np.array(soundRight)
soundLeft = np.array(soundLeft)

print('Print Bed:', printBed.shape)
print('Nozzle:', nozzle.shape)
print('Sound Right:', soundRight.shape)
print('Sound Left:', soundLeft.shape)

# %% Restructure data (chunking each dimension from each channel)
if len(printBed) % steps != 0:
    removal = len(printBed) % steps

    printBedNew, nozzleNew = printBed[:-removal], nozzle[:-removal]
    soundRightNew, soundLeftNew = soundRight[:-removal], soundLeft[:-removal]

    printBedNew, nozzleNew = printBedNew.reshape(-1, steps), nozzleNew.reshape(-1, steps)
    soundRightNew, soundLeftNew = soundRightNew.reshape(-1, steps), soundLeftNew.reshape(-1, steps)

    print(f'Data length is not divisible by window size. Removed {removal} samples to make the data length divisible by {steps}.')
else:
    printBedNew, nozzleNew = printBed.reshape(-1, steps), nozzle.reshape(-1, steps)
    soundRightNew, soundLeftNew = soundRight.reshape(-1, steps), soundLeft.reshape(-1, steps)

    print('Data length is divisible by window size.')

print('Print Bed:', printBedNew.shape)
print('Nozzle:', nozzleNew.shape)
print('Sound Right:', soundRightNew.shape)
print('Sound Left:', soundLeftNew.shape)

# %% Extract frequency domain features
from scipy.fft import fft
import scipy.stats as stats

printBedNew, nozzleNew = np.abs(fft(printBedNew**2) / len(printBedNew)), np.abs(fft(nozzleNew**2) / len(nozzleNew))
soundRightNew, soundLeftNew = np.abs(fft(soundRightNew**2) / len(soundRightNew)), np.abs(fft(soundLeftNew**2) / len(soundLeftNew))

printBedFDF, nozzleFDF = [], []
soundRightFDF, soundLeftFDF = [], []

for i in printBedNew:
    printBedFDF.append([np.max(i), np.sum(i), np.mean(i), np.var(i), np.max(np.abs(i)), stats.skew(i), stats.kurtosis(i)])
for j in nozzleNew:
    nozzleFDF.append([np.max(j), np.sum(j), np.mean(j), np.var(j), np.max(np.abs(j)), stats.skew(j), stats.kurtosis(j)])
for k in soundRightNew:
    soundRightFDF.append([np.max(k), np.sum(k), np.mean(k), np.var(k), np.max(np.abs(k)), stats.skew(k), stats.kurtosis(k)])
for l in soundLeftNew:
    soundLeftFDF.append([np.max(l), np.sum(l), np.mean(l), np.var(l), np.max(np.abs(l)), stats.skew(l), stats.kurtosis(l)])

printBedFDF, nozzleFDF = np.array(printBedFDF), np.array(nozzleFDF)
soundRightFDF, soundLeftFDF = np.array(soundRightFDF), np.array(soundLeftFDF)

print('Print Bed:', printBedFDF.shape)
print('Nozzle:', nozzleFDF.shape)
print('Sound Right:', soundRightFDF.shape)
print('Sound Left:', soundLeftFDF.shape)

#%% Combine data
tdf = np.concatenate([printBedFDF, nozzleFDF, soundRightFDF, soundLeftFDF])
print(tdf.shape)

#%% Preprocessed features in the time domain
np.save('4 Machine Learning' + f'/1 {date}_{pattern}_{sampling}_{channels}_{run}_FrequencyDomainFeatures_{steps}.npy', tdf)

# %% Visualize windows
import matplotlib.pyplot as plt

i, j = 0, steps  # i and j are the start and end of the window

signal_0 = printBedNew[:, 0][i:j]
signal_1 = nozzleNew[:, 1][i:j]
signal_2 = soundRightNew[:, 2][i:j]
signal_3 = soundLeftNew[:, 3][i:j]

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
