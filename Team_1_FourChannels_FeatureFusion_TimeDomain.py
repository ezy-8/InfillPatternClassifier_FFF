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

# %% Extract time domain features
printBedTDF, nozzleTDF = [], []
soundRightTDF, soundLeftTDF = [], []

import scipy.stats as stats

for i in printBedNew:
    printBedTDF.append([
        np.min(i), np.max(i), np.mean(i), np.sqrt(np.mean(i**2)), np.var(i),
        np.std(i), np.mean(i**2), np.max(np.abs(i)), np.ptp(i),
        np.max(np.abs(i))/np.sqrt(np.mean(i**2)), stats.skew(i), stats.kurtosis(i),
        np.sqrt(np.mean(i**2))/np.mean(i), np.max(np.abs(i))/np.mean(i)
        ])
for j in nozzleNew:
    nozzleTDF.append([
        np.min(j), np.max(j), np.mean(j), np.sqrt(np.mean(j**2)), np.var(j),
        np.std(j), np.mean(j**2), np.max(np.abs(j)), np.ptp(j),
        np.max(np.abs(j))/np.sqrt(np.mean(j**2)), stats.skew(j), stats.kurtosis(j),
        np.sqrt(np.mean(j**2))/np.mean(j), np.max(np.abs(j))/np.mean(j)
        ])
for k in soundRightNew:
    soundRightTDF.append([
        np.min(k), np.max(k), np.mean(k), np.sqrt(np.mean(k**2)), np.var(k),
        np.std(k), np.mean(k**2), np.max(np.abs(k)), np.ptp(k),
        np.max(np.abs(k))/np.sqrt(np.mean(k**2)), stats.skew(k), stats.kurtosis(k),
        np.sqrt(np.mean(k**2))/np.mean(k), np.max(np.abs(k))/np.mean(k)
        ])
for l in soundLeftNew:
    soundLeftTDF.append([
        np.min(l), np.max(l), np.mean(l), np.sqrt(np.mean(l**2)), np.var(l),
        np.std(l), np.mean(l**2), np.max(np.abs(l)), np.ptp(l),
        np.max(np.abs(l))/np.sqrt(np.mean(l**2)), stats.skew(l), stats.kurtosis(l),
        np.sqrt(np.mean(l**2))/np.mean(l), np.max(np.abs(l))/np.mean(l)
        ])

printBedTDF, nozzleTDF = np.array(printBedTDF), np.array(nozzleTDF)
soundRightTDF, soundLeftTDF = np.array(soundRightTDF), np.array(soundLeftTDF)

print('Print Bed:', printBedTDF.shape)
print('Nozzle:', nozzleTDF.shape)
print('Sound Right:', soundRightTDF.shape)
print('Sound Left:', soundLeftTDF.shape)

#%% Combine data
tdf = np.concatenate([printBedTDF, nozzleTDF, soundRightTDF, soundLeftTDF])
print(tdf.shape)

#%% Preprocessed features in the time domain
np.save('4 Machine Learning' + f'/0 {date}_{pattern}_{sampling}_{channels}_{run}_TimeDomainFeatures_{steps}.npy', tdf)

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