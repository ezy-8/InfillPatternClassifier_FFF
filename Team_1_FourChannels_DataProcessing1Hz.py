#%% 1. libraries and dataset
import pandas as pd
import matplotlib.pyplot as plt

filePath = 'Team_1_SixChannels'

date = 20250430
pattern = 'concentric'  # 'concentric', 'hilbert', 'honeycomb', 'rectilinear', 'triangle'
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

# %% Data Restructuring (sample, window size, and features)
import numpy as np

totalSamples = []

for i in range(dropRowValue, len(df)):
    totalSamples.append(np.array([xP.loc[i], yP.loc[i], zP.loc[i], yN.loc[i], sR.loc[i], sL.loc[i]]))
    
totalSamples = np.array(totalSamples)
print(f'Total samples: {totalSamples.shape}')

# %% visualize windows
i, j = 0, 5  # i and j are the start and end of the window
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.plot(totalSamples[:, 0][i:j], label='X print bed')
plt.plot(totalSamples[:, 1][i:j], label='Y print bed')
plt.plot(totalSamples[:, 2][i:j], label='Z print bed')
plt.plot(totalSamples[:, 3][i:j], label='Y nozzle')
plt.plot(totalSamples[:, 4][i:j], label='Sound Right')
plt.plot(totalSamples[:, 5][i:j], label='Sound Left')
plt.legend()
plt.title(f'Window of {j} samples')

# %% FIXED (unadaptive) windows, adjust later and do not make adaptive
steps = 50

if len(totalSamples) % steps != 0:
    print('Data length is not divisible by window size.')
    # Truncate the data to make it divisible by the window size
    removal = len(totalSamples) % steps
    # Remove the last few samples
    totalSamplesNew = totalSamples[:-removal]
    print(f'Removed {removal} samples to make the data length divisible by {steps}.')
    # Reshape into 3d array
    totalSamplesNew = totalSamplesNew.reshape(-1, steps, 6) # 6 channels
    print(totalSamplesNew.shape)
else:
    print('Data length is divisible by window size.')
    totalSamplesNew = totalSamples.reshape(-1, steps, 6)
    print(totalSamplesNew.shape)

# %% save preprocessed data
np.save('4 Machine Learning' + f'/{date}_{pattern}_{sampling}_{run}_PreprocessedWith{steps}Windows.npy', totalSamplesNew)

print(steps)
# %%
