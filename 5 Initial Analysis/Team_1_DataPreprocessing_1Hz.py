#%% 1. libraries and dataset
import pandas as pd
import matplotlib.pyplot as plt

date = 20250429
run = 'triangle'
sampling = '1Hz'
trial = '1'

# p - print bed accelerometer; n - nozzle; s - sound sensor (Right, Left)
df = pd.read_csv('Team_1_IncreasedBaudRate' + f'/{date}_{run}_{sampling}_{trial}.csv',
                        names=['Time', 'Xp', 'Yp', 'Zp', 'Xn', 'Yn', 'Zn', 'SoundR', 'SoundL'])
dfNew = df['Time'].str.split(',', expand=True)
dfNew = dfNew.apply(pd.to_numeric, errors='coerce')

# define channels and eliminate bad rows (the first few where no printing is indicated)
dropRowValue = 40
dfNew = dfNew[dropRowValue:]

time = dfNew[0]  # time in seconds
xP, yP, zP = dfNew[1], dfNew[2], dfNew[3]
xN, yN, zN = dfNew[4], dfNew[5], dfNew[6]
sR, sL = dfNew[7], dfNew[8]

print(f'Loaded all data, dropped first 40 rows (or equivalent to almost {dropRowValue} seconds at {sampling})')

# %% Data Restructuring (sample, window size, and features)
import numpy as np

totalSamples = []

for i in range(dropRowValue, len(df)):
    totalSamples.append(np.array([xP.loc[i], yP.loc[i], zP.loc[i],
                              xN.loc[i], yN.loc[i], zN.loc[i],
                              sR.loc[i], sL.loc[i]]))
    
totalSamples = np.array(totalSamples)
print(f'Total samples: {totalSamples.shape}')

# %% visualize windows
i, j = 0, 10  # i and j are the start and end of the window
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(totalSamples[:, 0][i:j], label='X print bed')
plt.plot(totalSamples[:, 1][i:j], label='Y print bed')
plt.plot(totalSamples[:, 2][i:j], label='Z print bed')
plt.plot(totalSamples[:, 3][i:j], label='X nozzle')
plt.plot(totalSamples[:, 4][i:j], label='Y nozzle')
plt.plot(totalSamples[:, 5][i:j], label='Z nozzle')
plt.plot(totalSamples[:, 6][i:j], label='Sound Right')
plt.plot(totalSamples[:, 7][i:j], label='Sound Left')
plt.legend()
plt.title('Window of 100 samples')

# %% FIXED (unadaptive) windows, adjust later and do not make adaptive
steps = 5

if len(totalSamples) % steps != 0:
    print('Data length is not divisible by window size.')
    # Truncate the data to make it divisible by the window size
    removal = len(totalSamples) % steps
    # Remove the last few samples
    totalSamplesNew = totalSamples[:-removal]
    print(f'Removed {removal} samples to make the data length divisible by {steps}.')
    # Reshape into 3d array
    totalSamplesNew = totalSamplesNew.reshape(-1, steps, 8)
    print(totalSamplesNew.shape)
else:
    print('Data length is divisible by window size.')
    totalSamplesNew = totalSamples.reshape(-1, steps, 8)
    print(totalSamplesNew.shape)

# %% save preprocessed data
np.save('4 Machine Learning' + f'/{date}_{run}_{sampling}_{trial}_PreprocessedWith{steps}Windows.npy', totalSamplesNew)

# %%
