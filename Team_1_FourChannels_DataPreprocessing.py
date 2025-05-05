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
i, j = 0, 50  # i and j are the start and end of the window
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.plot(totalSamples[:, 0][i:j], label='Y nozzle')
plt.plot(totalSamples[:, 1][i:j], label='Y print bed')
plt.plot(totalSamples[:, 2][i:j], label='Sound Right')
plt.plot(totalSamples[:, 3][i:j], label='Sound Left')
plt.legend()
plt.title(f'Window of {j} samples')

# %% FIXED (unadaptive) windows, adjust later and do not make adaptive
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

# %% save preprocessed data
np.save('4 Machine Learning' + f'/{date}_{pattern}_{sampling}_{channels}_{run}_PreprocessedWith{steps}Windows.npy', totalSamplesNew)

print(steps)
# %%
