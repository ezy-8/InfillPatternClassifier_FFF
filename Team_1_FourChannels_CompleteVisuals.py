#%% 1. libraries and dataset
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
figureOne.savefig('3 Figures' + f'/Acoustic Channels for {date}_{pattern}_{sampling}_{channels}_{run}.png')

#%% vibration channels of print bed accelerometer
figureTwo, axes = plt.subplots(2, 1, figsize=(20, 20))

vibrationTitles = ['Print Bed Movement [Y]', 'Nozzle Movement [Y]']

allVibrations = [yP, yN]

for i, ax in enumerate(axes.flat):
    ax.set_title(vibrationTitles[i])
    ax.plot(allVibrations[i]) 
    ax.set_xlabel('Samples')
    ax.set_ylabel('Acceleration (m/s^2)')
plt.tight_layout()

# save results
figureTwo.savefig('3 Figures' + f'/Accelerometer Channels for {date}_{pattern}_{sampling}_{channels}_{run}.png')

print(len(yP), len(yN), len(sR), len(sL))

# %%
