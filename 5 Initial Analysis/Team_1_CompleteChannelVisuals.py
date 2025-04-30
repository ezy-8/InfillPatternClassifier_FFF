#%% 1. libraries and dataset
import pandas as pd
import matplotlib.pyplot as plt

date = 20250429
run = 'concentric'
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

#%% 3. Visualization
# acoustic channels
figureOne, axes = plt.subplots(2, 1, figsize=(20, 20))

acousticTitles = ['Sound Sensor Right', 'Sound Sensor Left']
allSounds = [sR, sL]

for i, ax in enumerate(axes.flat):
    ax.set_title(acousticTitles[i])
    ax.plot(time, allSounds[i]) 
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude (volts)')
plt.tight_layout()

# save results
figureOne.savefig(f'Acoustic Channels for {date}_{run}_{sampling}_{trial}.png')

#%% vibration channels of print bed accelerometer
figureTwo, axes = plt.subplots(6, 1, figsize=(20, 20))

vibrationTitles = ['Print Bed Movement [X]', 'Print Bed Movement [Y]', 'Print Bed Movement [Z]',
          'Nozzle Movement [X]', 'Nozzle Movement [Y]', 'Nozzle Movement [Z]']

allVibrations = [xP, yP, zP, xN, yN, zN]

for i, ax in enumerate(axes.flat):
    ax.set_title(vibrationTitles[i])
    ax.plot(time, allVibrations[i]) 
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude (volts)')
plt.tight_layout()

# save results
figureTwo.savefig(f'Accelerometer Channels for {date}_{run}_{sampling}_{trial}.png')

# %% Data preprocessing [METHOD - FIND THE START OF THE PRINT]

