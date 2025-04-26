#%% 1. libraries and dataset
import pandas as pd
import matplotlib.pyplot as plt

date = 20250426
run = 'fiveMin'
trial = '0'

dataFrame = pd.read_csv('Team_1\\Data' + f'/{date}_{run}_{trial}.csv')
print(dataFrame.shape)

#%% 2. define channels [METHOD - DROP USELESS ROWS]
dropRowValue = 1 # dropped first row of data if it deviates too much

time = dataFrame.iloc[:, 0] / (1000 * 60) # convert to minutes
time = time[1:]

acousticOne = dataFrame['Sound'][dropRowValue:]

vibX1 = dataFrame['X'][dropRowValue:]
vibY1 = dataFrame['Y'][dropRowValue:]
vibZ1 = dataFrame['Z'][dropRowValue:]

print('Loaded all data, dropped first row')

#%% 3. Visualization
# acoustic channels
plt.figure(0, figsize=(15, 15))
plt.title('Acoustics [Side]')
plt.xlabel('Time (minutes)')
plt.ylabel('Amplitude (volts)')
plt.plot(time, acousticOne)

# vibration channels of print bed accelerometer
figureTwo, axes = plt.subplots(3, 1, figsize=(15, 15))

titles = ['Print Bed Movement [X]', 'Print Bed Movement [Y]', 'Print Bed Movement [Z]']
allVibration1 = [vibX1, vibY1, vibZ1]

for i, ax in enumerate(axes.flat):
    ax.set_title(titles[i])
    ax.plot(time, allVibration1[i]) 
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Amplitude (volts)')
plt.tight_layout()

# save results

figureTwo.savefig('C:\\Users\\dk1023\\Downloads\\0 Poster Images' + f'/Movements of Print Bed Accelerometer for {run} {trial}')

# %% Data preprocessing [METHOD - FIND THE START OF THE PRINT]

