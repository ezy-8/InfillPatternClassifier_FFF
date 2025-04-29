#%% 1. libraries and dataset
import pandas as pd
import matplotlib.pyplot as plt

date = 20250429
run = 'concentric'
sampling = '1Hz'
trial = '2'

# p - print bed accelerometer; n - nozzle; s - sound sensor (Right, Left)
df = pd.read_csv('Team_1_IncreasedBaudRate' + f'/{date}_{run}_{sampling}_{trial}.csv',
                        names=['Time', 'Xp', 'Yp', 'Zp', 'Xn', 'Yn', 'Zn', 'SoundR', 'SoundL'])
dfNew = df['Time'].str.split(',', expand=True)
dfNew = dfNew.apply(pd.to_numeric, errors='coerce')

# define channels and eliminate bad rows
dropRowValue = 1
time = dfNew[0][dropRowValue:]
xP, yP, zP = dfNew[1][dropRowValue:], dfNew[2][dropRowValue:], dfNew[3][dropRowValue:]
xN, yN, zN = dfNew[4][dropRowValue:], dfNew[5][dropRowValue:], dfNew[6][dropRowValue:]
sR, sL = dfNew[7][dropRowValue:], dfNew[8][dropRowValue:]

print('Loaded all data, dropped first row')

#split data into windows
windowSize = 1000 # 1000 samples = 1000ms = 1s
windowStart = 0
windowEnd = windowStart + windowSize

