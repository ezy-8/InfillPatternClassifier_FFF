#%% import libraries (done with help from perplexity.ai)
date = 20250428
pattern = 'concentric'
sampling = '0.5Hz'
trial = 1

import serial
import csv

with open(f'{date}_{pattern}_{sampling}_{trial}.csv', 'w', newline='') as csvfile:
    try:
        ser = serial.Serial('COM4', 2000000, timeout=1)
        writer = csv.writer(csvfile)
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line:
                print(line)  # or process/save the data
                writer.writerow([line])
    except Exception as e:
        print(f'Error: {e}')
    finally:
        ser.close()

# %%
