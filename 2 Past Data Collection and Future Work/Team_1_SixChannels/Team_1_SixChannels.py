#%% import libraries (done with help from perplexity.ai)
date = 20250502
pattern = 'concentric'
sampling = '5Hz'
run = 1

import serial
import csv

with open(f'{date}_{pattern}_{sampling}_{run}.csv', 'w', newline='') as csvfile:
    try:
        ser = serial.Serial('COM3', 2000000, timeout=1)
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
