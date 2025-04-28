#%% import libraries (done with help from perplexity.ai)
date = 20250428
pattern = 'hilbert'
trial = '0'

import serial
import csv

with open(f'{date}_{pattern}_{trial}.csv', 'w', newline='') as csvfile:
    try:
        ser = serial.Serial('COM3', 9600, timeout=1)  # Replace 'COM3' with your port
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
