#%% import libraries (done with help from perplexity.ai)
import serial

ser = serial.Serial('COM4', 9600, timeout=1)  # Replace 'COM3' with your port

while True:
    line = ser.readline().decode('utf-8').strip()
    if line:
        print(line)

# %%
