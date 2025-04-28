#%% import libraries
import csv
import serial

#create folder to store data
#folder = 'test'
date = 20250428
pattern = 'hilbert'
trial = '0'

# open the serial port in Arduino
arduinoPort = serial.Serial('COM3', 9600) # depends on the system
arduinoPort.flushInput() #discards all buffers

# initialize csv file
file = open(f'{date}_{pattern}_{trial}.csv', 'w', newline='')
#file.truncate() no reason for this

# gather data
while True:
    #arduinoPort.flush() Flush does not discard all contents in buffer

    # read a line of data and decode it
    serialBytes = arduinoPort.readline()
    decodeBytes = serialBytes.decode('ascii').strip('\r\n')

    # parse the data
    values = decodeBytes.split(',')
    print(values)

    # write data into csv file
    data = csv.writer(file)
    data.writerow(values)

#file.close()
# %%
