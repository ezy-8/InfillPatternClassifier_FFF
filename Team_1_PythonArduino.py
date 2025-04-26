#%% import libraries
import csv
import serial
import os
import time

#create folder to store data
folder = 'test'
date = 20250423
version = '1'

# make another folder to store data if this one already exists
count = 1
while os.path.exists(folder):
    folder = folder[:-1] + str(count)
    count += 1
os.mkdir(folder)

# Depends on system
comPort = 'COM3'
baudRate = 9600

# initialize csv file
file = open(folder + f'/{date}_{version}.csv', 'w', newline='')
#file.truncate()

# seconds of data collection and flush
t = float(input('How long do you want to collect the data for (in seconds)? '))  
timeout = time.time() + t + 1

# open the serial port in Arduino
arduinoPort = serial.Serial(comPort, baudRate)
arduinoPort.flush() ## fixed in 12/11/2023, moved to resolve issues

# gather data
while time.time() < timeout:
   # read a line of data and decode it
   serialBytes = arduinoPort.readline()
   decodeBytes = serialBytes.decode('ascii').strip('\r\n')
   
   # parse the data
   values = decodeBytes.split(',')
   print(values)
   
   # write data into csv file
   data = csv.writer(file)
   data.writerow(values)
file.close()
# %%
