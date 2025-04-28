#%% import libraries
import csv
import serial

#create folder to store data
#folder = 'test'
date = 20250428
pattern = 'interruptbutton'
trial = '1'

'''# make another folder to store data if this one already exists
count = 1
while os.path.exists(folder):
    folder = folder[:-1] + str(count)
    count += 1
os.mkdir(folder)'''

# Depends on system
comPort = 'COM3'
baudRate = 9600

# open the serial port in Arduino
arduinoPort = serial.Serial(comPort, baudRate)
arduinoPort.flushInput() #discards all buffers
arduinoPort.flushOutput()

# initialize csv file
file = open(f'{date}_{pattern}_{trial}.csv', 'w', newline='')
#file.truncate() no reason for this

# seconds of data collection and flush
#t = float(input('How long do you want to collect the data for (in seconds)? '))  
#timeout = t * 1000 #time.time()

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
