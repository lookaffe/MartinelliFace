'''
Created on 26/ott/2013

@author: Cento
'''
import serial

serialPort = 'COM11'
scanning = True

ser = serial.Serial(serialPort, 9600)
while scanning:
    print ser.readline()