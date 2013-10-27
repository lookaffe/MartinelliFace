'''
Created on 19/ott/2013

@author: Laboratorio Creativo Geppetto
'''

import cv2
import numpy as np
import imageUtils
import training,time
import serial

def normalize(X, low, high, dtype=None):
    """Normalizes a given array in X to a value between low and high."""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

#prende un'immagine dalla webcam
def cumShot():
    vidFile = cv2.VideoCapture(0)
    ret, im = vidFile.read()
    cv2.imwrite('../data_pictures/picture/image.jpg',im)
    cv2.VideoCapture(0).release

#creazione del modello del riconoscitore, eventuale training e predizione
def main():    
    
    global scanning
    global ser
    
    # Creazione modello
    model = training.createModel()

    #TOGLIERE IL COMMENTO ALLA RIGA QUI SOTTO PER AGGIUNGERE LA FASE DI TRAINING
    #training.training(paintsPath, size, model)
    model.load("model.xml")
    
    #Legge l'immagine da riconoscere
    [W, w] = imageUtils.read_images(picPath, size, 1)
    if W is None:
        print("Faccia non riconosciuta")
        scanning = True
        ser = serial.Serial(serialPort, 9600)
        return None
    
    #Effettua la classificazione
    [p_label, p_confidence] = model.predict(np.asarray(W[0]))

    #mostra l'immagine
    vis = imageUtils.showImage(W, paintsPath, p_label, size)
    cv2.imshow("merge", vis)
    
    #mostra le immagini di confronto per 5 secondi, poi 
    #chiude le finestre e si rimette in ascolto dell'arduino
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    scanning = True
    ser = serial.Serial(serialPort, 9600)


#VARIABILI DA MODIFICARE 

#Path immagini di training
paintsPath = '../data_paintings'
#Path immagine da analizzare
picPath = '../data_pictures'
#dimensioni delle immagini usate dal riconoscitore
size = (259,360)
#porta seriale dell'Arduino
serialPort = 'COM11'
scanning = True

#Loop di ascolto dell'Arduino
ser = serial.Serial(serialPort, 9600)
while scanning:
    print ser.read()
    if ser.read()=='1':
        ser.close()
        scanning = False
        cumShot()
        main()


