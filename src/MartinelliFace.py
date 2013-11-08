'''
Created on 19/ott/2013

@author: Laboratorio Creativo Geppetto
'''

import cv2
import numpy as np
import imageUtils
import training, os

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
def camShot():
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
    training.training(paintsPath, size, model)
    model.load("model.xml")
    
    #Legge l'immagine da riconoscere
    [W, w] = imageUtils.read_images(picPath, size, 1)
    if W is None:
        print("Faccia non riconosciuta")
        scanning = True
        #ser = serial.Serial(serialPort, 9600)
        return None
    
    #Effettua la classificazione
    [p_label, p_confidence] = model.predict(np.asarray(W[0]))

    
    #mostra l'immagine

    #creo un'immagine nera
    bg = np.zeros(scsize, np.uint8)
    vis = imageUtils.showImage(W, paintsPath, p_label, size)
    #metto al centro dell'immagine nera il confronto tra volto rilevato e quadro
    bg[bg.shape[0]/2-vis.shape[0]/2:bg.shape[0]/2+vis.shape[0]/2, bg.shape[1]/2-vis.shape[1]/2:bg.shape[1]/2+vis.shape[1]/2] = vis
    cv2.imshow("display", bg)

    
    #mostra le immagini di confronto per 5 secondi, poi 
    #chiude le finestre e si rimette in ascolto dell'arduino
    cv2.waitKey(5000)
    
    siz = cv2.cv.GetSize(cv2.cv.fromarray(vis))
    
    #thumbnail = cv2.cv.CreateImage( ( siz[0] / 10, siz[1] / 10), 8, 1)
    newsz = (siz[0]/2, siz[1]/2)
    print "newsz", newsz
    thumbnail=cv2.resize(vis, newsz)
    # mostra in dipinto originale

    filename = paintsOriginalPath+ "/" +str(p_label) +"/" + str(p_label)+ ".jpg"
    im = cv2.imread(filename, cv2.CV_LOAD_IMAGE_COLOR)
    #carico il confronto tra le due facce in alto a sinistra
    im[0:thumbnail.shape[0], 0:thumbnail.shape[1],0] = thumbnail
    im[0:thumbnail.shape[0], 0:thumbnail.shape[1],1] = thumbnail
    im[0:thumbnail.shape[0], 0:thumbnail.shape[1],2] = thumbnail
    cv2.imshow('display',im)
    cv2.waitKey(15000)

    cv2.destroyAllWindows()
    scanning = True
    #ser = serial.Serial(serialPort, 9600)


#VARIABILI DA MODIFICARE 

#Path immagini di training
paintsPath = '../data_paintings'
#Path immagine da analizzare
picPath = '../data_pictures'
#Path quadri totali
paintsOriginalPath = '../original_paintings'

#dimensioni delle immagini usate dal riconoscitore
size = (388,540) #259, 360

scsize= (800,1280)


"""
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
        """
#carico le immagini dell'interfaccia
home = cv2.imread("home.jpg", cv2.IMREAD_GRAYSCALE)
wait = cv2.imread("wait.jpg", cv2.IMREAD_GRAYSCALE)

#creo la finestra in cui visualizzare le immagini
cv2.namedWindow("display",cv2.WINDOW_NORMAL);
#cv2.setWindowProperty( "display", 0,1);

#visualizzo l'immagine di benvenuto
cv2.imshow('display',home)
cv2.waitKey()
#visualizzo l'immagine di attesa
cv2.imshow('display',wait)
cv2.waitKey(1)

camShot()
main()