import numpy as np
from multiprocessing import Process, Queue
from Queue import Empty
import cv2
import cv2.cv as cv
from PIL import Image, ImageTk
import time
import Tkinter as tk
from threading import Timer


#testo per tempo mancante
myText=-1
#istante t per i lcalcolo del tempo mancante
#face detector
faceCascade = cv2.CascadeClassifier("faceDet.xml")

#funzione di riconoscimento facciale
def detect(img):
   
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = faceCascade.detectMultiScale(gray_image, 1.1, 2, cv2.cv.CV_HAAR_SCALE_IMAGE, (10,10))
    cv2.cv.CV_HAAR_SCALE_IMAGE
    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    return rects, img

#disegno i rettangoli
def box(rects, img):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
    return img


#tkinter GUI functions----------------------------------------------------------
def quit_(root):
   cv2.VideoCapture(0).release
  
   root.destroy()

#prendo una nuova immagine
def update_image(image_label):
   frame=image_capture()
   im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

   #face detection
   rects, img=detect(im)
   a=box(rects, img)

   #aggiorno l'immagine visualizzata
   raw = Image.fromarray(a)
   b = ImageTk.PhotoImage(image=raw)
   image_label.configure(image=b)
   image_label._image_cache = b  # avoid garbage collection
   root.update()


#aggiornamento del tempo mancante al prossimo scatto e dell'immagine da visualizzare
def update_all(root, image_label,time_label):
   update_image(image_label)
   time_label.configure(text="%d" % myText)
   #root.after(0, func=lambda: update_all(root, image_label,queue))
   root.after(0, func=lambda: update_all(root, image_label, time_label))

#scatto dell'immagine da webcam
def image_capture():

   vidFile = cv2.VideoCapture(0)
   ret, im = vidFile.read()

   return im

#calcolo del countdown
def countdown():

   
   global myText

   if myText<=-0:
      myText=10

   else:
      myText=myText-1

   t = Timer(1.0, countdown)
   t.start()


if __name__ == '__main__':

   root = tk.Tk()
   print 'GUI initialized...'
   image_label = tk.Label(master=root)# label for the video frame
   image_label.pack()

   time_label = tk.Label(master=root)# label for countdown
   time_label.pack()
   print 'GUI image label initialized...'

   #p = Process(target=image_capture, args=(queue,))
   #p.start()


   print 'image capture process has started...'
   # quit button
   quit_button = tk.Button(master=root, text='Quit',command=lambda: quit_(root,))
   quit_button.pack()
   print 'quit button initialized...'
   # setup the update callback
   root.after(0, func=lambda: update_all(root, image_label, time_label))

   print 'root.after was called...'
   countdown()
   root.mainloop()
   print 'mainloop exit'
   print 'image capture process exit'