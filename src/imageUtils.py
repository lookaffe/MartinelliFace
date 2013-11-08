'''
Created on 26/ott/2013

@author: Laboratorio Creativo Geppetto
'''

import cv2
import numpy as np
import os, sys, copy

#face detector
faceCascade = cv2.CascadeClassifier("faceDet.xml")

#funzione di riconoscimento facciale
def detect(img):
   
    gray_image = img
    rects = faceCascade.detectMultiScale(gray_image, 1.1, 2, cv2.cv.CV_HAAR_SCALE_IMAGE, (10,10))
    if len(rects) == 0:
        return [], img
    #im22 = copy.deepcopy(img)
    #im22 = img[rects[0][1]:rects[0][1]+rects[0][3], rects[0][0]:rects[0][0]+rects[0][2]]
    #cv2.imwrite('../data_pictures/provaFace.jpg',im22)
    #re20 = int(rects[0][3]*0.2)
    #rects[:, 2:] += rects[:, :2]
    #rects[:, 1] += -re20
    #rects[:, 3] += +re20
    rects[:, 2:] += rects[:, :2]
    return rects, img

def read_images(path, sz=None, cr=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):

                if filename.endswith('.jpg'):
                    try:
                        im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                        #print os.path.join(subject_path, filename)
                        # crop the image on the face
                        if (cr is not None):
                            rect, img = detect(im)
                            if len(rect) == 0:
                                return [None,None]
                            im = img[rect[0][1]:rect[0][3], rect[0][0]:rect[0][2]]
                            
                            #im = Image.fromarray(img)
                        # resize to given size (if given)
                        if (sz is not None):
                            #print im, sz
                            im = cv2.resize(im, sz)
                            cv2.imwrite('../data_pictures/prova'+str(c)+'.jpg',im)
                        X.append(np.asarray(im, dtype=np.uint8))
                        y.append(c)
                    except IOError, (errno, strerror):
                        print "I/O error({0}): {1}".format(errno, strerror)
                    except:
                        print "Unexpected error:", sys.exc_info()[0]
                        raise


            c = c+1
    return [X,y]

#mostra due immagini affiancate
def showImage(W, paintsPath, p_label, size):
    h1, w1 = W[0].shape[:2]
    vis = np.zeros((max(h1, h1), w1+w1), np.uint8)
    print paintsPath + "/" + str(p_label)+ "/" + str(p_label) +".jpg"
    onlyfiles = [ f for f in os.listdir(paintsPath+  "/"+str(p_label)) if os.path.isfile(os.path.join(paintsPath+  "/"+str(p_label),f)) ]
    print "onlyfiles",onlyfiles
    img = cv2.imread(paintsPath + "/" + str(p_label)+ "/" +onlyfiles[-1], cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(img, size)
    vis[:h1, :w1] = np.asarray(im, dtype=np.uint8)
    vis[:h1, w1:w1+w1] = W[0]
    return vis