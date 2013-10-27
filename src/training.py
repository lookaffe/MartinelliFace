'''
Created on 26/ott/2013

@author: Cento
'''
import cv2
import numpy as np
import imageUtils

def training(paintsPath, size, model):
    # Now read in the image data. This must be a valid path!
    [X,y] = imageUtils.read_images(paintsPath, size)
    # Convert labels to 32bit integers. This is a workaround for 64bit machines,
    # because the labels will truncated else. This will be fixed in code as
    # soon as possible, so Python users don't need to know about this.
    # Thanks to Leo Dirac for reporting:
    y = np.asarray(y, dtype=np.int32)
    
    # If a out_dir is given, set it:
    #if len(sys.argv) == 3:
    #    out_dir = sys.argv[2]
        # Readc
    # Learn the model. Remember our function returns Python lists,
    # so we use np.asarray to turn them into NumPy lists to make
    # the OpenCV wrapper happy:
    model.train(np.asarray(X), np.asarray(y))
    model.save("model.xml")
    
def createModel():
    # Create the Eigenfaces model. We are going to use the default
    # parameters for this simple example, please read the documentation
    # for thresholding:
    model = cv2.createFisherFaceRecognizer()
    return model