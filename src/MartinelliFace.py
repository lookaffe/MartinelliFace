'''
Created on 19/ott/2013

@author: Laboratorio Creativo Geppetto
'''
#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2012, Philipp Wagner <bytefish[at]gmx[dot]de>.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the author nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# ------------------------------------------------------------------------------------------------
# Note:
# When using the FaceRecognizer interface in combination with Python, please stick to Python 2.
# Some underlying scripts like create_csv will not work in other versions, like Python 3.
# ------------------------------------------------------------------------------------------------

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

def cumShot():
    vidFile = cv2.VideoCapture(0)
    ret, im = vidFile.read()
    cv2.imwrite('../data_pictures/picture/image.jpg',im)
    cv2.VideoCapture(0).release

def main():    
    # Create a model and training
    model = training.createModel()
    #training.training(paintsPath, size, model)
    model.load("model.xml")
    
    # We now get a prediction from the model! In reality you
    # should always use unseen images for testing your model.
    # But so many people were confused, when I sliced an image
    # off in the C++ version, so I am just using an image we
    # have trained with.
    #
    # model.predict is going to return the predicted label and
    # the associated confidence:
    [W, w] = imageUtils.read_images(picPath, size, 1)
    if W is None:
        print("Faccia non riconosciuta")
        exit()
    [p_label, p_confidence] = model.predict(np.asarray(W[0]))
    # Print it:
    #print "Predicted label = %d (confidence=%.2f)" % (p_label, p_confidence)
    
    #cv2.imshow("me", W[0])
    #cv2.imshow("Doppelganger", W[0])
    
    
    vis = imageUtils.showImage(W, paintsPath, p_label, size)
    cv2.imshow("merge", vis)
    # Cool! Finally we'll plot the Eigenfaces, because that's
    # what most people read in the papers are keen to see.
    #
    # Just like in C++ you have access to all model internal
    # data, because the cv::FaceRecognizer is a cv::Algorithm.
    #
    # You can see the available parameters with getParams():
    #print model.getParams()
    # Now let's get some data:
    #mean = model.getMat("mean")
    #eigenvectors = model.getMat("eigenvectors")
    # We'll save the mean, by first normalizing it:
    #mean_norm = normalize(mean, 0, 255, dtype=np.uint8)
    #mean_resized = mean_norm.reshape(X[0].shape)
    #if out_dir is None:
    #    cv2.imshow("mean", mean_resized)
    #else:
    #    cv2.imwrite("%s/mean.png" % (out_dir), mean_resized)
    # Turn the first (at most) 16 eigenvectors into grayscale
    # images. You could also use cv::normalize here, but sticking
    # to NumPy is much easier for now.
    # Note: eigenvectors are stored by column:
    
    #for i in xrange(min(len(X), 16)):
    #    eigenvector_i = eigenvectors[:,i].reshape(X[0].shape)
    #    eigenvector_i_norm = normalize(eigenvector_i, 0, 255, dtype=np.uint8)
        # Show or save the images:
    #    if out_dir is None:
    #        cv2.imshow("%s/eigenface_%d" % (out_dir,i), eigenvector_i_norm)
    #    else:
    #        cv2.imwrite("%s/eigenface_%d.png" % (out_dir,i), eigenvector_i_norm)
    # Show the images:
    #if out_dir is None:
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    global scanning
    scanning = True
    global ser
    ser = serial.Serial(serialPort, 9600)


# Take a picture
paintsPath = '../data_paintings'
picPath = '../data_pictures'
size = (259,360)

serialPort = 'COM11'
scanning = True

ser = serial.Serial(serialPort, 9600)
while scanning:
    print ser.read()
    if ser.read()=='1':
        ser.close()
        scanning = False
        cumShot()
        main()


