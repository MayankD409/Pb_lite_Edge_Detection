#!/usr/bin/env python3

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Mayank Deshpande (msdeshp4@umd.edu)
M.Eng. in Robotics,
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import tensorflow._api.v2.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import cv2
import os
import sys
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import *
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from io import StringIO
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
    """
    Inputs: 
    BasePath - Path to images
    Outputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]
    DataPath = []
    NumImages = len(glob.glob(BasePath+'*.png'))
    SkipFactor = 1
    for count in range(1,NumImages+1,SkipFactor):
        DataPath.append(BasePath + str(count) + '.png')

    return ImageSize, DataPath
    
def ReadImages(ImageSize, DataPath):
    """
    Inputs: 
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    
    ImageName = DataPath
    
    I1 = cv2.imread(ImageName)
    mean = np.mean(I1, axis=(1,2), keepdims=True)
    std = np.std(I1, axis=(1,2), keepdims=True)
    standardized_image = (I1 - mean) / (std + 0.0001) 
    #standardized_image = I1
    if(standardized_image is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
    
    # if(I1 is None):
    #     # OpenCV returns empty list if image is not read! 
    #     print('ERROR: Image I1 cannot be read')
    #     sys.exit()
        
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################

    # I1S = iu.StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(standardized_image, axis=0)

    return I1Combined, I1
                

def TestOperation(ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred, numEpochs):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    Length = ImageSize[0]
    # Predict output with forward pass, MiniBatchSize for Test is 1
    _, prSoftMaxS = Resnet(ImgPH, ImageSize, 1)

    # Setup Saver
    Saver = tf.train.Saver()
    startEpoch = 0
    
    with tf.Session() as sess:
        for epoch in range(startEpoch,numEpochs,1):
            print("EPOCH = ", epoch)
            ModelPath_ = ModelPath + "/" + str(epoch) + "model.ckpt"
            Saver.restore(sess, ModelPath)
            print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
            
            OutSaveT = open(LabelsPathPred, 'w')

            for count in tqdm(range(np.size(DataPath))):            
                DataPathNow = DataPath[count]
                Img, ImgOrg = ReadImages(ImageSize, DataPathNow)
                FeedDict = {ImgPH: Img}
                PredT = np.argmax(sess.run(prSoftMaxS, FeedDict))

                OutSaveT.write(str(PredT)+'\n')
                
            OutSaveT.close()

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    LabelsTrue = list(LabelsTrue)
    LabelsPred = list(LabelsPred)
    cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print('Accuracy: '+ str(Accuracy(LabelsPred, LabelsTrue)), '%')

        
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='./Phase2/Code/Checkpoints/resnet/24model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='./Phase2/CIFAR10/Test/', help='Path to load images from, Default:BasePath')
    # Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./Phase2/Code/TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--Plotpath', default='./Phase2/Code/plots/resnet', help='Path to save plots')
    Parser.add_argument('--NumEpochs', type=int, default=10, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--TxtPath', dest='TxtPath', default='./Phase2/Code/TxtFiles/', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    TxtPath = Args.TxtPath
    NumEpochs = Args.NumEpochs
    LabelsPath = TxtPath + "LabelsTest.txt"
    PlotPath = Args.Plotpath

    # Setup all needed parameters including file reading
    ImageSize, DataPath = SetupAll(BasePath)
    numEpochs = NumEpochs
    startEpoch = 0
    accuracy_epochs = []
    epochs = [] 

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
    # LabelsPathPred = 'Phase2/Code/TxtFiles/PredOut.txt' # Path to save predicted labels
    LabelsPathPred = TxtPath + "PredOut.txt"

    TestOperation(ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred, numEpochs)

    # Plot Confusion Matrix
    for epoch in range(startEpoch, numEpochs, 1):
        LabelsPathPred_ = LabelsPathPred
        LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred_)
        LabelsTrue = list(LabelsTrue)
        LabelsPred = list(LabelsPred)
        ConfusionMatrix(LabelsTrue, LabelsPred)
        acc_epoch = Accuracy(LabelsTrue, LabelsPred)
        accuracy_epochs.append(acc_epoch)
        epochs.append(epoch)

    plt.figure()
    plt.plot(epochs, accuracy_epochs)
    plt.xlabel("epochs")
    plt.ylabel("testing accuracy")
    plt.ylim((30, 65))


    plt.savefig(PlotPath+"/test.png")
    # LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    # ConfusionMatrix(LabelsTrue, LabelsPred)
     
if __name__ == '__main__':
    main()
 
