"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow._api.v2.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def CIFAR10Model(Img):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    
    #############################
    # Fill your network here!
    #############################
    model = tf.layers.conv2d(Img, name='conv1', filters = 32, kernel_size = 3, activation = tf.nn.relu)
    model = tf.layers.conv2d(model, name='conv2', filters = 64, kernel_size = 3, activation = tf.nn.relu)

    model = tf.layers.flatten(model)

    model = tf.layers.dense(model, name='fc1', units = 100, activation = tf.nn.relu)
    model = tf.layers.dense(model, name='fc2', units = 10, activation = None)

    prLogits = model
    prSoftMax = tf.nn.softmax(prLogits)  

    return prLogits, prSoftMax

##############################################################################################################
# Modified Network
##############################################################################################################
def ModifiedModel(Img):

    model = Img
    model = tf.layers.conv2d(model, name='conv1', filters = 32, kernel_size = 5, activation = tf.nn.relu)
    model = tf.layers.batch_normalization(model)

    model = tf.layers.conv2d(model, name='conv2', filters = 64, kernel_size = 3, activation = tf.nn.relu)
    model = tf.layers.batch_normalization(model)

    model = tf.layers.conv2d(model, name='conv3', filters = 128, kernel_size = 3, activation = tf.nn.relu)
    model = tf.layers.batch_normalization(model)

    model = tf.layers.flatten(model)
    model = tf.layers.dense(model, name='fc1', units = 100, activation = None)
    model = tf.nn.dropout(model, 0.7)
    model = tf.layers.batch_normalization(model)

    model = tf.layers.dense(model, name='fc2', units = 10, activation = None)
    model = tf.nn.dropout(model, 0.7)
    model = tf.layers.batch_normalization(model)

    prLogits = model
    prSoftMax = tf.nn.softmax(logits = prLogits) 
    return prLogits, prSoftMax

##############################################################################################################
# ResNet!!
##############################################################################################################

def Residual_connection(input_image, num_filters, kernel_size, block_number, layer_number):

    x = tf.layers.conv2d(inputs =input_image, name=str(layer_number)+'conv', padding='same',filters = num_filters, kernel_size = kernel_size, activation = tf.nn.relu)
    x  = tf.layers.batch_normalization(inputs = x ,axis = -1, center = True, scale = True, name = str(layer_number)+'bn')

    f_x = tf.layers.conv2d(input_image, name=str(block_number)+'conv1', padding = 'same', filters = num_filters, kernel_size = kernel_size, activation = tf.nn.relu)
    f_x = tf.layers.batch_normalization(f_x, name=str(block_number)+'bn1')

    f_x = tf.layers.conv2d(f_x, name=str(block_number)+'conv2', padding = 'same',filters = num_filters, kernel_size = kernel_size, activation = None)
    f_x = tf.layers.batch_normalization(f_x, name=str(block_number)+'bn2')

    h_x = tf.math.add(x, f_x)
    h_x = tf.nn.relu(h_x, name='relu'+str(layer_number))
    return h_x


def Resnet(Img):

    model = Img
    
    model = Residual_connection(model, num_filters=32, kernel_size=5, block_number=1, layer_number=2)
    model = Residual_connection(model, num_filters=64, kernel_size=5, block_number=2, layer_number=3)

    model = tf.layers.flatten(model)
    model = tf.layers.dense(model, name='fc1', units = 100, activation = None)
    model = tf.layers.dense(model, name='fc2', units = 10, activation = None)
    
    prLogits = model
    prSoftMax = tf.nn.softmax(logits = prLogits)


    return prLogits, prSoftMax

##############################################################################################################
# ResNext!!
##############################################################################################################

def bottleneck(model): 
    model = tf.layers.conv2d(model, filters=32,kernel_size=1, padding='SAME', activation = tf.nn.relu)
    model = tf.layers.batch_normalization(model)
    
    model = tf.layers.conv2d(model, filters=64,kernel_size=3, padding='SAME', activation = tf.nn.relu)
    model = tf.layers.batch_normalization(model)
    
    model = tf.layers.conv2d(model, filters=32,kernel_size=1, padding='SAME', activation = tf.nn.relu)
    model = tf.layers.batch_normalization(model)

    return model

def ResBlock(model,cardinality=5):
    I =  model
    branch = []
    for _ in range(cardinality) :
        split = bottleneck(model)
        branch.append(split)

    model = tf.math.add_n(branch)
    model = tf.add(model,I)
    return model 

def ResNext(Img, num_classes=10, cardinality=5):

    # first layers
    model = tf.layers.conv2d(inputs = Img, filters=32,kernel_size=3, padding='SAME', activation = None)
    model = tf.layers.batch_normalization(model)
    model = tf.nn.relu(model)

    model = ResBlock(model,cardinality=5)
    model = ResBlock(model,cardinality=5)

    model = tf.layers.flatten(model)
    model = tf.layers.dense(inputs = model,units=256, activation=None)
    prLogits = tf.layers.dense(model,units=10, activation=None)

    prSoftMax = tf.nn.softmax(prLogits, axis=None, name=None)

    return prLogits, prSoftMax

##############################################################################################################
# DenseNet!!
##############################################################################################################

def transferLayer(x, layer_number, dropout_rate):

    x = tf.layers.batch_normalization(x, axis = -1, center = True, scale = True, name = str(layer_number)+'bn')
    x = tf.nn.relu(x, name=str(layer_number)+'relu')
    x = tf.layers.dropout(x, rate=dropout_rate)          
    return x

def denseBlock(input_image, num_filters, kernel_size, density, block_number):
    layers = []
    x = tf.layers.conv2d(input_image, padding='valid',filters = num_filters, kernel_size = kernel_size, activation = None)
    layers.append(x)
    for d in range(density-1):
        x = tf.concat(layers, axis = 3)
        x = tf.layers.conv2d(x, name='conv_'+str(block_number)+str(d), padding = 'same', filters = num_filters, kernel_size = kernel_size, activation = tf.nn.relu)
        layers.append(x)
    x = tf.concat(layers, axis = 3)
    return x

def densenet(Img):
    model = Img
    
    model = tf.layers.conv2d(model, name='conv1', filters = 16, kernel_size = 5, activation = None)
    model = tf.layers.batch_normalization(model)

    model = denseBlock(model, 32, 5, 3, 1)
    model = transferLayer(model, 2, 0.5)

    model = denseBlock(model, 32, 5, 3, 3)
    model = transferLayer(model, 4, 0.5)

    model = tf.layers.flatten(model)
    model = tf.layers.dense(model, name='fc1', units = 100, activation = None)
    model = tf.layers.dense(model, name='fc2', units = 10, activation = None)
    
    prLogits = model
    prSoftMax = tf.nn.softmax(logits = prLogits)


    return prLogits, prSoftMax
