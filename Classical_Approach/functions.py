#!/usr/bin/env python3

"""
CMSC733 Spring 2024: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Mayank Deshpande (msdeshp4@umd.edu)
M.Eng. in Robotics,
University of Maryland, College Park

"""

# Code starts here:

import numpy as np
import sklearn.cluster
import math
import cv2
import imutils
import os
import matplotlib.pyplot as plt

def gaussian_kernel(sigma, size):
  """Generates a 2D Gaussian kernel.

  Args:
    size: The width and height of the kernel.
    sigma: The standard deviation of the kernel.

  Returns:
    A 2D Gaussian kernel.
  """

  s_x, s_y = sigma
  kernel = np.zeros([size, size])
  if (size%2) == 0:
    index = size/2
  else:
    index = (size - 1)/2

  x, y = np.meshgrid(np.linspace(-index, index, size), np.linspace(-index, index, size))
  pow = (np.square(x)/np.square(s_x)) + (np.square(y)/np.square(s_y))
  pow /= 2
  kernel = (0.5/(np.pi * s_x * s_y)) * np.exp(-pow)
  return kernel

def sin_funct(frequency, size, angle):
  if (size%2) == 0:
    index = size/2
  else:
    index = (size - 1)/2

  x, y = np.meshgrid(np.linspace(-index, index, size), np.linspace(-index, index, size))
  mu = x * np.cos(angle) + y * np.sin(angle)
  sin = np.sin(mu * 2 * np.pi * frequency/size)

  return sin

def half_disk(radius, angle):
	size = 2*radius + 1
	c = radius
	halfDisk = np.zeros([size, size])
	for i in range(radius):
		for j in range(size):
			d = np.square(i-c) + np.square(j-c)
			if d <= np.square(radius):
				halfDisk[i,j] = 1
    
	
	halfDisk = imutils.rotate(halfDisk, angle)
	halfDisk[halfDisk<=0.5] = 0
	halfDisk[halfDisk>0.5] = 1
	return halfDisk

def apply_ft(image, filter_bank):
    img = []
    for f in filter_bank:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ig = cv2.filter2D(grayscale,-1, f)
        img.append(ig)

    return img

def KMeans(input, bins, init):
    kmeans = sklearn.cluster.KMeans(n_clusters = bins, n_init = init)
    kmeans.fit(input)
    labels = kmeans.predict(input)

    return labels

def loadImages(folder_name, files):
	print("Loading images from ", folder_name)
	images = []
	if files == None:
		files = os.listdir(folder_name)
	print(files)
	for file in files:
		image_path = folder_name + "/" + file
		image = cv2.imread(image_path)
		if image is not None:
			images.append(image)
			
		else:
			print("Error in loading image ", image)

	return images

def jpg2pngList(jpg_list):
	png_list = []
	for n in range(len(jpg_list)):
		s = jpg_list[n]
		f_name = str()
		for i in range(len(s)):
			if s[i] == '.':
				break

			f_name += str(s[i])	

		png_list.append(str(f_name) + ".png")
	
	return png_list
    
def printFilterbank(filter_bank, file_name, cols):
	#cols = 6
	rows = math.ceil(len(filter_bank)/cols)
	plt.subplots(rows, cols, figsize=(15,15))
	for index in range(len(filter_bank)):
		plt.subplot(rows, cols, index+1)
		plt.axis('off')
		plt.imshow(filter_bank[index], cmap='gray')
	
	plt.savefig(file_name)
	plt.close()

