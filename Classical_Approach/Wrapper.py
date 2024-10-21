#!/usr/bin/env python3

"""
CMSC733 Spring 2021: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Mayank Deshpande (msdeshp4@umd.edu)
M.Eng. in Robotics,
University of Maryland, College Park
"""

# Code starts here:
import sys
import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from functions import *
import imutils

sys.path.append(str(Path( __file__ ).parent.joinpath('..')))

def DoG_Filter(num_orientations, scales, size):

	flt_bank = []

	sobel_x = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
	sobel_y = np.array([[1, 2, 1],[0, 0, 0], [-1, -2, -1]])
    
	for s in scales:
		sigma = [s, s]
		G = gaussian_kernel(sigma, size)
		Gx = cv2.filter2D(G,-1, sobel_x)
		Gy = cv2.filter2D(G,-1, sobel_y)
		for o in range(num_orientations):
			angle = o * 360 / num_orientations 
			dog = (Gx * np.cos(angle)) +  (Gy * np.sin(angle))
			flt_bank.append(dog)
        
		
	return flt_bank

def LM_Filter(scales, num_orientations, size):
	dog_scale = scales[0:3]
	gaussian_scale = scales
	log_scale = scales + [i * 3 for i in scales]

	flt_bank = []
	DoG = []
	D2oG = []
	gaussian = []
	LoG = []

	sobel_x = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
	sobel_y = np.array([[-1, -2, -1],[0, 0, 0], [1, 2, 1]])

	for s in dog_scale:
		sigma = [3*s, s]
		G = gaussian_kernel(sigma, size)

		dog = cv2.filter2D(G, -1, sobel_x) + cv2.filter2D(G, -1, sobel_y)
		d2og = cv2.filter2D(dog, -1, sobel_x) + cv2.filter2D(dog, -1, sobel_y)

		for o in range(num_orientations):
			angel = o * 180 / num_orientations
     	
			dog =  imutils.rotate(dog, angel)
			DoG.append(dog)

			d2og = imutils.rotate(d2og, angel)
			D2oG.append(d2og)
	
	for s in gaussian_scale:
		sigma = [s, s]
		gaussian.append(gaussian_kernel(sigma, size))

	for s in log_scale:
		sigma = [s, s]
		G = gaussian_kernel(sigma, size)
		log_kernal = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
		LoG.append(cv2.filter2D(G, -1, log_kernal))


	flt_bank = DoG + D2oG + LoG + gaussian
	return flt_bank

def gb_filter(scales, num_orientations, frequencies, size):
	flt_bank = []
	for s in scales:
		sigma = [s, s]
		G = gaussian_kernel(sigma, size)
		for f in frequencies:
			for o in range(num_orientations):
				angels = o * 180 / num_orientations
				sinusoidal = sin_funct(f, size, angels)
				gabor_filter = G * sinusoidal
				flt_bank.append(gabor_filter)

	return flt_bank

def halfdiskFilters(radius, num_orientations):
	flt_bank = []
	for r in radius:
		pairs = []
		t = []
		for o in range(num_orientations):
			angle = o * 360 / num_orientations
			half_disk_filter = half_disk(r, angle)
			t.append(half_disk_filter)

		i = 0
		while i < num_orientations/2:
			pairs.append(t[i])
			pairs.append(t[i+int((num_orientations)/2)])
			i = i+1

		flt_bank+=pairs
	
	
	return flt_bank

def texton(image, flt_bank):
	img = apply_ft(image, flt_bank)	
	img = np.array(img)
	f,x,y = img.shape
	input = img.reshape([f, x*y])
	input = input.transpose()

	return input, x, y

def brightness(image):
	grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	x,y = grayscale.shape
	input = grayscale.reshape([x*y,1])

	return input, x, y

def color(image):
	x,y,c = image.shape
	input = image.reshape([x*y,c])

	return input, x, y

def gradient(map, bins, flt_bank):
	grad = chi_distance(map, bins, flt_bank)
	grad = np.array(grad)
	grad = np.mean(grad, axis = 0)
	return grad

def chi_distance(input, bins, flt_bank):

	distances = []
	L = len(flt_bank)
	n = 0
	while n < L:
		l_m = flt_bank[n]
		r_m = flt_bank[n+1]		
		temp = np.zeros(input.shape)
		chi_sq_dist = np.zeros(input.shape)
		min_bin = np.min(input)
	

		for b in range(bins):
			temp[input == b+min_bin] = 1
			g_i = cv2.filter2D(temp,-1,l_m)
			h_i = cv2.filter2D(temp,-1,r_m)
			chi_sq_dist += (g_i - h_i)**2/(g_i + h_i + np.exp(-7))

		chi_sq_dist /= 2
		distances.append(chi_sq_dist)
		n = n+2
    	

	return distances

def pbLite(Tg, Bg, Cg, Canny_edge, Sobel_edges, weights):
	Canny_edge = cv2.cvtColor(Canny_edge, cv2.COLOR_BGR2GRAY)
	Sobel_edges = cv2.cvtColor(Sobel_edges, cv2.COLOR_BGR2GRAY)
	T1 = (Tg + Bg + Cg)/3
	w1 = weights[0]
	w2 = weights[1]
	T2 = (w1 * Canny_edge) + (w2 * Sobel_edges)

	pb_lite_op = np.multiply(T1, T2)
	return pb_lite_op


def main():
	get_t_map = True
	get_brightness_map = True
	get_color_map = True

	texton_bins = 64
	brightness_bins = 16
	color_bins = 16
 
	t_map = []
	b_map = []
	c_map = []

	t_grd = []
	b_grd = []
	c_grd = []

	folder_prefix = "./"
	image_folder = folder_prefix + "Phase1/BSDS500/Images"
	sobel_baseline_folder = folder_prefix + "Phase1/BSDS500/SobelBaseline"
	canny_baseline_folder = folder_prefix + "Phase1/BSDS500/CannyBaseline"
	


	print("Generating filters...")
	sys.stdout.flush()
	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	Dog_fb = DoG_Filter(16, [2,3], 25)
	printFilterbank(Dog_fb, folder_prefix + "Phase1/results/Filters/DoG.png", cols = 8)
	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	LMS_fb = LM_Filter([1, np.sqrt(2), 2, 2*np.sqrt(2)], 6, 25)
	printFilterbank(LMS_fb,folder_prefix + "Phase1/results/Filters/LMS.png", 6)
	LML_fb = LM_Filter([np.sqrt(2), 2, 2*np.sqrt(2), 4], 6, 25)
	printFilterbank(LML_fb, folder_prefix + "Phase1/results/Filters/LML.png", 6)
	LM_fb = LMS_fb + LML_fb
	printFilterbank(LM_fb, folder_prefix + "Phase1/results/Filters/LM.png", 6)
	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	gb_fb = gb_filter([10,25], 6, [2,3,4], 25)
	printFilterbank(gb_fb, folder_prefix + "Phase1/results/Filters/Gabor.png",6)

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	hd_fb = halfdiskFilters([2,5,10,20,30], 16)
	printFilterbank(hd_fb, folder_prefix + "Phase1/results/Filters/HDMasks.png", 6)
	print("generating texton maps..")
	sys.stdout.flush()
	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	edit: used Dog + LM + gabor
	"""	
	images = loadImages(image_folder, files=None)
	sys.stdout.flush()
	file_names = os.listdir(image_folder)
	flt_bank = Dog_fb + LM_fb + gb_fb
    
	
	if get_t_map:
		for i,image in enumerate(images):
			input_mat,x,y = texton(image, flt_bank)
			"""
			Generate texture ID's using K-means clustering
			Display texton map and save image as TextonMap_ImageName.png,
			use command "cv2.imwrite('...)"
			"""
			labels = KMeans(input_mat, texton_bins, 2)
			texton_image = labels.reshape([x,y])
			t_map.append(texton_image)
			plt.imsave(folder_prefix + "Phase1/results/Texton_map/TextonMap_"+ file_names[i], texton_image)     

	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	print("generating texton gradients..")
	sys.stdout.flush()
	for i,texton_map in enumerate(t_map):
		T_grad = gradient(texton_map, texton_bins, hd_fb)
		t_grd.append(T_grad)
		plt.imsave(folder_prefix + "Phase1/results/T_g/tg_" + file_names[i], T_grad)		
	print("t_grad ", t_grd)
	sys.stdout.flush()	 

    
	"""
	Generate Brightness Map
	Perform brightness binning 
	"""
	print("generating brightness maps..")
	sys.stdout.flush()
	if get_brightness_map:
		for i,image in enumerate(images):
			input_mat,x,y = brightness(image)
			labels = KMeans(input_mat, brightness_bins, 4)
			brightness_image = labels.reshape([x,y])
			b_map.append(brightness_image)
			plt.imsave(folder_prefix + "Phase1/results/Brightness_map/BrightnessMap_" + file_names[i], brightness_image)
   			

	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	print("generating brightness gradient..")
	sys.stdout.flush()
	for i,brightness_map in enumerate(b_map):
		B_grad = gradient(brightness_map, brightness_bins, hd_fb)
		b_grd.append(B_grad)
		plt.imsave(folder_prefix + "Phase1/results/B_g/bg_" + file_names[i], B_grad)
	print("b_grad ", b_grd)
	sys.stdout.flush()

	"""
	Generate Color Map
	Perform color binning or clustering
	"""
	print("generating color maps..")
	sys.stdout.flush()
	if get_color_map:
		for i,image in enumerate(images):
			input_mat, x, y = color(image)
			labels = KMeans(input_mat, color_bins, 4)
			color_image = labels.reshape([x,y])
			c_map.append(color_image)
			plt.imsave(folder_prefix + "Phase1/results/Color_map/ColorMap_"+ file_names[i], color_image) 


	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	print("generating color gradient..")
	sys.stdout.flush()
	for i,color_map in enumerate(c_map):
		C_grad = gradient(color_map, color_bins, hd_fb)
		c_grd.append(C_grad)
		plt.imsave(folder_prefix + "Phase1/results/C_g/cg_" + file_names[i], C_grad)
	print("c_grad ", c_grd)
	sys.stdout.flush()
	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""
	baseline_files = jpg2pngList(file_names)
	print(baseline_files)
	sys.stdout.flush()
	sobel_baseline = loadImages(sobel_baseline_folder, baseline_files)

	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""

	canny_baseline = loadImages(canny_baseline_folder, baseline_files)


	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
	print("pb lite output..")
	sys.stdout.flush()
	if get_t_map and get_brightness_map and get_color_map:
		for i in range(len(images)):	
			print("generating edges for image ", baseline_files[i])
			sys.stdout.flush()
			if i >= len(t_grd):
				print("t_grd index out of range")
				sys.stdout.flush()
			if i >= len(b_grd):
				print("b_grd index out of range")
				sys.stdout.flush()
			if i >= len(c_grd):
				print("c_grd index out of range")
				sys.stdout.flush()
			if i >= len(canny_baseline):
				print("canny_baseline index out of range")
				sys.stdout.flush()
			if i >= len(sobel_baseline):
				print("sobel_baseline index out of range")
				sys.stdout.flush()
			pb_edge = pbLite(t_grd[i], b_grd[i], c_grd[i], canny_baseline[i], sobel_baseline[i], [0.5,0.5])
			plt.imsave("Phase1/results/pb_lite_output/" + baseline_files[i], pb_edge, cmap = "gray")

if __name__ == '__main__':
    main()
 


