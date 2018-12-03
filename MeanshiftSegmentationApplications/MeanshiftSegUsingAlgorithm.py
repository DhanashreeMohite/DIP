'''
Algorithm:
Feature space: (L,u,v,x,y) Intensity + (u,v)
color channels + Position in image (x,y)
• Apply meanshift in the 5-dimensional space
• For each pixel (xi,yi) of intensity Li and color
(ui,vi), find the corresponding mode ck
• All of the pixel (xi,yi) corresponding to the same
mode ck are grouped into a single region
'''

'''
Input Arguments:
        imagePath : Path of the image which needs to be segmented
        h: blob size i.e. no. of pixels can be used to find the mean (h = 140)
        tolerance: Shift in previous and current mean can be tolerated (tolerence = 20)
'''
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
from numpy import nan
import numpy as np
import matplotlib
import random
import time
import cv2


image = cv2.imread('baboon.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h = 60 #100 not working
tolerance = 10
    
start = time.time()
# Stoaring dimension of image
height, width, _ = image.shape

# Declaration and initialization of 5 dimensional feature array
# 0 to 2  for pixel value, 3 and 4 for x and y pixel position 
featureMatrix = np.zeros((height * width, 5), dtype='float')

# Declaration and initialization of new segmented image
segmentImage = np.zeros_like(image)

# Index to access FeatureMatrix elements i.e pixel
featureIndex = 0
# Store image pixel values and positions in featurematrix
for row in range(height):
	for col in range(width):
		# x position
		featureMatrix[featureIndex, 3] = row
		# y position
		featureMatrix[featureIndex, 4] = col
		# Store R, G, B value of pixel 
		featureMatrix[featureIndex, 0:3] = image[row, col, 0:3]
		featureIndex += 1

featureIndex = 1
prev_meanShift = 0

# Run till featureMatrix is empty
while featureMatrix.shape[0] >= 1:
	# Choose new random element for fresh start only if we have reached the dense area otherwise use last mean for iteration
	if featureIndex == 1:
		# Select random row from featureMatrix
		index = random.randint(0, featureMatrix.shape[0] - 1)
		# Select row elements as current means
		curr_mean = featureMatrix[index,:].reshape(-1, 5)
		featureIndex = 0

	# Find ecludian distance from current mean to all other elemnts in featureMatrix
	eucldn_dist = euclidean_distances(curr_mean, featureMatrix)
	# Sort all ecludian distances by value in ascending order 
	dist_sortAscend = np.sort(eucldn_dist)
	# Sort all ecludian distances by index in ascending order 
	dist_sortAscend_index = np.argsort(eucldn_dist)
	# Choose all elemnts whose distance is less than size of blob (h)
	_, interestElements = np.where(dist_sortAscend < h)   

	if interestElements.any() == 0:
		# If there are no elemnts in blob region assign current mean as new mean
		new_mean = curr_mean
	else:
		# else calculate new mean for elements within blob(h) (Vectorized)
		new_mean = np.mean(featureMatrix[dist_sortAscend_index[0, 0:interestElements[-1]]], axis = 0).reshape(-1,5)

	# Calculate mean shift for previous and current mean (Vectorized)
	meanShift = euclidean_distances(curr_mean, new_mean)

	# Assign current meanShift to variable
	prev_meanShift = meanShift

	if meanShift == 0: # noise, some speacial value not having similar surrounding

		# If the meanshift is zero assign current means to selected pixels within blob
		f_row = featureMatrix[dist_sortAscend_index[0,0], 3].astype('uint16') 
		f_col = featureMatrix[dist_sortAscend_index[0,0], 4].astype('uint16')
		segmentImage[f_row,f_col, 0:3] = curr_mean[0, 0:3]

		# Delete all the pixels from featurematrix after assigning to segemented matrix above, row wise delete axis 0
		featureMatrix = np.delete(featureMatrix, dist_sortAscend_index[0,0].reshape(-1, 1), axis = 0)

		# Set variable to one to select new random row in updated feature matrix
		featureIndex = 1

	elif meanShift < tolerance:

		# If the meanshift is less than tolerance value then assign new means to selected pixels within blob (Vectorized)
		f_row = featureMatrix[dist_sortAscend_index[0, 0:interestElements[-1]], 3].astype('uint16') 
		f_col = featureMatrix[dist_sortAscend_index[0, 0:interestElements[-1]], 4].astype('uint16')
		segmentImage[f_row, f_col, 0:3] = new_mean[0, 0:3]
		# Delete all the pixels from featurematrix after assigning to segemented matrix above
		featureMatrix = np.delete(featureMatrix, dist_sortAscend_index[0, 0:interestElements[-1]].reshape(-1, 1), axis = 0)
		# Set variable to one to select new random row in updated feature matrix
		featureIndex = 1

	else:
		# If meanshift is more than tolerance values then assign new means to current means and iterate the process
		curr_mean = new_mean

print("Time taken to complete Segmentation is {0}seconds.".format(time.time()-start))

# Display segmented image
plt.figure(figsize=(10,5))
plt.imshow(segmentImage)
plt.title("Segmented Image")
plt.show()
