import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from PIL import Image
import imp
from resizeimage import resizeimage
import time

image = cv2.imread('baboon.jpg')
image = np.array(image)

start = time.time()
# Shape of original image
originShape = image.shape

#Converting image into feature array of dimension [nb of pixels in originImage, 3] based on r g b intensities
flat_image=np.reshape(image, [-1, 3])#.tolist()
#print(flat_image)

#Estimate bandwidth
# quantile : smoothening parameter. 
#cut points dividing the range of a probability distribution into continuous intervals with equal probabilities, or dividing the observations in a sample in the same way.
#should be between [0, 1] 0.5 means that the median of all pairwise distances is used.

# n_samples : The number of samples to use. If not given, all samples are used.
#bandwidth increases by very less amount by increasing no. of samples and not much visible difference will be there
bandwidth1 = estimate_bandwidth(flat_image, quantile=.1, n_samples=500)

#print(bandwidth1)


ms1 = MeanShift(bandwidth1, bin_seeding=True)

#Performing meanshift on flatImg
ms1.fit(flat_image)

#(r,g,b) vectors corresponding to the different clusters after meanshift
labels1=ms1.labels_


#Remaining colors after meanshift
cluster_centers1 = ms1.cluster_centers_


#Finding and diplaying the number of clusters
labels_unique1 = np.unique(labels1)
n_clusters_1 = len(labels_unique1)
#print("number of estimated clusters : %d" % n_clusters_1)

# Displaying segmented image
seg1 = cluster_centers1[np.reshape(labels1, originShape[:2])]
segImage1 = seg1.astype(np.uint8)

imageR = cv2.resize(image, (250, 300))
segImage1_R =  cv2.resize(segImage1, (250, 300))
print("Time takend to complete Segmentation is {0}seconds.".format(time.time()-start))

#Result = np.hstack((imageR,segImage1_R,segImage2_R,segImage3_R,segImage4_R))
Result = np.hstack((imageR,segImage1_R))

cv2.imshow('Image', Result)
cv2.waitKey(0)
cv2.destroyAllWindows()
