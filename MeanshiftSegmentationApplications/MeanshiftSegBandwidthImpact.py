import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from PIL import Image
import imp
from resizeimage import resizeimage
import time

image = cv2.imread('jerry.jpg')
image = np.array(image)

start = time.time()
# Shape of original image
originShape = image.shape

#Converting image into feature array of dimension [nb of pixels in originImage, 3] based on r g b intensities
flat_image=np.reshape(image, [-1, 3])#.tolist()
#print(flat_image)

#Estimate bandwidth
#bandwidth increases by very less amount by increasing no. of samples and not much visible difference will be there
bandwidth1 = estimate_bandwidth(flat_image, quantile=.1, n_samples=500)

'''
bandwidth2 = estimate_bandwidth(flat_image, quantile=.2, n_samples=500)
#bandwidth3 = estimate_bandwidth(flat_image, quantile=.1, n_samples=1000) 
#bandwidth4 = estimate_bandwidth(flat_image, quantile=.3, n_samples=1000)
bandwidth3 = estimate_bandwidth(flat_image, quantile=.3, n_samples=500)
bandwidth4 = estimate_bandwidth(flat_image, quantile=.4, n_samples=500)


print(bandwidth1)
print(bandwidth2)
print(bandwidth3)
print(bandwidth4)
'''

ms1 = MeanShift(bandwidth1, bin_seeding=True)
'''
ms2 = MeanShift(bandwidth2, bin_seeding=True)
ms3 = MeanShift(bandwidth3, bin_seeding=True)
ms4 = MeanShift(bandwidth4, bin_seeding=True)
#print(ms1)
'''

#Performing meanshift on flatImg
ms1.fit(flat_image)
'''
ms2.fit(flat_image)
ms3.fit(flat_image)
ms4.fit(flat_image)
'''
#(r,g,b) vectors corresponding to the different clusters after meanshift
labels1=ms1.labels_
'''
labels2=ms2.labels_
labels3=ms3.labels_
labels4=ms4.labels_
#print(labels)
'''

#Remaining colors after meanshift
cluster_centers1 = ms1.cluster_centers_
'''
cluster_centers2 = ms2.cluster_centers_
cluster_centers3 = ms3.cluster_centers_
cluster_centers4 = ms4.cluster_centers_
'''

#Finding and diplaying the number of clusters
labels_unique1 = np.unique(labels1)
n_clusters_1 = len(labels_unique1)
print("number of estimated clusters : %d" % n_clusters_1)
'''
labels_unique2 = np.unique(labels2)
n_clusters_2 = len(labels_unique2)
print("number of estimated clusters : %d" % n_clusters_2)

labels_unique3 = np.unique(labels3)
n_clusters_3 = len(labels_unique3)
print("number of estimated clusters : %d" % n_clusters_3)

labels_unique4 = np.unique(labels4)
n_clusters_4 = len(labels_unique4)
print("number of estimated clusters : %d" % n_clusters_4)
'''

# Displaying segmented image
seg1 = cluster_centers1[np.reshape(labels1, originShape[:2])]
segImage1 = seg1.astype(np.uint8)
'''
seg2 = cluster_centers2[np.reshape(labels2, originShape[:2])]
segImage2 = seg2.astype(np.uint8)

seg3= cluster_centers3[np.reshape(labels3, originShape[:2])]
segImage3 = seg3.astype(np.uint8)

seg4 = cluster_centers4[np.reshape(labels4, originShape[:2])]
segImage4 = seg4.astype(np.uint8)
'''
imageR = cv2.resize(image, (250, 300))
segImage1_R =  cv2.resize(segImage1, (250, 300))
print("Time takend to complete Segmentation is {0}seconds.".format(time.time()-start))
'''
segImage2_R =  cv2.resize(segImage2, (250, 300))
segImage3_R =  cv2.resize(segImage3, (250, 300))
segImage4_R =  cv2.resize(segImage4, (250, 300))
'''
#Result = np.hstack((imageR,segImage1_R,segImage2_R,segImage3_R,segImage4_R))
Result = np.hstack((imageR,segImage1_R))


cv2.imshow('Image', Result)
cv2.waitKey(0)
cv2.destroyAllWindows()
