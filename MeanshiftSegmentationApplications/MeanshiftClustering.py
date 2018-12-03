import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from PIL import Image
import imp
from resizeimage import resizeimage

image = Image.open('jerry.jpg')
#image = resizeimage.resize_contain(image, [867, 1210])
image = np.array(image)

#Shape of original image
#originShape = originImg.shape

#Converting image into feature array of dimension [nb of pixels in originImage, 3] based on r g b intensities
flat_image=np.reshape(image, [-1, 3])#.tolist()
#print(flat_image)

#Estimate bandwidth
bandwidth2 = estimate_bandwidth(flat_image, quantile=.2, n_samples=500)
print(bandwidth2)

ms = MeanShift(bandwidth2, bin_seeding=True)
print(ms)

#Performing meanshift on flatImg
ms.fit(flat_image)

#(r,g,b) vectors corresponding to the different clusters after meanshift
labels=ms.labels_
print(labels)

#Remaining colors after meanshift
cluster_centers = ms.cluster_centers_

#Finding and diplaying the number of clusters
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)


from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
	my_members = labels == k
	cluster_center = cluster_centers[k]
	plt.plot(flat_image[my_members, 0], flat_image[my_members, 1], col + '.')
	plt.plot(cluster_center[0], cluster_center[1], 'x', markerfacecolor=col, markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
