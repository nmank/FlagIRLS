import numpy as np
import center_algorithms as ca
from sklearn import metrics

#from https://stanford.edu/~cpiech/cs221/handouts/kmeans.html

# Function: Should Stop
# -------------
# Returns True or False if k-means is done. K-means terminates either
# because it has run a maximum number of iterations OR the centroids
# stop changing.
def shouldStop(oldCentroids, centroids, iterations, max_itrs):
	if oldCentroids is None:
		return False
	elif iterations > max_itrs: 
		return True

	return np.allclose(oldCentroids,centroids)



# Function: Get Labels
# -------------
# Returns a label for each piece of data in the dataset. 
def getLabels(dataSet, centroids, med):
    # For each element in the dataset, chose the closest centroid. 
    # Make that centroid the element's label.
	labels = []
	r = dataSet[0].shape[1]
	for d in dataSet:
		dists = []
		for c in centroids:
			if med == 'flag':
				dists.append(r-np.trace(d.T @ c @ c.T @ d))
			elif med == 'sine' or med == 'cosine':
				sin_sq = r - np.trace(d.T @ c @ c.T @ d)
				if sin_sq < 0:
					sin_sq = 0
				dists.append(np.sqrt(sin_sq))
			# s_vals = np.linalg.svd(d.T @ c)[1][:r]
			# if (s_vals < 0).any:
			# 	idx = np.where(s_vals < 0)
			# 	s_vals[idx] = 0
			# if (s_vals > 1).any:
			# 	idx = np.where(s_vals > 1)
			# 	s_vals[idx] = 1
			# dists.append(np.sqrt(np.sum(np.arccos(s_vals)**2)))
		idx = np.argmin(dists)
		labels.append(idx)
	return labels


# Function: Get Centroids
# -------------
# Returns centroids, each of dimension n.
def getCentroids(dataSet, labels, centroids, med):
	# Each centroid is the geometric mean of the points that
	# have that centroid's label. Important: If a centroid is empty (no points have
	# that centroid's label) you should randomly re-initialize it.
	[n,r] = dataSet[0].shape

	new_centroids = []
	for ii in range(len(centroids)):
		idx = np.where(np.array(labels) == ii)[0]
		# if idx.size == 0:
		# 	centroids[ii] = getRandomCentorids(1,n,r)[0]
		if len(idx) != 0:
			X = [dataSet[i] for i in idx]
			if med == 'flag':
				new_centroids.append(ca.flag_mean(X, r, fast = False))
			elif med == 'sine':
				new_centroids.append(ca.irls_flag(X, r, 5, 'sine')[0])
			elif med == 'cosine':
				new_centroids.append(ca.irls_flag(X, r, 5, 'cosine')[0])	    

	return new_centroids


# Function: Get Random Centroids
# -------------
# Returns k random centroids, each of dimension n.
def getRandomCentroids(k,n,r,dataSet):
	# idx =  np.random.randint(0,len(dataSet),k)
	# centroids = [dataSet[i] for i in idx]

	centroids = []
	for i in range(k):
		centroids.append(np.linalg.qr(np.random.rand(n,r)-.5)[0][:,:r])
	return centroids


# Function: Cluster Purity
# -------------
# Returns the average purity of the clusters
def clusterPurity(labels_true, dataSet, centroids, med):

	labels = getLabels(dataSet, centroids, med)

	purity = []
	for l in np.unique(labels):
		idx = np.where(np.array(labels) == l)[0]
		Ltrue = np.array([labels_true[i] for i in idx])
		Lcounts = [len(np.where(Ltrue == ll)[0]) for ll in np.unique(Ltrue)]
		purity.append(np.max(Lcounts)/len(idx))

	return np.mean(purity)
	# contingency_matrix = metrics.cluster.contingency_matrix(labels_true, labels)
	# return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
	

# Function: K Means
# -------------
# K-Means is an algorithm that takes in a dataset and a constant
# k and returns k centroids (which define clusters of data in the
# dataset which are similar to one another).
def kmeans(dataSet, k, max_itrs, med):

	[n,r] = dataSet[0].shape
	
	# Initialize centroids randomly
	centroids = getRandomCentroids(k,n,r,dataSet)


	# Initialize book keeping vars.
	iterations = 0
	oldCentroids = None

	# Run the main k-means algorithm
	while not shouldStop(oldCentroids, centroids, iterations, max_itrs):
		# Save old centroids for convergence test. Book keeping.
		oldCentroids = centroids.copy()
		iterations += 1

		# Assign labels to each datapoint based on centroids
		labels = getLabels(dataSet, centroids, med)

		# Assign centroids based on datapoint labels
		centroids = getCentroids(dataSet, labels, centroids, med)
	
	# print(iterations)
	    
	# We can get the labels too by calling getLabels(dataSet, centroids)
	return centroids