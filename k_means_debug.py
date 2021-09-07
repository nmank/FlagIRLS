import scipy.io as sio
import numpy as np
import mat73
import center_algorithms as ca
import matplotlib.pyplot as plt
import k_means as km
import seaborn as sns
import pandas as pd

labels_raw = sio.loadmat('./data/kmeans_action_labels.mat')['kmeans_action_labels']

labels_true = [l[0][0] for l in labels_raw['labels'][0][0]]
# labelidxs =labels_raw['labelidxs'][0][0][0]


raw_data = mat73.loadmat('./data//kmeans_pts.mat')

gr_list = [t[0] for t in raw_data['Data']['gr_pts']]




n_itrs = 20

k=15

centers = km.kmeans(gr_list, k, n_itrs, 'flag')
cluster_purity = km.clusterPurity(labels_true, gr_list, centers, 'flag')
