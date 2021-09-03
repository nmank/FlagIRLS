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

#finish this and run by tomorrow morning.
cluster_purities = pd.DataFrame(columns = ['Algorithm','NumberClusters','ClusterPurity'])
for k in range(5,60,5):
    print('.')
    print('.')
    print('.')
    for exp in range(10):
    # for exp in range(2):
        centers = km.kmeans(gr_list, k, n_itrs, 'flag')
        cluster_purity = km.clusterPurity(labels_true, gr_list, centers, 'flag')
        cluster_purities = cluster_purities.append({'Algorithm':'Flag Mean','NumberClusters':k,'ClusterPurity':cluster_purity},ignore_index = True)
        print("Flag trial"+str(exp+1)+" finished")
        print('.')

        # centers = km.kmeans(gr_list, k, n_itrs, 'sine')
        # cluster_purity = km.clusterPurity(labels_true, gr_list, centers, 'sine')
        # cluster_purities = cluster_purities.append({'Algorithm':'Sine Median','NumberClusters':k,'ClusterPurity':cluster_purity},ignore_index = True)
        # print("Sine trial"+str(exp+1)+" finished")
        # print('.')

        # centers = km.kmeans(gr_list, k, n_itrs, 'cosine')
        # cluster_purity = km.clusterPurity(labels_true, gr_list, centers, 'cosine')
        # cluster_purities = cluster_purities.append({'Algorithm':'Maximum Cosine','NumberClusters':k,'ClusterPurity':cluster_purity}, ignore_index = True)
        # print("Cosine trial"+str(exp+1)+" finished")
        # print('.')

    print('Experiment '+str(k)+' finished.')



# import IPython; IPython.embed()
    

# ax = sns.violinplot(x='NumberClusters', y='ClusterPurity', hue='Algorithm', data = cluster_purities)
# # ax = sns.catplot(x='NumberClusters', y='ClusterPurity', hue='Algorithm', kind = 'bar', data = cluster_purities)
# ax.set_ylabel("Cluster Purities")
# ax.set_xlabel("Number of Clusters")
# plt.savefig('k_means_violins.png')

cluster_purities.to_csv(index = False)

sns.boxplot(x='NumberClusters', y='ClusterPurity', hue='Algorithm', data = cluster_purities)

plt.savefig('k_means_boxplots.png')