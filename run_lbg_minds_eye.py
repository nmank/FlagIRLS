import scipy.io as sio
import numpy as np
import mat73
import center_algorithms as ca
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
from sklearn.manifold import MDS


'''
datapoints rows and centers columns

sine
cosine
sine_sq
'''
def distance_matrix(X, C, opt_type = 'sine'):
    n = len(X)
    m = len(C)
    Distances = np.zeros((m,n))

    # if opt_type == 'cosine':
    #     opt_type = 'sinesq'

    for i in range(m):
        for j in range(n):
            Distances[i,j] = ca.calc_error_1_2([C[i]], X[j], 'sine')
            
    return Distances

def cluster_purity(X, centers, opt_type, labels_true):
    #calculate distance matrix
    d_mat = distance_matrix(X, centers, opt_type)

    #find the closest center for each point
    index = np.argmin(d_mat, axis = 0)
    
    count = 0
    for i in range(len(centers)):
        idx = np.where(index == i)[0]
        if len(idx) != 0:
            cluster_labels = [labels_true[i] for i in idx]
            most_common_label = max(set(cluster_labels), key = cluster_labels.count)
            # count += cluster_labels.count(most_common_label)
            count += cluster_labels.count(most_common_label)/len(idx)

    # return count/len(X)
    return count/len(centers)


def lbg_subspace(X, epsilon, n_centers = 17, opt_type = 'sine', n_its = 10, seed = 1):
    n_pts = len(X)
    error = 1
    r = 48
    distortions = []

    #init centers
    np.random.seed(seed)
    centers = []
    for i in range(n_centers):
        centers.append(X[np.random.randint(n_pts)])

    #calculate distance matrix
    d_mat = distance_matrix(X, centers, opt_type)

    #find the closest center for each point
    index = np.argmin(d_mat, axis = 0)

    #calculate first distortion
    new_distortion = np.sum(d_mat[index])

    distortions.append(new_distortion)


    errors = []
    while error > epsilon:

        #set new distortion as old one
        old_distortion = new_distortion

        m = len(centers)

        #calculate new centers
        centers = []
        for c in range(m):
            idx = np.where(index == c)[0]
            if len(idx) > 0:
                if opt_type == 'sinesq':
                    centers.append(ca.flag_mean([X[i] for i in idx], r, fast = False))
                elif opt_type == 'l2_med':
                    centers.append(ca.l2_median([X[i] for i in idx], .1, r, 1000)[0])
                else:
                    centers.append(ca.irls_flag([X[i] for i in idx], r, n_its, opt_type, opt_type)[0])
        #         centers.append(np.mean([X[i] for i in idx], axis = 0))

        #calculate distance matrix
        d_mat = distance_matrix(X, centers, opt_type)

        #find the closest center for each point
        index = np.argmin(d_mat, axis = 0)

        #new distortion
        new_distortion = np.sum(d_mat[index])

        distortions.append(new_distortion)

        if new_distortion <0.00000000001:
            error = 0
        else:
            error = np.abs(new_distortion - old_distortion)/old_distortion
        errors.append(error)
        print(error)

    return centers, errors, distortions





n_its= 10
seed = 0
n_trials = 20

f_name = './lbg_'+str(n_trials)+'trials.png'

labels_raw = sio.loadmat('/data4/mankovic/MindsEye/kmeans_action_labels.mat')['kmeans_action_labels']

labels_true = [l[0][0] for l in labels_raw['labels'][0][0]]
# labelidxs =labels_raw['labelidxs'][0][0][0]


raw_data = mat73.loadmat('/data4/mankovic/MindsEye/kmeans_pts.mat')

X = [t[0] for t in raw_data['Data']['gr_pts']]



idx = []
for the_labels in ['run', 'pickup', 'bend','follow', 'ride-bike']:
# for the_labels in ['run', 'stand', 'pickup']:
# for the_labels in ['run', 'stand', 'walk-rifle']: #for winning sine median
    idx += list(np.where(np.array(labels_true) == the_labels)[0])

labels_true = [labels_true[i] for i in idx]
X = [X[i] for i in idx]


Purities = pandas.DataFrame(columns = ['Algorithm','Codebook Size','Cluster Purity'])

for n in range(4, 24, 4):
    sin_purities = []
    cos_purities = []
    flg_purities = []
    for trial in range(10):
        print('cluster '+str(n)+' trial '+str(trial))
        print('.')
        print('.')
        print('.')
        print('sin start')
        centers_sin, error_sin, dist_sin = lbg_subspace(X, .0001, n_centers = n, opt_type = 'sine', n_its = 10, seed = trial)
        sin_purity = cluster_purity(X, centers_sin, 'sine', labels_true)
        print('l2 start')
        centers_l2, error_l2, dist_l2 = lbg_subspace(X, .0001, n_centers = n, opt_type = 'l2_med', n_its = 10, seed = trial)
        l2_purity = cluster_purity(X, centers_l2, 'l2_med', labels_true)
        print('flg start')
        centers_flg, error_flg, dist_flg = lbg_subspace(X, .0001, n_centers = n, opt_type = 'sinesq', seed = trial)
        flg_purity = cluster_purity(X, centers_flg, 'sinesq', labels_true)


        Purities = Purities.append({'Algorithm': 'Flag Median', 
                                'Codebook Size': n,
                                'Cluster Purity': sin_purity},
                                ignore_index = True)
        Purities = Purities.append({'Algorithm': 'L2 Median', 
                                'Codebook Size': n,
                                'Cluster Purity': l2_purity},
                                ignore_index = True)
        Purities = Purities.append({'Algorithm': 'Flag Mean', 
                                'Codebook Size': n,
                                'Cluster Purity': flg_purity},
                                ignore_index = True)
    print(Purities)
    Purities.to_csv('LBG_results_newest'+str(n)+'.csv')
        


sns.boxplot(x='Codebook Size', y='Cluster Purity', hue='Algorithm', data = Purities)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.savefig(f_name, bbox_inches='tight')

Purities.to_csv('LBG_results_newest.csv')