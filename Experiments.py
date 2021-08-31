import scipy.io as sio
import numpy as np
import mat73
import center_algorithms as ca
import matplotlib.pyplot as plt

labels_raw = sio.loadmat('./smaller_action_labels_2345.mat')['smaller_action_labels']

labellist = [l[0][0] for l in labels_raw['labellist'][0][0]]
labelidxs = [l[0] for l in labels_raw['labelidxs'][0][0]]

raw_data = mat73.loadmat('./DARPA_tracklets_2345_09_05.mat')

data_list = [t[0]['data'] for t in raw_data['tracklets']]

gr_list = []
for vid in data_list:
    X = np.hstack([vid[:,:,i].reshape((-1, 1), order="F") for i in range(vid.shape[2])])
    gr_list.append(np.linalg.svd(X)[0][:,:48])


def experiments(data, labellist, labelidxs, num_clusters):

    def find_closest_point(data, Y, labellist, labelidx):
        err = []
        for x in data:
            r = x.shape[1]
            err.append(np.sqrt(r - np.trace(Y.T @ x @ x.T @ Y)))
        idx = np.argmin(err)

        closest_label = [labellist[ii] for ii in np.where(labelidxs == idx)[0]]

        return closest_label

    p_correct = {'Flag Median': 0, 'Sine Median': 0, 'Max Cosine': 0}


    clusters = np.random.randint(0,num_clusters,len(data))

    for ii in range(num_clusters):
        n_its = 30

        idx = np.where(clusters == ii)[0]
        print(len(idx))
        X = [data[i] for i in idx]
        labels = [labellist[i] for i in idx]
        labelid = [labelidxs[i] for i in idx]

        most_common = max(set(labels), key=labels.count) #from stack exchange

        k = X[0].shape[1]

        flagmean = ca.flag_mean(X, k, fast = False)
        print('Flag Mean finished')

        sin_median = ca.irls_flag(X, k, n_its, 'sine', fast = False)[0]
        print('Sine Median finished')

        max_cosine = ca.irls_flag(X, k, n_its, 'cosine', fast = False)[0]
        print('Max Cos finished')


        p_correct['Flag Median'] += int(most_common in find_closest_point(data, flagmean, labellist, labelidxs) )
        p_correct['Sine Median'] += int(most_common in find_closest_point(data, sin_median, labellist, labelidxs))
        p_correct['Max Cosine'] += int(most_common in find_closest_point(data, max_cosine, labellist, labelidxs))

        print('Iteration '+str(ii+1)+' out of '+str(num_clusters)+' finished')
        print('--------------------------------------------')

    p_correct['Flag Median'] = p_correct['Flag Median']/num_clusters  
    p_correct['Sine Median'] = p_correct['Sine Median']/num_clusters
    p_correct['Max Cosine'] = p_correct['Max Cosine']/num_clusters

    
    return p_correct



# answer = experiments(gr_list, labellist, labelidxs, 350)
# print(answer['Flag Median'])
# print(answer['Sine Median'])
# print(answer['Cosine Max Median'])
# print(answer['Max Cosine'])

flag = []
sine_med = []
max_cos = []
for c in range(1,7):
    answer = experiments(gr_list, labellist, labelidxs, c*50)
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
    print('experiment '+str(c*50)+' finished')
    flag.append(answer['Flag Median'])
    sine_med.append(answer['Sine Median'])
    max_cos.append(answer['Max Cosine'])


plt.plot(flag, label = 'Flag Mean')
plt.plot(sine_med, label = 'Sine Median')
plt.plot(max_cos, label = 'Maximum Cosine')
plt.xlabel('Cluster Size')
plt.ylabel('Percent of exemplars that match their cluster label')
plt.legend()
plt.savefig('./Figures/cluster_examplars.png')