import numpy as np
import center_algorithms as ca
import matplotlib.pyplot as plt
import time
import pandas




num_points = 20

n_its = 500

n_trials = 10

n = 10000
r = 10
ks = [i for i in range(1,21)]
mean_times = []
std_times = []
for j in range(20):
    k = ks[j]

    trial_times = []
    for seed in range(n_trials):
        np.random.seed(seed)
        center = np.random.rand(n,k)*10
        center_rep = np.linalg.qr(center)[0][:,:k]

        #generate dataset of points in Gr(k,n)
        data = []
        for i in range(num_points):
            Y_raw = center_rep + (np.random.rand(n,k)-.5)*.01
            Y = np.linalg.qr(Y_raw)[0][:,:k]
            data.append(Y)

        np.random.seed(1)
        Y_init = np.linalg.qr(np.random.rand(n,n))[0][:,:k]
        
        start = time.time()

        errors = ca.irls_flag(data, k, n_its, 'sine', opt_err = 'sine', fast = False, init = Y_init)[1]

        trial_times.append(time.time()- start)

    

    mean_times.append(np.mean(trial_times))
    std_times.append(np.std(trial_times))

    # iterations.append(len(errors))
    print(str(j)+' done.')

import pandas
trial_stats = pandas.DataFrame(columns = ['n','k','Iterations', 'Time'])
trial_stats['k'] = ks
trial_stats['Mean'] = mean_times
trial_stats['Std'] = std_times
trial_stats.to_csv('run_times_trial.csv', index_label = False)


# num_points = 20


# n_its = 500


# ns = [10**(i) for i in range(8)]
# rs = [1]
# rs += [10*i for i in range(1,8)]
# ks = [1]
# ks += [10*i for i in range(1,8)]
# times = []
# iterations = []
# for j in range(20):
#     n = ns[j]
#     k = ks[j]
#     r = rs[j]

#     center = np.random.rand(n,k)*10
#     center_rep = np.linalg.qr(center)[0][:,:k]

#     #generate dataset of points in Gr(k,n)
#     data = []
#     for i in range(num_points):
#         Y_raw = center_rep + (np.random.rand(n,k)-.5)*.01
#         Y = np.linalg.qr(Y_raw)[0][:,:k]
#         data.append(Y)

#     np.random.seed(1)
#     Y_init = np.linalg.qr(np.random.rand(n,n))[0][:,:k]
    
#     start = time.time()

#     errors = ca.irls_flag(data, k, n_its, 'sine', opt_err = 'sine', fast = False, init = Y_init)[1]

#     times.append(time.time()- start)

#     iterations.append(len(errors))
#     print(str(j)+' done.')


# trial_stats = pandas.DataFrame(columns = ['n','k','Iterations', 'Time'])
# trial_stats['n'] = ns
# trial_stats['k'] = ks
# trial_stats['Iterations'] = iterations
# trial_stats['Time'] = times
# trial_stats.to_csv('run_times_trials.csv')
