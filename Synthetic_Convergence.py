import numpy as np
import center_algorithms as ca
import matplotlib.pyplot as plt


'''
plotting function
'''
def add_line(data, lstyle, mkr, lbl, color = 'b', ci = 95):
    med = np.median(data, axis = 0)

    lower_ci = []
    upper_ci = []
    for i in range(data.shape[1]):
        lower_ci.append(np.percentile(data[:,i],50-ci/2))
        upper_ci.append(np.percentile(data[:,i],50+ci/2))
    plt.fill_between(list(np.arange(len(med))), lower_ci, upper_ci, alpha=0.25, color = color)
    plt.xticks(np.arange(0, len(med), 2))
    # plt.rcParams["text.usetex"] =True
    plt.plot(med, color = color, linestyle=lstyle, marker=mkr, markevery = 2, linewidth=.5, label = lbl)




def convergence_check(gr_list, n_its):

    [n,k] = gr_list[0].shape

    irls_sin_median = []
    gd_sin_median = []

    # irls_geodesic_median = []
    # gd_geodesic_median = []

    irls_max_cosine = []
    gd_max_cosine = []

    Y_raw = np.random.rand(n,k)
    Y = np.linalg.qr(Y_raw)[0][:,:k]

    for i in range(10): 

        irls_sin_median.append(ca.irls_flag(gr_list, k, n_its, 'sine', fast = False, init = Y)[1])
        gd_sin_median.append(ca.gradient_descent(gr_list, k, -.01, n_its, 'sine', init = Y)[1])
        print('Sine Median finished')

        # irls_geodesic_median.append(ca.irls_flag(gr_list, k, n_its, 'geodesic', fast = False, init = Y)[1])
        # gd_geodesic_median.append(ca.gradient_descent(gr_list, k, .01, n_its, 'geodesic', init = Y)[1])
        # print('Geodesic finished')

        irls_max_cosine.append(ca.irls_flag(gr_list, k, n_its, 'cosine', fast = False, init = Y)[1])
        gd_max_cosine.append(ca.gradient_descent(gr_list, k, -.01, n_its, 'cosine', init = Y)[1])
        print('Max Cos finished')

    irls_sin_median = np.vstack(irls_sin_median)
    gd_sin_median = np.vstack(gd_sin_median)

    # irls_geodesic_median = np.vstack(irls_geodesic_median)
    # gd_geodesic_median = np.vstack(gd_geodesic_median)

    irls_max_cosine = np.vstack(irls_max_cosine)
    gd_max_cosine = np.vstack(gd_max_cosine)

    #make the plots
    LINESTYLES = ["-", "--", ":", "-."]
    MARKERS = ['D', 'o', 'X', '*', '<', 'd', 'S', '>', 's', 'v']
    COLORS = ['b','k','c','m','y']

    add_line(irls_sin_median, LINESTYLES[0], MARKERS[0], 'IRLS', 'g')
    add_line(gd_sin_median, LINESTYLES[1], MARKERS[1], 'Gradient Descent', 'b')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Objective function value')
    plt.savefig('./Figures/sin_median_convergence.png')
    plt.close()

    # add_line(irls_geodesic_median, LINESTYLES[0], MARKERS[0], 'IRLS', 'g')
    # add_line(gd_geodesic_median, LINESTYLES[1], MARKERS[1], 'Gradient Descent', 'b')
    # plt.legend()
    # plt.savefig('./Figures/geodesic_median_convergence.png')
    # plt.close()

    add_line(irls_max_cosine, LINESTYLES[0], MARKERS[0], 'IRLS', 'g')
    add_line(gd_max_cosine, LINESTYLES[1], MARKERS[1], 'Gradient Descent', 'b')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Objective function value')
    plt.savefig('./Figures/max_cosine_convergence.png')
    plt.close()



n = 100
r = 3
m = 10

n_its = 5

data = []
#sample from uniform distribution
for i in range(m):
	[u,t] = np.linalg.qr(np.random.rand(n,n))
	data.append(u[:,:r])



convergence_check(data, n_its)