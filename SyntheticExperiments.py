import numpy as np
import center_algorithms as ca
import matplotlib.pyplot as plt


def visual_2D(num1, num2):

    k=1
    n_its = 20

    Process1 = np.vstack([np.random.normal(0, .2, num1), np.random.normal(1, .2, num1)])
    if num2 != 0:
        Process2 = np.vstack([np.random.normal(1, .2, num2), np.random.normal(0, .2, num2)])
        data_array = np.hstack([Process1, Process2])
    else:
        data_array = Process1

    gr_list = []
    for i in range(data_array.shape[1]):
        point = data_array[:,[i]]
        gr_list.append(point/np.linalg.norm(point))
        plt.plot([-gr_list[i][0,0], gr_list[i][0,0]],[-gr_list[i][1,0],gr_list[i][1,0]], color = '.5', linestyle = 'dashed')
        

    flagmean = ca.flag_mean(gr_list, k, fast = False)
    print('Flag Mean finished')

    sin_median = ca.irls_flag(gr_list, k, n_its, 'sine', fast = False)[0]
    print('Sine Median finished')

    geodesic_median = ca.irls_flag(gr_list, k, n_its, 'geodesic', fast = False)[0]
    print('Geodesic finished')

    max_cosine = ca.irls_flag(gr_list, k, n_its, 'cosine', fast = False)[0]
    print('Max Cos finished')

    l0 = plt.plot([-flagmean[0,0], flagmean[0,0]], [-flagmean[1,0], flagmean[1,0]], label = 'Flag Mean', color = 'b')
    l1 = plt.plot([-sin_median[0,0], sin_median[0,0]], [-sin_median[1,0], sin_median[1,0]], label = 'Sine Median', color = 'g')
    l2 = plt.plot([-geodesic_median[0,0], geodesic_median[0,0]], [-geodesic_median[1,0], geodesic_median[1,0]], label = 'Geodesic Median', color = 'r')
    l3 = plt.plot([-max_cosine[0,0], max_cosine[0,0]], [-max_cosine[1,0], max_cosine[1,0]], label = 'Maximum Cosine', color = 'y')

    plt.xlim(-1,1)
    plt.ylim(-1,1)

    plt.savefig('./Figures/2example_2D_'+str(num1)+'_'+str(num2)+'.png')
    plt.close()

    return l0, l1, l2, l3



visual_2D(30, 0)
visual_2D(30, 8)
visual_2D(30, 15)
visual_2D(30, 23)
lines = visual_2D(30, 30)
lines = [l[0] for l in lines]

labels = ['Flag Mean', 'Sine Median', 'Geodesic Median', 'Maximum Cosine']

import pylab
fig = pylab.figure()
figlegend = pylab.figure(figsize=(3,2))
ax = fig.add_subplot(111)
figlegend.legend(lines, labels, 'center')
figlegend.savefig('./Figures/legend.png')

