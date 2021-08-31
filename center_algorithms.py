import numpy as np

'''
- fix PLS code for fast == True
- horst's and maybe one more algorith
- 2d checks
'''


def calc_error_cos(data, Y):
    s = np.linalg.svd(Y.T @ data)[1]
    return np.sum(s)

def calc_error_1_2(data, Y, sin_cos):
    k = Y.shape[1]
    err = 0
    if sin_cos == 'cosine':
        for x in data:
            r = np.min([k,x.shape[1]])
            err += np.sqrt(np.trace(Y.T @ x @ x.T @ Y))
    elif sin_cos == 'sine':
        for x in data:
            r = np.min([k,x.shape[1]])
            sin_sq = r - np.trace(Y.T @ x @ x.T @ Y)
            if sin_sq < 0:
                sin_sq = 0
            err += np.sqrt(sin_sq)
    elif sin_cos == 'sinesq':
        for x in data:
            r = np.min([k,x.shape[1]])
            sin_sq = r - np.trace(Y.T @ x @ x.T @ Y)
            if sin_sq < 0:
                sin_sq = 0
            err += sin_sq
    elif sin_cos == 'geodesic':
        for x in data:
            cos = np.sqrt(Y.T @ x @ x.T @ Y)[0][0]
            if cos > 1:
#                 print(cos)
                cos= 1
            elif cos < 0:
#                 print(cos)
                cos = 0
            err += np.arccos(cos)
    return err


def flag_mean(data, k, fast = False):
    X = np.hstack(data)
    if fast:
        #Initialize
        # U  = X[:,:k][:] 
        mean = np.random.rand(X.shape[0],k)
        V = np.zeros((X.shape[1],k))

        #err[j] stores 2-norm squared difference between ith and (i+1)th iterate of U[:,j]
        err = []
        #iteration of partial least squares
        for j in range(k):
            err1 = []
            err1.append(1)
            while err1[-1] > .000000001:
            # for _ in range(n_iters):
                old_U = mean[:,[j]][:]
                V[:,j] = mean[:,j] @ X / (mean[:,j] @ mean[:,[j]])
                V[:,j] = V[:,j]/np.linalg.norm(V[:,j])
                new_U =  X @ V[:,[j]] / (V[:,j] @ V[:,[j]])
                err1.append(np.linalg.norm(old_U - new_U)**2)
                print(err1[-1])
                mean[:,[j]] = new_U[:]
            err.append(err1[1:])
        
    else:
        mean = np.linalg.svd(X)[0][:,:k]
    return mean

#change eps to .0000001 to reproduce figures in paper
def flag_mean_iteration(data, Y0, weight, fast = False, eps = .0000001):
    k = Y0.shape[1]
    aX = []
    al = []

    ii=0

    for x in data:
        if weight == 'cosine':
            cossq = np.trace(Y0.T @ x @ x.T @ Y0)
            al.append((cossq+eps)**(-1/4))
        elif weight == 'sine':
            r = np.min([k,x.shape[1]])
            sinsq = r - np.trace(Y0.T @ x @ x.T @ Y0)
            al.append((sinsq+eps)**(-1/4))
        elif weight == 'geodesic':
            r = np.min([k,x.shape[1]])
            sinsq = 1 - Y0.T @ x @ x.T @ Y0
            cossq = Y0.T @ x @ x.T @ Y0
            al.append((sinsq*cossq + eps)**(-1/4))
        else:
            print('sin_cos must be geodesic, sine or cosine')
        aX.append(al[-1]*x)
        ii+= 1

    Y = flag_mean(aX, k, fast)

    return Y

def irls_flag(data, k, n_its, sin_cos, fast = False, init = 'random', seed = 0): 
    err = []
    n = data[0].shape[0]


    #initialize
    if init == 'random':
        #randomly
        np.random.seed(seed)
        Y_raw = np.random.rand(n,k)
        Y = np.linalg.qr(Y_raw)[0][:,:k]
    else:
        Y = init

    err.append(calc_error_1_2(data, Y, 'geodesic'))

    #flag mean iteration function
    for _ in range(n_its):
        Y = flag_mean_iteration(data, Y, sin_cos, fast)
        err.append(calc_error_1_2(data, Y, 'geodesic'))

    return Y, err


#This is wrong!!!
# def irls_flag_1_infty_cosine(data, k, n_its):
#     n = data[0].shape[0]

#     #initialize
#     #randomly
#     Y_raw = np.random.rand(n,k)
#     Y = np.linalg.qr(Y_raw)[0][:,:k]

#     Beta = []
    
#     for x in data:
#         for i in range(n_its):
#             w = np.trace(Y.T @ x @ x.T @ Y)**(-1/4)
#             B = np.linalg.svd(w*x)[0][:,[0]]
#         Beta.append(B)
#     Beta = np.hstack(Beta)

#     Y = np.linalg.svd(Beta)[0][:,:k]
#     return Y

def calc_gradient(data, Y0, weight):
    k = Y0.shape[1]
    aX = []
    al = []
    for x in data:
        if weight == 'cosine':
            al.append(np.trace(Y0.T @ x @ x.T @ Y0)**(-1/4))
        elif weight == 'sine':
            r = np.min([k,x.shape[1]])
            sin_sq = r - np.trace(Y0.T @ x @ x.T @ Y0)
            if sin_sq < .0000000001 :
                sin_sq = 0
            al.append(sin_sq**(-1/4))
        elif weight == 'geodesic':
            r = np.min([k,x.shape[1]])
            al.append(((1 - Y0.T @ x @ x.T @ Y0)**(-1/4))*((Y0.T @ x @ x.T @ Y0)**(-1/4)))
        else:
            print('sin_cos must be geodesic, sine or cosine')
        aX.append(al[-1]*x)

    big_X = np.hstack(aX)

    return big_X @ big_X.T @ Y0


def gradient_descent(data, k, alpha, n_its, sin_cos, init = 'random'):
    n = data[0].shape[0]

    #initialize
    if init == 'random':
        #randomly
        Y_raw = np.random.rand(n,k)
        Y = np.linalg.qr(Y_raw)[0][:,:k]
    else:
        Y = init

    err = []
    err.append(calc_error_1_2(data, Y, sin_cos))

    for _ in range(n_its):
        Fy = calc_gradient(data,Y,sin_cos)
        # project the gradient onto the tangent space
        G = (np.eye(n)-Y@Y.T)@Fy
        # %error check
        # if sum(G) == 0 || any(isnan(G))
        #     Fy
        #     G
        #     break
        # end
        # move to a point that is alpha along the geodesic at Y0 in the
        # direction of G
        [U,S,V] = np.linalg.svd(G)
        cosin = np.diag(np.cos(-alpha*S))
        sin = np.vstack([np.diag(np.sin(-alpha*S)), np.zeros((n-k,k))])
        if cosin.shape[0] == 1:
            Y = Y*V*cosin*V.T+U@sin *V.T
        else:
            Y = Y@V@cosin@V.T+U@sin@V.T
        
        err.append(calc_error_1_2(data, Y, sin_cos))
    return Y, err