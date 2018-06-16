# Author: Kilian Fatras <kilian.fatras@ensta-paristech.fr>
#
# License: MIT License
###Implementation of the paper [Genevay et al., 2016]: (https://arxiv.org/pdf/1605.08527.pdf)

import numpy as np
import ot
import matplotlib.pylab as pl
import time



def coordinate_gradient(eps, nu, v, C, i):
    '''
    Compute the coordinate gradient update for regularized semi continuous
        distributions for (i, :)

    Parameters
    ----------

    epsilon : float number,
        Regularization term > 0
    nu : np.ndarray(nt,),
        target measure
    v : np.ndarray(nt,),
        optimization vector
    C : np.ndarray(ns, nt),
        cost matrix
    i : number int,
        picked number i

    Returns
    -------

    coordinate gradient : np.ndarray(nt,)
    '''

    r = c[i,:] - v
    exp_v = np.exp(-r/eps) * nu
    khi = exp_v/(np.sum(exp_v)) #= [exp(r_l/eps)*nu[l]/sum_vec for all l]
    return nu - khi #grad

def averaged_sgd_entropic_transport(epsilon, mu, nu, C, n_source, n_target, nb_iter, lr):
    '''
    Compute the ASGD algorithm to solve the regularized semi continuous measures
        optimal transport max problem

    Parameters
    ----------

    epsilon : float number,
        Regularization term > 0
    mu : np.ndarray(ns,),
        source measure
    nu : np.ndarray(nt,),
        target measure
    C : np.ndarray(ns, nt),
        cost matrix
    n_source : int number
        size of the source measure
    n_target : int number
        size of the target measure
    nb_iter : int number
        number of iteration
    lr : float number
        learning rate


    Returns
    -------

    ave_v : np.ndarray(nt,)
        optimization vector
    '''

    cur_v = np.zeros(n_target)
    ave_v = np.zeros(n_target)
    for cur_iter in range(nb_iter):
        k = cur_iter + 1
        i = np.random.randint(n_source)
        cur_coord_grad = coordinate_gradient(epsilon, nu, cur_v, C, i)
        cur_v += (lr/np.sqrt(k)) * cur_coord_grad #max -> Ascent
        ave_v = (1./k) * cur_v + (1 - 1./k) * ave_v
    return ave_v

def c_transform_entropic(epsilon, nu, v, C, n_source, n_target):
    '''
    The goal is to recover u from the c-transform

    Parameters
    ----------

    epsilon : float
        regularization term > 0
    nu : np.ndarray(nt,)
        target measure
    v : np.ndarray(nt,)
        dual variable
    C : np.ndarray(ns, nt)
        cost matrix
    n_source : np.ndarray(ns,)
        size of the source measure
    n_target : np.ndarray(nt,)
        size of the target measure

    Returns
    -------

    u : np.ndarray(ns,)
    '''

    u = np.zeros(n_source)
    for i in range(n_source):
        r = c[i,:] - v
        exp_v = np.exp(-r/epsilon) * nu
        u[i] = - epsilon * np.log(np.sum(exp_v))
    return u

def transportation_matrix_entropic(epsilon, mu, nu, C, n_source, n_target, nb_iter, lr):
    '''
    Compute the transportation matrix to solve the regularized discrete measures
        optimal transport problem

    Parameters
    ----------

    epsilon : float number,
        Regularization term > 0
    mu : np.ndarray(ns,),
        source measure
    nu : np.ndarray(nt,),
        target measure
    C : np.ndarray(ns, nt),
        cost matrix
    n_source : int number
        size of the source measure
    n_target : int number
        size of the target measure
    nb_iter : int number
        number of iteration
    lr : float number
        learning rate

    Returns
    -------

    pi : np.ndarray(ns, nt)
        transportation matrix
    '''

    opt_v = averaged_sgd_entropic_transport(epsilon, mu, nu, C, n_source, n_target, nb_iter, lr)
    opt_u = c_transform_entropic(epsilon, nu, opt_v, C, n_source, n_target)
    pi = np.exp((opt_u[:, None] + opt_v[None, :] - c[:, :])/eps) * mu[:, None] * nu[None, :]
    return pi

if __name__ == '__main__':
#Constants
    n_source = 7
    n_target = 4
    eps = 1
    nb_iter = 10000
    lr = 0.1

#Initialization
    mu = np.random.random(n_source)
    mu *= (1./np.sum(mu))
    X_source = np.arange(n_source)
    nu = np.random.random(n_target)
    nu *= (1./np.sum(nu))
    Y_target = np.arange(0, n_target)

    c = np.abs(X_source[:, None] - Y_target[None, :])
    #print("The cost matrix is : \n", c)

#Check Code
    start_asgd = time.time()
    asgd_pi = transportation_matrix_entropic(eps, mu, nu, c, n_source, n_target, nb_iter, lr)
    end_asgd = time.time()
    print("The transportation matrix from SAG is : \n", asgd_pi)


####TEST result from POT library
    start_sinkhorn = time.time()
    sinkhorn_pi = ot.sinkhorn(mu, nu, c, 1)
    end_sinkhorn = time.time()
    print("According to sinkhorn and POT, the transportation matrix is : \n", sinkhorn_pi)

    print("difference of the 2 methods : \n", asgd_pi - sinkhorn_pi)

    print("asgd time : ", end_asgd - start_asgd)
    print("sinkhorn time : ", end_sinkhorn - start_sinkhorn)

#### Plot Results
    pl.figure(4, figsize=(5, 5))
    ot.plot.plot1D_mat(mu, nu, asgd_pi, 'OT matrix ASGD')
    pl.show()

    pl.figure(4, figsize=(5, 5))
    ot.plot.plot1D_mat(mu, nu, sinkhorn_pi, 'OT matrix Sinkhorn')
    pl.show()
