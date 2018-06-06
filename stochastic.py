# Author: Kilian Fatras <kilian.fatras@ensta-paristech.fr>
#
# License: MIT License
###Implementation of the paper [Genevay et al., 2016]: (https://arxiv.org/pdf/1605.08527.pdf)

import numpy as np
import ot
import time



def coordinate_gradient(eps, nu, v, C, i):
    '''
    Compute the coordinate gradient update for regularized discrete
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


def sag_entropic_transport(epsilon, mu, nu, C, n_source, n_target, nb_iter, lr):
    '''
    Compute the SAG algorithm to solve the regularized discrete measures
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

    v : np.ndarray(nt,)
        dual variable
    '''

    v = np.zeros(n_target)
    stored_gradient = np.zeros((n_source, n_target))
    sum_stored_gradient = np.zeros(n_target)
    for _ in range(nb_iter):
        i = np.random.randint(n_source) #SAG over the source points
        cur_coord_grad = mu[i] * coordinate_gradient(epsilon, nu, v, C, i)
        sum_stored_gradient += (cur_coord_grad - stored_gradient[i])
        stored_gradient[i] = cur_coord_grad
        v += lr * (1./n_source) * sum_stored_gradient #Max --> Ascent
    return v

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

def transportation_matrix_entropic(method, epsilon, mu, nu, C, n_source, n_target, nb_iter, lr):
    '''
    Compute the transportation matrix to solve the regularized discrete measures
        optimal transport max problem

    Parameters
    ----------

    methode : str,
        used method (SAG or ASGD)
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
    if method.lower() == "sag":
        opt_v = sag_entropic_transport(epsilon, mu, nu, C, n_source, n_target, nb_iter, lr)
    else :
        opt_v = averaged_sgd_entropic_transport(epsilon, mu, nu, C, n_source, n_target, nb_iter, lr)
    opt_u = c_transform_entropic(epsilon, nu, opt_v, C, n_source, n_target)
    pi = np.exp((opt_u[:, None] + opt_v[None, :] - c[:,:])/eps) * mu[:, None] * nu[None, :]
    return pi

if __name__ == '__main__':
#Constants
    n_source = 7
    n_target = 4
    eps = 1
    nb_iter = 10000
    lr = 0.1
    method = "Sag"

#Initialization
    mu = (1./n_source) * np.ones(n_source)
    X_source = np.arange(n_source)
    nu = (1./n_target) * np.ones(n_target)
    Y_target = np.arange(0, 2 * n_target, 2)

    c = np.abs(X_source[:, None] - Y_target[None, :])
    #print("The cost matrix is : \n", c)

#Check Code
    print(np.sum(mu), np.sum(nu))
    start_sag = time.time()
    pi = transportation_matrix_entropic(method, eps, mu, nu, c, n_source, n_target, nb_iter, lr)
    end_sag = time.time()
    print("The transportation matrix from ASGD is : \n", pi)


####TEST result from POT library
    start_sinkhorn = time.time()
    test = ot.sinkhorn(mu, nu, c, 1)
    end_sinkhorn = time.time()
    print("According to sinkhorn and POT, the transportation matrix is : \n", test)

    print("difference of the 2 methods : \n", pi - test)

    print("asgd time : ", end_sag - start_sag)
    print("sinkhorn time : ", end_sinkhorn - start_sinkhorn)
