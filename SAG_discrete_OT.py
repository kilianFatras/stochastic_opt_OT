# Author: Kilian Fatras <kilian.fatras@ensta-paristech.fr>
#
# License: MIT License
###Implementation of the paper [Genevay et al., 2016]: (https://arxiv.org/pdf/1605.08527.pdf)

import numpy as np
import ot
import time



def coordinate_gradient(eps, nu, v, C, i):
    '''
    Compute the coordinate gradient update for regularized discrete distributions
        for (i, :)

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

    exp_v = np.zeros(n_target)
    r = c[i,:] - v
    exp_v = np.exp(-r/eps) * nu
    khi = exp_v/(np.sum(exp_v)) #= [exp(r_l/eps)*nu[l]/sum_vec for all l]
    return nu - khi #grad


def sag(epsilon, mu, nu, C, n_source, n_target, nb_iter, lr):
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
            optimization vector
    '''

    v = np.zeros(n_target)
    stored_gradient = np.zeros((n_source, n_target))
    avg_stored_gradient = np.zeros(n_target)
    for _ in range(nb_iter):
        i = np.random.randint(n_source) #SAG over the source points
        cur_stoc_grad = mu[i] * coordinate_gradient(epsilon, nu, v, C, i)
        avg_stored_gradient += (cur_stoc_grad - stored_gradient[i])
        stored_gradient[i] = cur_stoc_grad
        v += lr * (1./n_source) * avg_stored_gradient #Max --> Ascent
    return v

def recovered_u(epsilon, nu, v, C, n_source, n_target):
    """
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
    """
    exp_v = np.zeros(n_target)
    u = np.zeros(n_source)
    for i in range(n_source):
        r = c[i,:] - v
        exp_v = np.exp(-r/epsilon) * nu
        u[i] = - epsilon * np.log(np.sum(exp_v))
    return u


if __name__ == '__main__':
#Constants
    n_source = 2
    n_target = 7
    eps = 1
    nb_iter = 10000
    lr = 0.1


#Initialization
    mu = np.array([1./n_source for i in range(n_source)])
    X_source = np.array([i for i in range(n_source)])
    nu = np.array([1./n_target for i in range(n_target)])
    Y_target = np.array([i for i in range(n_target)])
    c = np.zeros((len(X_source), len(Y_target)))
    for i in range(len(X_source)):
        for j in range(len(Y_target)):
            c[i][j] = abs(X_source[i] - Y_target[j]) #c(x,y) = |x-y|
    #print("The cost matrix is : \n", c)

#Check Code
    #print(np.sum(mu), np.sum(nu))
    start_sag = time.time()
    opt_v = sag(eps, mu, nu, c, n_source, n_target, nb_iter, lr)
    opt_u = recovered_u(eps, nu, opt_v, c, n_source, n_target)
    end_sag = time.time()
    pi = np.zeros((len(X_source), len(Y_target)))
    for i in range(n_source):
        for j in range(n_target):
            pi[i][j] = np.exp((opt_u[i] + opt_v[j] - c[i,j])/eps) * mu[i] * nu[j]
    #print(opt_v)
    #print(opt_u)
    print(pi)

####TEST result from POT library
    start_sinkhorn = time.time()
    test = ot.sinkhorn(mu, nu, c, 1)
    end_sinkhorn = time.time()
    print("According to sinkhorn and POT, the transportation matrix is : \n", test)

    print("difference of the 2 methods : \n", pi - test)

    print("asgd time : ", start_sag - end_sag)
    print("sinkhorn time : ", start_sinkhorn - end_sinkhorn)
