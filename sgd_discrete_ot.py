# Author: Kilian Fatras <kilian.fatras@ensta-paristech.fr>
#
# License: MIT License
###Implementation of the paper [Genevay et al., 2016]: (https://arxiv.org/pdf/1605.08527.pdf)

import numpy as np
import ot



def partial_gradient_F_u(eps, u, v, M, n_source, n_target, batch_u, batch_v, batch_size):
    grad = np.zeros(n_source)
    for i in batch_u:
        for j in batch_v:
            grad[i] += - np.exp((u[i] + v[j] - M[i,j])/eps)
        grad[i] += 1
    return grad


def partial_gradient_F_v(eps, u, v, M, n_source, n_target, batch_u, batch_v, batch_size):
    grad = np.zeros(n_target)
    for j in batch_v:
        for i in batch_u:
            grad[j] += - np.exp((u[i] + v[j] - M[i,j])/eps)
        grad[j] += 1
    return grad


def sgd_entropic_regularization(eps, a, b, M, batch_size, n_source, n_target, numItermax, lr):
    '''
    Compute the SAG algorithm to solve the regularized discrete measures
        optimal transport max problem

    Parameters
    ----------

    epsilon : float bmber,
        Regularization term > 0
    a : np.ndarray(ns,),
        source measure
    b : np.ndarray(nt,),
        target measure
    C : np.ndarray(ns, nt),
        cost matrix
    batch_size : int bmber
        size of the batch
    n_source : int bmber
        size of the source measure
    n_target : int bmber
        size of the target measure
    numItermax : int bmber
        bmber of iteration
    lr : float bmber
        learning rate

    Returns
    -------

    u : np.ndarray(ns,)
        dual variable
    v : np.ndarray(nt,)
        dual variable
    '''

    u = np.zeros(n_source)
    v = np.zeros(n_target)
    for cur_iter in range(numItermax):
        k = cur_iter + 1
        batch_u = np.random.choice(n_source, batch_size, replace=False)
        batch_v = np.random.choice(n_target, batch_size, replace=False)
        cur_subgrad_u = partial_gradient_F_u(eps, u, v, M, n_source, n_target, batch_u, batch_v, batch_size)
        u += (lr/np.sqrt(k)) * cur_subgrad_u
        cur_subgrad_v = partial_gradient_F_v(eps, u, v, M, n_source, n_target, batch_u, batch_v, batch_size)
        v += (lr/np.sqrt(k)) * cur_subgrad_v
    return u, v


def transportation_matrix_entropic(epsilon, a, b, M, batch_size, n_source, n_target, numItermax, lr):
    '''
    Compute the transportation matrix to solve the regularized discrete measures
        optimal transport max problem

    Parameters
    ----------

    epsilon : float bmber,
        Regularization term > 0
    a : np.ndarray(ns,),
        source measure
    b : np.ndarray(nt,),
        target measure
    C : np.ndarray(ns, nt),
        cost matrix
    n_source : int bmber
        size of the source measure
    n_target : int bmber
        size of the target measure
    numItermax : int bmber
        bmber of iteration
    lr : float bmber
        learning rate

    Returns
    -------

    pi : np.ndarray(ns, nt)
        transportation matrix
    '''

    opt_u, opt_v = sgd_entropic_regularization(epsilon, a, b, M, batch_size, n_source, n_target, numItermax, lr)
    print(opt_v)
    print(opt_u)
    pi = np.exp((opt_u[:, None] + opt_v[None, :] - M[:,:])/eps) * a[:, None] * b[None, :]
    return pi

if __name__ == '__main__':
    n_source = 7
    n_target = 4
    eps = 1
    numItermax = 10000
    lr = 0.1
    batch_size = 2


    a = (1./n_source) * np.ones(n_source)
    b = (1./n_target) * np.ones(n_target)
    X_source = np.arange(n_source)
    Y_target = np.arange(0, 2 * n_target, 2)
    M = np.abs(X_source[:, None] - Y_target[None, :])
    sgd_pi = transportation_matrix_entropic(eps, a, b, M, batch_size, n_source, n_target, numItermax, lr)
    print(sgd_pi)

    test = ot.sinkhorn(a, b, M, 1)
    print("According to sinkhorn and POT, the transportation matrix is : \n", test)

    #print("Difference between my solution and POT : \n", sgd_pi - test)
