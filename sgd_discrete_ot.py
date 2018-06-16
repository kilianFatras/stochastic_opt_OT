# Author: Kilian Fatras <kilian.fatras@ensta-paristech.fr>
#
# License: MIT License
###Implementation of the paper [Genevay et al., 2016]: (https://arxiv.org/pdf/1605.08527.pdf)

import numpy as np
import matplotlib.pylab as pl
import ot



def partial_gradient_dF_du(M, reg, u, v, batch_u, batch_v):
    '''Computes the partial gradient of F_\W_varepsilon

        Compute the partial gradient of the dual problem:
        ..Math:
            \forall i in batch_u,
                grad_U_i = 1 + sum_{j in batch_v} exp((u_i + v_j - M_{i,j})/reg)

        where :
        - M is the (ns,nt) metric cost matrix
        - u, v are dual variables in R^ixR^J
        - reg is the regularization term
        - batch_u and batch_v are list of index

    Parameters
    ----------

    reg : float bmber,
        Regularization term > 0
    a : np.ndarray(ns,),
        source measure
    b : np.ndarray(nt,),
        target measure
    C : np.ndarray(ns, nt),
        cost matrix
    batch_u : np.ndarray(bs,)
        batch of index of u
    batch_v : np.ndarray(bs,)
        batch of index of v

    Returns
    -------

    grad : np.ndarray(ns,)
        partial grad_u of F
    '''

    grad_u = np.zeros(np.shape(M)[0])
    grad_u[batch_u] = 1
    for j in batch_v:
        grad_u[batch_u] -= np.exp((u[batch_u] + v[j] - M[batch_u, j])/reg)
    return grad_u

def partial_gradient_dF_dv(M, reg, u, v, batch_u, batch_v):
    '''Computes the partial gradient of F_\W_varepsilon

        Compute the partial gradient of the dual problem:
        ..Math:
            \forall j in batch_v,
                grad_U_j = 1 + sum_{i in batch_u} exp((u_i + v_j - M_{i,j})/reg)

        where :
        - M is the (ns,nt) metric cost matrix
        - u, v are dual variables in R^ixR^J
        - reg is the regularization term
        - batch_u and batch_v are list of index

        Parameters
        ----------

        reg : float bmber,
            Regularization term > 0
        a : np.ndarray(ns,),
            source measure
        b : np.ndarray(nt,),
            target measure
        C : np.ndarray(ns, nt),
            cost matrix
        batch_u : np.ndarray(bs,)
            batch of index of u
        batch_v : np.ndarray(bs,)
            batch of index of v

        Returns
        -------

        grad : np.ndarray(ns,)
            partial grad_u of F
    '''

    grad_v = np.zeros(np.shape(M)[1])
    grad_v[batch_v] = 1
    for i in batch_u:
        grad_v[batch_v] -= np.exp((u[i] + v[batch_v] - M[i, batch_v])/reg)
    return grad_v


def sgd_entropic_regularization(M, reg, batch_size, numItermax, lr):
    '''
    Compute the sgd algorithm to solve the regularized discrete measures
        optimal transport max problem

    Parameters
    ----------

    reg : float bmber,
        Regularization term > 0
    a : np.ndarray(ns,),
        source measure
    b : np.ndarray(nt,),
        target measure
    C : np.ndarray(ns, nt),
        cost matrix
    batch_size : int bmber
        size of the batch
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

    u = np.random.randn(n_source)
    v = np.random.randn(n_target)
    for cur_iter in range(numItermax):
        k = np.sqrt(cur_iter + 1)
        batch_u = np.random.choice(n_source, batch_size, replace=False)
        batch_v = np.random.choice(n_target, batch_size, replace=False)
        u += lr * partial_gradient_dF_du(M, reg, u, v, batch_u, batch_v)
        v += lr * partial_gradient_dF_dv(M, reg, u, v, batch_u, batch_v)
    return u, v


def transportation_matrix_entropic(a, b, M, reg, batch_size, numItermax, lr):
    '''
    Compute the transportation matrix to solve the regularized discrete measures
        optimal transport max problem

    Parameters
    ----------

    reg : float bmber,
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

    opt_u, opt_v = sgd_entropic_regularization(M, reg, batch_size, numItermax,
                                               lr)
    pi = (np.exp((opt_u[:, None] + opt_v[None, :] - M[:, :])/reg) *
          a[:, None] * b[None, :])
    return pi

if __name__ == '__main__':
    n_source = 7
    n_target = 4
    reg = 1
    numItermax = 700000
    lr = 0.1
    batch_size = 3


    a = ot.utils.unif(n_source)
    b = ot.utils.unif(n_target)
    #a = np.random.random(n_source)
    #b = np.random.random(n_target)
    X_source = np.arange(n_source)
    Y_target = np.arange(0, 2 * n_target, 2)
    M = np.abs(X_source[:, None] - Y_target[None, :])
    sgd_pi = transportation_matrix_entropic(a, b, M, reg, batch_size, numItermax, lr)
    print(sgd_pi)

    sinkhorn_pi = ot.sinkhorn(a, b, M, reg)
    print("According to sinkhorn and POT, the transportation matrix is : \n", sinkhorn_pi)

    print("Difference between my solution and POT : \n", sgd_pi - sinkhorn_pi)

    pl.figure(4, figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, sgd_pi, 'OT matrix ASGD')
    pl.show()

    pl.figure(4, figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, sinkhorn_pi, 'OT matrix ASGD')
    pl.show()
