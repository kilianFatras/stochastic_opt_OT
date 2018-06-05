# Author: Kilian Fatras <kilian.fatras@ensta-paristech.fr>
#
# License: MIT License
###Implementation of the paper [Genevay et al., 2016]: (https://arxiv.org/pdf/1605.08527.pdf)

import numpy as np

def stochatic_gradient(eps, nu, v, C, x_i):
    '''
    Compute the stochatic gradient update for regularized semi continuous
        distributions for (x_i, :)

    inputs :
        epsilon : float number,
            Regularization term > 0
        nu : np.ndarray,
            target measure
        v : np.ndarray,
            optimization vector
        C : np.ndarray,
            cost matrix
        i : number int,
            picked number i

    outputs :
        gradient : np.ndarray
    '''

    exp_v = np.zeros(n_target)
    r = c[x_i,:] - v
    exp_v = np.exp(-r/eps) * nu
    khi = exp_v/(np.sum(exp_v)) #= [exp(r_l/eps)*nu[l]/sum_vec for all l]
    return nu - khi #grad

def averaged_sgd(epsilon, mu, nu, C, n_source, n_target, nb_iter, lr):
    '''
    Compute the ASGD algorithm to solve the regularized semi continuous measures
        optimal transport max problem

    inputs :
        epsilon : float number,
            Regularization term > 0
        mu : np.ndarray,
            source measure
        nu : np.ndarray,
            target measure
        C : np.ndarray,
            cost matrix
        n_source : int number
            size of the source measure
        n_target : int number
            size of the target measure
        nb_iter : int number
            number of iteration
        lr : float number
            learning rate

    outputs :
        ave_v : np.ndarray
            optimization vector
    '''
    cur_v = np.zeros(n_target)
    ave_v = np.zeros(n_target)
    for cur_iter in range(nb_iter):
        k = cur_iter + 1
        x_i = np.random.randint(n_source)
        cur_stoc_grad = stochatic_gradient(epsilon, nu, cur_v, C, x_i)
        cur_v += (lr/np.sqrt(k)) * cur_stoc_grad #max -> Ascent
        ave_v = (1./(k)) * cur_v + (1 - 1./(k)) * ave_v
    return ave_v

def test_code(epsilon, mu, nu, C, n_source, n_target, nb_iter, lr):
    """
    Test the code
    parameters : same than averaged_sgd
    outputs : v and bar_H_e
    """
    bar_H_e = 0
    bar_h_e = 0
    v = averaged_sgd(epsilon, mu, nu, C, n_source, n_target, nb_iter, lr)
    #\bar{h_e} calculus
    for j in range(n_target):
        bar_h_e += v[j] * nu[j]

    #regularization calculus
    exp_v_final = np.zeros(n_target)
    r_final = C[i,:] - v
    exp_v_final = np.exp(-r_final/eps) * nu
    sum_exp_v_final = np.sum(exp_v_final)
    bar_h_e -= eps * (1 + np.log(sum_exp_v_final))

    #\bar{H_e} calculus
    bar_H_e = np.sum(bar_h_e * mu)

    return v, bar_H_e

if __name__ == '__main__':
#Constants
    n_source = 4
    n_target = 4
    eps = 1
    nb_iter = 30000
    lr = 0.1
    bar_H_e = 0

#Initialization
    mu = np.array([0.1 for i in range(n_source)])
    X_source = np.array([i for i in range(n_source)])
    nu = np.array([0.1 for i in range(n_target)])
    Y_target = np.array([i for i in range(n_target)])
    c = np.zeros((len(X_source), len(Y_target)))
    for i in range(len(X_source)):
        for j in range(len(Y_target)):
            c[i][j] = abs(X_source[i] - Y_target[j]) #c(x,y) = |x-y|
    print("The cost matrix is : ")
    print(c)

#Check Code
    v, bar_H_e = test_code(eps, mu, nu, c, n_source, n_target, nb_iter, lr)
    print(v)
    print(bar_H_e)
