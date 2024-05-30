#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:33:36 2024

@author: andrewlowy
"""

import random
import math
import numpy as np
from numpy import mean
from numpy import median
from numpy import percentile
import os
import signal 


beta = 1 # smoothnes parameter
L = 5*beta #Lipschitz parameter if ||X|| \leq 1 and ||W|| \leq 2
N = 125 # total samples
n = int(4*N/5)  # training samples
d = 100 # dim of model
n_test = int(N/5)
delta = 1/n**1.5
#C is diameter of W
###### function f(w,x) = beta


#######################  DATA GENERATION  #####################################
# Generate x  

#X_vectors = [np.random.uniform(-1/np.sqrt(d), 1/np.sqrt(d), size=d) for _ in range(N)]




####################### HELPER FUNCTIONS ##############################
# Compute norm of gradient of training or test loss
# def grad_norm(A_list, b_list, w):
#     total_sum = np.zeros_like(w)  # Initialize total sum to zeros
#     num_matrices = len(A_list)
#     for i in range(num_matrices):
#         A = A_list[i]
#         b = b_list[i]
#         total_sum += np.dot(A, w) - b
#     return np.linalg.norm(total_sum / num_matrices)

def grad(w,x_batch):
    squared_norm = np.dot(w,w)
    xbar = np.mean(x_batch, axis=0)
    return beta*(w + np.cos(squared_norm)*w + xbar)

def grad_norm(w, X): #X is a list of vectors (train or test set)
    return beta*np.linalg.norm(grad(w, X))


#def sample_indices(n, p):  # poisson subsampling
   # return [i for i in range(n) if np.random.choice([True, False], p=[p, 1-p])]

def sample_indices(n, p):
    def handler(signum, frame):
        raise TimeoutError("Timeout occurred")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(10)  # Set a timeout of 10 seconds

    try:
        indices = [i for i in range(n) if np.random.choice([True, False], p=[p, 1-p])]
    except TimeoutError:
        print("Sampling indices timed out. Returning empty list.")
        indices = []

    signal.alarm(0)  # Reset the alarm
    return indices


def clip(w, C):
    if np.linalg.norm(w) > C:
        w = C*w/np.linalg.norm(w)
    return w

def random_vector_in_ball(d, C=2):
    """
    Generate a random vector inside a d-dimensional Euclidean ball of radius 2:
    """
    x = np.random.normal(0, 1, d)
    s = np.random.uniform(0, 1)**(1/d)
    return C*(s * x)/ np.linalg.norm(x)

###################### DATA GENERATION ############################

X_vectors = [random_vector_in_ball(d, C=1) for _ in range(N)]
# Train test split:
X_train = X_vectors[0:n]
X_test = X_vectors[n: N+1]
Xbar = np.mean(X_train, axis=0)
####################### ALGORITHMS #####################################



def dp_sgd(w0, T, X, K, eta, eps, delta, C = 2):#w0 needs to have norm \leq 2 
    sigma = np.sqrt(8*T*(L**2)*math.log(1/delta)/((eps*n)**2))
    K = min(max(n*eps/(2*np.sqrt(T)), 1), n)
    p = K/n
    #print(p)
    #w = np.random.rand(d)
    w = w0 
    for t in range(T):
        if t % 10 == 0:
            print(f"iteration {t}: w = {w}; grad norm = {grad_norm(w,X)}")
        S = []
        while len(S) == 0:
            S = sample_indices(n, p)
        k = len(S)
        grads = []
        for i in S:
            grad_i = grad(w, X[i]) # Gradient computation
            grads.append(grad_i)
        G = sum(grads)/k + np.random.normal(scale=sigma, size=d)
        # print(np.shape(G))
        w = clip(w - eta*G, C)
        #w = w - (eta/np.sqrt(t+1))*G
    return w, grad_norm(w, X_train), grad_norm(w, X_test)
# random.choice(Ws)


def dp_spider(w0, T, q, X, K1, K2, eta, eps, delta, C=2): #X should have length n (e.g. X = X_train); ||w0|| < C=2
    n = len(X)
    d = len(w0)
    sigma1 = np.sqrt(8*(T/q)*(L**2)*math.log(1/delta)/((eps*n)**2))
    sigma2 = np.sqrt(8*(T - T/q)*((beta*(2 + C**2))**2)*math.log(1/delta)/((eps*n)**2))
    sigma3 = np.sqrt(32*(T/q)*(L**2)*math.log(1/delta)/((eps*n)**2))
    Tprime = int(T/q)
    K1 = n
    if Tprime == 0 or eps == 0:
    # Handle the case where Tprime or eps is zero to avoid division by zero
        K2 = 1
    else:
        K2 = int(min(max(n * eps / (2 * np.sqrt(Tprime)), 1), n))
    p1 = K1/n
    p2 = K2/n
    #w = np.random.rand(d)
    w = w0
    u = w
    for t in range(Tprime):
        if t % 10 == 0:
            print(f"iteration {t}: w = {w}; grad norm = {grad_norm(w,X)}")
        #eta = eta/np.sqrt(t+1)
        #print(f"Phase {t}: grad_norm = {grad_norm(A_list, b_list, w)}")
        S = []
        while len(S) == 0:
            S = sample_indices(n, p1)
        k = len(S)
        grads = []
        for i in S:
            g = grad(w, X[i])
            grads.append(g)
        v = sum(grads)/k + np.random.normal(scale=sigma1, size=d)
        #print("v0 = ", v)
        u = clip(w - eta*v, C)
        #print("w1 =", u)
        for s in range(q):
            S2 = []
            while len(S2) == 0:
                S2 = sample_indices(n, p2)
            k2 = len(S2)
            diffs = []
            for i in S2:
                g1 = grad(u, X[i])
                g2 = grad(w, X[i])
                diff = g1 - g2
                diffs.append(diff)
            Delta = sum(diffs)/k2 + np.random.normal(scale=min(sigma2 *
                                                               np.linalg.norm(w - u), sigma3), size=d)
            #print(f"Iteration {s+1}: Delta = {Delta}")
            v = v + Delta
            #print(f"Iteration {s+1}: v_{s+1} = {v}")
            uold = u
            w = uold
            #print(f"Iteration {s+1}: w_{s + 1} = {w}")
            u = clip(w - eta*v, C)
            #print(f"Iteration {s+2}: w_{s + 2} = {u}")
        w = u
    return w, grad_norm(w, X_train), grad_norm(w, X_test)


# spider after sgd; warmstart(w0, X_train, Tsgd, Tspider, q_ws, Ksgd, K1, K2, eta_ws_sgd, eta_ws_spi, eps1, eps2, delta)
def warmstart(winit, X, Tsgd, Tspider, q, Ksgd, K1, K2, eta_sgd, eta_spider, eps1, eps2, delta, Csgd = 2, Cspider = 2):
    w0 = dp_sgd(winit, Tsgd, X, Ksgd, eta_sgd, eps1, delta/2)[0]
    return dp_spider(w0, Tspider, q, X, K1, K2, eta_spider, eps2, delta/2)


##################### HYPERPARAMETER TUNING ##################################
# tune eps1 and eps2 separately
# functions that take validation data and opt alg/problem parameters and grids of parameters returns alg parameters that perform best (grad norm) on hold-out portion of validation set
def sgd_tuning(winit, T, eps, X_val, C=2):
    #X_val = [np. random.uniform(-1/np.sqrt(d), 1/np.sqrt(d), size=d) for _ in range(n)]
    #b_val = generate_random_standard_normal_vectors(int(N/5), d)
    etas = [.05, 0.025, 0.005, 0.0025, 0.001, .0005]
    w_values = np.zeros((len(etas), d))
    for i in range(len(etas)):
        eta = etas[i]
        w_values[i] = dp_sgd(winit, T, X_val[0:n], n, eta, eps, delta)[0]
    best_grad_norm = np.inf
    best_index = -np.inf
    for i in range(len(etas)):
        w = w_values[i]
        if grad_norm(w, X_val[0:n]) < best_grad_norm:
            best_grad_norm = grad_norm(w, X_val[0:n])
            best_index = i
    return  etas[best_index], grad_norm(w_values[best_index], X_val[0:n])


# just return best eta and q for given initialization, T, K, etc.
def spi_tuning(winit, T, eps, X_val, C=2):
    etas = [.05, 0.025, 0.005, 0.0025, 0.001, .0005]
    qs = [1, int(T/20), int(T/10), int(T/5), T]
    w_values = np.zeros((len(etas), len(qs), d))
    for i in range(len(etas)):
        eta = etas[i] 
        for j in range(len(qs)):
            q = qs[j]
            w_values[i, j] = dp_spider(winit, T, q, X_val[0:n], n, n, eta, eps, delta)[0]
    best_grad_norm = np.inf
    best_index = -np.inf, -np.inf
    for i in range(len(etas)):
        for j in range(len(qs)):
            # print(best_grad_norm)
            w = w_values[i, j]
            if grad_norm(w, X_val[0:n]) < best_grad_norm:
                best_grad_norm = grad_norm(w, X_val[0:n])
                best_index = i, j
    return etas[best_index[0]], qs[best_index[1]], grad_norm(w_values[best_index[0], best_index[1]], X_val[0:n])


# return best eta_sgd, eta_spider and q
def ws_tuning(winit, T, eps1, eps2, X_val, Csgd =2, Cspider=2):
    etas = etas = [.05, 0.025, 0.005, 0.0025, 0.001, .0005]
    qs = [1, 5, 10, 25, 50, 100, 200]
    Tsgd = [1, 25, 50, 100, 250]
    w_values = np.zeros((len(etas), len(etas), len(qs), len(Tsgd), d))
    #w_values = 
    best_grad_norm = np.inf
    best_index = -np.inf, -np.inf, -np.inf, -np.inf
    for i in range(len(etas)):
        for j in range(len(etas)):
            for m in range(len(Tsgd)):
                Tspider = T - Tsgd[m]
                #qs = [1, int(Tspider/20), int(Tspider/10), int(Tspider/5), int(Tspider)]
                for k in range(len(qs)):
                    if qs[k] <= Tspider:
                        w_values[i, j, k, m] = warmstart(winit, X_val[0:n], Tsgd[m], Tspider, qs[k],  n, n, n, etas[i], etas[j], eps1, eps2, delta)[0]
                        w = w_values[i, j, k, m]
                        if grad_norm(w, X_val[0:n]) < best_grad_norm:
                            best_grad_norm = grad_norm(w, X_val[0:n])
                            best_index = i, j, k, m
    return etas[best_index[0]], etas[best_index[1]], qs[best_index[2]], Tsgd[best_index[3]], grad_norm(w_values[best_index[0], best_index[1], best_index[2], best_index[3]], X_val[0:n])


####Tuning###

best_eta_sgd_ = []
best_grad_norm_sgd_ = []
best_eta_spi_ = []
best_q_spi_ = []
best_grad_norm_spi_ = []
best_etasgd_ws_ = []
best_etaspi_ws_ = []
best_q_ws_ = []
best_grad_norm_ws_ = []
eps = 1 #do it for each eps
C = 2 #2 
T = n
Csgd = 2#2
Cspider = 2 #2
eps1 = eps/8 #try eps1 in eps/100, eps/20, eps/8, eps/4, eps/2
eps2 = eps - eps1
n_trials = 10
tune = False #change Tune to Try in order to tune
if tune == True: 
    for trial in range(n_trials):
        winit = random_vector_in_ball(d)
        X_val = [random_vector_in_ball(d, C=1) for _ in range(2*n)]
    # Call tuning function
        best_eta_sgd, best_grad_norm_sgd = sgd_tuning(winit, T, eps, X_val)
        best_eta_spi, best_q_spi, best_grad_norm_spi = spi_tuning(winit, T, eps, X_val)
        best_etasgd_ws, best_etaspi_ws, best_q_ws, best_Tsgd_ws, best_grad_norm_ws = ws_tuning(winit, T, eps1, eps2, X_val)

    # Append results to lists
        best_eta_sgd_.append(best_eta_sgd)
        best_grad_norm_sgd_.append(best_grad_norm_sgd)
        best_eta_spi_.append(best_eta_spi)
        best_q_spi_.append(best_q_spi)
        best_grad_norm_spi_.append(best_grad_norm_spi)
        best_etaspi_ws_.append(best_etaspi_ws)
        best_etasgd_ws_.append(best_etasgd_ws)
        best_q_ws_.append(best_q_ws)
        best_grad_norm_ws_.append(best_grad_norm_ws)
    
        print(
        f"Trial {trial+1}: SGD Best Eta: {best_eta_sgd_[trial]}, SGD Best Gradient Norm: {best_grad_norm_sgd_[trial]}.")
        print(f"Spider: best eta: {best_eta_spi_[trial]}, best q: {best_q_spi_[trial]}, best grad: {best_grad_norm_spi_[trial]}.")
        print(f"Warmstart: best etasgd: {best_etasgd_ws_[trial]}, best etaspi: {best_etaspi_ws_[trial]}, best q: {best_q_ws_[trial]}, best grad: {best_grad_norm_ws_[trial]}.")
    

# Print results
    for trial in range(n_trials):
        print(
        f"Trial {trial+1}: SGD Best Eta: {best_eta_sgd_[trial]}, SGD Best Gradient Norm: {best_grad_norm_sgd_[trial]}.")
        print(f"Spider: best eta: {best_eta_spi_[trial]}, best q: {best_q_spi_[trial]}, best grad: {best_grad_norm_spi_[trial]}.")
        print(f"Warmstart: best etasgd: {best_etasgd_ws_[trial]}, best etaspi: {best_etaspi_ws_[trial]}, best q: {best_q_ws_[trial]}, best Tsgd: {best_Tsgd_ws}, best grad: {best_grad_norm_ws_[trial]}.")
  



##################### EXPERIMENTS - store results and then plot ####################



def run_experiment(T, T_sgd, T_spider, q_spi, q_ws, K_sgd, K1, K2, eta_sgd, eta_spider, eta_ws_sgd, eta_ws_spi, eps1, eps2, delta):
    # Initialize weights
    w_init = random_vector_in_ball(d) 
    X_vectors = [random_vector_in_ball(d, C=1) for _ in range(N)]
    # Train test split:
    X_train = X_vectors[0:n]
    X_test = X_vectors[n: N+1]

    # Run dp-sgd
    w_sgd, _ , __ = dp_sgd(w_init, T, X_train, K_sgd, eta_sgd, eps, delta)
    #dp_sgd(w0, T, X, K, eta, eps, delta, C = 2) 
    sgd_train_grad_norm, sgd_test_grad_norm = grad_norm(w_sgd, X_train), grad_norm(w_sgd, X_test)

    # Run dp-spider
    w_spider, _ , __ = dp_spider(w_init, T, q_spi, X_train, K1, K2, eta_spider, eps, delta)
    spider_train_grad_norm, spider_test_grad_norm = grad_norm(w_spider, X_train), grad_norm(w_spider, X_test)

    # Run warmstart
    w_ws, _ , __ = warmstart(w_init, X_train, T_sgd, T_spider, q_ws, K_sgd, K1, K2, eta_ws_sgd, eta_ws_spi, eps1, eps2, delta)
    ws_train_grad_norm, ws_test_grad_norm = grad_norm(w_ws, X_train), grad_norm(w_ws, X_test)

    return sgd_train_grad_norm, sgd_test_grad_norm, spider_train_grad_norm, spider_test_grad_norm, ws_train_grad_norm, ws_test_grad_norm


#Change these parameters according to tuning results or as desired:
T = n
T_sgd = 25
T_spider = T - T_sgd
q_spi = 25
q_ws = 25
#or 5 or 25 or 1 
K_sgd = n
K1 = n
K2 = n
eta_sgd = 0.001
eta_spider = 0.001
eta_ws_sgd =  0.005
eta_ws_spi =  0.001
eps1 = eps/4
eps2 = eps - eps1
num_trials = 10
do_experiments = True


if do_experiments == True: 
    sgd_train_grad_norm, sgd_test_grad_norm, spider_train_grad_norm, spider_test_grad_norm, ws_train_grad_norm, ws_test_grad_norm = 0, 0, 0, 0, 0, 0
    for trial in range(num_trials): 
        sgd_train_grad_norm_, sgd_test_grad_norm_, spider_train_grad_norm_, spider_test_grad_norm_, ws_train_grad_norm_, ws_test_grad_norm_ = run_experiment(T, T_sgd, T_spider, q_spi, q_ws, K_sgd, K1, K2, eta_sgd, eta_spider, eta_ws_sgd, eta_ws_spi, eps1, eps2, delta)
        sgd_train_grad_norm += sgd_train_grad_norm_/num_trials
        sgd_test_grad_norm += sgd_test_grad_norm_/num_trials
        spider_train_grad_norm += spider_train_grad_norm_/num_trials
        spider_test_grad_norm  += spider_test_grad_norm_/num_trials
        ws_train_grad_norm += ws_train_grad_norm_/num_trials
        ws_test_grad_norm  += ws_test_grad_norm_/num_trials
    print(sgd_train_grad_norm, sgd_test_grad_norm, spider_train_grad_norm, spider_test_grad_norm, ws_train_grad_norm, ws_test_grad_norm)


 