import numpy as np
import matplotlib.pyplot as plt

from numpy import cos, sin, diag, eye, ones, zeros, dot, kron, isscalar, outer, vstack, hstack
# TODO: replace numpy Guassian with scipy
# from scipy.stats import multivariate_normal
from numpy.random import multivariate_normal
from scipy import linalg

def hx_approx(x_k):
    '''

    :param x_k:
    :return:
    '''

    len_effector_1 = 0.8  # r1
    len_effector_2 = 0.2  # r2

    # TODO: rewrite as a matrix mux
    return np.array([x_k[0]])
    
    
class CKF(object):

    def __init__(self, x_kk=None, P_kk=None, Q_approx = None, R_approx=None):

        self.x_kk = x_kk
        self.P_kk = P_kk

        self.Q_approx = Q_approx
        self.R_approx = R_approx


        dim_x = P_kk.shape[0]

        # create a set of normalized cubature points and weights given the state vector dim.
        self.num_points = (2 * dim_x)
        self.cubature_points = np.concatenate((np.eye(dim_x), -1 * np.eye(dim_x)), axis=1)
        self.cubature_weights = kron(ones((1, self.num_points)), 1/self.num_points)


    def predict(self, x_kk, P_kk):

        # TODO: Can you use fx_approx?
        #x_kk1 = fx_approx(self.x_kk)
        x_kk1 = x_kk
        P_kk1 = P_kk + self.Q_approx

        self.x_kk = x_kk1
        self.P_kk = P_kk1

    def update(self, x_kk1, P_kk1, z_k, R):

        # calculate Xi_k from x_kk1 and P_kk1

        P_kk1 = 0.5*(P_kk1 + P_kk1.T)

        # TODO: Apply chol. and QR
        U, S, Vdash = linalg.svd(P_kk1, full_matrices=True)

        S_kk1 = 0.5 * dot((U + Vdash.T), np.sqrt(diag(S)))

        # matrix to hold cubature points
        Xi = kron(ones((1, self.num_points)), x_kk1) + dot(S_kk1, self.cubature_points)

        # pass Xi_k through the measurement function to obtain z_kk, Pzz_kk,and Pxz_kk
        Zi = [hx_approx(Xi[:,i]) for i in range(self.num_points)]

        # convert a list of np.array to np.matrix
        Zi = vstack(Zi).T

        # predicted meas
        z_kk1 = vstack(Zi.sum(axis=1)/self.num_points)
           
        # W = np.diag([self.num_points]*len(self.num_points))    

        X_ = (Xi - kron(ones((1,self.num_points)), x_kk1))/(self.num_points**0.5)

        Z_ = (Zi - kron(ones((1,self.num_points)), z_kk1))/(self.num_points**0.5)

        # innovation cov
        # Generic way for cubature rule with unequal weights
        # Pzz = Z_.dot(W).dot(Z_.T)
        
        Pzz = dot(Z_, Z_.T) + R

        # cross cov
        Pxz = dot(X_, Z_.T)

        # CKF gain
        G = dot(Pxz, linalg.pinv(Pzz))

        # update
        self.x_kk = x_kk1 + dot(G, (z_k - z_kk1))

        self.P_kk = P_kk1 - G.dot(Pzz).dot(G.T)