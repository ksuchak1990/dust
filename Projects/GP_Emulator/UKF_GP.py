#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:16:08 2019

@author: minhlkieu
"""

from __future__ import (absolute_import, division)

from copy import deepcopy
#from math import log, exp, sqrt
#mport sys
import numpy as np
from numpy import eye, zeros, dot, isscalar, outer
from numpy.linalg import norm
#from math import atan2
import math 
from scipy.linalg import cholesky,expm,block_diag
#from filterpy.kalman import unscented_transform
#from __future__ import division
import matplotlib.pyplot as plt

#Now define the Van der Merwe's algorithm to choose sigma points
class MerweScaledSigmaPoints(object):
    """
    Generates sigma points and weights according to Van der Merwe
    """
    def __init__(self, n, alpha, beta, kappa):        
        self.n = n   #number of dimension in the data      
        self.alpha = alpha #varies from 0 to 1
        self.beta = beta
        self.kappa = kappa        
        self.sqrt = cholesky        
        self.subtract = np.subtract        
        self._compute_weights()

    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2*self.n + 1  #this is a rule-of-thumb

    def sigma_points(self, x, P):    #x is the starting position, P is the starting covariance error matrix

        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, np.size(x)))
        n = self.n
        #rescale x and P for multiplication 
        if np.isscalar(x):
            x = np.asarray([x])
        if  np.isscalar(P):
            P = np.eye(n)*P
        else:
            P = np.atleast_2d(P)
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = self.sqrt((lambda_ + n)*P)
        sigmas = np.zeros((2*n+1, n))
        sigmas[0] = x
        for k in range(n):            
            sigmas[k+1]   = self.subtract(x, -U[k])
            sigmas[n+k+1] = self.subtract(x, U[k])
        return sigmas

    def _compute_weights(self):
        """ Computes the weights for the scaled unscented Kalman filter."""
        n = self.n
        lambda_ = self.alpha**2 * (n +self.kappa) - n
        c = .5 / (n + lambda_)
        self.Wc = np.full(2*n + 1, c)
        self.Wm = np.full(2*n + 1, c)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)

def Q_discrete_white_noise(dim, dt=1., var=1., block_size=1, order_by_dim=True):
    """
    Returns the Q matrix 
    """
    if not (dim == 2 or dim == 3 or dim == 4):
        raise ValueError("dim must be between 2 and 4")
    if dim == 2:
        Q = [[.25*dt**4, .5*dt**3],
             [ .5*dt**3,    dt**2]]
    elif dim == 3:
        Q = [[.25*dt**4, .5*dt**3, .5*dt**2],
             [ .5*dt**3,    dt**2,       dt],
             [ .5*dt**2,       dt,        1]]
    else:
        Q = [[(dt**6)/36, (dt**5)/12, (dt**4)/6, (dt**3)/6],
             [(dt**5)/12, (dt**4)/4,  (dt**3)/2, (dt**2)/2],
             [(dt**4)/6,  (dt**3)/2,   dt**2,     dt],
             [(dt**3)/6,  (dt**2)/2 ,  dt,        1.]]
    if order_by_dim:
        return block_diag(*[Q]*block_size) * var
    return order_by_derivative(array(Q), dim, block_size) * var

def unscented_transform(sigmas, Wm, Wc,Q):
    """
    Computes unscented transform of a set of sigma points and weights.
    returns the mean and covariance in a tuple.    
    """
    kmax, n = sigmas.shape    
    # new mean is just the sum of the sigmas * weight
    x = np.dot(Wm, sigmas)    # dot = \Sigma^n_1 (W[k]*Xi[k])    
    # new covariance is the sum of the outer product of the residuals
    # times the weights
    y = sigmas - x[np.newaxis, :]
    P = np.dot(y.T, np.dot(np.diag(Wc), y))
    P += Q
    return (x, P)

class UnscentedKalmanFilter(object):    
    """
    This is the main class of UKF
    """
    def __init__(self, dim_x, dim_z, dt, hx, fx, points):
        """
        Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.
        """
        self.x = zeros(dim_x)
        self.P = eye(dim_x)
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        self.Q = eye(dim_x)
        self.R = eye(dim_z)
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.points_fn = points  #call Van Der Merwe Sigma Points algorithm
        self._dt = dt
        self._num_sigmas = points.num_sigmas()
        self.hx = hx
        self.fx = fx        
        self.msqrt = cholesky        
        # weights for the means and covariances.
        self.Wm, self.Wc = points.Wm, points.Wc        
        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update
        self.sigmas_f = zeros((self._num_sigmas, self._dim_x))
        self.sigmas_h = zeros((self._num_sigmas, self._dim_z))
        self.K = np.zeros((dim_x, dim_z))    # Kalman gain
        self.y = np.zeros((dim_z))           # residual
        self.z = np.array([[None]*dim_z]).T  # measurement
        self.S = np.zeros((dim_z, dim_z))    # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))   # inverse system uncertainty
        self.inv = np.linalg.inv
        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
    def predict(self, **fx_args):
        """
        Performs the predict step of the UKF. On return, self.x and
        self.P contain the predicted state (x) and covariance (P). '
        Important: this MUST be called before update() is called for the first
        time.        
        """
        dt = self._dt
        UT = unscented_transform
        fx=self.fx

        # calculate sigma points for given mean and covariance
        self.compute_process_sigmas(dt, fx, **fx_args)
        #and pass sigmas through the unscented transform to compute prior
        self.x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q)
        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z, **hx_args):
        """
        Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.
        **hx_args : keyword argument
            arguments to be passed into h(x) after x -> h(x, **hx_args)
        """
        if z is None:
            self.z = np.array([[None]*self._dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return      
        hx = self.hx
        UT = unscented_transform           

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(hx(s, **hx_args))
        self.sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = UT(self.sigmas_h, self.Wm, self.Wc, self.R)
        self.SI = self.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)


        self.K = dot(Pxz, self.SI)        # Kalman gain
        self.y = np.subtract(z, zp)   # residual

        # update Gaussian state estimate (x, P)
        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot(self.K, dot(self.S, self.K.T))

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        """
        Compute cross variance of the state `x` and measurement `z`.
        """
        Pxz = zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        N = sigmas_f.shape[0]
        for i in range(N):
            dx = np.subtract(sigmas_f[i], x)
            dz = np.subtract(sigmas_h[i], z)
            Pxz += self.Wc[i] * outer(dx, dz)
        return Pxz
    def compute_process_sigmas(self, dt,fx, **fx_args):
        """
        computes the values of sigmas_f. Normally a user would not call
        this, but it is useful if you need to call update more than once
        between calls to predict (to update for multiple simultaneous
        measurements), so the sigmas correctly reflect the updated state
        x, P.
        """

        fx = self.fx

        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.P)
        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = fx(s, dt, **fx_args)
			
#define function fx and hx
def f_x(x, GP):
    """ state transition function is a pretrained Gaussian Process"""
    y_pred, y_unc, _ = GP.predict(np.array([x]))
		    
    return y_pred, y_unc
    

#measurement function
def h_x(x):
    return x
	
dt=1
# Step 1: Find sigma points, follow Van Der Merwe
points = MerweScaledSigmaPoints(n=4,alpha=.1,beta=2,kappa=1)
kf = UnscentedKalmanFilter(4,2, dt, fx=f_x, hx=h_x, points=points)
kf.Q[0:2,0:2]=Q_discrete_white_noise(2,dt=dt,var=0.1)
kf.Q[2:4,2:4]=Q_discrete_white_noise(2,dt=dt,var=0.1)
#kf.R = np.diag([range_std**2,elevation_angle_std**2])
kf.x = np.array([0,90,1100,0])
kf.P = np.diag([300**2,30**2,150**2,3**2])

np.random.seed(200)
pos=(0,0)

time = np.arange(0,360+dt,dt)
xs, ys = [],[]
for t in time:
    if t >= 60:
        ac.vel[1]=300/60 #300m/min increase in elevation    
    ac.update(dt) #move the aircraft
    r = radar.noisy_reading(ac.pos)
    ys.append(ac.pos[1])
    kf.predict()
    kf.update([r[0],r[1]])
    xs.append(kf.x)
xs = np.asarray(xs)

#now plotting
plt.plot(time, xs[:,2], label='filter', )
plt.plot(time, ys, label='Aircraft', lw=2, ls='--', c='k')
plt.xlabel('time(sec)')
plt.ylabel('altitude')
plt.legend(loc=4)


		