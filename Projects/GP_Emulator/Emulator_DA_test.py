#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:11:42 2019

@author: minhlkieu
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import gp_emulator
os.chdir("/Users/geomik/Dropbox/Minh_UoL/DA/Emulators")
from GaussianProcess import GaussianProcess

def rmse(dataset1, dataset2, ignore=None):

   # Make sure that the provided data sets are numpy ndarrays, if not
   # convert them and flatten te data sets for analysis
   if type(dataset1).__module__ != np.__name__:
      d1 = np.asarray(dataset1).flatten()
   else:
      d1 = dataset1.flatten()

   if type(dataset2).__module__ != np.__name__:
      d2 = np.asarray(dataset2).flatten()
   else:
      d2 = dataset2.flatten()

   # Make sure that the provided data sets are the same size
   if d1.size != d2.size:
      raise ValueError('Provided datasets must have the same size/shape')

   # Check if the provided data sets are identical, and if so, return 0
   # for the root-mean-squared error
   if np.array_equal(d1, d2):
      return 0

   # If specified, remove the values to ignore from the analysis and compute
   # the element-wise difference between the data sets
   if ignore is not None:
      index = np.intersect1d(np.where(d1 != ignore)[0], 
                                np.where(d2 != ignore)[0])
      error = d1[index].astype(np.float64) - d2[index].astype(np.float64)
   else:
      error = d1.astype(np.float64) - d2.astype(np.float64)

   # Compute the mean-squared error
   meanSquaredError = np.sum(error**2) / error.size

   # Return the root of the mean-square error
   return np.sqrt(meanSquaredError)

#load simulated data
data = np.genfromtxt('data_emulator.csv', delimiter=',')

#process data to make predictions
x = data[1:-1,:4]
y=data[2:,4] 

x_train = x[:600,:]
y_train = y[:600]

x_test = x[600:]
y_test = y[600:]

#model 1: gp_emulator
gp1 = gp_emulator.GaussianProcess(x_train,y_train)
gp1.learn_hyperparameters(n_tries=25)
y_pred, y_unc, _ = gp1.predict(x_test,do_unc=True, do_deriv=False)
plt.plot(y_test,y_pred, 'ro', lw=2., label="gp_emulator")
RMSE1 = rmse(y_pred,y_test)
#gp_emulator: RMSE = 0.016

#model 2: mogp_emulator
gp2 = GaussianProcess(x_train,y_train)
gp2.learn_hyperparameters(n_tries=25)
y_pred, y_unc, _ = gp2.predict(x_test,do_unc=True, do_deriv=False)
x=np.arange(0,1.2,0.1)
y=x
plt.plot(y_test,y_pred, 'k.', lw=2., label="mogp_emulator")
plt.plot(x,y,'-',lw=2., label="y=x")
plt.legend()
plt.xlabel('Gaussian Process Predicted X-axis force')
plt.ylabel('Synthetic X-axis force')
RMSE2 = rmse(y_pred,y_test)
# mogp_emulator:  RMSE = 0.04



