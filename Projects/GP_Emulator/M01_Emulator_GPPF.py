#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:57:35 2019

@author: geomik
"""
import os
os.chdir("/Users/geomik/Documents/GitHub/dust/Projects/GP_Emulator")

import numpy as np
import matplotlib.pyplot as plt
import pickle
from ParticleFilter_GP import ParticleFilter
from copy import deepcopy
import pandas as pd

#Step 1: Load the Emulator model


file = open("gp_emulator_1.sav",'rb')
Gpm = pickle.load(file)
file.close()

#Step 2: Apply Particle Filter

def apply_PF(model0,new_data):
    filter_params = {
        'number_of_particles': 10000,
        'std': 0.0005,
        'resample_window': 1,
        'do_copies': True,
        'do_save': True
        }
    model = deepcopy(model0)
    pf = ParticleFilter(model, **filter_params)

    for niter in range(len(new_data)):
        model.step()
        true_state = GroundTruth0[niter,:]
        measured_state = true_state #+ np.random.normal(0, 0., true_state.shape)  #to add noise in the measured_state if needed
        pf.step(measured_state, true_state)
    x = np.array([bus.trajectory for bus in pf.models[np.argmax(pf.weights)].buses]).T    
    return x
