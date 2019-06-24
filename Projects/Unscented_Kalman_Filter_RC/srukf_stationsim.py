# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:13:26 2019

@author: RC

first attempt at a square root UKF class
class built into 5 steps
-init
-Prediction SP generation
-Predictions
-Update SP generation
-Update

SR filter generally the same as regular UKF for efficiency 
but more numerically stable wrt rounding errors 
and preserving PSD covariance matrices

based on
citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.1421&rep=rep1&type=pdf
"""


import numpy as np
from math import floor
from StationSim_Wiggle import Model,Agent
from ukf_plots import plots,animations
from copy import deepcopy
import pickle
from srukf import srukf
import matplotlib.pyplot as plt
plt.style.use("dark_background")


class srukf_ss:
    def __init__(self,model_params,filter_params,srukf_params):
        """various inits for parameters,storage, indexing and more"""
        #call params
        self.model_params = model_params #stationsim parameters
        self.filter_params = filter_params # ukf parameters
    
        self.base_model = Model(self.model_params) #station sim

        """
        calculate how many agents are observed and take a random sample of that
        many agents to be observed throughout the model
        """
        self.pop_total = self.model_params["pop_total"] #number of agents
        #number of batch iterations
        self.number_of_iterations = model_params['batch_iterations']
        self.sample_rate = self.filter_params["sample_rate"]
        #how many agents being observed
        if self.filter_params["do_restrict"]==True: 
            self.sample_size= floor(self.pop_total*self.filter_params["prop"])
        else:
            self.sample_size = self.pop_total
            
        #random sample of agents to be observed
        self.index = np.sort(np.random.choice(self.model_params["pop_total"],
                                                     self.sample_size,replace=False))
        self.index2 = np.empty((2*self.index.shape[0]),dtype=int)
        self.index2[0::2] = 2*self.index
        self.index2[1::2] = (2*self.index)+1
        
        self.srukf_histories = []
   
    def F_x(self,state,**fx_args):
        """
        Transition function for each agent. where it is predicted to be.
        For station sim this is essentially gradient * v *dt assuming no collisions
        with some algebra to get it into cartesian plane.
        
        to generalise this to every agent need to find the "ideal move" for each I.E  assume no collisions
        will make class agent_ideal which essential is the same as station sim class agent but no collisions in move phase
        
        New fx here definitely works as intended.
        I.E predicts lerps from each sigma point rather than using
        sets of lerps at 1 point (the same thing 5 times)
        """
        #maybe call this once before ukf.predict() rather than 5? times. seems slow
        
        f = open("temp_pickle_model","rb")
        model = pickle.load(f)
        f.close()
            
        model.step()
        state = model.agents2state()[self.index2]
        return state
   
    def H_x(location,z):
        """
        Measurement function for agent.
        !!im guessing this is just the output from base_model.step
        """
        return z
    
    def init_srukf(self):
        state = self.base_model.agents2state(self)
        Q = np.eye(self.pop_total*2)
        R = np.eye(self.pop_total*2)
        self.srukf = srukf(srukf_params,state,self.F_x,self.H_x,Q,R)
        
        self.srukf_histories.append(self.srukf.x)
    
    
    def main(self):
        np.random.randint(8)
        self.init_srukf() 
        for _ in range(self.number_of_iterations-1):
            if _%100 ==0: #progress bar
                print(f"iterations: {_}")

            f_name = f"temp_pickle_model"
            f = open(f_name,"wb")
            pickle.dump(self.base_model,f)
            f.close()
            
            self.srukf.predict() #predict where agents will jump
            self.base_model.step() #jump stationsim agents forwards
            

            if self.base_model.time_id%self.sample_rate == 0: #update kalman filter assimilate predictions/measurements
                
                state = self.base_model.agents2state()[self.index2] #observed agents states
                self.srukf.update(z=state) #update UKF
                self.srukf_histories.append(self.srukf.x) #append histories
        
            if self.base_model.pop_finished == self.pop_total: #break condition
                break
        
        
    def data_parser(self,do_fill):
        #!! with nans and 
        sample_rate = self.sample_rate
        "partial true observations"
        a = {}
        "UKF predictions for observed above"
        b = np.vstack(self.srukf_histories)
        for k,index in enumerate(self.index):
            a[k] =  self.base_model.agents[index].history_loc
            
        max_iter = max([len(value) for value in a.values()])
        
        a2= np.zeros((max_iter,self.sample_size*2))*np.nan
        b2= np.zeros((max_iter,self.sample_size*2))*np.nan

        #!!possibly change to make NAs at end be last position
        for i in range(int(a2.shape[1]/2)):
            a3 = np.vstack(list(a.values())[i])
            a2[:a3.shape[0],(2*i):(2*i)+2] = a3
            if do_fill:
                a2[a3.shape[0]:,(2*i):(2*i)+2] = a3[-1,:]


        for j in range(int(b2.shape[0]//sample_rate)):
            b2[j*sample_rate,:] = b[j,:]
            
        "all agent observations"
        a_full = {}

        for k,agent in  enumerate(self.base_model.agents):
            a_full[k] =  agent.history_loc
            
        max_iter = max([len(value) for value in a_full.values()])
        a2_full= np.zeros((max_iter,self.pop_total*2))*np.nan

        for i in range(int(a2_full.shape[1]/2)):
            a3_full = np.vstack(list(a_full.values())[i])
            a2_full[:a3_full.shape[0],(2*i):(2*i)+2] = a3_full
            if do_fill:
                a2_full[a3_full.shape[0]:,(2*i):(2*i)+2] = a3_full[-1,:]

        return a2,b2,a2_full



if __name__ == "__main__":
    """
            a - alpha scaling parameter determining how far apart sigma points are spread. Typically between 1e-4 and 1
            b - beta scaling paramater incorporates prior knowledge of distribution of state space. b=2 optimal for gaussians
            k - kappa scaling parameter typically 0 for state space estimates and 3-dim(x) for parameter estimation
            init_x- initial state space
    """

    model_params = {
                    'width': 200,
                    'height': 100,
                    'pop_total': 3,
                    'entrances': 3,
                    'entrance_space': 2,
                    'entrance_speed': 1,
                    'exits': 2,
                    'exit_space': 1,
                    'speed_min': .1,
                    'speed_desire_mean': 1,
                    'speed_desire_std': 1,
                    'separation': 4,
                    'wiggle': 1,
                    'batch_iterations': 10000,
                    'do_save': True,
                    'do_plot': False,
                    'do_ani': False
                    }
        
    filter_params = {         
                    "Sensor_Noise":  1, # how reliable are measurements H_x. lower value implies more reliable
                    "Process_Noise": 1, #how reliable is prediction F_x lower value implies more reliable
                    'sample_rate': 1,   #how often to update kalman filter. higher number gives smoother (maybe oversmoothed) predictions
                    "do_restrict": True, #"restrict to a proportion prop of the agents being observed"
                    "do_animate": False,#"do animations of agent/wiggle aggregates"
                    "do_wiggle_animate": False,
                    "do_density_animate":False,
                    "do_pair_animate":False,
                    "prop": 1,#proportion of agents observed. 1 is all <1/pop_total is none
                    "heatmap_rate": 2,# "after how many updates to record a frame"
                    "bin_size":10,
                    "do_batch":False
                    }
    
    srukf_params = {
            "a":0.1,#alpha between 1 and 1e-4 typically
            "b":2,#beta set to 2 for gaussian 
            "k":0,#kappa usually 0 for state estimation and 3-dim(state) for parameters
            }
    
    
    sr = srukf_ss(model_params,filter_params,srukf_params)
    sr.main()
    a,b,a_full = sr.data_parser(True)
    plots.diagnostic_plots(sr)
    res = a-b