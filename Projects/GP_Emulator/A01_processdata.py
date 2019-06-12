# -*- coding: utf-8 -*-
"""
Created on Thu May  2 09:49:27 2019

@author: geomlk
"""


import os
import numpy as np

#get a list of data files
#os.chdir("/Users/geomik/Dropbox/Minh_UoL/DA/Emulators")
#prefixed = [filename for filename in os.listdir('.') if filename.startswith("r-individual")]
prefixed = ["raw_data_new.csv"]

#collect the data from each file
data = [0,0,0,0,0]
for f in prefixed:
    df = np.genfromtxt(f, delimiter=',', skip_header=1)
    a = np.array(df[:,1].astype(int), dtype=np.str)
    b = np.array(df[:,7].astype(int), dtype=np.str)
    c = np.array(np.zeros(np.size(df[:,1])).astype(int), dtype=np.str)
    d = np.char.add(a,c).astype(str)
    d2 = np.char.add(d,c).astype(str)
    d3 = np.char.add(d2,c).astype(str)
    e = np.char.add(d3,b).astype(int)
    df[:,1]=e
    df = df[~np.isnan(df[:,4]),:6]
    
    #now loop through time
    for t in range(0,2000*1000,60*1000):
        df_interval = df[(df[:,0]>=t) & (df[:,0]<t+100*1000),:]
        
        #find out how many people walk through the entrance door (door coordinate = 10m)
        count_doorin = np.size(np.unique(df_interval[(df_interval[:,2]>= 9.5)& (df_interval[:,2]< 10.5),1]))
        #print("walk in = ",df_doorin)
        #find out how many people walk through the exit door (door coordinate = 90m)
        count_doorout = np.size(np.unique(df_interval[(df_interval[:,2]>= 89.5)& (df_interval[:,2]< 90.5),1]))
        #print("walk out= ",df_doorout)
        #Number of active passengers
        count_pes=np.size(np.unique(df_interval[:,1]))
        #average speed of pedestrian
        mean_speed = np.mean(df_interval[:,4])
        #average x-force
        mean_xforce = np.mean(df_interval[:,5])
        #collect all the result
        data = np.vstack((data,[count_doorin,count_doorout,count_pes,mean_speed,mean_xforce]))
        #print(t)
    #data = data[data[:,1]>200,:]
    #np.savetxt("data_emulator.csv", data,fmt='%10.3f', delimiter=",")
    np.savetxt("processed_data_new.csv", data,fmt='%10.3f', delimiter=",")
         
        
        
        

