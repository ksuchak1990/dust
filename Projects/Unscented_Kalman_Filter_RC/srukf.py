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

SR filter generally the same as regular filters for efficiency 
but more numerically stable wrt rounding errors 
and preserving PSD covariance matrices

based on
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6179312
"""



import numpy as np
from choldate import cholupdate,choldowndate
#"pip install git+git://github.com/jcrudy/choldate.git"
"""
for cholesky update/downdate.
see cholupdate matlab equivalent not converted to numpy from LAPACK yet (probably never will be)
"""

class SRUKF:
    
    def __init__(self,srukf_params):
        """this needs to:
            - init x0, S_0,S_v and S_n     
        """
        
        #init initial state
        self.x = srukf_params["init_x"]  #!!initialise some positions and covariances
        self.n = self.x.shape[0] #state space dimension

        self.P = np.array([[1,0.5],[0.5,1]])
        #self.P = np.linalg.cholesky(self.x)
        self.S = np.linalg.cholesky(self.P)
        self.Q = np.eye(self.n)
        self.R = np.eye(self.n)
        
        #init further parameters based on a through el
        self.lam = srukf_params["a"]**2*(self.n+srukf_params["k"]) - self.n #lambda paramter calculated viar
        self.g = np.sqrt(self.n+self.lam) #gamma parameter

        
        #init weights based on paramters a through el
        self.wm = np.zeros(((2*self.n)+1))
        self.wm[0] = self.lam/(self.n+self.lam)
        self.wc = np.zeros(((2*self.n)+1,1))
        self.wc[0] = self.wm[0] + (1-srukf_params["a"]**2+srukf_params["b"])

        other_weights =  1/(2*(self.n+self.lam))
        for i in range(1,(2*self.n)+1):        
            self.wc[i] = other_weights
            self.wm[i] = other_weights  
            
        self.sqrtQ = np.linalg.cholesky(self.Q)
        self.sqrtR = np.linalg.cholesky(self.R)
         
        self.xhat = None
        self.Sxx = None

        self.xs = []
        self.Ss = []

    def Sigmas(self,mean,S):
        "sigma point calculations based on current x and P"
        "double loop here is probably convoluted but easiest way to maintain structure"
     
        #sigmas build around mean and confidence in each dimension   
        
        sigmas = np.zeros((self.n,(2*self.n)+1))
        sigmas[:,0] = mean
        for i in range(self.n):
            sigmas[:,(i+1)] = mean + self.g*S[:,i]
            
        for i in range(self.n):
            sigmas[:,self.n+i+1] = mean - self.g*S[:,i]
        return sigmas
    

    def Fx(self,sigma):
        """
        (non-)linear transition function taking current state space and predicting 
        innovation
        !! make this user defined?
        """
        sigma+= np.array([1,1])
        return sigma
  

    def predict(self):
        """
        predict transitions
        calculate estimates of new mean in the usual way
        calculate predicted covariance using qr decomposition
        """
        #calculate NL projection of sigmas
        self.sigmas = self.Sigmas(self.x,self.S)
        nl_sigmas = np.zeros((self.sigmas.shape))
        
        for i in range(self.sigmas.shape[1]):
            nl_sigmas[:,i] = self.Fx(self.sigmas[:,i])
        


        
        wnl_sigmas = nl_sigmas.copy() #weighted
        for i in range(nl_sigmas.shape[1]):
            wnl_sigmas[:,i]*=self.wm[i]

        xhat = np.sum(wnl_sigmas,axis=1)        
        Pxx =np.vstack([np.sqrt(self.wc[1][0])*(nl_sigmas.transpose()-xhat),self.sqrtQ])[1:,:]
        Sxx = np.linalg.qr(Pxx)[1]
        #up/downdating as necessary depending on sign of first covariance weight
        u =  np.sqrt(np.sqrt(np.abs(self.wc[0][0])))*(nl_sigmas[:,0]-xhat)
        if self.wc[0][0]>0:    
            cholupdate(Sxx,u)
        if self.wc[0][0]<0:    
            choldowndate(Sxx,u)   
        self.Sxx= Sxx #update xx UT
        self.xhat = xhat #update xhat

        
    
    
    def Hx(self,x):
        """
        measurement function converting state space into same dimensions 
        as measurements to assess residuals
        """

        return x
    
    
    def update(self,z):     
        """
        calculate residuals
        calculate kalman gain
        merge prediction with measurements in the Kalman style using
        cholesky update function
        """
        
        
        sigmas = self.Sigmas(self.xhat,self.Sxx) #update interim sigmas
        nl_sigmas = np.empty((sigmas.shape))
        for i in range(sigmas.shape[1]):
            nl_sigmas[:,i] = self.Hx(sigmas[:,i])
            
        wnl_sigmas = nl_sigmas.copy() #weighted
        for i in range(nl_sigmas.shape[1]):
            wnl_sigmas[:,i]*=self.wm[i]

        yhat = np.sum(wnl_sigmas,axis=1)
        Pyy =np.vstack([np.sqrt(self.wc[1])*(nl_sigmas.transpose()-yhat),self.sqrtR])[1:,:]
        Syy = np.linalg.qr(Pyy)[1]
        u =  np.sqrt(np.sqrt(np.abs(self.wc[0])))*(nl_sigmas[:,0]-yhat)
        cholupdate(Syy,u)
        if self.wc[0][0]>0:    
            cholupdate(Syy,u)
        if self.wc[0][0]<0:    
            choldowndate(Syy,u)   
        self.Syy= Syy
        
        Pxy= np.zeros((self.n,self.n))
        for i,wc in enumerate(self.wc):
            Pxy += wc*np.outer((sigmas[:,i].transpose()-self.xhat).transpose(),(nl_sigmas[:,i].transpose()-yhat))
            
        
        K = np.matmul(Pxy,np.linalg.inv(np.matmul(Syy,Syy.T)))
        U = np.matmul(K,Syy)
        
        #update xhat
        self.xhat += np.matmul(K,(z-yhat))
        self.x=self.xhat
        
        "U is a matrix (not a vector) and so requires dim(U) updates of Sxx using each column of U as a 1 step cholup/down/date as if it were a vector"
        for j in range(U.shape[0]):
            choldowndate(self.Sxx,U[:,j])
        self.S = Syy
        
        self.Ss.append(self.S)
        self.xs.append(self.xhat)
        
        
    def batch(self):
        """
        batch
        """
        return


ys = np.zeros((10,2))
for i in range(ys.shape[0]):
    ys[i,:] = np.array([i,i]) + 0.5*np.random.normal(size=2)


if __name__ == "__main__":
    srukf_params = {
            "a":0.1,#alpha between 1 and 1e-4 typically
            "b":2,#beta set to 2 for gaussian 
            "k":0,#kappa usually 0 for state estimation and 3-dim(state) for parameters
            "init_x":ys[0,:]
            }
    
    xs = []
    xs.append(srukf_params["init_x"])
    srukf = SRUKF(srukf_params)

    for j in range(1,10):
       
        
        srukf.predict()
        y = ys[j,:]
        srukf.update(y)
        xs.append(srukf.xhat)
        
        
        
    res = np.array(xs)-np.array(ys)
    
    import matplotlib.pyplot as plt
    xs=np.array(xs)
    ys=np.array(ys)
    
    f = plt.figure()
    plt.plot(xs[:,0],xs[:,1],label = "kf")
    plt.plot(ys[:,0],ys[:,1],label="truth")
    plt.legend()
