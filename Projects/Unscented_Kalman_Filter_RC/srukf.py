

import numpy as np
from choldate import cholupdate,choldowndate
#"pip install git+git://github.com/jcrudy/choldate.git"
"""
for cholesky update/downdate.
see cholupdate matlab equivalent not converted into numpy.linalg from LAPACK yet (probably never will be)
but I found this nice package in the meantime.
"""

class srukf:
    
    def __init__(self,srukf_params,init_x,fx,hx,Q,R):
        """this needs to:
            - init x0, S_0,S_v and S_n     
        """
        
        #init initial state
        self.x = init_x #!!initialise some positions and covariances
        self.n = self.x.shape[0] #state space dimension

        self.P = np.eye(self.n)
        #self.P = np.linalg.cholesky(self.x)
        self.S = np.linalg.cholesky(self.P)
        self.fx=fx
        self.hx=hx
        
        #init further parameters based on a through el
        self.lam = srukf_params["a"]**2*(self.n+srukf_params["k"]) - self.n #lambda paramter calculated viar
        self.g = np.sqrt(self.n+self.lam) #gamma parameter

        
        #init weights based on paramters a through el
        main_weight =  1/(2*(self.n+self.lam))
        self.wm = np.ones(((2*self.n)+1))*main_weight
        self.wm[0] *= 2*self.lam
        self.wc = self.wm.copy()
        self.wc[0] += (1-srukf_params["a"]**2+srukf_params["b"])

    
            
        self.Q=Q
        self.R=R
        self.sqrtQ = np.linalg.cholesky(self.Q)
        self.sqrtR = np.linalg.cholesky(self.R)
         
        self.xs = []
        self.Ss = []

    def Sigmas(self,mean,S):
        """sigma point calculations based on current mean x and  UT (upper triangular) 
        decomposition S of covariance P"""
        
     
        sigmas = np.ones((self.n,(2*self.n)+1)).T*mean
        sigmas=sigmas.T
        sigmas[:,1:self.n+1] += self.g*S #'upper' confidence sigmas
        sigmas[:,self.n+1:] -= self.g*S #'lower' confidence sigmas
        return sigmas 

    def predict(self,**fx_args):
        """
        - calculate sigmas using prior mean and UT element of covariance S
        - predict interim sigmas X for next timestep using transition function Fx
        - predict unscented mean for next timestep
        - calculate interim S using concatenation of all but first column of Xs
            and square root of process noise
        - cholesky update to nudge on unstable 0th row
        - calculate futher interim sigmas using interim S and unscented mean
        """
        #calculate NL projection of sigmas
        sigmas = self.Sigmas(self.x,self.S) #calculate current sigmas using state x and UT element S
        nl_sigmas = np.apply_along_axis(self.fx,0,sigmas)
        wnl_sigmas = nl_sigmas*self.wm
            
        xhat = np.sum(wnl_sigmas,axis=1)#unscented mean for predicitons
        
        
        """
        a qr decompositions of the compound matrix contanining the weighted predicted sigmas points
        and the matrix square root of the additive process noise.
        """
        Pxx =np.vstack([np.sqrt(self.wc[1])*(nl_sigmas[:,1:].T-xhat),self.sqrtQ])
        Sxx = np.linalg.qr(Pxx,mode="r")
        "up/downdating as necessary depending on sign of first covariance weight"
        u =  np.sqrt(np.sqrt(np.abs(self.wc[0])))*(nl_sigmas[:,0]-xhat)
        if self.wc[0]>0:    
            cholupdate(Sxx,u)
        if self.wc[0]<0:    
            choldowndate(Sxx,u)   
            
        self.S = Sxx #update Sxx
        self.x = xhat #update xhat
    
    def update(self,z):     
        """
        Does numerous things in the following order
        - calculate interim sigmas using Sxx and unscented mean estimate
        - calculate measurement sigmas Y = h(X)
        - calculate unscented mean of Ys
        - calculate qr decomposition of concatenated columns of all but first Y scaled 
            by w1c and square root of sensor noise to calculate interim S
        - cholesky update to nudge interim S on potentially unstable 0th 
            column of Y
        - calculate sum of scaled cross covariances between Ys and Xs Pxy
        - calculate kalman gain
        - calculate x update
        - calculate S update
        """
        sigmas = self.Sigmas(self.x,self.S) #update using Sxx and unscented mean
        nl_sigmas = np.apply_along_axis(self.hx,0,sigmas)
        wnl_sigmas = nl_sigmas*self.wm

        yhat = np.sum(wnl_sigmas,axis=1) #unscented mean for measurements
        """
        a qr decompositions of the compound matrix contanining the weighted measurement sigmas points
        and the matrix square root of the additive sensor noise.
        """
        Pyy =np.vstack([np.sqrt(self.wc[1])*(nl_sigmas[:,1:].T-yhat),self.sqrtR]).T
        Syy = np.linalg.qr(Pyy,mode="r")
        u =  np.sqrt(np.sqrt(np.abs(self.wc[0])))*(nl_sigmas[:,0]-yhat)
        if self.wc[0]>0:    
            cholupdate(Syy,u)
        if self.wc[0]<0:    
            choldowndate(Syy,u)   
        

        Pxy =  self.wc[0]*np.outer((sigmas[:,0].T-self.x),(nl_sigmas[:,0].transpose()-yhat))
        for i in range(1,len(self.wc)):
            Pxy += self.wc[i]*np.outer((sigmas[:,i].T-self.x),(nl_sigmas[:,i].transpose()-yhat))
            
        
        K = np.matmul(Pxy,np.linalg.inv(np.matmul(Syy,Syy.T)))
        #K= np.linalg.lstsq(np.matmul(Syy,Syy.T),Pxy.T)[0].T
        
        U = np.matmul(K,Syy)
        
        #update xhat
        self.x += np.matmul(K,(z-yhat))
        
        "U is a matrix (not a vector) and so requires dim(U) updates of Sxx using each column of U as a 1 step cholup/down/date as if it were a vector"
        Sxx = self.S
        for j in range(U.shape[1]):
            choldowndate(Sxx,U[:,j])
        
        self.S = Sxx
        self.Ss.append(self.S)
        self.xs.append(self.x)
        
        
        
    def batch(self):
        """
        batch function maybe build later
        """
        return

