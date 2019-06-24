

import numpy as np
from choldate import cholupdate,choldowndate
#"pip install git+git://github.com/jcrudy/choldate.git"
"""
for cholesky update/downdate.
see cholupdate matlab equivalent not converted into numpy.linalg from LAPACK yet (probably never will be)
but I found this nice package in the meantime.
"""

class srukf:
    
    def __init__(self,srukf_params,init_x,Fx,Hx,Q,R):
        """this needs to:
            - init x0, S_0,S_v and S_n     
        """
        
        #init initial state
        self.x = init_x #!!initialise some positions and covariances
        self.n = self.x.shape[0] #state space dimension

        self.P = np.eye(self.n)
        #self.P = np.linalg.cholesky(self.x)
        self.S = np.linalg.cholesky(self.P)
        self.Fx=Fx
        self.Hx =Hx
        
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
            
        self.Q=Q
        self.R=R
        self.sqrtQ = np.linalg.cholesky(self.Q)
        self.sqrtR = np.linalg.cholesky(self.R)
         
        self.xhat = None
        self.Sxx = None

        self.xs = []
        self.Ss = []

    def Sigmas(self,mean,S):
        """sigma point calculations based on current mean x and  UT (upper triangular) 
        decomposition S of covariance P"""
        
        "double loop here is probably convoluted but easiest way to maintain structure"
     
        
        sigmas = np.zeros((self.n,(2*self.n)+1))
        sigmas[:,0] = mean
        for i in range(self.n):
            sigmas[:,(i+1)] = mean + self.g*S[:,i]
            
        for i in range(self.n):
            sigmas[:,self.n+i+1] = mean - self.g*S[:,i]
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
        self.sigmas = self.Sigmas(self.x,self.S)
        nl_sigmas = np.zeros((self.sigmas.shape))
        
        for i in range(self.sigmas.shape[1]):
            nl_sigmas[:,i] = self.Fx(self.sigmas[:,i],**fx_args)
        


        
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
        batch function maybe build later
        """
        return

