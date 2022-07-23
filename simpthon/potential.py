"""
potential.py : class implementing potential

parameters
----------
x : array_like
    position [x,y,z].

returns
-------
pot : scalar
    potential.

	
|		
"""

#History: 16-12-2020 -    written                              - Hasanuddin
#         08-11-2021 -  __str__ added                          - Hasanuddin 
#         01-07-2022 -  potentials class, cluster_pot added    - Hasanuddin

from abc import ABC, abstractmethod
import numpy as np
from math import sqrt,log

class potential(ABC):
    @abstractmethod
    
	
    def pot(x):
        r""" return scalar potential :math:`\Phi` at given position :math:`x`. """
        pass

    def acc(x):
        r""" return vector acceleration :math:`a` at given position :math:`x`. """
        pass

    def tid(x):
        r""" return tensor tidal force :math:`T` at given position :math:`x`. 
        
        |
        """
        pass
		
    def mass(x):
        r""" return scalar mass :math:`m` inside radius :math:`r`. """
        pass

class osilator(potential):
    r""" Harmonik oscilator potential:
    
    .. math:: \Phi (x) = \frac{1}{2} k x^2. 
    
    """
	
    def __init__(self,k,m):
        self.k = k 
        self.m = m
    def __str__(self):
	    return "potential:osilator(k="+str(self.k)+",m="+str(self.m)+")"		
    def pot(self,x):
        return 0.5*self.k*x[0]*x[0]
    def acc(self,x) :
        return -self.k*x/self.m	
    def tid(self,x):
        return np.array([[-self.k/self.m,0,0],[0,0,0],[0,0,0]])
    def mass(self,x):
        a = self.acc(x)
        return x.dot(x)*sqrt(a.dot(a))
		
    r"""
	
    
    \n \n
    """
    
class pointmass(potential):
    r""" Point mass potential: 
    
    .. math:: \Phi (x) = -\frac{GM}{r}.
    
    """
    
    def __init__(self,GM):
        self.GM = GM
    def __str__(self):
	    return "potential:pointmass(GM="+str(self.GM)+")"
    def pot(self,x):
        return -self.GM/sqrt(x.dot(x))
    def acc(self,x):
        return self.pot(x)*x/x.dot(x)
    def tid(self,x):
        rq = x.dot(x)
        r = sqrt(rq)
        k = self.GM/rq/r
        xx = x[0]*x[0]/rq
        xy = x[0]*x[1]/rq
        xz = x[0]*x[2]/rq
        yy = x[1]*x[1]/rq
        yz = x[1]*x[2]/rq
        zz = x[2]*x[2]/rq
        return k*np.array([[3*xx-1, 3*xy, 3*xz],[3*xy, 3*yy-1, 3*yz],[3*xz, 3*yz, 3*zz-1]])		
    def mass(self,x):
        return self.GM

class plummer(potential):
    r"""Plummer potential :
    
    .. math:: \Phi (x) = -\frac{GM}{\sqrt{r^2 + b^2}}, 
    
    with :math:`b` scale factor.
	
    """
    
    def __init__(self,GM,b):
        self.GM = GM
        self.b  = b
    def __str__(self):
        return "potential:plummer(GM="+str(self.GM)+",b="+str(self.b)+")"
    def pot(self,x):
        return -self.GM/sqrt(x.dot(x)+self.b**2)   
    def acc(self,x):
        return self.pot(x)*x/(x.dot(x)+self.b**2)
    def tid(self,x):
        #need to be implemented
        pass
    def mass(self,x):
        #need implementation
        pass

class jaffe(potential):
    r""" Jaffe potential:

    .. math:: \Phi (x) = -\frac{GM}{b} \frac{\ln (1+r/b)}{(r/b)^2},
    
    with :math:`b` scale factor. 	
    """
	
    def __init__(self,GM,b):
        self.GM = GM
        self.b  = b
    def __str__(self):
        return "potential:jaffe(GM="+str(self.GM)+",b="+str(self.b)+")"
    def pot(self,x):
        rq = x.dot(x)/self.b/self.b
        r = sqrt(rq)
        return -self.GM*log(1+r)/rq/self.b
    def acc(self,x):
        # need implementation
        pass
    def tid(self,x):
        #need to be implemented
        pass
    def mass(self,x):
        #need implementation
        pass
    
        
class potentials(potential):
    r""" list of potential:

    .. math:: \Phi (x) = \Sigma_i \Phi_i (x).
     	
    """
	
    def __init__(self,pots=[]):
        self.__pots = pots
        
    def __str__(self):
        return str([str(pi) for pi in self.__pots])
    def pot(self,x):
        p = 0
        for pi in self.__pots:
            p = p + pi.pot(x)
        return p 
        
    def acc(self,x):
        a = np.zeros(len(x))
        for pi in self.__pots:
            a = a + pi.acc(x)
        return a 
        
    def tid(self,x):
        #need to be implemented
        pass
    def mass(self,x):
        #need implementation
        pass
        
    def pots(self):
        return self.__pots
        

class cluster_potential(potential):
    def __init__(self, pot, X=np.array([0,0,0])):
        self.__pot = pot
        self.__X = X 
    def __str__(self):
	    return "cluster with"+str(self.__pot)
    def pot(self,x):
        return self.__pot.pot(x-self.__X)
    def acc(self,x):
        return self.__pot.acc(x-self.__X)
    def tid(self,x):
        return self.__pot.tid(x-self.__X)		
    def mass(self,x):
        return self.__pot.mass(x-self.__X)
    def set_cluster_pos(self,X):
        self.__X = X 
