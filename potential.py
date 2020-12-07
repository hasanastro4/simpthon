'''
    list of potential
	input: numpy    position      x
	output: scalar  potential     pot
	        numpy   acceleration  acc
            numpy   tide          	
'''
from abc import ABC, abstractmethod
import numpy as np
from math import sqrt

class potential(ABC):
    @abstractmethod
    def pot(x):
        pass

    def acc(x):
        pass

    def tid(x):
        pass


class osilator(potential):
    def __init__(self,k,m):
        self.k = k 
        self.m = m
		
    def pot(self,x):
        return 0.5*self.k*x[0]*x[0]
    def acc(self,x) :
        return -self.k*x/self.m	
    def tid(self,x):
        return np.array([[-self.k/self.m,0,0],[0,0,0],[0,0,0]])
    def mass(self,x):
        a = self.acc(x)
        return x.dot(x)*sqrt(a.dot(a))
		
class pointmass(potential):
    def __init__(self,GM):
        self.GM = GM
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
    def __init__(self,GM,b):
        self.GM = GM
        self.b  = b
    def pot(self,x):
        return -self.GM/sqrt(x.dot(x)+self.b**2)
    
		
