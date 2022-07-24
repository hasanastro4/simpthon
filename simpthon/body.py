'''
   body.py
'''

import numpy as np 

class Body():
    """class representing a body
        
    """
    def __init__(self, m, x, v):
        if isinstance(x,list):
            x = np.array(x)
        if isinstance(v,list):
            v = np.array(v)
        self.__mass= m
        self.__pos = x
        self.__vel = v

    def get_mass(self):
        return self.__mass
    def get_pos (self):
        return self.__pos
    def get_vel(self):
        return self.__vel
		
    def set_mass(self,m):
        self.__mass = m
    
    def set_pos(self,x):
        if isinstance(x,list):
            x = np.array(x)
        self.__pos = x
		
    def set_vel(self,v):
        if isinstance(v,list):
            v = np.array(v)
        self.__vel = v

    def set_phase_space(self,x,v):
        if isinstance(x,list):
            x = np.array(x)
        if isinstance(v,list):
            v = np.array(v)
        self.__pos = x
        self.__vel = v
	

def mass(b):
    return b.get_mass() 
	
def pos(b):
    return b.get_pos()
	
def vel(b):
    return b.get_vel()
    
def angmom(b):
    r"""
    angular momentum per unit mass of body `b`
    
    .. math:: \mathbf{L} = \mathbf{r \times v}   
        
    """
    return np.cross(pos(b),vel(b))
    
def kinetic(b):
    r"""
    kinetic energy per unit mass of body `b`
    
    .. math:: K = \frac{1}{2} \mathbf{v} \cdot \mathbf{v} 
    
    """
    return 0.5*vel(b).dot(vel(b))


class NBody():
    '''
        n-body
    '''
    def __init__(self):
        self.__N=0
        self.__member=[]
	
    def add(self, b):
        self.__member.append(b)
        self.__N +=1
		
    def member(self):
        return self.__member

    def number(self):
        return len(self.__member)
