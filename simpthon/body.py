'''
   body.py
'''

import numpy as np 
from math import sqrt 

class Body():
    r"""class representing a body/particle
        
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
        r"""return mass of a body/particle
        
        """
        return self.__mass

    def get_pos (self):
        r"""return position of a body/particle
        
        """
        return self.__pos

    def get_vel(self):
        r"""return velocity of a body/particle
        
        """
        return self.__vel
		
    def set_mass(self,m):
        r"""set mass of a body/particle as m
        
        """
        self.__mass = m
    
    def set_pos(self,x):
        r"""set position of a body/particle as :math:`\mathbf{x}`
        
        """
        if isinstance(x,list):
            x = np.array(x)
        self.__pos = x
		
    def set_vel(self,v):
        r"""set velocity of a body/particle as :math:`\mathbf{v}`
        
        """
        if isinstance(v,list):
            v = np.array(v)
        self.__vel = v

    def set_phase_space(self,x,v):
        r"""set phase space of a body/particle as :math:`\mathbf{x,v}`
        
        """
        if isinstance(x,list):
            x = np.array(x)
        if isinstance(v,list):
            v = np.array(v)
        self.__pos = x
        self.__vel = v
	

def mass(b):
    r"""mass of a body
    
    Args:
        b (class Body): body.
    
    Returns:
        float: mass of a body.

    """
    return b.get_mass() 
	
def pos(b):
    r"""position of a body
    
    Args:
        b (class Body): body.
    
    Returns:
        numpy array : position of a body.

    """
    return b.get_pos()
	
def vel(b):
    r"""velocity of a body
    
    Args:
        b (class Body): body.
    
    Returns:
        numpy array : velocity of a body.

    """
    return b.get_vel()
    
def angmom(b):
    r"""
    angular momentum per unit mass of body `b`
    
    .. math:: \mathbf{L} = \mathbf{r \times v}   
    
    Args:
        b (class Body): body.
    
    Returns:
        numpy array : angular momentum per unit mass of a body.
        
    """
    return np.cross(pos(b),vel(b))
    
def kinetic(b):
    r"""
    kinetic energy per unit mass of body `b`
    
    .. math:: K = \frac{1}{2} \mathbf{v} \cdot \mathbf{v} 
    
    Args:
        b (class Body): body.
    
    Returns:
        float : kinetic energy per unit mass of a body.
    
    """
    return 0.5*vel(b).dot(vel(b))

def radius(b):
    r"""
    radius of position of a body `b` i.e. :math:`r=\sqrt{\mathbf{x}}`.
	
	Args:
	    b (class Body): body.
    
    Returns:
        float : radius of position  of a body.
    
    """
    return sqrt(pos(b).dot(pos(b)))
	
def vradial(b):
    r"""
	radial velocity
    
    Args:
        b (class Body): body.
    
    Returns:
        numpy array : radial velocity of a body.
    
    """
    rq = radius(b)*radius(b)
    return vel(b).dot(pos(b))/rq * pos(b)
	
def vtangent(b):
    r"""
    tangential velocity
    
    Args:
        b (class Body): body.
    
    Returns:
        numpy array : radial velocity of a body.
    
    """
    return vel(b)-vradial(b)

def omega(b):
    r"""
	angular frequency of body `b`
	
	.. math:: \Omega = \frac{v_{\mathrm{tangential}}}{r}
    
    Args:
	    b (class Body): body.
    
    Returns:
        float : angular frequency of a body.
    
    """ 
    return sqrt(vtangent(b).dot(vtangent(b)))/radius(b)


class NBody():
    r"""class representing a bunch of bodies or particles (N-body)
        
    """
    def __init__(self):
        self.__N=0
        self.__member=[]
	
    def add(self, b):
        r""" append a body to bodies
        
        """
        self.__member.append(b)
        self.__N +=1
		
    def member(self):
        r""" return a list of body in bodies
        
        """
        return self.__member

    def number(self):
        r""" return number of bodies
        
        """
        return len(self.__member)
