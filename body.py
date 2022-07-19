'''
   body.py
'''

import potential

class Body():
    """
        body
    """
    def __init__(self, m, x, v):
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
        self.__pos = x
		
    def set_vel(self,v):
        self.__vel = v

    def set_phase_space(self,x,v):
        self.__pos = x
        self.__vel = v
	

def mass(b):
    return b.get_mass() 
	
def pos(b):
    return b.get_pos()
	
def vel(b):
    return b.get_vel()

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
