'''
    time_step.py
    function: lists of time-step class
'''
from abc import ABC, abstractmethod
import potential
from math import sqrt
import numpy as np

#abstract class
class time_step(ABC):
    @abstractmethod
    def size(x):
        pass

#time-step based on dynamical time
# T = n sqrt(r^3/M(r))
class orbital_period(time_step):
    def __init__(self, eta,potential):
        self.eta = eta
        self.pot = potential
    def size(self, x):
        r = sqrt(x.dot(x))
        return self.eta* sqrt(r**3/self.pot.mass(x))

# constant time step
class constant_time(time_step):
    def __init__(self,eta):
        self.eta = eta
    def size(self,x):
        return self.eta

#time-step based on position
# T = eta*[abs(x)/A)+1]/2
class position(time_step):
    def __init__(self,eta,amp=1.,eps=0.1):
        self.eta = eta
        self.amp = amp
        self.eps = eps
    def size(self,x):
        return 0.5*self.eta*(sqrt(x.dot(x))/self.amp+self.eps)