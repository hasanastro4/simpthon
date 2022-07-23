r"""
    Integrators
    input: numpy  position           x 
           numpy  velocity           v
           scalar initial time       ti
           scalar final time         tf
           scalar timestep           dt
           class  potential          pot
"""

from abc import ABC, abstractmethod
import numpy as np


class integrator(ABC):
    @abstractmethod

    def integrate(x,v,ti,tf,dt,pot):
        r""" return time, position, and velocity from ti to tf.
        ...
        """
        pass

    def step(x,v,dt):
        r""" return position and velocity after 1 step
        ...
        """
        pass
		
class leapfrog(integrator):
    r"""
        a Kick-Drift-Kick version of leapfrog integrator 
        for solving dynamical sistem:
        ..math::
        \frac{dv}{dt} = a(x)
        \frac{dx}{dt} = v	
        
        Attributes
        ----------

        		
    """
	
    def __init__(self,pot):
        self.pot= pot
    def integrate(self,xi,vi,ti,tf,dt):
        t = ti
        x = xi
        v = vi
        while t<tf:
            x,v = self.step(x,v,dt)
            t+=dt
        return t,x,v 
			
    
    def step(self,x,v,dt):
        v =v+self.pot.acc(x)*dt*0.5 
        x =x+v*dt 
        v =v+self.pot.acc(x)*dt*0.5
        return x,v
    
    
class RungeKutta4(integrator):
    r"""
        a Runge-Kutta 4th Order integrator 
        for solving dynamical sistem:
        dv/dt = a(x)
        dx/dt = v		
    """
    def __init__(self,pot) :
        self.pot=pot
    def integrate(self,xi,vi,ti,tf,dt):
        t = ti
        x = xi
        v = vi
        while t<tf:
            x,v = self.step(x,v,dt)
            t+=dt
        return t,x,v 
			
    
    def step(self,x,v,dt):
        k1 = self.pot.acc(x)*dt
        l1 = v*dt
        k2 = self.pot.acc(x+0.5*l1)*dt
        l2 = (v+0.5*k1)*dt
        k3 = self.pot.acc(x+0.5*l2)*dt
        l3 = (v+0.5*k2)*dt
        k4 = self.pot.acc(x+l3)*dt
        l4 = (v+k3)*dt	
        v = v + (k1+2*k2+2*k3+k4)/6.
        x = x + (l1+2*l2+2*l3+l4)/6.		
        return x,v	
    
    
class Forward4OSymplectic(integrator):
    r"""
        a Forward 4th Order Symplectic Integrator (Chin & Chen, 2005)
		for solving dynamical sistem:
        dv/dt = a(x)
        dx/dt = v	
    """
    def __init__(self, pot):
        self.pot = pot
	
    def integrate(self,xi,vi,ti,tf,dt):
        t = ti
        x = xi
        v = vi
        while t<tf:
            x,v = self.step(x,v,dt)
            t+=dt
        return t,x,v 
			
    def step(self,x,v,dt):
        #kick by 1/6
        v = v + self.pot.acc(x)*dt/6.
        #drift by 1/2
        x = x + 0.5*dt*v
        #calculate pseudo-acceleration
        ac =  self.pot.acc(x)
        pseudo_acc = ac + dt*dt*np.dot(ac,self.pot.tid(x))/48.
        #kick by 2/3
        v = v + 4.*pseudo_acc*dt/6.
        #drift by 1/2
        x = x + 0.5*dt*v
        #kick by 1/6
        v = v + self.pot.acc(x)*dt/6.		
        return x,v  		
		
class Euler(integrator):
    r"""
        Euler Method w(t+dt) = w(t) + dt*f(w(t)) for solving dynamical sistem:
        dv/dt = a(x)
        dx/dt = v		
    """
	
    def __init__(self,pot):
        self.pot= pot
    def integrate(self,xi,vi,ti,tf,dt):
        t = ti
        x = xi
        v = vi
        while t<tf:
            x,v = self.step(x,v,dt)
            t+=dt
        return t,x,v 
			
			
    def step(self,x,v,dt):
        return x+v*dt,v+self.pot.acc(x)*dt

