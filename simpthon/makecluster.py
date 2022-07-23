"""
    makecluster.py
"""
import numpy as np
import pickle
import random as rnd
import body as bod
import time
from math import pi, cos, sin, acos, sqrt


def random_orientation(r):
    '''
        randomize the orientation a vector magnitude r
    '''
    theta = acos(rnd.uniform(-1,1))
    phi   = rnd.uniform(0,2*pi)
    x = r * sin(theta)*cos(phi)
    y = r * sin(theta)*sin(phi)
    z = r * cos(theta)
    return [x,y,z]

def plummer(n,a=1,G=1,M=1,seed=None,diagnose_energy=False, check_quartile=False,
            fname=' '):
    """
        make isotropic plummer model with uniform mass
    """
    start_time =time.time()
    rnd.seed(seed)
    nb = bod.NBody()
    mass = 1./n
    for i in range(n):
        m = rnd.random()
        r = 1./sqrt(m**(-2./3.)-1)
        pos = random_orientation(r)
        q = 0.0
        g = 0.1
        while g > q*q*(1.0-q*q)**3.5:
            q = rnd.random()
            g = rnd.uniform(0,0.1)
        v = sqrt(2)*q*(1.+r*r)**(-0.25)
        vel = random_orientation(v)
        b = bod.Body(mass,pos,vel)
        nb.add(b)
    if diagnose_energy:
        KE = 0
        for b in nb.member():
            KE += bod.mass(b)*(bod.vel(b)[0]**2+bod.vel(b)[1]**2+bod.vel(b)[2]**2)
        KE = 0.5*KE
        PE = 0
        i=0
        
        while i < n:
            j = i+1
            while j < n:
                xi = bod.pos(nb.member()[i])
                xj = bod.pos(nb.member()[j])
                distq = (xi[0]-xj[0])**2+(xi[1]-xj[1])**2+(xi[2]-xj[2])**2
                dist = sqrt(distq)
                PE -=bod.mass(nb.member()[i])*bod.mass(nb.member()[j])/dist
                j +=1
            i +=1				
        """
		for i in range(n):
            for j in range(i+1,n):
                xi = bod.pos(nb.member()[i])
                xj = bod.pos(nb.member()[j])
                distq = (xi[0]-xj[0])**2+(xi[1]-xj[1])**2+(xi[2]-xj[2])**2
                dist = sqrt(distq)
                PE -=bod.mass(nb.member()[i])*bod.mass(nb.member()[j])/dist
	    """
        PE = G*PE
        """
        x = [[b.get_pos()[0]] for b in nb.member()]
        y = [[b.get_pos()[1]] for b in nb.member()]
        z = [[b.get_pos()[2]] for b in nb.member()]
        m = [ b.get_mass() for b in nb.member()] 
		
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        m = np.array(m)
		
        dx = x.T-x
        dy = y.T-y
        dz = z.T-z
		
		# matrix that stores 1/r for all particle pairwise particle separations 
        inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
        inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]

        # sum over upper triangle, to count each interaction only once
        PE = G * np.sum(np.sum(np.triu(-(m*m.T)*inv_r,1)))
		"""
        EP = -0.09375*pi*G*M*M/a
        print ("Kinetic Energy   = "+str(KE))
        print ("Potential Energy = "+str(PE))	
        print ("Total energy     = "+str(KE+PE))
        print ("Plummer potential Energy = "+str(EP))
    if check_quartile:
        ordered_squared_radii = []
        for b in nb.member():
            rq = bod.pos(b)[0]**2 + bod.pos(b)[1]**2+bod.pos(b)[2]**2
            ordered_squared_radii.append(rq)
        ordered_squared_radii.sort()
        r14 = sqrt(ordered_squared_radii[round(0.25*n)-1])
        r12 = sqrt(ordered_squared_radii[round(0.50*n)-1])
        r34 = sqrt(ordered_squared_radii[round(0.75*n)-1])
        print ("r(1/4)= "+ str(r14))
        print ("r(1/2)= "+ str(r12))
        print ("r(3/4)= "+ str(r34))
    print("Execution time %.2f second"%(time.time()-start_time))
    if fname !=' ' :
        with open(fname,'wb') as file:
            pickle.dump(nb,file)
    return nb
