'''
    orbits simulation generating tidal streams 
	by sprinkling method in any conservative potentials
'''
# Filename : 
#         stream.py
# purpose  :
#         generating stream and orbit by sprinkilig method 
# History  :  
#         01-07-2022 -  first stab                        - Hasanuddin
#         24-07-2022 -  change name from norbit to stream - Hasanuddin


import pickle
import numpy             as np
import matplotlib.pyplot as plt
import sys
import time              as CPUtime

from  math       import sqrt,pi
from .integrator import leapfrog  
from .potential  import pointmass  
from .potential  import plummer    
from .potential  import cluster_potential 
from .potential  import potentials  
from .body       import mass, pos, vel, Body, NBody



def omega(b):
    xc = pos(b)
    vc = vel(b)
    Rck = xc.dot(xc)  # jarak cluster kuadrat
    Rc  = sqrt(Rck)   # jarak cluster
    # kecepatan radial
    radial = np.dot(vc,xc)/Rck * xc 
    tangent = vc - radial 
    return sqrt(tangent.dot(tangent))/Rc

Mgal = 1
G = 1
Mclu =0.00001
CoG = (Mclu/Mgal)**(1/3.)
# initial cluster 	
Xclu = np.array([1,0,0])
Vclu = np.array([0,0.5,0])
B = Body(Mclu,Xclu,Vclu)

massa_titik = pointmass(G*Mgal)
cluster_as_plummer = plummer(G*Mclu,0.01)
cluster_as_point = pointmass(G*Mclu)

#cluster_pot = cluster_potential(cluster_as_point)
cluster_pot = cluster_potential(cluster_as_plummer)

galaxy_cluster = potentials([massa_titik,cluster_pot])

def rtid(xc):
    return CoG*sqrt(xc.dot(xc)) 
   
def create2stars(B):
    X = pos(B)
    Rq = X.dot(X)
    R  = sqrt(Rq)
    V = vel(B)
    rt = rtid(X)
    uX = X/R
    pos1 = X + rt*uX 
    pos2 = X - rt*uX
    #
    #
    '''
    radial = np.dot(V,X)/Rq * X 
    tangent = V - radial
    uT = tangent/sqrt(tangent.dot(tangent))
    vel1 = omega(B)*(R+rt)*uT
    vel2 = omega(B)*(R-rt)*uT 
    '''
    #
    # similar velocity to that of cluster 
    #vel1 =V 
    #vel2 =V 
    
    # for v propto 1/r^{1/2}
    vel1 = V*sqrt(R/(R+rt))
    vel2 = V*sqrt(R/(R-rt))
    
    b1 = Body(mass(B)/1000.,pos1,vel1)
    b2 = Body(mass(B)/1000.,pos2,vel2)
    return b1,b2 

def run(ti, tf, dt=0.01, cluster=B, include_cluster_grav=False):
    pot = massa_titik
    if include_cluster_grav:
        pot = galaxy_cluster
    
    stream = NBody()
    #  
    b1,b2 = create2stars(cluster)
	# mass is tcre 
    b1.set_mass(ti)
    b2.set_mass(ti)
    stream.add(b1)
    stream.add(b2)
    igc = leapfrog(massa_titik)  # cluster integrator
    ig  = leapfrog(pot)
    t = ti 
    Xo = [pos(cluster)]
    Vo = [vel(cluster)]
	
    while t < tf :
        X = pos(cluster)
        V = vel(cluster)
        if include_cluster_grav:
            pot.pots()[1].set_cluster_pos(X)
        Xnew,Vnew = igc.step(X,V,dt)
       
        #update cluster
        cluster.set_pos(Xnew)
        cluster.set_vel(Vnew)
        Xo.append(Xnew)
        Vo.append(Vnew)
        for b in stream.member():
            #update stars properties
            x = pos(b)
            v = vel(b)
            xnew,vnew = ig.step(x,v,dt)
            b.set_pos(xnew)
            b.set_vel(vnew)
        t = t + dt
        b1,b2 = create2stars(cluster)
        b1.set_mass(t)
        b2.set_mass(t)
        stream.add(b1)
        stream.add(b2)
    print("number of stars = "+ str(stream.number()))
    XX = [X[0] for X in Xo]
    YY = [X[1] for X in Xo]	

    xs = [pos(b)[0] for b in stream.member()]
    ys = [pos(b)[1] for b in stream.member()]
    tcre = [ b.get_mass() for b in stream.member()]
	
    plt.plot(XX,YY,'k')
    plt.scatter(0,0,s=16)
    plt.scatter(xs,ys,s=4, c=tcre, cmap='cool', vmin=ti,vmax=tf, edgecolor=['none'])
    #plt.scatter(xs,ys,s=4, c=tcre, cmap='cool', vmin=ti,vmax=tf, edgecolor=' ' )
    plt.colorbar()
    plt.xlabel(r'$X$')
    plt.ylabel(r'$Y$')
    #plt.axes().set_aspect('equal')
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.show()
    #>>> norbit.run(0,3.,dt=0.001)
    print("OK")

