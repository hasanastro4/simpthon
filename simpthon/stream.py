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
from .body       import mass, pos, vel, Body, NBody, omega, radius 


class GalaxyCluster():
    def __init__(self, Mgal, Mclu):
        self.galaxyMass = Mgal
        self.clusterMass = Mclu
    def rtid(self,B):
        fac = (self.clusterMass/self.galaxyMass)**(1./3)
        x = pos(B)
        return fac*sqrt(x.dot(x))
    def create2stars(self,B,vel_adhoc="vcirc"):
        X = pos(B)
        R = radius(B)
        Rq = R*R 
        V = vel(B)
        rt = self.rtid(B)
        uX = X/R
        pos1 = X + rt*uX 
        pos2 = X - rt*uX
        # this for vel_adhoc=="vcluster"  
        vel1 = V 
        vel2 = V
        if vel_adhoc=="vcirc":
            # for v propto 1/r^{1/2}
            vel1 = V*sqrt(R/(R+rt))
            vel2 = V*sqrt(R/(R-rt))
        elif vel_adhoc=="vomega":
            tangent = vtangent(B)
            uT = tangent/sqrt(tangent.dot(tangent))
            vel1 = omega(B)*(R+rt)*uT
            vel2 = omega(B)*(R-rt)*uT
        b1 = Body(mass(B)/1000.,pos1,vel1)
        b2 = Body(mass(B)/1000.,pos2,vel2)
        return b1,b2     
       
    def clustermass(self):
        return self.clusterMass

        

mclu =0.00001

# initial cluster 	
Xclu = np.array([1,0,0])
Vclu = np.array([0,0.5,0])
B = Body(mclu,Xclu,Vclu)


 

def run(ti, tf, dt=0.01, cluster=B, include_cluster_grav=False, G=1, mgal=1):
    # setting galaxy+cluster
    gc = GalaxyCluster(mgal,mass(cluster))
    assert mass(cluster)==gc.clustermass()
    
    # setting potential
    massa_titik = pointmass(G*mgal)
    cluster_as_plummer = plummer(G*mass(cluster),0.02)
    cluster_as_point = pointmass(G*mass(cluster))

    #cluster_pot = cluster_potential(cluster_as_point)
    cluster_pot = cluster_potential(cluster_as_plummer)

    galaxy_cluster = potentials([massa_titik,cluster_pot])
    
    pot = massa_titik
    if include_cluster_grav:
        pot = galaxy_cluster
    
    stream = NBody()
    #  
    b1,b2 = gc.create2stars(cluster)
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
        b1,b2 = gc.create2stars(cluster)
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
    plt.colorbar()
    plt.xlabel(r'$X$')
    plt.ylabel(r'$Y$')
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.show()
    #>>> norbit.run(0,3.,dt=0.001)
    print("OK")
    return t, cluster, stream 

