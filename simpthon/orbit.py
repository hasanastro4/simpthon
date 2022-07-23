'''
    orbit.py
    : particle simulation in any conservative potentials
'''
import pickle
import integrator
import potential
import time_step
import numpy as np
import matplotlib.pyplot as plt
import sys
from math import sqrt,pi
import time as CPUtime

def orbit(pot, x, v, t0, tf, dt, method='leapfrog', fname=' ',timestep='constant_time'):
    #Name: orbit
    """Simulate a particle in given potential
    
    parameters
    ----------
    pot : object class
        potential.
    x : list
        initial position.
    v : list
        initial velocity.
    t0 : float
        initial time.
    tf : float
        final time.
    dt : float
        time-step (constant) or accuracy parameter (variabel).
    method : str
        integrator.
    fname  : str
        saving file.
    timestep : str
        time-step.

    returns
    -------
    none
		none if fname not given.    
    file : file
        a binary file if fname given.		
    
    
    |
    """
     
    #History:
    #    16-12-2020-written-Hasanuddin
    #    08-11-2020-added  - Hasanuddin : runinfo added to pickle file
    
    runinfo = ("orbit(pot="+str(pot)+", x="+str(x)+", v="+str(v)+", t0="+str(t0)
	          +", tf="+str(tf)+",dt="+str(dt)+", method="+str(method)+", fname="
			  +str(fname)+", timestep="+str(timestep)+")")
    
    x = np.array(x)
    v = np.array(v)
    	
    ig = integrator.leapfrog(pot)
    if method=='rungekutta4':
        ig = integrator.RungeKutta4(pot)
    if method=='forward4osymplectic':
        ig = integrator.Forward4OSymplectic(pot)
    ts = time_step.constant_time(dt) 
    if timestep =='orbital_period':
        ts = time_step.orbital_period(dt,pot) 
    time = [t0] 
    pos  = [x]
    vel  = [v]
    step = 0
    while t0 < tf:
        X=x
        V=v
        dt = ts.size(x)        
        x,v=ig.step(X,V,dt)
        t0 +=dt
        step = step+1
        time.append(t0)
        pos.append(x)
        vel.append(v)
    #storing time, pos, vel,runinfo in data
    data = [time, pos, vel, runinfo]
    if fname !=' ' :
        with open(fname,'wb') as file:
            pickle.dump(data,file)
    print('number of steps= '+str(step))		
#

def runinfo(fname):
    #Name :
    #     runinfo
    """read run information
    
    parameters
    ----------
	fname : str 
        name of file 
    
    returns
    -------
    None
    
    """	
    #History
    # 08-11-2021  -  written - Hasanuddin
    
    with open(fname,'rb') as file:
        data = pickle.load(file)
    print(data[-1])
    
def plot_XY(fname,xlim=[0,0],ylim=[0,0], savefig=' '):
    #Name: 
    #    plot_XY
    """simply plot orbit

    parameters
    ----------
    fname : str
        name of file containing t,x,v.
    xlim  : list, optional
        x-axis limit.
    ylim  : list, optional
        y-axis limit.
    savefig: str, optional
        name of saving figure.
        
    returns
    -------
    none
        none if savefig not given.
    figure: file
        show figure and save figure if savefig given.
    
    
    |
    """

    #History
    # 17-12-2020  written Hasanuddin
    
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract data
    #time = data[0]
    pos = data[1]
    #print (pos)
    #vel = data[2]
    x = [xt[0] for xt in pos]
    y = [xt[1] for xt in pos]
    
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    plt.axes().set_aspect('equal')
    if savefig!=' ' :
        plt.savefig(savefig)    
    plt.show()
#	
#def plot_ratio_VxVy_time(fname,xlim=[0,0],ylim=[0,0], savefig=' '):
#    #open data
#    with open(fname,'rb') as file:
#        data = pickle.load(file)
#    #extract data
#    time = data[0]
#    pos = data[1]
#    vel = data[2]
#    X = np.array([xt[0] for xt in pos])
#    Y = np.array([xt[1] for xt in pos])
#    Vx = np.array([vt[0] for vt in vel])
#    Vy = np.array([vt[1] for vt in vel])
#    Ratio = np.arctan2(Vy,Vx) 
#	
#    plt.plot(time, Ratio)
#    plt.plot(time, X )
#    plt.xlabel('t')
#    plt.ylabel(r'$\theta$')
#    if xlim !=[0,0]:
#        plt.xlim(xlim)
#    if ylim !=[0,0]:
#        plt.ylim(ylim)
#    if savefig!=' ' :
#        plt.savefig(savefig)    
#    plt.show()    

def plot_action_time(fname,xlim=[0,0],ylim=[0,0], savefig=' '):
    """ plot action (:math:`J_r , L_z`) over time
    
    parameters
    ----------
    fname : str
        name of file containing at least t,x,v in respective order
	xlim  : list, optional
        x-axis limit.
    ylim  : list, optional
        y-axis limit.
    savefig: str, optional
        name of saving figure.
		
    returns
    -------
    none
        none if savefig not given.
    figure : file
        a file containing saving figure if savefig given.
    """
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract data
    time = data[0]
    pos = data[1]
    vel = data[2]
    x = np.array([xt[0] for xt in pos])
    y = np.array([xt[1] for xt in pos])
    vx = np.array([v[0] for v in vel])
    vy = np.array([v[1] for v in vel])
    Lz = x*vy - y*vx
    rq = x*x+y*y
    rq0 = rq[0]	
    vrq = (x*vx+y*vy)**2/rq
	
    t0  = time[0]
    tf  = time[-1]
    N = len(time)
    time = np.array(time)
    dts = [time[i+1]-time[i] for i in range(N-1)]
    dts = [t0]+dts

    #this is pointmass potential only
    print("orbit properties")
    print("L = "+str(Lz[0]))
    E0 = 0.5*(vy[0]*vy[0]+vx[0]*vx[0])-1./sqrt(rq[0])
    print("E = "+str(E0))
    a = -1/(2*E0) 
    print("a= "+str(a))
    e = sqrt(1+2*E0*Lz[0]*Lz[0])
    print("e= "+str(e))
    T = 2*pi*a*sqrt(a/1.)
    print("T = "+str(T))
    Jr0the =  sqrt(2/-E0)*pi - 2*pi*Lz[0] 
    print("Jr (theory) = "+str(Jr0the))
    print(" ")
    
    #start
    CPUstart = CPUtime.time()
    flip = 0
    sumJr=0	
    Jr = []
    ti = []
    # 0 -> 100%
    didel = "\b"*3   # backspace 3 times
    print("Calculation in progress ... ")
	
    for i in range(1,N-1):
        sumJr = sumJr+vrq[i]*dts[i] 
        drq = rq[i]-rq[i-1]
        drqnext=rq[i+1]-rq[i]
        if drq*drqnext <0:
            flip +=1
        if drq*drqnext <0 and flip==2:
            Jr.append(sumJr)
            ti.append(time[i])
            sumJr = 0
            flip = 0
            
        progress = int(i/(N-2)*100)
        print("{0}{1:{2}}".format(didel, progress,3), end="")
        sys.stdout.flush()
		

    Jr = np.array(Jr)
    ti = np.array(ti)
    Jrmean = np.mean(Jr)
    Lzmean = np.mean(Lz)
    print(" ")
    print("Jr (numeric) = "+str(Jr[0]))
    print("<Jr> ="+str(Jrmean))
    print("<Lz> ="+str(Lzmean))
    print("<dJr> ="+str(np.std(Jr)))
    print("<dLz> ="+str(np.std(Lz)))
	
    # CPU time information
    print("Execution time %.3f second"%(CPUtime.time()-CPUstart))
    print(" ")
	
	# remove savefig extension 
    idot = savefig.find('.')
    if idot>-1:
        savefigWOExten=savefig[:idot]
        extfig = savefig[idot:]
    else:
        savefigWOExten=savefig
        extfig = '.jpg'

    plt.figure(1)
    plt.plot(time/T,Lz-Lzmean)
    plt.ylabel(r'$L_z-<L_z>$',fontsize=14)
    plt.xlabel(r'$t/T$',fontsize=14)
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    plt.tight_layout()
    if savefig!=' ' :
        plt.savefig(savefigWOExten+'Lz'+extfig) 
	
    plt.figure(2)
    plt.plot(ti/T,Jr-Jrmean,'.')
    plt.ylabel(r'$J_r-<J_r>$',fontsize=14)
    plt.xlabel(r'$t/T$',fontsize=14)
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    plt.tight_layout()
    if savefig!=' ' :
        plt.savefig(savefigWOExten+'Jr'+extfig)
     	
    plt.show()   
    
	

def plot_energy_time(fname,pot, type='de', xlim=[0,0],ylim=[0,0], savefig=' '):
    """ plot energy('E') :math:`E` or error ('dE') :math:`\Delta E`, relative energy error ('de') :math:`\Delta E / E_0` over time.
    
    parameters
    ----------
    fname : str
        name of file containing t,x,v.
    type : str
        plot energy ('E'), energy error ('dE'), or relative error energy ('de').
    pot : class
        potential class object.
    xlim  : list, optional
        x-axis limit.
    ylim  : list, optional
        y-axis limit.
    savefig: str, optional
        name of saving figure.
    
    returns
    -------
    none
        none if savefig not given.
    figure : file
        a file containing saving figure if savefig given.
    
    
    |
    """
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract data
    time = data[0]
    pos = data[1]
    vel = data[2]
    KE =  np.array([0.5*vi.dot(vi) for vi in vel])
    PE =  np.array([pot.pot(xi) for xi in pos])
    Et = KE + PE
    E0 = KE[0] + PE[0]
    dE = Et-E0
    Error = dE/E0
    
    if type =='de':
        plt.plot(time, Error)
        plt.ylabel(r'$\Delta E / E_0$')
    elif type=='dE':
        plt.plot(time,dE)
        plt.ylabel(r'$\Delta E$')
    else:
        plt.plot(time,Et)
        plt.ylabel(r'$E$')

    plt.xlabel(r'$t$')
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if savefig!=' ' :
        plt.savefig(savefig)    
    plt.show()