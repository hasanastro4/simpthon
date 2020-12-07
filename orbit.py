'''
    orbit.py
    simulasi orbit di berbagai potensial
'''
import pickle
import integrator
import potential
import time_step
import numpy as np
import matplotlib.pyplot as plt

def orbit(pot, x, v, t0, tf, dt, method='leapfrog', fname=' ',timestep='constant_time'):
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
    #storing time, pos, vel in data
    data = [time, pos, vel]
    if fname !=' ' :
        with open(fname,'wb') as file:
            pickle.dump(data,file)
    print('number of steps= '+str(step))		
			
def plot_XY(fname,xlim=[0,0],ylim=[0,0], savefig=' '):
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
	
def plot_ratio_VxVy_time(fname,xlim=[0,0],ylim=[0,0], savefig=' '):
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract data
    time = data[0]
    pos = data[1]
    vel = data[2]
    X = np.array([xt[0] for xt in pos])
    Y = np.array([xt[1] for xt in pos])
    Vx = np.array([vt[0] for vt in vel])
    Vy = np.array([vt[1] for vt in vel])
    Ratio = np.arctan2(Vy,Vx) 
	
    plt.plot(time, Ratio)
    plt.plot(time, X )
    plt.xlabel('t')
    plt.ylabel(r'$\theta$')
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if savefig!=' ' :
        plt.savefig(savefig)    
    plt.show()    


def plot_energy_time(fname,pot,xlim=[0,0],ylim=[0,0], savefig=' '):
    '''
       plot relatif energy error over time
    '''
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
    plt.plot(time, Error)
    plt.xlabel('t')
    plt.ylabel('Error')
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if savefig!=' ' :
        plt.savefig(savefig)    
    plt.show()