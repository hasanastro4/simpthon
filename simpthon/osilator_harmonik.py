''' 
    osilator_harmonik.py
    simulasi osilator harmonik pegas
'''

import pickle
from .integrator import Euler, leapfrog, RungeKutta4, Forward4OSymplectic
from .potential  import osilator as osilator_pot
import numpy as np
import math
import matplotlib.pyplot as plt
from .time_step import constant_time as const_timestep
from .time_step import position as position_based_ts

def simulate(k,m,x0,v0,t0,tf,dt,fname=' ',method='leapfrog',timestep='constant_time',eps=0.1,amp=1):
    oh = osilator_pot(k,m)
    ig = leapfrog(oh)
    if method == 'rungekutta4':
        ig = RungeKutta4(oh)
    if method == 'forward4osymplectic':
        ig = Forward4OSymplectic(oh)
    if method == 'euler':
        ig = Euler(oh)
    ts = const_timestep(dt) 
    if timestep =='position':
        ts = position_based_ts(dt,abs(x0))
        if amp!=1:
            ts = position_based_ts(dt,amp,eps) 		
    x = np.array([x0,0,0])
    v = np.array([v0,0,0])
    time = [t0] 
    pos  = [x0]
    vel  = [v0]
    steps = 0
    while t0 < tf:
        X=x
        V=v
        dt = ts.size(X)
        x,v=ig.step(X,V,dt)
        t0 +=dt
        steps+=1
        time.append(t0)
        pos.append(x[0])
        vel.append(v[0])
    #storing time, pos, vel in data
    data = [time, pos, vel]
    if fname !=' ' :
        with open(fname,'wb') as file:
            pickle.dump(data,file)
    print('number of steps= '+str(steps))

def plot_pos_time(fname,xlim=[0,0],ylim=[0,0], savefig=' '):
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract data
    time = data[0]
    pos  = data[1]
    #vel = data[2]
    
    plt.plot(time, pos)
    plt.xlabel('time')
    plt.ylabel('x')
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if savefig!=' ' :
        plt.savefig(savefig)    
    plt.show()

def plot_period(fname,xlim=[0,0],ylim=[0,0], savefig=' ', dpi=300, dt=1e-3):
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract data
    time = np.array(data[0])
    pos  = np.array(data[1])
    #vel = data[2]
    #omegaq = k/m 
    #omega = math.sqrt(omegaq) 
    #tm = np.arange(time[0],time[len(time)-1],dt)
    #X = pos[0]*np.cos(omega*tm)
    #period = 2*math.pi/omega
    period = []
    T = []
    for i in range(len(time)-2):
        if (pos[i+2]-pos[i+1])*(pos[i+1]*pos[i]) < 0:
            period.append(time[i+1])
    for i in range(len(period)-1):
        T.append(period[i+1]-period[i])
    print (T)
    plt.plot(T)		
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if savefig!=' ' :
        plt.savefig(savefig,dpi=dpi)    
    plt.show()	
	
def plot_pos_error(fname, k,m, fname2=' ',fname3=' ', label1=' ',label2=' ',label3=' ',labelloc='lower left', xlim=[0,0],ylim=[0,0], savefig=' ', dpi=300, dt=1e-3,**kwargs):
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    if fname2!=' ' :
        with open(fname2,'rb') as file2:
            data2 = pickle.load(file2)
        time2 = np.array(data2[0])
        pos2  = np.array(data2[1])
    if fname3!=' ' :
        with open(fname3,'rb') as file3:
            data3 = pickle.load(file3)
        time3 = np.array(data3[0])
        pos3  = np.array(data3[1])
    #extract data
    time = np.array(data[0])
    pos  = np.array(data[1])
    #vel = data[2]
    omegaq = k/m 
    omega = math.sqrt(omegaq) 
    tm = np.arange(time[0],time[len(time)-1],dt)
    X = pos[0]*np.cos(omega*tm)
    period = 2*math.pi/omega
	
    plt.plot(time, pos, 'r.',label=label1)
    if fname2 !=' ':
        plt.plot(time2,pos2,'g^',label=label2)
    if fname3 !=' ':
        plt.plot(time3,pos3,'mo',label=label3)
    plt.plot(tm, X, label='analitik',**kwargs)
    plt.legend(loc=labelloc)
    plt.vlines([period,0.5*period],-pos[0],pos[0])
    plt.xlabel(r'$t$',fontsize='xx-large')
    plt.ylabel(r'$x$',fontsize='xx-large')
    plt.tight_layout()
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    ax = plt.gca()
    ticklabels = ax.get_yticklabels()+ax.get_xticklabels()
    for label in ticklabels:
        label.set_fontsize('x-large')
    if savefig!=' ' :
        plt.savefig(savefig,dpi=dpi)    
    plt.show()

	
def plot_vel_error(fname,k,m,xlim=[0,0],ylim=[0,0], savefig=' ',dt=1e-3):
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract data
    time = np.array(data[0])
    pos  = np.array(data[1])
    vel = data[2]
    omegaq = k/m 
    omega = math.sqrt(omegaq) 
    tm = np.arange(time[0],time[len(time)-1],dt)
    V = -pos[0]*omega*np.sin(omega*tm)
    period = 2*math.pi/omega
	
    plt.plot(time, vel, 'r.')
    plt.plot(tm, V)
    plt.vlines([period,0.5*period],-pos[0],pos[0])
    plt.xlabel('time')
    plt.ylabel('x')
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if savefig!=' ' :
        plt.savefig(savefig)    
    plt.show()
	
def arccos2(x,r):
    u = np.arccos(x/r)
    u = list(u)
    for i in range(1,len(u)):
        while u[i]<u[i-1]:
            u[i] = u[i]+2*math.pi
    return np.array(u)
      
		
def phase_error(fname, k, m, fname2=' ', xlim=[0,0],ylim=[0,0], savefig=' ',**kwargs):
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    omegaq = k/m 
    omega = math.sqrt(omegaq) 
    if fname2 !=' ':
        with open(fname2,'rb') as fileo:
            data2 = pickle.load(fileo)
        time2 = np.array(data2[0])
        pos2  = np.array(data2[1])
        #vel2 = np.array(data2[2])
        X2 = pos2[0]*np.cos(omega*time2)
    #extract data
    time = np.array(data[0])
    pos  = np.array(data[1])
    #vel = data[2]

    X  = pos[0]*np.cos(omega*time)     
    #theta0 = arccos2(X,pos[0])
    #theta  = arccos2(pos,pos[0]) 
    #dtheta = theta -theta0
	
    plt.plot(time, np.abs(pos-X),**kwargs)
    if fname2!=' ':
        plt.plot(time2, np.abs(pos2-X2),c='r')
    plt.xlabel('time')
    plt.ylabel('x')
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if savefig!=' ' :
        plt.savefig(savefig)    
    plt.show()

    
	
def plot_vel_time(fname,xlim=[0,0],ylim=[0,0], savefig=' '):
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract data
    time = data[0]
    #pos = data[1]
    vel = data[2]
    
    plt.plot(time, vel)
    plt.xlabel('time')
    plt.ylabel('v')
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if savefig!=' ' :
        plt.savefig(savefig)    
    plt.show()

def plot_energy_pos_time(fname, k, m, xlim=[0,0], ylim1=[0,0], ylim2=[0,0], savefig=' ', dpi=300):
    '''
        plot energy error dE/E0 vs time dan plot pos vs time
    '''
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract data
    time = data[0]
    pos = data[1]
    vel = data[2]
	
    E0 = 0.5*m*vel[0]*vel[0] + 0.5*k*pos[0]*pos[0]
    pos = np.array(pos)
    vel = np.array(vel)
    Et = 0.5*m*vel*vel + 0.5*k*pos*pos
    error = (Et-E0)/E0
    
    fig=plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(time,error)
    if ylim1 !=[0,0]:
        plt.ylim(ylim1)
    plt.ylabel(r'$\varepsilon$',fontsize='x-large')
    plt.setp(ax1.get_xticklabels(),visible=False)
	
    ax2 = fig.add_subplot(212,sharex=ax1)
    ax2.plot(time,pos)
    plt.hlines(0,xlim[0],xlim[1])
    if ylim2 !=[0,0]:
        plt.ylim(ylim2)
    plt.ylabel(r'$x$',fontsize='x-large')
    
    plt.xlabel(r'$t$', fontsize='x-large')
    ticklabels = ax1.get_yticklabels()+ax2.get_xticklabels()+ax2.get_yticklabels()
    for label in ticklabels:
        label.set_fontsize('x-large')
    plt.tight_layout(pad=1.)
    plt.subplots_adjust(hspace=0)
    if xlim !=[0,0]:
        plt.xlim(xlim)
    
    if savefig!=' ' :
        plt.savefig(savefig,dpi=dpi)    
    plt.show()
	
def plot_energy_time(fname, k, m, xlim=[0,0], ylim=[0,0], savefig=' '):
    '''
        plot relative energi error over time
    '''
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract data
    time = data[0]
    pos = data[1]
    vel = data[2]
	
    E0 = 0.5*m*vel[0]*vel[0] + 0.5*k*pos[0]*pos[0]
    pos = np.array(pos)
    vel = np.array(vel)
    Et = 0.5*m*vel*vel + 0.5*k*pos*pos
    error = (Et-E0)/E0
    print("Final Error = "+str(error[len(error)-1]))
    plt.plot(time,error)	
    plt.xlabel('time')
    plt.ylabel('error')
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if savefig!=' ' :
        plt.savefig(savefig)    
    plt.show()
	
def compare_energy_time(fname1, fname2, k, m, xlim=[0,0], ylim=[0,0], savefig=' '):
    '''
        plot relative energi error over time
    '''
    #open data
    with open(fname1,'rb') as file1:
        data1 = pickle.load(file1)
    with open(fname2,'rb') as file2:
        data2 = pickle.load(file2)
    #extract data
    time1 = data1[0]
    pos1 = data1[1]
    vel1 = data1[2]
    time2 = data2[0]
    pos2 = data2[1]
    vel2 = data2[2]
	
    E01 = 0.5*m*vel1[0]*vel1[0] + 0.5*k*pos1[0]*pos1[0]
    E02 = 0.5*m*vel2[0]*vel2[0] + 0.5*k*pos2[0]*pos2[0]
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    vel1 = np.array(vel1)
    vel2 = np.array(vel2)
    Et1 = 0.5*m*vel1*vel1 + 0.5*k*pos1*pos1
    Et2 = 0.5*m*vel2*vel2 + 0.5*k*pos2*pos2
    error1 = (Et1-E01)/E01
    error2 = (Et2-E02)/E02
    print("Final Error 1 = "+str(error1[len(error1)-1]))
    print("Final Error 2 = "+str(error2[len(error2)-1]))
    plt.plot(time1,error1)
    plt.plot(time2,error2)	
    plt.xlabel('time')
    plt.ylabel('error')
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if savefig!=' ' :
        plt.savefig(savefig)    
    plt.show()

	
def compare_3energy(fname1, fname2, fname3, k, m, xlim=[0,0], ylim=[0,0], savefig=' ',dpi=300,label1='1',label2='2',label3='3',legendloc='center right'):
    '''
        plot relative energi error over time
    '''
    #open data
    with open(fname1,'rb') as file1:
        data1 = pickle.load(file1)
    with open(fname2,'rb') as file2:
        data2 = pickle.load(file2)
    with open(fname3,'rb') as file3:
        data3 = pickle.load(file3)
    #extract data
    time1 = data1[0]
    pos1 = data1[1]
    vel1 = data1[2]
    time2 = data2[0]
    pos2 = data2[1]
    vel2 = data2[2]
    time3 = data3[0]
    pos3 = data3[1]
    vel3 = data3[2]
	
    E01 = 0.5*m*vel1[0]*vel1[0] + 0.5*k*pos1[0]*pos1[0]
    E02 = 0.5*m*vel2[0]*vel2[0] + 0.5*k*pos2[0]*pos2[0]
    E03 = 0.5*m*vel3[0]*vel3[0] + 0.5*k*pos3[0]*pos3[0]
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    pos3 = np.array(pos3)
    vel1 = np.array(vel1)
    vel2 = np.array(vel2)
    vel3 = np.array(vel3)
    Et1 = 0.5*m*vel1*vel1 + 0.5*k*pos1*pos1
    Et2 = 0.5*m*vel2*vel2 + 0.5*k*pos2*pos2
    Et3 = 0.5*m*vel3*vel3 + 0.5*k*pos3*pos3
    error1 = (Et1-E01)/E01
    error2 = (Et2-E02)/E02
    error3 = (Et3-E03)/E03
    print("Final Error 1 = "+str(error1[len(error1)-1]))
    print("Final Error 2 = "+str(error2[len(error2)-1]))
    print("Final Error 3 = "+str(error3[len(error3)-1]))
    plt.semilogx(time1,error1,label=label1)
    plt.semilogx(time2,error2,label=label2)	
    plt.semilogx(time3,error3,label=label3)
    plt.legend(loc=legendloc)
    plt.xlabel(r'$t$',fontsize='x-large')
    plt.ylabel(r'$\varepsilon$',fontsize='x-large')
    plt.tight_layout(pad=2.0)
    ax = plt.gca()
    ticklabels=ax.get_xticklabels()+ax.get_yticklabels()
    for label in ticklabels:
        label.set_fontsize('x-large')
    
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if savefig!=' ' :
        plt.savefig(savefig,dpi=dpi)    
    plt.show()
	
def plot_vel_pos(fname, xlim=[0,0],ylim=[0,0], savefig=' '):
    #open data
    with open(fname,'rb') as file:
        data = pickle.load(file)
    #extract data
    #time = data[0]
    pos = data[1]
    vel = data[2]
    
    plt.plot(pos,vel)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$v$')
    if xlim !=[0,0]:
        plt.xlim(xlim)
    if ylim !=[0,0]:
        plt.ylim(ylim)
    if savefig!=' ' :
        plt.savefig(savefig)    
    plt.show()
	
    
        		
        		



