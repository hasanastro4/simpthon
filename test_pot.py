###############################################################################

"""
    test_pot.py
    
    test potential by comparing acceleration at a point from formula and 
    numerical differentiation 
"""

###############################################################################

import potential 
import numpy as np
import random as rnd
from math import pi,sqrt


def test(pot,rang=[1,10],points=100, dr = 0.01, limit=0.01, printlog=' '):
    r""" test potential
    
    Parameters
    ----------
    pot : object class
        potential
    rang : list,optional
        range of data points
    points : int,optional
        number of points
    dr : float,optional
        interval 
    limit : float, optional
        acceptable error
    printlog : str, optional
        how deep to print ('e' -> error only, 'f' -> full print, 'r' -> report only. 

    Returns
    -------
	none
	
	"""
    
    # HISTORY:
    #     27-12-2020   - Started  -   Hasan
    #
    
    # generate 1-d data points
    r = np.array([rnd.uniform(rang[0],rang[1]) for i in range(points)])
    #import matplotlib.pyplot as plt
    #plt.hist(r,10)
    #plt.show()
    costheta = np.array([rnd.uniform(-1.0,1.0) for i in range(points)])
    theta  = np.arccos(costheta)
    phi = np.array([rnd.uniform(0,2*pi) for i in range(points)])
    pos = np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), 
                    r*np.cos(theta) ])
	
    pos = pos.transpose()
    
    acc = np.array([pot.acc(pos[i]) for i in range(points)])
	
    dx = np.array([dr,0,0])
    dy = np.array([0,dr,0])
    dz = np.array([0,0,dr])
    nac = np.array([np.array([0.5*(pot.pot(pos[i]-dx)-pot.pot(pos[i]+dx))/dr, 
        0.5*(pot.pot(pos[i]-dy)-pot.pot(pos[i]+dy))/dr  , 
        0.5* (pot.pot(pos[i]-dz)- pot.pot(pos[i]+dz))/dr ]) 
        for i in range(points)])
    error = (nac-acc)/acc
    print(" ")
    if printlog=='f':
        for i in range(points):
            print ('pos:' +str(pos[i])+'with r= '+str(r[i])+' has error'+ 
            str(error[i])+' of acc '+str(acc[i]) )
    elif printlog=='e':
        for i in range(points):
            print ('error: '+str(error[i]))
    elif printlog=='r':
        for i in range(points):
            print('acceleration: '+str(acc[i]) +' compared with '+str(nac[i]))
    print(" ")
    passtest = True
    for i in range (points):
        if (sqrt(error[i].dot(error[i]))> limit) :
            passtest = False
    if passtest: 
        print('All potential and acceleration match. No error found!') 
    else: 
        print('Some or all potential and acceleration mismatch. Error found!')
