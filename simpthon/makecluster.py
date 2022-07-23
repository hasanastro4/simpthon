"""
    makecluster.py
"""
import numpy as np
import pickle
import random as rnd
from .body import Body, NBody, mass, vel, pos  
import time
from math import pi, cos, sin, acos, sqrt
import matplotlib.pyplot as plt 
from matplotlib import colors
import matplotlib.ticker

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

def single_mass_plummer(n,a=1,G=1,M=1,seed=None,diagnose_energy=False, 
    check_quartile=False, fname=' ', potential_method='matrix', uni =False):
    """
        make isotropic plummer model with uniform mass
        nb: 
        we use mass as potential energy of body
        potential method -> matrix or for 		
		uni -> uniform distribution random 
        typical command:
        >>> cluster = mc.single_mass_plummer(10000,diagnose_energy=1,fname='plum10k.pkl')
    """
    start_time =time.time()
    runinfo = "single_mass_plummer(n="+str(n)+",a="+str(a)+",G="+str(G)+",M="+str(M)+\
               ",seed="+str(seed)+",diagnose_energy="+str(diagnose_energy)+ \
               ",check_quartile="+str(check_quartile)+",fname="+fname+ \
			   ", potential_method ="+potential_method+")"
    start_time =time.time()
    rnd.seed(seed)
    nb = NBody()
    mas = 1./n
    for i in range(n):
        if uni:
            m = rnd.uniform(0,1)
        else:
            m = rnd.random()
        r = 1./sqrt(m**(-2./3.)-1)
        posi = random_orientation(r)
        q = 0.0
        g = 0.1
        while g > q*q*(1.0-q*q)**3.5:
            q = rnd.random()
            g = rnd.uniform(0,0.1)
        v = sqrt(2)*q*(1.+r*r)**(-0.25)
        velo = random_orientation(v)
        b = Body(mas,posi,velo)
        nb.add(b)
    if diagnose_energy:
        # 0 -> 100%
        didel = "\b"*3   # backspace 3 times
        print("Simulation in progress ... ")
        KE = 0
        for b in nb.member():
            KE += (vel(b)[0]**2+vel(b)[1]**2+vel(b)[2]**2)
        KE = 0.5*mas*KE
        if potential_method!='matrix': 
            PE = 0
            i=0
        
            while i < n:
                j = i+1
                while j < n:
                    xi = pos(nb.member()[i])
                    xj = pos(nb.member()[j])
                    distq = (xi[0]-xj[0])**2+(xi[1]-xj[1])**2+(xi[2]-xj[2])**2
                    dist = sqrt(distq)
                    PE -=mas*mas/dist
                    j +=1
                i +=1
                progress = int(i/n*100)
                print("{0}{1:{2}}".format(didel, progress,3), end="")
                sys.stdout.flush()				
            """
		    for i in range(n):
                for j in range(i+1,n):
                    xi = pos(nb.member()[i])
                    xj = pos(nb.member()[j])
                    distq = (xi[0]-xj[0])**2+(xi[1]-xj[1])**2+(xi[2]-xj[2])**2
                    dist = sqrt(distq)
                    PE -=mass(nb.member()[i])*mass(nb.member()[j])/dist
            """
            PE = G*PE
        
        else : 
            x = [[b.get_pos()[0]] for b in nb.member()]
            y = [[b.get_pos()[1]] for b in nb.member()]
            z = [[b.get_pos()[2]] for b in nb.member()]
            m = [ mas for b in nb.member()] 
		
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
            #print(-(m*m.T)*inv_r)
            i = 0
            GMqir = G*np.sum(-(m*m.T)*inv_r,1)
            for b in nb.member():
                b.set_mass(GMqir[i])
                i+=1
            # sum over upper triangle, to count each interaction only once
            PE = G * np.sum(np.sum(np.triu(-(m*m.T)*inv_r,1)))
		
        EP = -0.09375*pi*G*M*M/a   # -3/32 pi GM^2/a 
        print (" ")
        print ("Kinetic Energy   = "+str(KE))
        print ("Potential Energy = "+str(PE))	
        print ("Total energy     = "+str(KE+PE))
        print ("Plummer potential Energy = "+str(EP))
    if check_quartile:
        ordered_squared_radii = []
        for b in nb.member():
            rq = pos(b)[0]**2 + pos(b)[1]**2+pos(b)[2]**2
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
        #data = [nb,runinfo]
        data = nb 
        with open(fname,'wb') as file:
            pickle.dump(data,file)
    return nb

'''mc.single_mass_plummer(10000,diagnose_energy=1,fname='plum10k_mat_uni.pkl',uni=1, potential_method='matrix')
Simulation in progress ...

Kinetic Energy   = 0.14762962700149415
Potential Energy = -0.29717682737611173
Total energy     = -0.14954720037461758
Plummer potential Energy = -0.2945243112740431
Execution time 3.07 second
<body.NBody object at 0x0000016BD3EF20D0>
'''

def plot_surface_density(cluster,xlim=[0,0], ylim=[0,0],dx=0, amp=1,savefig=' ',
    dpi=150):
    '''
        plot histogram 2d cluster in the plane XY 
        cluster might be file name string contains class NBody and runinfo
            or class NBody itself
        amp -> amplitudo for surface density 1/pi * (1+r^2)^{-2} 
        typical command
        >>> mc.plot_surface_density(cluster,xlim=[-5,5],ylim=[-5,5],dx=0.1,amp=600)
		>>> mc.plot_surface_density('plum10k_uni.pkl',xlim=[-5,5],ylim=[-5,5],dx=0.16,
		amp=900,savefig='sd_10k.jpg',dpi=300)
    '''
    if isinstance(cluster,str):
        with open(cluster,'rb') as file:
            data = pickle.load(file)
            glob_cluster = data        
    elif isinstance(cluster,NBody):
        glob_cluster = cluster 
    else : 
        return None
    # extract data
    x = np.array([b.get_pos()[0] for b in glob_cluster.member()])
    y = np.array([b.get_pos()[1] for b in glob_cluster.member()])
    
    if dx <= 0:
        Nbin = 100
    else:
        Nbin = int((xlim[1]-xlim[0])/dx)
    xx = np.linspace(xlim[0],xlim[1],1000)
    
    def scatter_hist(x, y, ax, ax_histx, ax_histy):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        
        # now determine nice limits by hand:
        binwidth = dx #0.25
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        
        # the scatter plot:
        h2d= ax.hist2d(x, y,bins=bins,cmap='Greys',norm=colors.LogNorm())
        ax.set_xlim(xlim[0],xlim[1])
        ax.set_ylim(ylim[0],ylim[1])
        ax.set_xlabel(r'$X$', fontsize=16)
        ax.set_ylabel(r'$Y$', fontsize=16)

       
        ax_histx.hist(x, bins=bins)
        ax_histx.plot(xx,amp*(1+xx*xx)**(-2))
        ax_histy.hist(y, bins=bins, orientation='horizontal')
        ax_histy.plot(amp*(1+xx*xx)**(-2),xx)
		
        return h2d  
    
    # start with a square Figure
    fig = plt.figure(figsize=(6, 6))

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.01, hspace=0.01)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # use the previously defined function
    h2d=scatter_hist(x, y, ax, ax_histx, ax_histy)
	
    cbax = fig.add_subplot(gs[0,1])
    fig.colorbar(h2d[3],cax=cbax)
    """ 
	https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
    """
    #plt.tight_layout()
    
    #plt.hist2d(x,y,bins=bins,cmap='Greys',norm=colors.LogNorm())
    #plt.xlim(xlim[0],xlim[1])
    #plt.ylim(ylim[0],ylim[1])
    #plt.gca().set_aspect(1)
    if savefig!=' ':
        plt.savefig(savefig,dpi=dpi)
    
    plt.show()    
    
def profile(cluster, potfilename=' ', xlim=[0,0],amphi=1,ampho=1,dx=0.16, 
            savefig=' ', dpi=720):
    '''
        plot potensial, density vs r 
        typical command:
        mc.profile('plum1m_uni.pkl','plum10k_mat_uni.pkl',xlim=[0,6],ampho=3,savefig='denspot.jpg')
    '''
    if isinstance(cluster,str):
        with open(cluster,'rb') as file:
            data = pickle.load(file)
            glob_cluster = data        
    elif isinstance(cluster,NBody):
        glob_cluster = cluster 
    else : 
        return None
		
    if potfilename != ' ':
        with open(potfilename,'rb') as file:
            gc = pickle.load(file)
    #extract data
    x = np.array([b.get_pos()[0] for b in glob_cluster.member()])
    y = np.array([b.get_pos()[1] for b in glob_cluster.member()])
    z = np.array([b.get_pos()[2] for b in glob_cluster.member()])
    R = np.sqrt(x*x+y*y+z*z)
    # mass corresponding to potential energy 
    if potfilename !=' ':
        p = np.array([b.get_mass() for b in gc.member()])
        xx = np.array([b.get_pos()[0] for b in gc.member()])
        yy = np.array([b.get_pos()[1] for b in gc.member()])
        zz = np.array([b.get_pos()[2] for b in gc.member()])
        RR = np.sqrt(xx*xx+yy*yy+zz*zz)
    else:
        p = np.array([b.get_mass() for b in glob_cluster.member()])
        RR = R 
	
    r = np.linspace(xlim[0],xlim[1],1000)
    rq = r*r
    phi = amphi/np.sqrt(1+rq)
    # rho = 3/4*pi (1+r**2)^(-5/2) *4*pi/3 *r*r dx 
    rho = ampho*rq*(1+rq)**(-5/2)
    
    binwidth = dx #0.25
    xymax = np.max(np.abs(x))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    plt.figure(figsize=(6,3))
    plt.plot(r,phi,'r', label=r'$-\Phi_{\mathrm{plummer}}$')
    # negatif phi dan phi = p / mass of stars 
    plt.scatter(RR,-len(p)*p,s=4,color='green',edgecolor=None, label=r'$-\Phi_{\mathrm{model}}$') 
    plt.hist(R,bins=bins,density=True, histtype='step',label=r'$n$')
    plt.plot(r,rho,'b', label=r'$r^2 \rho \Delta r$')
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.04))
    plt.xlim(xlim)
    plt.ylim([0,1.04])
    plt.xlabel(r'$r$',fontsize=16)
    plt.ylabel(r'$-\Phi, n $', fontsize=16)
    plt.legend()
    plt.tight_layout()
    if savefig!=' ':
        plt.savefig(savefig,dpi=dpi)
    plt.show()
    
def anisotropic(cluster):
    '''
       calculate anisotropic parameter beta
    '''
    if isinstance(cluster,str):
        with open(cluster,'rb') as file:
            data = pickle.load(file)
            glob_cluster = data        
    elif isinstance(cluster,NBody):
        glob_cluster = cluster 
    else : 
        return None
    #extract data
    vr = []
    vth = []
    vph = []
    for b in glob_cluster.member():
        x = np.array(b.get_pos())
        v = np.array(b.get_vel())
        rq = x.dot(x)
        r = sqrt(rq)
        R = sqrt(x[0]*x[0]+x[1]*x[1])
        costheta = x[2]/r 
        sintheta = R/r 
        cosphi = x[0]/R 
        sinphi = x[1]/R 
        svrad = v.dot(x)/r 
        vrad = svrad*x/r 
        vtan = v - vrad 
        utheta = np.array([costheta*cosphi,costheta*sinphi,-sintheta])
        uphi = np.array([-sinphi,cosphi,0])
        #svtheta = vtan.dot(utheta)
        svtheta = v.dot(utheta)
        vtheta = svtheta*utheta 
        #vphi = vtan - vtheta 
        #svphi = sqrt(vphi.dot(vphi))     
        svphi = v.dot(uphi)        
        vr.append(svrad)
        vth.append(svtheta)
        vph.append(svphi)         
        #if v.dot(v) != (svrad**2+svtheta**2+svphi**2):
        #    print (str(v.dot(v))+'<>'+str(svrad**2+svtheta**2+svphi**2) )       
    vr = np.array(vr)
    vth = np.array(vth)
    vph = np.array(vph)
    print(np.var(vr))
    print(np.var(vth))
    print(np.var(vph))
    return 1 - 0.5*(np.var(vth)+np.var(vph))/np.var(vr)
    
def plot_log_surface_density(observed,a=1,M=1,savefig=' ',dpi=720):
    '''
       plot observed surface density
    '''
    # r    -> rad in arc
    # lsd  -> log surface density 
    # elsd -> error log surface density
    r, lsd,elsd = np.loadtxt(observed,usecols=(0,1,2),unpack=True) 
    #data = np.loadtxt(observed)
    #print(data)
    lr = np.log10(r)
    #model
    x = np.linspace(min(r),max(r),10000)
    iaq = 1/a/a  
    S = M/pi*iaq*(1+x*x*iaq)**(-2)
    LS = np.log10(S)
    
    plt.scatter(lr,lsd,s=16,edgecolor=None, alpha=1,label='data observasi')
    plt.errorbar(lr, lsd, yerr=elsd, fmt='o')
    plt.plot(np.log10(x),LS, label='profil Plummer')
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    plt.ylim(min(lsd-elsd)-0.1,max(lsd+elsd)+0.1)
    plt.xlabel(r'$\log (r/ \mathrm{arcsec}) $',fontsize=16)
    plt.ylabel(r'$\log (\Sigma / \mathrm{arcsec}^{-2}) $',fontsize=16)
    plt.legend()
    #plt.text(-0.3,-1,r'$\Sigma = \frac{M}{\pi \mathit{a}^2} \left[ 1+ \left(\frac{r}{\mathit{a}}\right)^2\right]^{-2}$  untuk $\mathit{a} =$'+str(a)+'\,& M = '+str(M))
    plt.tight_layout()
    if savefig !=' ':
        plt.savefig(savefig,dpi=dpi)
    plt.show()
    # last command: 
    #>>> mc.plot_log_surface_density('47tuc.txt',M=37000,a=40,savefig='47tuc.jpg')