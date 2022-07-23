##
#   PRIMARY NOTE :
#   
#   This file enables Python to recognise this as a package.
#   
#   
#   ADDITIONAL NOTE:
#   -----
#   
#   One way to enable Python to recognise this module from any directory
#   
#   is by adding a .pth file in the site package where Python is installed.
#   
#   For Example:
#
#   The path of this package has been added to
#
#   C:\Users\ASUS\Anaconda3\Lib\site-packages\hasanpath.pth
#
#   where hasanpath.pth file consists only of package directories row by row. 
#
#   In this case, D:\HPython.  
#   -----
#   
#   another way is by setup 

from . import body, hplot, integrator, makecluster, orbit, osilator_harmonik 
from . import potential, test_pot, time_step

#from .body import *  
#from .hplot import *
#from .integrator import *
#from .makecluster import *
#from .orbit import * 
#from .osilator_harmonik import *
#from .potential import *
#from .test_pot import *
#from .time_step import * 