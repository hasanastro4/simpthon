API reference
=============

`simpthon`
------------

``simpthon.body`` 
-----------------

.. autoclass:: simpthon.body.Body
   :members:

.. autoclass:: simpthon.body.NBody
   :members:
   
.. autofunction:: simpthon.body.mass

.. autofunction:: simpthon.body.pos

.. autofunction:: simpthon.body.vel
  
``simpthon.integrator`` 
-----------------------
  
.. autoclass:: simpthon.integrator.integrator
   :members:
   
.. autoclass:: simpthon.integrator.leapfrog
   :show-inheritance:
   
.. autoclass:: simpthon.integrator.Euler
   :show-inheritance:
   
.. autoclass:: simpthon.integrator.RungeKutta4
   :show-inheritance:

.. autoclass:: simpthon.integrator.Forward4OSymplectic
   :show-inheritance:


``simpthon.potential`` 
----------------------

.. autoclass:: simpthon.potential.potential
   :members:
   
.. autoclass:: simpthon.potential.potentials
   :show-inheritance:
   
.. autoclass:: simpthon.potential.osilator
   :show-inheritance:
   
.. autoclass:: simpthon.potential.pointmass
   :show-inheritance:
   
.. autoclass:: simpthon.potential.plummer
   :show-inheritance:
   
.. autoclass:: simpthon.potential.jaffe
   :show-inheritance:
   
.. autoclass:: simpthon.potential.cluster_potential
   :show-inheritance:


``simpthon.hplot`` 
------------------

.. autofunction:: simpthon.hplot.plot

.. autofunction:: simpthon.hplot.plotf

