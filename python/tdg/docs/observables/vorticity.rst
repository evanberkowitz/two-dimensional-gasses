.. _vorticity:

Vorticity
=========

Baryon Number Vorticity
-----------------------

.. autofunction:: tdg.observable.vorticity.vorticity
.. autofunction:: tdg.observable.vorticity.vorticity_squared

.. autofunction:: tdg.observable.vorticity.vorticity_vorticity

There is a great deal of subtlety involved in understanding the low-energy behavior of vortex correlations.
Because by periodic boundary conditions :math:`\sum_r \texttt{vorticity_vorticity} = 0` and its Fourier transform goes like momentum squared at high momentum, we need to study moments.
We define moments :math:`B`,

.. math::
   B_n(k) = \int d^2r\; e^{-i k r} |r|^n \Omega(r) 

:math:`B_n(k)` has dimensions :math:`[M^{-2} L^{-(6-n)}]`.

So, :math:`M^2 L^{6-n} B_n(k)` is dimensionless.  We can divide by appropriate powers of :math:`2\pi N = k_F^2 L^2` to eliminate :math:`L`.

.. autofunction:: tdg.observable.vorticity.b2_by_kF4
.. autofunction:: tdg.observable.vorticity.b4_by_kF2
.. autofunction:: tdg.observable.vorticity.b6

