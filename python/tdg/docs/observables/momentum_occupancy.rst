.. _momentum_occupancy:

Momentum Occupancy
==================

The momentum occupancies are like the corresponding fermion bilinears but are not site-local but momentum-local.

.. warning::
   These are NOT the fourier trasnforms of :func:`~.n` and :func:`~.spin`!
   Should you want those quantities, just measure them and take the fourier transform.

.. autofunction:: tdg.observable.momentum_occupancy.n_momentum
.. autofunction:: tdg.observable.momentum_occupancy.spin_momentum

