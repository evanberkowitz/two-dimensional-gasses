.. _number:

Number
======

Baryon Number
-------------

These measure the local and global baryon number.
Even though they are *callable*, under the hood they are cached.

.. autofunction:: tdg.observable.number.n
.. autofunction:: tdg.observable.number.N

Bosonic Estimators
------------------

Because we do the Hubbard-Stratanovich transformation in the number channel we can use a Ward-Takahashi-like identity to estimate the number density from the auxiliary field.

.. autofunction:: tdg.observable.number.n_bosonic
.. autofunction:: tdg.observable.number.N_bosonic

Density-density two-point functions
-----------------------------------

.. autofunction:: tdg.observable.nn.nn
.. autofunction:: tdg.observable.nn.density_density_fluctuations

