.. _parameters:

************************
Leading-Order Parameters
************************

We here focus on the low-energy theory which has scattering that can be entirely described by the scattering length.
This offers a simplification; to :ref:`tune a whole effective range expansion <tuning>` we would need to provide one parameter per shape parameter; in the leading-order case it's common to just quote the binding energy instead.

We group the parameters into those valid in the thermodynamic (infinite volume) limit.
To get a set of finite-volume we can start with a set of thermodynamic parameters and, given one finite-volume parameter, can construct a whole specification for the finite-volume parameters.


========================
Dimensionless Parameters
========================

There are three independent dimensionless parameters that stay finite in the infinite-volume limit.

tdg.parameters.Thermodynamic
----------------------------

.. autoclass:: tdg.parameters.Thermodynamic
   :members:
   :undoc-members:

To get a set of finite-volume parameters we need to provide one dimensionless quantity that knows about L.

tdg.parameters.FiniteVolume
---------------------------

.. autoclass:: tdg.parameters.FiniteVolume
   :members:
   :undoc-members:
   :show-inheritance:

=======================
Dimensionful Parameters
=======================

Finally, laboratory experiments have "real units", so to speak.  A common choice of units is electronvolts, micrometrs, and nanokelvin, {eV, µm, nK}.

We provide a consistent set of units, converting everything to eV using :math:`1=c=\hbar=k_B`.

.. automodule :: tdg.units
   :members:

With these units in hand we can construct

.. autoclass :: tdg.parameters.Dimensionful
   :members:
   :undoc-members:
