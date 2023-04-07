.. _observables:

********************
Physical Observables
********************

Observables are physical quantities that can be measured on ensembles.
We distinguish between *primary observables* and *derived quantities*, using language from :cite:`Wolff:2003sm`.
A primary observable can be measured directly on a single configuration.
A derived quantity is a generally nonlinear function of primary observables which can only be estimated using a whole ensemble.

For example, while the :func:`baryon number <tdg.observable.number.n>` is a primary observable, the :func:`two-point fluctuations <tdg.observable.nn.density_density_fluctuations>`

.. math::

   \left\langle \delta n*\delta n \right\rangle = \left\langle n*n \right\rangle - \left\langle n \right\rangle * \left\langle n \right\rangle

is not because it requires combining different expectation values.

Primary observables are `descriptors`_ of the :class:`~.GrandCanonical` ensemble (and therefore also of the :class:`~Canonical` ensemble).
Derived quantities, because they can only be estimated from expectation values, are `descriptors`_ of the :class:`~.Bootstrap`.
You can add :ref:`custom observables <custom observables>` without modifying the internals of these classes.

We adopt a convention where local densities start with a lowercase letter and global quantities with an Uppercase.


.. toctree::
   :maxdepth: 2

   propagator.rst
   action.rst
   fermi.rst
   number.rst
   spin.rst
   momentum_occupancy.rst
   energy.rst
   contact.rst
   currents.rst
   vorticity.rst
   custom-observables.rst

.. _descriptors : https://docs.python.org/3/howto/descriptor.html
