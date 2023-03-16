.. _observables:

***********
Observables
***********

Observables are physical quantities that can be measured on ensembles.
They are smuggled into the :class:`~.GrandCanonical` ensemble as `descriptors`_, so that you may call :code:`ensemble.observable`.
This allows you to :ref:`add custom observables <custom observables>`.

The Propagator
==============

Many observables are expectation values of fermionic operators.
Once normal-ordered, those operators may be replaced by a sum of Wick contractions.
Each pair of contracted fermionic operators yields a propagator.

.. autofunction:: tdg.observable.UU.G

Local and Global Observables
============================

We adopt a convention where local densities start with a lowercase letter and global quantities with an Uppercase.

Simple auxiliary field observables
----------------------------------

.. autofunction:: tdg.observable.field.average_field

The Action
----------

.. note:: *NOT* named :code:`.Action` because ensembles already have an :attr:`~.GrandCanonical.Action`.

.. autofunction:: tdg.observable.action.S

Baryon Number
-------------

These measure the local and global baryon number.
Even though they are *callable*, under the hood they are cached.

.. autofunction:: tdg.observable.number.n
.. autofunction:: tdg.observable.number.N

Spin
----

.. autofunction:: tdg.observable.spin.spin
.. autofunction:: tdg.observable.spin.Spin

The Contact
-----------

Tan's *contact* universally describes the large-momentum tails of certain correlation functions.

.. autofunction:: tdg.observable.contact.doubleOccupancy
.. autofunction:: tdg.observable.contact.DoubleOccupancy
.. autofunction:: tdg.observable.contact.Contact

Two-Point Correlations
======================

We can compute equal-time two-point correlation functions.
By translation invariance these may be reduced to a function of one space variable.

.. autofunction:: tdg.observable.nn.nn
.. autofunction:: tdg.observable.ss.ss

.. _custom observables:

Adding Custom Observables
=========================

Observables come in two types: *pure observables*, which need no parameters, and *callable observables* which are passed arguments.

In other words, pure observables only need the ensemble itself.
Pure observables are automatically cached.
A *callable* observable requires arguments and has no built-in caching, though there may be caching as an implementation detail.

To write a custom observable is quite simple: you write a function that takes an ensemble and computes the observable you're interested in, and `decorate`_ it with :code:`@observable`.

For example, here is the actual implementation of :func:`~.average_field`

.. literalinclude:: ../../tdg/observable/field.py

Writing a callable observable is much the same, but rather than just a single :code:`ensemble` argument, the :code:`@observable` should take additional parameters.
See `the implementation`_ of :func:`~.number.n` for an example implementation of a callable :code:`@observable` which caches under the hood.

Naming collisions are a risk.  Only the name of the actual function is used for the :class:`~.GrandCanonical` attribute/method.
So, even though they are defined in different submodules, the action :func:`~.action.S` and total spin :func:`~.spin.Spin` get different names.



.. _descriptors : https://docs.python.org/3/howto/descriptor.html
.. _decorate :    https://peps.python.org/pep-0318/
.. _the implementation: /_modules/tdg/observable/number.html#n
