.. _custom observables:

Adding Custom Observables
=========================

Observables need no parameters beyond the ensemble itself and are automatically cached.

To write a custom observable is quite simple: you write a function that takes an ensemble and computes the observable you're interested in, and `decorate`_ it with :code:`@observable`.

For example, here is the actual implementation of :func:`~.average_field`

.. literalinclude:: ../../tdg/observable/field.py

Naming collisions are a risk.  Only the name of the actual function is used for the :class:`~.GrandCanonical` attribute/method.
So, even though they are defined in different submodules, the action :func:`~.action.S` and total spin :func:`~.spin.Spin` get different names.

A derived quantity can be constructed the same way, but with the :code:`@derived` decorator.
These are attached to the :class:`~.Bootstrap` class, as they require expectation values for evaluation.

.. _decorate :    https://peps.python.org/pep-0318/
.. _the implementation: /_modules/tdg/observable/number.html#n
