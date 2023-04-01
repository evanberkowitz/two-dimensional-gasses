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


.. _decorate :    https://peps.python.org/pep-0318/
.. _the implementation: /_modules/tdg/observable/number.html#n
