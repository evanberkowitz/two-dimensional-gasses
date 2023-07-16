.. _custom observables:

Adding Custom Observables
=========================

Observables need no parameters beyond the ensemble itself and are automatically cached.

To write a custom observable is quite simple: you write a function that takes an ensemble and computes the observable you're interested in, and `decorate`_ it with :code:`@observable`.

For example, here is the actual implementation of :func:`~.average_field`

.. literalinclude:: ../../tdg/observable/field.py

You can use the value of another observable inside your observable, just call :code:`ensemble.other`.
For example, the :func:`N <tdg.observable.number.N>` observable is just a sum of :func:`n <tdg.observable.number.n>`,

.. literalinclude:: ../../tdg/observable/number.py
   :pyobject: N

Observables will be evaluated lazily but then cached for reuse.
You should be careful to avoid creating a circular dependency.

Naming collisions are a risk.  Only the name of the actual function is used for the :class:`~.GrandCanonical` attribute/method.
So, even though they are defined in different submodules, the action :func:`~.action.S` and total spin :func:`~.spin.Spin` get different names.

Sometimes you want to share a common step between different observables but don't need to treat the commonality as a "full" observable.
In this case you can mark the function :code:`@intermediate` instead.
For example, :code:`_vorticity_vorticity` is an :code:`@intermediate`; it is used in both :func:`~.vorticity_squared` and :func:`~.vorticity_vorticity` but is otherwise not needed for further analysis.

Intermediate quantities are also not written to or read from disk.
One example intermediate quantity is :func:`the propagator G <tdg.observable.UU.G>` which takes memory like the spacetime volume squared to store.
In this case marking it :code:`@intermediate` rather than :code:`@observable` is a tradeoff between paying storage costs now or potential computation costs later.

A derived quantity can be constructed the same way, but with the :code:`@derived` decorator.
These are attached to the :class:`~.Bootstrap` class, as they require expectation values for evaluation.

.. _decorate :    https://peps.python.org/pep-0318/
.. _the implementation: /_modules/tdg/observable/number.html#n
