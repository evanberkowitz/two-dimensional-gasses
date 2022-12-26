Many-Body
=========

As we increase the system size beyond the one- and two- body sectors an exact calculation becomes more and more difficult.
Instead, we turn to studying thermal averages, which ultimately yields a euclidean path integral.
The discretization of the thermal circle yields a temporal direction.
We encapsulate the spatial lattice and the time direction as a :class:`~.Spacetime`.

tdg.Spacetime
-------------

.. autoclass:: tdg.spacetime.Spacetime
   :members:
   :undoc-members:
   :show-inheritance:

tdg.FermionMatrix
-----------------

The fermions are tricky to handle numerically, as they are anticommuting.
Instead we introduce an auxiliary field to linearize the action in fermion operators, and then integrate the fermions out, yielding a :class:`~.FermionMatrix`.

.. autoclass:: tdg.FermionMatrix
   :members:
   :undoc-members:
   :show-inheritance:

tdg.Action
----------

The :class:`~.Action` is combination of the fermion determinant and the gaussian piece of the auxiliary field.


.. autoclass:: tdg.action.Action
   :members:
   :undoc-members:
   :show-inheritance:


