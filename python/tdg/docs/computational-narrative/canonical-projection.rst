
.. _canonical:

**********************
The Canonical Ensemble
**********************

A `canonical ensemble`_ with fixed conserved quantities may be built from the :class:`~.GrandCanonical` by projection,

.. math::
    \text{tr}[e^{-\beta H}]_{N, S_h} = \text{tr}[ e^{-\beta H} P_N P_{S_h} ]

where :math:`P` indicates a projection operator of the indicated quantum number to a particular value.
A :class:`~.Sector` is a choice of for the conserved quantities; one may construct sectors with fixed particle number :math:`N`, spin projection :math:`S_h` or both.

.. warning::
   Canonical projection is *very* expensive.
   In addition to the normal measurement cost the cost scales with an *extra* factor the spatial volume for every quantum number projected.
   Constructing multiple :class:`~.Sector`s from a :class:`~.Canonical` ensemble shares work under the hood and amortizes these costs over the sectors.
   Accelerated techniques exist :cite:`Gilbreth20151` but are not (yet?) implemented here, and offer better, but still large, costs!

tdg.ensemble.Canonical
----------------------

.. autoclass:: tdg.ensemble.Canonical
   :no-special-members:
   :members: Sector
   :show-inheritance:

tdg.ensemble.Sector
-------------------

.. autoclass:: tdg.ensemble.Sector
   :no-special-members:
   :members: weight, Sh
   :show-inheritance:


Because the reweighting leverages the :class:`~.GrandCanonical` observables, we can test the grand-canonical sampling and observables on small few-body examples where we can calculate exactly.
See, for example, ``sanity-checks/canonical-double-occupancy.py``, where we compare an exact two-body spin-0 calculation of the :func:`~.DoubleOccupancy` on a 3×3 lattice at finite temperature.



.. _canonical ensemble: https://en.wikipedia.org/wiki/Canonical_ensemble
