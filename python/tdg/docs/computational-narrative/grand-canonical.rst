.. _grand-canonical:

****************************
The Grand Canonical Ensemble
****************************

With the auxiliary field :class:`~.Action` in our hands we can sample the distribution :math:`\exp(-S)`.
An assignment for the auxiliar field is a *configuration*, and set of configurations is called an *ensemble*.
Since our :class:`~.Action` represents an approximation of :math:`\text{tr}[ \exp(-\beta(H - \mu N - h \cdot S))]` it is a `grand canonical ensemble`_.

We can :func:`~.GrandCanonical.generate` an ensemble using a sampling scheme (described in :ref:`monte-carlo`) or build one :func:`~.GrandCanonical.from_configurations` we already have.

tdg.ensemble.GrandCanonical
---------------------------

.. autoclass:: tdg.ensemble.GrandCanonical
   :members:
   :undoc-members:
   :show-inheritance:

.. _grand canonical ensemble: https://en.wikipedia.org/wiki/Grand_canonical_ensemble
