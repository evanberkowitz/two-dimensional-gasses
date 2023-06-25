.. _grand-canonical:

****************************
The Grand Canonical Ensemble
****************************

Our general approach to the quantum many-body problem is to discretize an approximation of the partition function :math:`Z = \text{tr}[ \exp(-\beta(H - \mu N - h \cdot S))]`, a `grand canonical ensemble`_, via a `Suzuki-Trotter decomposition`_ or simply a *Trotterization*.

For a discretization of space into :math:`N_x` sites on a side we regroup all the dimensionful parameters into

.. math::
   \begin{align}
    \tilde{H} &= HML^2
    &
    \tilde{\beta} &= \beta/ML^2
    \\
    \tilde{\mu} &= \mu ML^2
    &
    \tilde{\vec{h}} &= \vec{h} ML^2
   \end{align}

With an on-site interaction, holding :math:`\tilde{a}=2\pi a/L` fixed yields a dimensionless Hamiltonian that depends only on :math:`N_x` and the scattering through the :class:`~.AnalyticTuning`.
This convention makes it simple to understand the Hamiltonian limit (:math:`N_t\rightarrow\infty` holding :math:`\tilde{\beta}` fixed) in which the Trotterization disappears, and the spatial continuum limit (:math:`N_x\rightarrow\infty` holding the physical parameters, and therefore, :math:`\tilde{a}`, :math:`\tilde{\beta}`, :math:`\tilde{\mu}`, :math:`\tilde{\vec{h}}` fixed).

The Fermionic Discretization
============================

Trotterizing the *thermal circle* yields a euclidean time, which can be combined with the :class:`~.Lattice` into a :class:`~.SpaceTime`.

tdg.Spacetime
-------------

.. autoclass:: tdg.spacetime.Spacetime
   :members:
   :undoc-members:
   :show-inheritance:

tdg.FermionMatrix
-----------------

Fermions are tricky to handle numerically, as they are anticommuting.
Instead we introduce an auxiliary field to linearize the action in fermion operators, and then integrate the fermions out, yielding a :class:`~.FermionMatrix`.
Including the determinant of the :class:`~.FermionMatrix` in the sampling includes the fermions, even though the auxiliary fields are bosonic.

.. note::
   The arguments are dimensionless, as above!
   Henceforth with only write the tilde if clarity requires.

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


The Grand Canonical Ensemble
============================

With the auxiliary field :class:`~.Action` in our hands we can sample the distribution :math:`\exp(-S)`.
An assignment for the auxiliar field is a *configuration*, and set of configurations is called an *ensemble*.

We can :func:`~.GrandCanonical.generate` an ensemble using a sampling scheme (described in :ref:`hmc`) or build one :func:`~.GrandCanonical.from_configurations` we already have.

tdg.ensemble.GrandCanonical
---------------------------

.. autoclass:: tdg.ensemble.GrandCanonical
   :no-special-members:
   :members: Action, from_configurations, generate, measure, to_h5, extend_h5, from_h5, cut, every, binned, bootstrapped
   :show-inheritance:

.. _grand canonical ensemble: https://en.wikipedia.org/wiki/Grand_canonical_ensemble
.. _Suzuki-Trotter decomposition: https://en.wikipedia.org/wiki/Hamiltonian_simulation#Product_formulas
