.. _tuning:

******
Tuning
******

==============================
Matching to Observable Physics
==============================

Given the form of the Hamiltonian we are trying to construct, how do we pick the parameters it contains?
This is a particularly pressing question as the Hamiltonian knows about UV things like the discretization and unphysical IR things like the finite volume.
One way is to try to match some observable to a fixed continuum value / experimental observation.

We chose to try to match two-particle *scattering data*, in the form of the Effective Range Expansion.
Adjusting Hamiltonian parameters to match the desired physics is called *tuning* the Hamiltonian [coefficients, or parameters].

tdg.EffectiveRangeExpansion
---------------------------

.. autoclass:: tdg.ere.EffectiveRangeExpansion
   :members:
   :undoc-members:
   :show-inheritance:

===============================
Tuning More Than One LegoSphere
===============================

We first describe the generic method for tuning a potential.
If the effective range is entirely parameterized by the scattering length, and you are happy tuning the on-site interaction [a :class:`~.LegoSphere` with radius (0,0)] then you should use the :class:`~.AnalyticTuning` and need not know the generic story.

tdg.Luescher
------------

The :class:`~tdg.ere.EffectiveRangeExpansion` gives information about continuum, infinite-volume two-particle scattering.
Even the continuum limit of our Hamiltonian represents a finite-volume.
The Lüscher quantization condition bridges the gap between infinite-volume scattering and finite-volume energy levels.

.. automodule:: tdg.Luescher
   :members:
   :undoc-members:
   :show-inheritance:

tdg.a1.ReducedTwoBodyHamiltonian
--------------------------------

Luckily, the two-body sector of our Hamiltonian is small enough that it does not require stochastic methods to analyze.
In fact, we can construct the two-body Hamiltonian in the :math:`A_1` sector explicitly; restricting to :math:`A_1` saves between a factor of 4 and 8 in the number of states.
By finding eigenvalues of the Hamiltonian we get the finite-volume information we need to use the Lüscher quantization condition to get at the infinite-volume scattering.

.. automodule:: tdg.a1
   :members:
   :undoc-members:
   :show-inheritance:

tdg.Tuning
----------

Finally, with the known targeted two-body physics (in the form of the :class:`~tdg.ere.EffectiveRangeExpansion`), all the information except the Wilson coefficients needed to construct the :class:`~tdg.a1.ReducedTwoBodyHamiltonian`, and the Lüscher :class:`~tdg.Luescher.Zeta2D` implementing the quantization condition, we can attempt to *tune* the Hamiltonian.

.. autoclass:: tdg.tuning.Tuning
   :members:
   :undoc-members:
   :show-inheritance:

==============================
Tuning The On-Site Interaction
==============================

In the case of an on-site interaction (with radius (0,0)), we can evaluate the :math:`T` matrix as a geometric series given by a bubble sum.
Then, using the two-dimensional quantum-mechanical expression for the :math:`T` matrix in terms of :math:`\cot \delta`, we can match and solve for :math:`C_{(0,0)}`.

The result is *almost* the standard two-dimensional result.
It is slightly modified by taking the square lattice of the Hamiltonian seriously.
Typically the cutoff is applied to the momentum squared; instead we should cut off each component independently.
The result is given by eg. Ref. :cite:`Korber:2019cuq`; see (C19).

tdg.AnalyticTuning
------------------

.. autoclass:: tdg.tuning.AnalyticTuning
   :members:
   :undoc-members:
   :show-inheritance:

