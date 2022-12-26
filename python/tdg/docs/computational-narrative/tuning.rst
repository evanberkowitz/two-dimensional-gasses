******
Tuning
******

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

