.. _hmc:

******************
Hybrid Monte Carlo
******************

Hybrid Monte Carlo :cite:`Duane1987`, *HMC*, or *Hamiltonian Monte Carlo*, is an importance-sampling algorithm with origins in lattice field theory.
Given a real 

tdg.HMC.Hamiltonian
-------------------

.. autoclass:: tdg.HMC.Hamiltonian
   :members:
   :undoc-members:
   :show-inheritance:

tdg.HMC Integrators
-------------------

Practically, integrating Hamilton's equations of motions for a finite amount of *molecular dynamics time* ``md_time`` requires discretizing the fictitious time into an integer number of ``md_steps`` and choosing a numerical method.
Integrators trace *trajectories* through the phase space of the auxiliary fields and associated momenta.

To maintain good Markov Chain properties the integrator should be reversible; if we do not incorporate any integration Jacobian into the accept/reject step :cite:`Foreman:2021rhs`, we need a `symplectic integrator`_.
Two classics are the :class:`~.LeapFrog` and :class:`~.Omelyan` integrators.

Specifying the integrator's discretization can be done manually.
Too fine a discretization and you waste time; too coarse and the HMC energy conservation is violated too dramatically, leading to a poor acceptance rate.
One can also use an :class:`~.Autotuner` to target a particular acceptance rate.

.. autoclass:: tdg.HMC.LeapFrog
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: tdg.HMC.Omelyan
   :members:
   :undoc-members:
   :show-inheritance:

It can be painfully time consuming to pick a good `md_steps`.
One can try to automatically find a good molecular dynamics discretization instead.

.. autoclass:: tdg.HMC.Autotuner
   :members:
   :undoc-members:
   :show-inheritance:

Here is a small and easy example which tunes relatively quickly.
We start at a fine molecular dynamics discretization of 15 steps.
The model then predicts 9 steps, which is found to still be fine.
It jumps down to 1 step, which has poor acceptance but updates the model's prediction to 2 steps.
Testing 2 steps shows it has an acceptance rate that satisfies the autotuner.

.. plot:: examples/plot/autotuner.py
   :include-source:

tdg.HMC.MarkovChain
-------------------

HMC is essentially a variant of the Metropolis-Hastings algorithm :cite:`Metropolis:1953,Hastings:1970`.
The refreshing of the momentum and subsequent integration form a proposal machine for an accept/reject step given by the Boltzmann factor determined by the :class:`~.Hamiltonian` :math:`\mathcal{H}`.

.. autoclass:: tdg.HMC.MarkovChain
   :members:
   :undoc-members:
   :show-inheritance:

.. _symplectic integrator: https://en.wikipedia.org/wiki/Symplectic_integrator
