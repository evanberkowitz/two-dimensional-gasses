.. _pairing:

Pairing
=======

The pairing matrix :cite:`PhysRevA.92.033603,PhysRevLett.123.136402,PhysRevLett.129.076403` gives access to the correlations of spin-singlet Cooper pairs.

We define the back-to-back two-baryon operators

.. math::
   \tilde{\Delta}^\dagger_{S,m,k} = \tilde{\psi}^\dagger_{\alpha,k} \left(P^\dagger_{S,m}\right)_{\alpha\beta} \tilde{\psi}^\dagger_{\beta,-k}

where :math:`S` and :math:`m` are spin quantum numbers.

Singlet Pairing
---------------

.. autodata:: tdg.observable.pairing.P_singlet

Triplet Pairing
---------------

There are three possible triplet operators.  These correspond to two-fermion operators with definite :math:`S` and :math:`S_z`.

.. autodata:: tdg.observable.pairing.P_triplet_plus
.. autodata:: tdg.observable.pairing.P_triplet_zero
.. autodata:: tdg.observable.pairing.P_triplet_minus


Pairing Matrices
----------------

We define pairing matrices of operators with good spin quantum numbers,

.. math::
   \begin{align}
        \tilde{M}^{S,m}_{k,q} =&
          \left\langle \tilde{\Delta}^\dagger_{S,m,k} \tilde{\Delta}_{S,m,q} \right\rangle
        \nonumber\\
        &
        - \left(P^\dagger_{S,m}\right)_{\alpha\beta} \left(P^{\phantom{\dagger}}_{S,m}\right)_{\sigma\tau} \left\{
                \left\langle \tilde{\psi}^\dagger_{\alpha, k} \tilde{\psi}_{\tau, q} \right\rangle \left\langle \tilde{\psi}^\dagger_{\beta, -k} \tilde{\psi}_{\sigma, -q} \right\rangle
            -   \left\langle \tilde{\psi}^\dagger_{\alpha, k} \tilde{\psi}_{\sigma, -q} \right\rangle \left\langle \tilde{\psi}^\dagger_{\beta, -k} \tilde{\psi}_{\tau, q} \right\rangle
        \right\}.
   \end{align}

.. note::
   :math:`\tilde{M}` is intentionally constructed to be zero in the free case, to ensure that it only encodes interactions.

We find that eigenvalues of :math:`\frac{\left\langle\Delta^\dagger_{S,m,k} \Delta_{S,m,q} \right\rangle}{L^4 (N/2)}` are dimensionless, have good continuum limits, and are at largest intensive.

Each ``channel ∈ {singlet, triplet_plus, triplet_minus, triplet_minus}`` has the same observables.
We present the documentation in a generic way; just replace ``channel`` appropriately.

.. py:function:: tdg.observable.pairing.pair_pair_channel(ensemble)

   :math:`\left\langle \tilde{\Delta}^\dagger_{S,m,k} \tilde{\Delta}_{S,m,q} \right\rangle`.

   Configurations first, then momenta :math:`k` and :math:`q`.

.. py:function:: tdg.observable.pairing.pair_pair_eigenvalues_channel(ensemble)

   Eigenvalues of :math:`\frac{\left\langle\Delta^\dagger_{S,m,k} \Delta_{S,m,q} \right\rangle}{L^4 (N/2)}`, which are dimensionless, have good continuum limits, and are at largest intensive.

   Bootstraps first, then eigenvalues from least to greatest.  The eigenvalues are positive definite.

.. py:function:: tdg.observable.pairing.pairing_channel(ensemble)

   :math:`\frac{M}{L^4 (N/2)}`.

   Bootstraps first, then momenta :math:`k` and :math:`q`.

.. py:function:: tdg.observable.pairing.condensate_fraction_channel(ensemble)

   Largest eigenvalue of :math:`\frac{M}{L^4 (N/2)} = \texttt{pairing_channel}`.

   One number per bootstrap.

.. py:function:: tdg.observable.pairing.pairing_wavefunction_channel(ensemble)

   Eigenvector of :math:`\frac{M}{L^4 (N/2)} = \texttt{pairing_channel}` that has eigenvalue :func:`condensate_fraction_channel`.

   Bootstrap first, then momentum.
