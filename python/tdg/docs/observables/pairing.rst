.. _pairing:

Pairing
=======

The pairing matrix :cite:`PhysRevA.92.033603,PhysRevLett.123.136402,PhysRevLett.129.076403` gives access to the correlations of spin-singlet Cooper pairs.

We define pairing matrices of operators with good spin quantum numbers,

.. math::
   M^{S,m}_{k,q} = \left\langle \tilde{\Delta}^\dagger_{S,m,k} \tilde{\Delta}_{S,m,q} - \delta_{kq} \tilde{\Delta}^\dagger_{S,m,k} \tilde{\Delta}_{S,m,k} \right\rangle

where the back-to-back two-baryon operators are given by

.. math::
   \tilde{\Delta}^\dagger_{S,m,k} = \tilde{\psi}^\dagger_{\alpha,k} \left(P^\dagger_{S,m}\right)_{\alpha\beta} \tilde{\psi}^\dagger_{\beta,-k}

The pairing matrix :math:`M` is Hermitian.

Singlet Pairing
---------------

.. autofunction:: tdg.observable.pairing.pairing_singlet

Triplet Pairing
---------------

There are three possible triplet operators.

.. autofunction:: tdg.observable.pairing.pairing_triplet_plus
.. autofunction:: tdg.observable.pairing.pairing_triplet_zero
.. autofunction:: tdg.observable.pairing.pairing_triplet_minus

