Background
==========

Ultracold atomic experiments can squeeze nonrelativistic fermionic atoms in one dimension, constructing a two-dimensional Fermi gas.
The leading-order effective field theory :cite:`Beane:2022wcn`

.. math::
   \begin{align}
    \mathcal{L} = \psi^\dagger\left(i\partial_0 + \frac{\nabla^2}{2M}\right)\psi - \frac{C_0}{2} (\psi^\dagger \psi)^2
   \end{align}

yields a Hamiltonian

.. math::
   \begin{align}
    H = - \psi^\dagger \frac{\nabla^2}{2M} \psi + \frac{C_0}{2} (\psi^\dagger \psi)^2
   \end{align}

invariant under scale and nonrelativistic-conformal transformations :cite:`Jackiw:1991je`.

The purpose of the tdg library is to compute observables of many-body systems in thermodynamic equilibrium governed by this Hamiltonian.
We pursue a lattice field theory-like approach, discretizing the two-dimensional space via a regular square lattice with periodic boundary conditions.

tdg.Lattice
-----------

A *lattice* is a two-dimensional square grid of points, a discretization of space.
For simplicity we restrict our attention to an isotropic lattice with the same number of points in each rectilinear direction; a list of :func:`~Lattice.coordinates` is provided.
We use periodic boundary conditions, and therefore can validly :func:`~Lattice.mod` points into the :func:`~Lattice.coordinates` of the lattice.
The periodic boundary conditions also inform the computation of :func:`~Lattice.distance_squared`, because the distance might be shorter than the naive route might suggest.
A lattice knows how big a spatial :func:`~Lattice.vector` is and knows how to Fourier transform them via :func:`~Lattice.fft` and :func:`~Lattice.ifft`.

We will use the *perfect* nonrelativistic dispersion relation, and therefore can use the :func:`~Lattice.adjacency_matrix` to construct the kinetic matrix :func:`~Lattice.kappa`.

The lattice has translational invariance and a :math:`D_4` point group.

.. autoclass:: tdg.lattice.Lattice
   :members:
   :undoc-members:
   :show-inheritance:

tdg.LegoSphere
--------------

On a square lattice, 'rotationally symmetric' has a limited meaning, because the lattice breaks SO(2) in both the UV (via the discretization) and in the IR (via the boundary conditions).
However, we still reliably have the smaller :math:`D_4` symmetry, and the trivial ireducible representation of :math:`D_4` is called :math:`A_1`.
For more details on this symmetry, and exactly why these LegoSpheres are :math:`A_1`, see :doc:`D4`.

.. autoclass:: tdg.LegoSphere.LegoSphere
   :members:
   :undoc-members:
   :show-inheritance:

tdg.potential
-------------

.. automodule:: tdg.potential
   :members:
   :undoc-members:
   :show-inheritance:


