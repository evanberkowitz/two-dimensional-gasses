#!/usr/bin/env python

import numpy as np
import tdg

def section(s):
    print(f"\n\n####\n#### {s.upper()}\n####\n")

section("Pauli matrices")
print("The four Pauli matrices are")
print(tdg.PauliMatrix)

section("spatial lattice")

lattice = tdg.Lattice(5, 3)
x = lattice.vector()
print(f"The spatial lattice is {lattice}.")
print(f"Spatial vectors have shape {x.shape}.")
print(f"The adjacency tensor has shape {lattice.adjacency_tensor.shape}; as a matrix it has shape {lattice.tensor_linearized(lattice.adjacency_tensor).shape}")
if( (lattice.adjacency_matrix == lattice.adjacency_matrix.T).all() ):
    print("The adjacency matrix is symmetric.")
else:
    print("The adjacency matrix is not symmetric.")
print(f"The sites have {np.sum(lattice.adjacency_matrix, axis=0)} neighbors.")

print(f"The integer x coordinates of this lattice are {lattice.x}")
print(f"The integer y coordinates of this lattice are {lattice.y}")
print(f"We also provide a broadcastable set of coordinates:")
print(f" lattice.X\n{lattice.X}")
print(f" lattice.Y\n{lattice.Y}")
x[0,0] = 1
k = lattice.fft(x)
print(f"The fourier transform of\n{x}\nis\n{k};\ninverting the fourier transform gives\n{lattice.ifft(k)}.")
if( np.abs( np.sum(np.conj(k) * k) - np.sum(np.conj(x) * x) ) < 1e-14):
    print(f"The fourier transform is unitary.")
else:
    print(f"You should think hard about the normalization of the fourier transform.")



section("spacetime")

nt   = 32
spacetime = tdg.Spacetime(nt, lattice)
v = spacetime.vector(complex)

print(f"The spacetime lattice is {spacetime}.")
print(f"Spacetime vectors have shape {v.shape}.")
print(f"The time coordinate takes values {spacetime.t}")
print(f"Broadcastable coordinates are provided .T, .X, and .Y")
print(f"A linearized list of coordinates has length {len(spacetime.coordinates)} which equals the number of sites {spacetime.sites}.")



section("Lego Spheres")

print("A LegoSphere just requires a radius and a Wilson coefficient.")
contact = tdg.LegoSphere([0,0], 2)
print(f"This sphere {contact} has radius {contact.r} and coefficient {contact.c}.")
print("The default coefficient is 1, and LegoSpheres may be multiplied by coefficients.")
sphere = 3 * tdg.LegoSphere([0,1])
print(f"This sphere {sphere} has radius {sphere.r} and coefficient {sphere.c}.")
print(f"The points on this LegoSphere are {sphere.points} from the center.")



section("Potentials")

print("A potential requires a lattice and a list of LegoSpheres.")
V = tdg.Potential(lattice, contact, sphere)
print(f"For example, {V=}.")
print(f"It has a spatial representation with shape {V.spatial.shape}.")
print(f"The inverse of the spatial representation has shape {V.inverse.shape}.")

one = lattice.tensor_linearized(np.einsum("abcd,cdef->abef", V.spatial, V.inverse))
zero = one - np.eye(lattice.sites)
if( (np.abs(zero) < 1e-14).all() ):
    print("We can check that V.spatial and V.inverse are floating-point inverses.")
else:
    print("However, V.spatial and V.inverse fail to be inverses of one another.")

print(f"The eigenvalues of V are\n{V.eigvals}.")
print(f"The spatial representation, inverse, and eigenvalues are cached, so it's cheap to reuse them.")
if( (V.eigvals >= 0).all() ):
    print("All the eigenvalues of V are positive, so this potential is amenable to our Hubbard-Stratonovich transformation.")
else:
    print("Not all the eigenvalues of V are positive, so the formal Hubbard-Stratonovich transformation is invalid for this potential.")
print(f"Since the contact potential needs to be treated specially, the potential provides a way to get the C0 coefficient {V.C0=}")

