#!/usr/bin/env python

import tdg

def section(s):
    print(f"\n\n####\n#### {s.upper()}\n####\n")


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
beta = 10
nt   = 32
spacetime = tdg.Spacetime(beta, nt, lattice)
v = spacetime.vector()

print(f"The spacetime lattice is {spacetime}.")
print(f"Spacetime vectors have shape {v.shape}.")



section("Lego Spheres")
print("A LegoSphere just requires a radius and a Wilson coefficient.")
S = tdg.LegoSphere([0,0], 2)
print(f"This sphere {S} has radius {S.r} and coefficient {S.c}.")
print("The default coefficient is 1, and LegoSpheres may be multiplied by coefficients.")
T = 3 * tdg.LegoSphere([0,1])
print(f"This sphere {T} has radius {T.r} and coefficient {T.c}.")
print(f"The points on this LegoSphere are {T.points} from the center.")


