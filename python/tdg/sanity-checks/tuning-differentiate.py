#!/usr/bin/env python

import torch
torch.set_default_tensor_type(torch.DoubleTensor)

import tdg

print(f'''This sanity check shows that the procedure for calculating the derivative of
the Wilson coefficients with respect to the effective range parameters is correct.

The strategy will be to compute three tunings:
      - one for the  ERE we're actually interested in
      - one for that ERE but with a slightly shifted scattering length a
      - one for that ERE but with a slightly shifted leading analytic momentum dependence r

WARNING: This takes some time to run! ~ 90 minutes on my laptop.
         It may be simple to improve this by optimizing the tuning routines.

''')
original= torch.tensor([1.0, 0.0])
radii   = [[0,0], [0,1]]
lattice = tdg.Lattice(7)

tuning = tdg.Tuning(
        ere     =   tdg.EffectiveRangeExpansion(original),
        lattice =   lattice,
        radii   =   radii,
        # from prior computation:
        starting_guess = torch.tensor([-4.7341,  1.9069], dtype=torch.float64),
        )

print(f"We start by tuning a starting set of Wilson coefficients C that correspond to")
print(f"{tuning}.")
print(f"The tuned coefficients are...")
print(tuning.C)

print(f"With the tuning now in hand we can differentiate them with respect to the ERE parameters.")
print(f"{tuning.dC_dERE}")
print(f"Each row is one Wilson coefficient; each column is one ERE parameter.")
print(f"The first column is the derivative with respect to the (dimensionless) scattering length a.")

print(f"\n\nNow we attempt to confirm these derivatives by finite differencing.")

print(f"\n\nFirst we detune the scattering length only, changing the ERE parameters by")
da = torch.tensor([0.001, 0.])
print(f"{da=}")
print(f"The a-detuned coefficients are...")
a_detuned = tdg.Tuning(
        ere     =   tdg.EffectiveRangeExpansion(original + da),
        lattice =   lattice,
        radii   =   radii,
        starting_guess = tuning.C,
        )
print(a_detuned.C)
print(f"The finite difference estimate for the derivative is")
finite_difference_a = (a_detuned.C - tuning.C) / da[0]
print(finite_difference_a)
print(f"The ratio of the finite differencing and exact derivative is")
ratio_a = tuning.dC_dERE[:,0] / finite_difference_a
print(ratio_a)

print(f"\n\nNext we detune the range only, changing the ERE parameters by")
dr = torch.tensor([0., 0.001])
print(f"{dr=}")
print(f"The r-detuned coefficients are...")
r_detuned = tdg.Tuning(
        ere     =   tdg.EffectiveRangeExpansion(original + dr),
        lattice =   lattice,
        radii   =   radii,
        starting_guess = tuning.C,
        )
print(r_detuned.C)
print(f"The finite difference estimate for the derivative is")
finite_difference_r = (r_detuned.C - tuning.C) / dr[1]
print(finite_difference_r)
print(f"The ratio of the finite differencing and exact derivative is")
ratio_r = tuning.dC_dERE[:,1] / finite_difference_r
print(ratio_r)


