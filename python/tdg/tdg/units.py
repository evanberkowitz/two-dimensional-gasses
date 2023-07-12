import torch

# The point is to be able to write things in preferred experimentalist units, {eV, nK, µm}.
# We will adopt the convention that everything should be cast into eV.
eV = torch.tensor(1., requires_grad=False)
r'''1 electronvolt'''

# Therefore we need conversions for nK and µm.

# To convert from length to energy we need to use 1 = hbar c.
# h and c are set exactly in SI units.
# In Mathematica,
#
#   h = Select[
#      EntityList@EntityClass["PhysicalConstant", "SIExact"],
#      EntityValue[#, "Name"] == "Planck constant" &][[1]]
#   c = Select[
#      EntityList@EntityClass["PhysicalConstant", "SIExact"],
#      EntityValue[#, "Name"] == "speed of light" &][[1]]
#   hbarc = N[UnitConvert[h["Value"] c["Value"], "eV*µm"]/(2 \[Pi]), 16]
#
hbarc = torch.tensor(0.1973269804593025, requires_grad=False) # eV µm
µm = 1./hbarc/eV
r'''1 = ℏc = 0.1973269804593025 eV µm'''
micrometers = µm
r'''Alias for :const:`tdg.units.µm`'''

# To convert from temperature to energy we need Boltzmann's constant
# Similarly kB is set exactly in SI units.
#
#   kB = Select[
#      EntityList@EntityClass["PhysicalConstant", "SIExact"],
#      EntityValue[#, "Name"] == "Boltzmann constant" &][[1]]
#   kB = N[UnitConvert[kB["Value"], "eV/nK"], 16]
kB = torch.tensor(8.617333262145177e-14, requires_grad=False) # eV/nK
nK = kB*eV
r'''1 = kB = 8.617333262145177e-14 eV / nK'''

# Finally, many references quote atomic masses in Daltons (Da), or unified atomic mass units (u),
# one twelfth of the rest mass of unbound, neutral, ground-state carbon-12.
dalton = torch.tensor(931.49410242e+6, requires_grad=False) # eV/c^2 # https://en.wikipedia.org/wiki/Dalton_(unit)
# There is an uncertainty of (28) on the last digits.

# We collect a variety of masses for convenience.
Mass = {
    "6Li": 6.0151228874*dalton, # https://en.wikipedia.org/wiki/Isotopes_of_lithium#List_of_isotopes
}
r'''
A dictionary of masses of select neutral atoms.
'''
