#!/usr/bin/env python

import torch
import tdg
from tdg.Luescher import Zeta2D

from functools import cached_property

class Tuning:
    r'''
    A :class:`~.Tuning` is a set of Wilson coefficients :math:`C` that reproduce chosen physics.
    In our case the physics being held fixed is the effective range expansion, or ERE.

    The tuning should fix all the free parameters in the Hamiltonian, so we should expect
    to tune as many coefficients as we have ERE parameters.  Our interaction is built from
    :class:`~.LegoSphere` s, each described by a radius and a Wilson coefficient.
    The purpose of tuning is to `discover` the Wilson coefficients, but we must provide the radii,
    a free choice.  Different radii give different Wilson coefficients, of course.

    If you have a ballpark estimate (from tuning to a similar ERE, or tuning on a similar lattice, for example),
    you can pass these as the ``starting_guess``.  If you know the actual results of the tuning you are describing
    you can save a lot of computational time by passing them at construction as ``C``;
    you can still calculate their derivative with respect to the ERE parameters.

    Parameters
    ----------
        ere:        :class:`~.EffectiveRangeExpansion`
        lattice:    :class:`~.Lattice` 
        radii:      list
            Should be one radius for each sphere and each parameter in ``ere`` formatted as ``[[x1, y1], [x2, y2], ...]``.
        zeta:       :class:`~.Zeta2D`
            Used in the finite-volume quantization condition.
        starting_guess: torch.tensor or None
            One starting guess for each ``C`` in the order that of the ``radii``.
        C:          torch.tensor or None
            If not `None`, the tuning computation will be skipped, though the other properties can be computed.
            Provide one for each ``ere`` parameter, in the order of the ``radii``.
    '''

    def __init__(self, ere, lattice, radii, zeta=Zeta2D(), starting_guess=None, C=None):
        
        self.ere = ere
        self.Lattice = lattice
        self.radii = radii
        self.zeta  = zeta
        self.start = starting_guess if starting_guess is not None else torch.ones(len(self.radii))
        
        assert len(self.radii) == len(self.ere.parameters), f'You must tune as many LegoSpheres ({len(self.radii)}) as there are ERE parameters ({len(self.ere.parameters)}).'

        # To avoid the expense of and jitter in retuning:
        if C is not None:
            self.C = C
            assert len(self.C) == len(self.radii), f'Radii ({len(self.radii)}) and coefficients ({len(self.C)}) must be one-to-one.'

    def __str__(self):
        return f'Tuning({self.ere}, {self.Lattice}, {self.radii})'

    # The tuning proceeds via the inverse Lüscher method.  The idea is that Lüscher's finite-volume quantization
    # condition translates finite-volume energy levels into scattering data.  Since we have two-body scattering data
    # we wish to reproduce, encoded in the ERE, we need to adjust Hamiltonian parameters until the energy levels come
    # out right.
    #
    # One subtlety is that in infinite volume we have good rotational symmetry and the angular momentum L is conserved.
    # But the finite volume with periodic boundary conditions we've got does not have full rotational symmetry; the 
    # SO(2) is broken down to the D_4 finite subgroup.
    #
    # While SO(2) has infinitely many irreps D_4 does not, so we know that each D_4 irrep will contain a mix of SO(2)
    # angular momentum irreps.
    #
    # While Lüscher's formula can be constructed for any D_4 irrep, it is most simple in the A_1 sector, which is
    # "like the S-wave" in the group-theory sense that it is the trivial irrep of dimension 1 and in the physics
    # sense that it is rotationally symmetric (but only D_4 rotations of the lattice are admissible)
    #
    # [
    #   Actually, not just any D_4 irrep.  The quantization condition depends on the total linear momentum;
    #   the quantization condition is in the little group of D_4 which preserves this conserved quantity.
    #   However, when the total momentum is zero the little group is the whole group itself!
    # ]
    #
    #
    # We begin by using Lüscher's finite-volume quantization condition to turn the ERE into energy levels.
    #
    @cached_property
    def _tuning_energies(self):
        # Caution: computationally intensive!
        # This solves an "inverse problem".
        return self.ere.target_energies(
                self.Lattice,
                len(self.ere.parameters),
                zeta=self.zeta,
                lr=1e-4, epochs=100000,
                )
    #
    # and now our problem is reduced to finding Hamiltonian parameters that
    # produce those levels in the two-body A1 sector.
    #
    # We construct a tunable two-body A1-projected Hamiltonian
    #
    @cached_property
    def _A1(self):
        return tdg.ReducedTwoBodyA1Hamiltonian(self.Lattice, [tdg.LegoSphere(r) for r in self.radii])
    #
    # which can be tuned until it yields the Wilson coefficients.
    #
    @cached_property
    def C(self): 
        r'''
        The Wilson coefficients :math:`C` of :class:`~.LegoSphere` s with the given ``radii``.

        .. note::
            Can be time-consuming to compute if unknown!
        '''
        return self._A1.tuning(
            self._tuning_energies.clone().detach(),
            start=self.start.clone().detach().requires_grad_(True),
            lr=1e-4, epochs=10000,
        )
    #
    # We can use these C to construct a potential,
    #
    @cached_property
    def Potential(self):
        r'''
        A potential constructed from LegoSpheres with the given ``radii`` and the tuned Wilson coefficients ``C``.
        '''
        return tdg.Potential(*[c*tdg.LegoSphere(r) for c, r in zip(self.C, self.radii)])
    #
    # and, with additional information, an action.
    #
    def Action(self, nt, beta, mu=torch.tensor(0.), h=torch.zeros(3), fermionMatrix=tdg.fermionMatrix.FermionMatrix):
        r'''
        An action with a :class:`~.Spacetime` given by the provided ``nt`` and the tuning's `Lattice`, a potential given by :attr:`~.Potential`, and other :class:`~.Action` parameters.

        Parameters
        ----------
            nt:     int
                Timeslices.
            others:
                Forwarded to :class:`~.Action` as expected.

        Returns
        -------
            :class:`~.Action` with ``.Tuning`` set.
        '''
        A = tdg.Action(
                tdg.Spacetime(nt, self.Lattice),
                self.Potential,
                beta,
                mu=mu,
                h=h,
                fermion=fermionMatrix
                )
        # additionally store the tuning itself.
        A.Tuning= self
        return A
    #
    # If this were our only goal, we would be finished.
    #
    # However, at least one observable we care about---the contact---depends on
    # knowing the derivative of the Wilson coefficients with respect to the
    # (log of) the scattering length.
    #
    # Calculating this derivative is pretty tricky.  We cannot "simply" use autograd
    # because calculating C required a numerical optimization step.
    #
    # Instead we must think.
    #
    # Our strategy is to compute dC/dERE via the chain rule.
    #
    #   dC/dERE = dC/d(finite-volume energies) * d(finite-volume energies)/dERE
    #
    # Actually, this doesn't look any better, both C as a function of the finite-volume energies and
    # the finite-volume energies as a function of the ERE parameters required numerical minimization.
    #
    # However, now that we have the tuned C in our hands (via the tuning!) it is easy to recompute
    # the finite-volume energies.
    #
    # In other words, we can calculate E(C); the tuning was C(E).
    #
    @cached_property
    def _eigenenergies(self):
        return self._A1.eigenenergies(self.C)[:len(self.radii)]
    #
    # Then we leverage the fact that
    #
    #   dC/dE = inverse(dE/dC)
    #
    @cached_property
    def _dC_dE(self):
        # C is rows
        # E is cols
        return torch.linalg.inv(torch.autograd.grad(
                    self._eigenenergies,
                    self.C,
                    torch.eye(len(self.radii)),
                    is_grads_batched=True,
                    retain_graph=True)[0])
    #
    # We will calculate the second part of the chain rule similarly.
    # Given the finite-volume energies we can use Lüscher's finite-volume
    # quantization condition to calculate the ERE parameters and then differentiate.
    #
    # The finite-volume quantization condition passes through an intermediate quantity
    #
    #   x = nx^2 E / (2π)^2
    #
    @cached_property
    def _x(self):
        return ((self.Lattice.nx/(2*torch.pi))**2 * self._eigenenergies).to(torch.float64)
    #
    # which corresponds to a simple numerical factor in the chain rule
    #
    @cached_property
    def _dE_dx(self):
        # just a number
        return (2*torch.pi / self.Lattice.nx)**2
    #
    # leaving us to calculate dx/dERE, which is nontrivial.
    #
    @cached_property
    def _dx_dERE(self):
        # x:   row
        # ERE: col
        
        # The quantization condition tell us that the x satisfy the constraint.
        #
        #   0 = ere.analytic(x) - zeta(x)/π^2
        #
        # How should we get dx/dERE?  Imagine infinitesimally perturbing the ere coefficients
        # 
        #   ere.coefficients --> ere.coefficients + dcoefficientsj
        #
        # so that the finite-volume xs change infinitesimally in response
        #
        #   x --> x + dxi
        #
        # Note that
        # - ere.analytic depends on c,
        # - ere.analytic depends on x, zeta depends on x
        #
        # Now we expand the constraint to first order in differentials and cancel the 0th order,
        # since the premise is that ERE and the xs we've got obey the constraint in the first place.
        #
        # The result is
        #
        #   d/dcj( ere.analytic ) dcj = ( zeta' / π^2 - d/dxi( ere.analytic ) ) dxi
        #
        # which then should be solved for
        #
        #   dxi / dcj = [ d/dxi( zeta/π^2 - ere.analytic ) ]^-1 [ d/dcj( ere.analytic )]
        #
        # We can calculate these generically using autograd.
        numerator = torch.autograd.grad(
                        self.ere.analytic(self._x),
                        self.ere.parameters,
                        torch.eye(len(self.ere.parameters), dtype=torch.float64),
                        is_grads_batched=True
                        )[0].to(torch.float64)
        
        denominator = torch.autograd.grad(
                        self.zeta(self._x)/torch.pi**2 - self.ere.analytic(self._x),
                        self._x,
                        torch.eye(len(self._x), dtype=torch.float64),
                        is_grads_batched=True,
                        )[0].to(torch.float64)
        
        # TODO: Use torch.linalg.solve instead?
        return torch.matmul(torch.linalg.inv(denominator), numerator)
    #
    # Now we have each term in the chain rule ready; we just string them together.
    #
    @cached_property
    def dC_dERE(self):
        r'''
        The derivative of the Wilson coefficients :attr:`~.C` with respect to the ``ere.parameters``.

        One row for each ``C``, one column for each parameter of ``ere``.
        '''
        return torch.matmul(self._dC_dE.to(torch.float64), self._dE_dx * self._dx_dERE.to(torch.float64))
    #
    # To calculate the contact we will want the derivative of the Wilson coefficients with respect
    # to the LOG of the scattering length only.
    #
    # This is just a straightforward extension of the derivative we've already computed via the chain rule!
    #
    @cached_property
    def dC_dloga(self):
        r'''
        The derivative of the Wilson coefficients :attr:`~.C` with respect to ``log(ere.a)``.
        '''
        # can compute dC_dERE, which includes dC_da
        # by the chain rule, we need to multiply by
        # da_dloga = 1/(dloga_da) = 1/(1/a) = a
        return self.dC_dERE[:,0] * self.ere.a

####
#### DEMO!
####

def _demo(recompute=True):
    # Nobody would use the demo if it took forever.
    # Therefore I precomputed the answer.
    # You can toss it and recompute if needed.
    return Tuning(
                ere             = tdg.ere._demo(),
                lattice         = tdg.lattice._demo(),
                radii           = [[0,0],[0,1]],
                # If you don't want to use the pre-computed values it's handy to have some ballpark
                # figures to help the tuning get going:
                starting_guess  = torch.tensor([-5., +1.7], dtype=torch.float64),
                # Pre-computed values to accelerate tuning-dependent examples:
                C=  (
                       None if recompute  else
                       torch.tensor([-5.05630981,  1.66827162], dtype=torch.float64, requires_grad=True)
                    )
                )

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute', action="store_true",  default=False)
    args = parser.parse_args()

    tuning = _demo(recompute=args.recompute)
    print(f"We start with the effective range expansion {tuning.ere}")
    print(f"and try to tune it on {tuning.Lattice}")

    if args.recompute:
        print(f"Computing the coefficients requires tuning and may take some time...")
        print(f"The resulting coefficients are " + ", ".join(f'{c:+.8f}' for c in tuning.C) +'.')
    else:
        print(f"The resulting (precomputed) coefficients are " + ", ".join(f'{c:+.8f}' for c in tuning.C) +'.')

    print(f"These give eigenenergies {tuning._eigenenergies}")
    print(f"The derivatives of these coefficients with respect to the ERE parameters are...")
    print(tuning.dC_dERE)
    print(f"and differentiating with respect to the log of the scattering length gives [" + ", ".join(f"{dc:+.8f}" for dc in tuning.dC_dloga) + "].")

    print(f"For a more extensive example and check, see sanity-checks/tuning-differentiate.py")
