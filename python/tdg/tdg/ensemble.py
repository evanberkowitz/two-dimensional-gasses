#!/usr/bin/env python

from functools import cached_property
from functools import lru_cache as cached

import torch
import functorch

import tdg
from tdg.h5 import H5able

def _no_op(x):
    return x

class GrandCanonical(H5able):
    r''' A grand-canonical ensemble of configurations and associated observables, importance-sampled according to :attr:`~.Action`.

    Parameters
    ----------
        Action: tdg.Action
    '''
    
    def __init__(self, Action):
        self.Action = Action

    def from_configurations(self, configurations):
        r'''
        Parameters
        ----------
            configurations: torch.tensor
                A set of pre-computed configurations.

        Returns
        -------
            the ensemble itself, so that one can do ``ensemble = GrandCanonical(action).from_configurations(phi)``.
        '''
        self.configurations = configurations
        return self
        
    def generate(self, steps, generator, start='hot', progress=_no_op):
        r'''
        Parameters
        ----------
            steps:  int
                Number of configurations to generate.
            generator
                Something which produces a new configuration if called as `generator.step(previous_configuration)`.
                Often an :class:`~.hmc` instance.
                May be provided with a default in the future.
            start:  'hot', 'cold', or torch.tensor
                A hot start begins with a configuration drawn from the quenched action.  A cold start beins with the zero configuration.
                If a tensor is passed that tensor is used as the first configuration.
            progress: something which wraps an iterator and provides a progress bar.
                In a script you might use `tqdm.tqdm`_, and in a notebook `tqdm.notebook`_.
                Defaults to no progress reporting.

        Returns
        -------
            the ensemble itself, so that one can do ``ensemble = GrandCanonical(action).generate(...)``.

        .. _tqdm.tqdm: https://pypi.org/project/tqdm/
        .. _tqdm.notebook: https://tqdm.github.io/docs/notebook/
        '''
        self.configurations = self.Action.Spacetime.vector(steps).to(torch.complex128)
        
        if start == 'hot':
            self.configurations[0] = self.Action.quenched_sample()
        elif start == 'cold':
            pass
        elif (type(start) == torch.Tensor) and (start.shape == self.configurations[0].shape):
            self.configurations[0] = start
        else:
            raise NotImplemented(f"start must be 'hot', 'cold', or a configuration in a torch.tensor.")
            
        for mcmc_step in progress(range(1,steps)):
            self.configurations[mcmc_step] = generator.step(self.configurations[mcmc_step-1]).real

        return self
    
    def cut(self, start):
        r'''
        Parameters
        ----------
            start:  int
                The number of configurations to drop from the beginning of the ensemble.

        Returns
        -------
            :class:`~.GrandCanonical` without some configurations at the start.  Useful for performing a thermalization cut.
        '''
        return GrandCanonical(self.Action).from_configurations(self.configurations[start:])

    def every(self, frequency):
        r'''
        Parameters
        ----------
            frequency:  int
                The frequency with which to keep configurations.

        Returns
        -------
            :class:`~.GrandCanonical` with configurations reduced in size by a factor of the frequency.  Useful for Markov Chain decorrelation.
        '''
        return GrandCanonical(self.Action).from_configurations(self.configurations[::frequency])

    # These utility functions help turn a doubly-struck sausage UU into a tensor, and back.
    def _matrix_to_tensor(self, matrix):
        V = self.Action.Spacetime.Lattice.sites
        return matrix.unflatten(-2, (V, 2)).unflatten(-1, (V, 2)).transpose(-3,-2)

    def _tensor_to_matrix(self, tensor):
        return tensor.transpose(-3,-2).flatten(start_dim=-4, end_dim=-3).flatten(start_dim=-2, end_dim=-1)

    # Most of the intermediates needed for the observables are only evaluated lazily, due to computational cost.
    # Once they are evaluated, they're stored.
    # This makes the creation of an ensemble object almost immediate.

    @cached_property
    def _UU(self):
        # A matrix for each configuration.
        return functorch.vmap(self.Action.FermionMatrix.UU)(self.configurations)
    
    @cached_property
    def _UUPlusOne(self):
        # A matrix for each configuration.
        return torch.eye(2*self.Action.Spacetime.Lattice.sites) + self._UU
    
    @cached_property
    def _detUUPlusOne(self):
        return torch.det(self._UUPlusOne)

    @cached_property
    def _UUPlusOneInverse(self):
        # A matrix for each configuration.
        return torch.linalg.inv(self._UUPlusOne)
    
    @cached_property
    def _UUPlusOneInverseUU(self):
        # A matrix for each configuration.
        return torch.matmul(self._UUPlusOneInverse,self._UU)
    
    @cached_property
    def average_field(self):
        r'''
        The average auxiliary field, one per configuration.
        '''
        return self.configurations.mean((1,2))

    @cached
    def n(self, method='fermionic'):
        r'''
        The local number density, one per site per configuration.

        Parameters
        ----------
            method: str
                The approach for calculating the local number densities, ``'fermionic'`` or ``'bosonic'``.

        Returns
        -------
            torch.tensor
                Configurations slowest, then sites.

        .. note ::
            The ``'fermionic'`` method computes a derivative of the fermion determinant, and seems to be positive-(semi?)definite.
            The ``'bosonic'`` method computes the derivative of the gauge action and is not positive-definite.

        '''
        if method == 'fermionic':
            return torch.einsum('caass->ca',
                                self._matrix_to_tensor(self._UUPlusOneInverseUU)
            )

        elif method == 'bosonic':
            Vinv = self.Action.Potential.inverse(self.Action.Spacetime.Lattice)
            return -torch.einsum('ab,ctb->cta', Vinv.to(torch.complex128), self.configurations).mean(1)/ (self.Action.beta/self.Action.Spacetime.nt)
        
        raise NotImplemented(f'Unknown {method=} for calculating n.')
    
    @cached
    def N(self, method='fermionic'):
        r'''
        The total number of particles for each configuration.
        
        Parameters
        ----------
            method: str
                The approach for calculating the local number densities, 'fermionic' or 'bosonic'.

        Returns
        -------
            torch.tensor
                One per configuration.
        '''
        return self.n(method).sum(1)
    
    @cached_property
    def spin(self):
        r'''
        The local spin density.
        Direction slowest, then configurations, then sites.  That makes it easy to do something with ``ensemble.s[1]``.

        .. note ::
            The indices on the spins match the indices on :data:`tdg.PauliMatrix`.
            The ``[0]`` entry matches ``ensemble.n('fermionic')``, a useful check.
        '''

        dt   = self.Action.dt
        hhat = self.Action.hhat.to(torch.complex128)
        absh = self.Action.absh.to(torch.complex128)
        
        if absh == 0.:
            Pspin = torch.diag(torch.tensor((0,0,0.5))).to(torch.complex128)
        else:
            Pspin = (
                0.5 * torch.outer(hhat,hhat)
                # sinh(x/2) * cosh(x/2) = 0.5 * sinh(x)
                + 0.5*torch.sinh(dt*absh) / (dt * absh) * (torch.eye(3) - torch.outer(hhat,hhat)) 
                # sinh^2(x/2) = 1/2 * (cosh(x) - 1)
                + 0.5j * (torch.cosh(0.5*dt*absh)-1) / (dt * absh) * torch.einsum('ijk,j->ik', tdg.epsilon.to(torch.complex128), hhat)
               )
        
        # expand to include 'all four' "spin" matrices.
        Pspin = torch.nn.ConstantPad2d((1,0,1,0), 0)(Pspin)
        Pspin[0,0] = 1.
        # and construct a vector of paulis.
        Pspin = torch.einsum('ik,kab->iab', Pspin, tdg.PauliMatrix)
        
        return torch.einsum('cxxab,iba->icx', self._matrix_to_tensor(self._UUPlusOneInverseUU), Pspin)
        
    @cached_property
    def Spin(self):
        r'''
        The total spin, summed over all sites.
        Direction slowest, then configurations.
        '''
        return self.spin.sum(-1)
    
    @cached_property
    def S(self):
        r'''
        The ``Action`` evaluated on the ensemble.
        '''
        return functorch.vmap(self.Action)(self.configurations)

    @cached_property
    def doubleOccupancy(self):
        r'''
        The double occupancy of a site is :math:`n_{\uparrow} n_{\downarrow}` on that site (or the equivalent along any direction; not just the :math:`z`-axis),
        which as an operator is equal to :math:`\frac{1}{2}(n^2-n)`, where :math:`n` is the total number operator :func:`~.n`.

        Configuration slowest, then space.
        '''

        # The double occupancy on site a is given by the expectation value of
        #
        #   2 * doubleOccupancy =
        #   sum_{st}
        #     + [ (1+UU)^-1 U ]^{ss}_{aa} [ (1+UU)^-1 U ]^{tt}_{aa}
        #     - [ (1+UU)^-1 U ]^{st}_{aa} [ (1+UU)^-1 U ]^{ts}_{aa}
        #
        # where the lower indices are spatial and the upper indices are spin.
        #
        # The first term is the square of the contraction of the fermionic n operator.
        first = self.n('fermionic')**2
        # The second term is a bit more annoying;
        UUPlusOneInverseUU = self._matrix_to_tensor(self._UUPlusOneInverseUU)
        second = torch.einsum('caast,caats->ca',
                                UUPlusOneInverseUU,
                                UUPlusOneInverseUU,
                             )

        # To get the double occupancy itself, take half the difference.
        return 0.5*(first - second)

    @cached_property
    def DoubleOccupancy(self):
        r'''
        The spatial sum of the :func:`~.doubleOccupancy`; one per configurtion.
        '''
        return self.doubleOccupancy.sum(axis=1)

    @cached
    def contact(self, method='bosonic'):
        r'''
        The `contact`, :math:`\frac{dH}{d\log a}`.

        Parameters
        ----------
            method: str
                The approach for calculating the number densities ``fermionic`` or ``bosonic``.

        Returns
        -------
            torch.tensor
                One per configuration.

        .. note::

            The method combines matrix elements with the derivative of the Wilson coefficients with respect to :math:`\log a` through :meth:`~.Tuning.dC_dloga`.
            Therefore the ``ensemble.Action`` must have a tuning!

        .. note::

            The ``'bosonic'`` method computes the derivative of the action with respect to the Wilson coefficients.
            This is faster but noiser.

        .. todo::

            The ``'fermionic'`` method computes matrix elements of number operators.
        '''
        if method=='bosonic':
            with torch.autograd.forward_ad.dual_level():
                C0_dual = torch.autograd.forward_ad.make_dual(self.Action.Tuning.C, self.Action.Tuning.dC_dloga)
                V_dual  = tdg.Potential(*[c * tdg.LegoSphere(r) for c,r in zip(C0_dual, self.Action.Tuning.radii)])
                S_dual  = tdg.Action(self.Action.Spacetime, V_dual, self.Action.beta, self.Action.mu, self.Action.h, self.Action.fermion)

                s_dual  = functorch.vmap(S_dual)(self.configurations)
                return  (2*torch.pi / self.Action.beta)* torch.autograd.forward_ad.unpack_dual(s_dual).tangent

        raise NotImplemented('Unknown {method=} for calculating the contact.')

####
#### Canonical
####

class Canonical(H5able):
    r'''
    Whereas the grand-canonical ensemble fixes thermodynamic variables (such chemical potential :math:`\mu` or external field :math:`\vec{h}`),
    a canonical ensemble fixes quantum numbers (like total particle number :math:`N` or total spin projection along :math:`\vec{h}`, :math:`S_\vec{h}`).
    
    We construct a canonical ensemble from a grand-canonical ensemble by `projecting` the desired quantum numbers.
    
    .. math::
        \begin{align}
            \left\langle\mathcal{O}\right\rangle_{N, S_h}
            &=
            \frac{\text{tr}\left[ e^{-\beta H} \mathcal{O}\right]_{N, S_h}}{\text{tr}\left[ e^{-\beta H}\right]_{N, S_h}}
            =
            \frac{\text{tr}\left[ e^{-\beta (H-\mu \hat{N}-h \cdot \hat{S})} \mathcal{O} \right]_{N, S_h}}{\text{tr}\left[ e^{-\beta (H-\mu \hat{N}-h \cdot \hat{S})}\right]_{N, S_h}} 
            \nonumber\\
            &=
            \frac{\text{tr}\left[ e^{-\beta (H-\mu\hat{N}-h\cdot \hat{S})} P_N P_{S_h} \mathcal{O}\right]}{\text{tr}\left[ e^{-\beta (H - \mu \hat{N} - h \cdot \hat{S})} P_N P_{S_h}\right]}
            =
            \frac{\left\langle P_N P_{S_h} \mathcal{O}\right\rangle}{\left\langle P_N P_{S_h}\right\rangle}
            \label{eq:canonical-grand canonical projection}
        \end{align}
    
    where the subscripted expectation values are canonical, the unsubscripted ones are grand-canonical, and :math:`P` operators are projectors of the respective quantum number to the subscript's value.
    '''
    def __init__(self, grandCanonical):
        self.GrandCanonical = grandCanonical
        self.configurations = grandCanonical.configurations

        self.V = self.GrandCanonical.Action.Spacetime.Lattice.sites
        self._n = torch.arange(2*self.V+1)
        self._s = torch.arange(-self.V, self.V+1).roll(-self.V)
        self.hhat = self.GrandCanonical.Action.hhat.to(torch.complex128)
        # In the grand canonical ensemble we project spin along the h direction.
        # Therefore the we may need σ.hhat with some frequency.
        self.hhatsigma = torch.einsum('i,ist->st', self.hhat, tdg.PauliMatrix[1:])

        # The projector is a sum over terms.
        # Those terms are specified by n and s, fourier-transform variables.
        #
        #   P = 1/(norm) ∑(n,s) e^{-2πi(nN+2sS)/(2V+1) det( 1 + exp[2πi(n + s hhat.σ)/(2V+1)] U ) / det(1+U)
        #
        # and observables that are normally functional derivatives of det(1+U) are functional
        # derivatives of just the numerator det(1+ exp(2πi(n + s hhat.σ) U).
        # Therefore we can think of separating each term in this sum into two pieces:
        #  - a piece independent of the N and S we're projecting to, reusable.
        #  - a piece that are independent of any particular N and S.

    @cached
    def _term(self, n, s):
        # Each term depends on n and s but not on N and S, and hence is reusable.
        # Since all observables are the same but for the U we're supposed to use
        #
        #   U -> exp[2πi(n+ s hhat.σ)/(2V+1)] U
        #
        # we can reuse all the grand-canonical machinery for making measurements.
        #
        # However, we can actually use the Action.projected(n, s) convenience function to construct
        # a grand-canonical ensemble with the appropriate shifts!
        return tdg.ensemble.GrandCanonical(
                self.GrandCanonical.Action.projected(n, s)
                ).from_configurations(self.GrandCanonical.configurations)

    @cached
    def _shifted_grand_canonical_ensemble(self, n, s):
        return GrandCanonical(self.GrandCanonical.Action.projected(n, s))

    @cached
    def Sector(self, Particles, Spin):
        r'''
        A :class:`~.Sector` in which to make measurements.
        Different :class:`~.Sector` s generated from the same canonical ensemble reuse
        intermediate results and can save a lot of computational effort.

        Parameters
        ----------
            Particles: int or None
            Spin: half-integer or None

        Returns
        -------
            :class:`~.Sector` in which to measure observables :math:`\mathcal{O}`.
        '''
        return Sector(self, Particles=Particles, Spin=Spin)

    # This is syntactic sugar to intercept calls, meant to be used in combination with
    # Sector._grid, where attributes of each term need to be evaluated as a function
    # of n and s and then passed their 'usual' arguments.
    #
    # For example, see Sector._canonical_weights, where we need a determinant in each
    # sector. We call
    #
    #   self.Canonical._detUUPlusOne
    #
    # which, as you can see in this implementation doesn't explicitly exist. Instead
    # we forward that call to each sector, which evaluates its own determinant.
    # Without this forwarding we'd need to write the same kind of for loop repeatedly.
    def __getattr__(self, name):
        def curry(n, s):
            return self._term(n,s).__getattribute__(name)
        return curry
    # This is in fact sort of magical because the . resolution happens first, so it even
    # forwards the arguments to method calls correctly.

class Sector(H5able):
    r'''
    A `sector` is a choice of quantum numbers that specify a particular canonical ensemble.
    We currently have the capacity to project particle number (``Particles``) or the spin
    projected along the external field :math:`\vec{h}` (if :math:`\vec{h}=0`, the z direction).
    
    The constructor checks for consistency between the choices.  You cannot, for example, specify
    an even number of particles and a half-integer spin.  Since the particles are spin-half there
    is no such sector.
    
    .. note::
        :class:`Sector` has machinery under the hood which allows the user to compute any measurements available in :class:`GrandCanonical`.
        So, even though a :class:`Sector` has no ``.N`` to call, you may call it nevertheless, with the options of :func:`GrandCanonical.N`.

        **The exception is** ``.Spin``, because the construction of the canonical sector requires observables to commute with the projector,
        so it does not make sense to measure spin along any axis except :math:`\hat{h}`; therefore, calculate :func:`~Sh` instead.

        For implementation details, see :func:`Sector.__getattr__` in the source.
    
    .. warning::
        Sectors should only rarely be constructed manually; typically a user should construct sectors by invoking :func:`Canonical.Sector`.

    Parameters
    ----------
        canonical: :class:`~.Canonical`
            The ensemble from which to construct the sector.
        Particles: integer or ``None``
            If an integer, the number of particles to project to.  If ``None`` no projection of the particle number is performed.
        Spin: half integer or ``None``
            If a half-integer, the spin projection along :math:`\vec{h}`.  If ``None`` no projection of the particle number is performed.

    '''
    def __init__(self, canonical, Particles, Spin):
        self.Canonical = canonical
        self.configurations = canonical.configurations
        
        self.Particles_projected = (Particles is not None)
        r'''
        Was ``Particles`` specified under construction?  In other words, should we project to a particular number of particles?
        ``True`` if ``Particles`` was a number, ``False`` if it was ``None``.
        '''
        self.Particles           = Particles if self.Particles_projected else 0
        r'''The number of particles in the canonical sector; 0 if not `.Particles_projected`.'''
        
        self.Spin_projected = (Spin is not None)
        r'''
        Was ``Spin`` specified under construction?  In other words, should we project to a particular number of particles?
        ``True`` if ``Spin`` was a number, ``False`` if it was ``None``.
        '''
        self.Spin = Spin if self.Spin_projected else 0
        r'''The total spin of the canonical sector, or `None`.'''
        
        if Particles and Particles < 0:
            raise ValueError(f"The number of particles must be nonnegative, not {Particles}.")

        if Spin and (Spin % 0.5) != 0.:
            raise ValueError(f"The spin must be an integer multiple of 1/2, not {Spin}.")
        
        if Spin and Particles:
            if abs(2*Spin) > Particles:
                raise ValueError(f"There is no canonical sector with {Particles} particle{'' if Particles==1 else 's'} and spin {Spin}, since the particles are spin-1/2.  With {Particles} particles the spin must be in [{-Particles/2}, {Particles/2}].")

            if (Particles/2 - Spin) % 1 != 0.:
                raise ValueError(f"There is no canonical sector with {Particles} particles and spin {Spin}, since the particles are spin-1/2.  "+
                                 ("With an even number of particles the spin must be integer." if Particles%2==0 else
                                  "With an odd number of particles the spin must be half-integer."))
        
        # When evaluating the canonical sum we will need to evaluate the same function for each n and s.
        # However, if we're only projecting one quantum number or the other we need not evaluate the terms
        # that do not contribute.  In other words, if we've only specified a particle number,
        # the sum over spin shouldn't be done.
        #   THIS IS A PROGRAMMING TRICK to make the following code simpler to write:
        #   We can replace the sum with a trivial sum over just one term.
        #   However, it allows us to always write code as though we're doing a projection of both quantum numbers.
        #   See _phases for an explanation of why this trick makes sense.
        self._n = torch.zeros(1, dtype=torch.int64) if not self.Particles_projected    else torch.arange(2*self.Canonical.V+1)
        self._s = torch.zeros(1, dtype=torch.int64) if not self.Spin_projected else torch.arange(-self.Canonical.V, self.Canonical.V+1).roll(-self.Canonical.V)
        
    # This utility helps us avoid writing the same for loop over and over again.
    # It takes a function f of the n and s that specifies a term of the canonical sum and produces a grid
    # of values of f with s slowest and n next slow.
    def _grid(self, f):
        return torch.stack(tuple(
               torch.stack(tuple(
                   f(n, s)
                # .numpy is required to ensure that n and s are passed as true integers.
                # otherwise the memoization of canonical._term fails (since n and s wind up as new views)
                for n in self._n.numpy()))
                for s in self._s.numpy()))

    # Each term in the canonical sum depends on an n and an s through some Fourier mode,
    # the phases exp(-2πi/(2V+1) (nN+2sS)).  If we're not projecting the number/spin we
    # want to omit the n or s term from these phases.  One way to just set n/s respectively to 0,
    # explaining the _n and _s trick.
    @cached_property
    def _phases(self):
        return torch.exp(self._grid(lambda n, s:
                    torch.tensor(-2j*torch.pi / (2*self.Canonical.V+1) * (n * self.Particles + 2* s * self.Spin))
                ))

    # Every term in the canonical sum gets a det(1+UU) in the downstairs
    # which comes from multiplying by det(1+UU) / det(1+UU) and using the upstairs
    # as part of the grand-canonical measure, to give a grand-canonical expectation value.
    # Also downstairs is one factor of (2V+1) for each quantum number projected.
    @cached_property
    def _norm(self):
        # A number for each configuration.

        # We can get (2V+1) to the right power by just counting the _phases.
        return 1 / self._phases.numel() / self.Canonical.GrandCanonical._detUUPlusOne
    
    # The '_canonical_weights' are the whole shebang:
    # the norm 1/(2V+1)^(# quantum numbers) / det(1+UU)
    # which correspond term-by-term to the _factors
    #
    # These get reused 'under the sum' while computing observables.
    @cached_property
    def _canonical_weights(self):
        # A matrix for every factor and every configuration.
        dets = self._grid( self.Canonical._detUUPlusOne )
        return torch.einsum('sn,snc,c->snc', self._phases, dets, self._norm)

    # The total weight is the thing that goes downstairs in the canonical expectation value.
    # As mentioned in the Canonical docstring a canonical expectation value is the ratio
    # of two grand-canonical expectation values,
    #
    #   < O >_{N,S} = < P_{N,S} O > / < P_{N,S} >
    #
    @cached_property
    def weight(self):
        r'''
        The operator :math:`\mathbb{P}` evaluated on each configuration.  The exepectation
        value goes downstairs in the grand-canonical formulation of the :class:`~.Canonical` expectation value.

        If you wish to perform a resampling analysis (like bootstrap or jackknife) these must
        be resampled and averaged independently from the projected observable upstairs in the expectation value.
        '''
        return self._canonical_weights.sum(axis=(0,1))
    
    # The upstairs observable expectation value is always a sum over the n and s terms.
    def _reweight(self, observable):
        # One observable for each configuration.
        return torch.einsum('snc,snc...->c...',
                            self._canonical_weights,
                            observable)
        
    @cached_property
    def Sh(self):
        r'''
        The spin projected along :math:`\vec{h}`.
        In expectation, you should get the ``Spin`` with which the sector was constructed, or 0 if the spin
        was not projected.
        '''
        return self._reweight(
            self._grid(lambda n, s:
                       torch.einsum('sc,s->c', self.Canonical._term(n,s).Spin[1:], self.Canonical.hhat)
                      ))

    # It may make sense to cache, but since the underlying terms cache this may be overkill.
    def __getattr__(self, name):
        # There are two kinds of attributes we want to fiddle with:
        #  - data, which we just want to reweight, and
        #  - callables, which we want to call with the right arguments and THEN reweight.

        # Since we have the built-in callable keyword we handle that case first.
        # Every term will be the same, and whether we are projecting number, spin, or both,
        # we'll always need the (0,0) contribution.  Therefore check
        if callable(self.Canonical._term(0,0).__getattribute__(name)):
            # and construct a function that forwards the arguments to the canonical evaluation.
            def curry(*args, **kwargs):
                # and reweights, see below.
                return self._reweight(
                    self._grid(
                            lambda n, s:
                            self.Canonical._term(n,s).__getattribute__(name)(*args, **kwargs)
                            )
                )
            return curry
        
        # If it's not callable, then it's just data, which we know how to handle.
        # Simply reweight it, evaluating the data term-by-term.
        return self._reweight(
            self._grid(
                lambda n, s:
                self.Canonical._term(n,s).__getattribute__(name)
                )
        )

    def __str__(self):
        if self.Particles_projected and self.Spin_projected:
            return f'CanonicalSector(Particles={self.Particles}, Spin={self.Spin})'
        if self.Particles_projected:
            return f'CanonicalSector(Particles={self.Particles})'
        if self.Spin_projected:
            return f'CanonicalSector(Spin={self.Spin})'
        return f'CanonicalSector()'

    def __repr__(self):
        return str(self)

####
#### Demo!
####

def _demo(steps=100, **kwargs):

    import tdg.action
    S = tdg.action._demo(**kwargs)

    import tdg.HMC as HMC
    H = HMC.Hamiltonian(S)
    integrator = HMC.Omelyan(H, 50, 1)
    hmc = HMC.MarkovChain(H, integrator)

    try:
        ensemble = GrandCanonical(S).generate(steps, hmc, progress=kwargs['progress'])
    except:
        ensemble = GrandCanonical(S).generate(steps, hmc)

    return ensemble

if __name__ == '__main__':
    from tqdm import tqdm
    ensemble = _demo(progress=tqdm)
    print(f"The fermionic estimator for the total particle number is {ensemble.N('fermionic').mean():+.4f}")
    print(f"The bosonic   estimator for the total particle number is {ensemble.N('bosonic'  ).mean():+.4f}")
    print(f"The Spin[0]   estimator for the total particle number is {ensemble.Spin[0].mean()       :+.4f}")
