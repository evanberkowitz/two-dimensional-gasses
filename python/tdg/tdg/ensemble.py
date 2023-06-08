#!/usr/bin/env python

from functools import cached_property
from functools import lru_cache as cached

import torch

from tdg import _no_op
import tdg
from tdg.h5 import H5able
from tdg.performance import Timer

import logging
logger = logging.getLogger(__name__)

class GrandCanonical(H5able):
    r''' A grand-canonical ensemble of configurations and associated observables, importance-sampled according to :attr:`~.GrandCanonical.Action`.

    .. note::
        :class:`~GrandCanonical` also supports :ref:`a large number of observables <observables>`.

    Parameters
    ----------
        Action: tdg.Action
            An action which describes a Euclidean path integral equal to a Trotterization of the physics of interest.
    '''
    
    _observables = set()
    # The _observables are populated by the @observable decorator.
    
    def __init__(self, Action):
        self.Action = Action
        r'''The action with which the ensemble was constructed.'''

    def from_configurations(self, configurations, weights=None, index=None):
        r'''
        Parameters
        ----------
            configurations: torch.tensor
                A set of pre-computed configurations.
            weights: torch.tensor
                Weights for each configuration.
            index:   torch.tensor
                Where in Markov chain time did each configuration come from.

        Returns
        -------
            the ensemble itself, so that one can do ``ensemble = GrandCanonical(action).from_configurations(phi)``.
            If :code:`weights` is :code:`None`, the weights are all 1.  If :code:`index` is :code:`None`, the index counts up from 0.
        '''
        self.configurations = configurations
        if weights is None:
            self.weights = torch.ones(self.configurations.shape[0])
        else:
            self.weights = weights
        assert self.configurations.shape[0] == len(self.weights)

        if index is None:
            self.index = torch.arange(self.configurations.shape[0])
        else:
            self.index = index
        assert self.configurations.shape[0] == len(self.index)

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

        Populates the :code:`index` attribute, a torch tensor counting up from 0, once for each call to the generator, so that each configuration has an index.
        This index is kept track of through :func:`cut`, :func:`every`, :func:`binned` (by the :class:`~.Binning`).

        .. _tqdm.tqdm: https://pypi.org/project/tqdm/
        .. _tqdm.notebook: https://tqdm.github.io/docs/notebook/
        '''
        self.configurations = self.Action.Spacetime.vector(steps) + 0j
        self.weights = torch.zeros(steps) + 0j
        self.index   = torch.arange(steps)
        
        if start == 'hot':
            seed = self.Action.quenched_sample()
        elif start == 'cold':
            seed = self.Action.Spacetime.vector()
        elif (type(start) == torch.Tensor) and (start.shape == self.configurations[0].shape):
            seed = start
        else:
            raise NotImplemented(f"start must be 'hot', 'cold', or a configuration in a torch.tensor.")
            
        configuration, weight = generator.step(seed)
        self.configurations[0] = configuration.real
        self.weights[0] = weight

        for mcmc_step in progress(range(1,steps)):
            configuration, weight = generator.step(self.configurations[mcmc_step-1])
            self.configurations[mcmc_step] = configuration.real
            self.weights[mcmc_step] = weight

        self.start = start
        self.generator = generator

        return self
    
    def measure(self, *observables):
        r'''
        Compute each :ref:`@observable <observables>` in `observables`; log an error for any :ref:`unregistered observable <custom observables>`.

        Parameters
        ----------
            observables: strings

        Returns
        -------
            :class:`~.GrandCanonical`; itself, now with some observables measured.

        .. note::

            If no `observables` are passed, evaluates **every** registered `@observable`.

        '''
        if not observables:
            observables = self._observables

        with Timer(logger.info, f'Measurement on {len(self)} configurations', per=len(self)):
            for observable in observables:
                if observable not in self._observables:
                    logger.error(f'No registered observable "{observable}"')
                    continue
                try:
                    getattr(self, observable)
                except AttributeError as error:
                    logger.error(str(error))
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
        return GrandCanonical(self.Action).from_configurations(
                self.configurations[start:],
                self.weights[start:],
                self.index[start:],
                )

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
        return GrandCanonical(self.Action).from_configurations(
                self.configurations[::frequency],
                self.weights[::frequency],
                self.index[::frequency]
                )

    def binned(self, width=1):
        r'''
        Parameters
        ----------
            width: int
                The width of the bins over which to average.

        Returns
        -------
            :class:`~.Binning` of the ensemble, with the width specified.
        '''
        return tdg.analysis.Binning(self, width)

    def bootstrapped(self, draws=100):
        r'''
        Parameters
        ----------
            draws: int
                Resamples for uncertainty estimation; see :class:`~.Bootstrap` for details.

        Returns
        -------
            :class:`~.Bootstrap` built from the ensemble, with the draws specified.
        '''
        return tdg.analysis.Bootstrap(self, draws)

    def __len__(self):
        r'''
        Returns
        -------
            The number of configurations.
        '''
        return len(self.configurations)


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

    Note that the chemical potential :math:`\mu` and external field :math:`\vec{h}` do not in principle alter canonical expectation values.
    However, in practice, if the grand canonical ensemble being projected has a expectation values very different from the sectors of interest you may encounter a numerical overlap problem, where the denominator is very small and noisy.
    In that sense, well-chosen :math:`\mu` and :math:`\vec{h}` can provide numerical stabilization.
    '''
    def __init__(self, grandCanonical):
        self.GrandCanonical = grandCanonical
        self.configurations = grandCanonical.configurations

        self.V = self.GrandCanonical.Action.Spacetime.Lattice.sites
        self._n = torch.arange(2*self.V+1)
        self._s = torch.arange(-self.V, self.V+1).roll(-self.V)
        self.hhat = self.GrandCanonical.Action.hhat + 0j
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
                The baryon number :math:`N`.
            Spin: half-integer or None
                The spin projection :math:`S_h` along the GrandCanonical :attr:`Action.h` (or :math:`\hat{z}` if :math:`\vec{h}=0`).

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
        The projection operator :math:`\mathbb{P}` evaluated on each configuration.  The exepectation
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
                       torch.einsum('cs,s->c',
                                    (self.Canonical._term(n,s).Spin),
                                    self.Canonical.hhat)
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

    import tdg.ensemble
    import tdg.action
    S = tdg.action._demo(**kwargs)

    import tdg.HMC as HMC
    H = HMC.Hamiltonian(S)
    integrator = HMC.Omelyan(H, 50, 1)
    hmc = HMC.MarkovChain(H, integrator)

    try:
        ensemble = tdg.ensemble.GrandCanonical(S).generate(steps, hmc, progress=kwargs['progress'])
    except:
        ensemble = tdg.ensemble.GrandCanonical(S).generate(steps, hmc)

    return ensemble

if __name__ == '__main__':
    from tqdm import tqdm
    import tdg, tdg.observable
    torch.set_default_dtype(torch.float64)
    ensemble = _demo(progress=tqdm)
    print(f"The fermionic estimator for the total particle number is {ensemble.N.mean():+.4f}")
    print(f"The bosonic   estimator for the total particle number is {ensemble.N_bosonic.mean():+.4f}")
