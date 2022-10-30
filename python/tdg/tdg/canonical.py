#!/usr/bin/env python3

from functools import cached_property
from functools import lru_cache as cached
import torch
import tdg

class Sector:
    r'''
    Represents the canonical sector with fixed particle number :math:`N` and spin :math:`S`.

    Raises a `ValueError` if `N` and `Spin` are incompatible based on physical principles.

    Parameters
    ----------
        N:  non-negative integer or None
            The number of particles in the sector.
        Spin:  signed half-integer or None
            The total spin of the sector, parallel to the external field :math:`\vec{h}` or :math:`\hat{z}` if :math:`h=0`.
    '''
    def __init__(self, N, Spin):
        self.N    = N
        r'''The number of particles in the canonical sector, or `None`.'''
        self.Spin = Spin
        r'''The total spin of the canonical sector, or `None`.'''
        
        if N and N < 0:
            raise ValueError(f"The number of particles must be nonnegative, not {N}.")

        if Spin and (Spin % 0.5) != 0.:
            raise ValueError(f"The spin must be an integer multiple of 1/2, not {Spin}.")
        
        if Spin and N:
            if abs(2*Spin) > N:
                raise ValueError(f"There is no canonical sector with {N} particle{'' if N==1 else 's'} and spin {Spin}, since the particles are spin-1/2.  With {N} particles the spin must be in [{-N/2}, {N/2}].")

            if (N/2 - Spin) % 1 != 0.:
                raise ValueError(f"There is no canonical sector with {N} particles and spin {Spin}, since the particles are spin-1/2.  "+
                                 ("With an even number of particles the spin must be integer." if N%2==0 else
                                  "With an odd number of particles the spin must be half-integer."))

    def __str__(self):
        if (self.N is not None) and (self.Spin is not None):
            return f'CanonicalSector(N={self.N}, Spin={self.Spin})'
        if (self.N is not None):
            return f'CanonicalSector(N={self.N})'
        if (self.Spin is not None):
            return f'CanonicalSector(Spin={self.Spin})'
        return f'CanonicalSector()'

    def __repr__(self):
        return str(self)

class Projection:
    r'''
    A canonical projection helps compute observables in the canonical ensemble.
    This base class should not be instantiated directly.
    Instead, it provides utilities for :class:`.ProjectionN`, :class:`ProjectionS`, and :class:`ProjectionNS`, which fix
    the total number of particles, the total spin, or both.

    .. note::
        Each conserved quantum number costs an additional factor of `V = Action.Spacetime.Lattice.sites` in performance.
        The run time of observable computation scales quite poorly: at least `~V^(3 + #specified quantum numbers)`.

    The general projection operator is used by :class:`.ProjectionNS` is

    .. math::
        \begin{align}
            \left\langle\mathbb{P}_{\texttt{N},\texttt{Spin}}\right\rangle
            &=
            \left\langle\frac{1}{(2\mathcal{V}+1)^2} \sum_{ns} e^{-\frac{2\pi i}{2\mathcal{V}+1} (n\texttt{N}+2s\texttt{Spin})}\; \frac{
            \det\left[ \mathbb{1} + \exp\left\{ \frac{2\pi i}{2\mathcal{V}+1}\left(n + s \hat{h} \cdot \sigma\right)\right\} \mathbb{U}\right]
            }{
            \det\left[ \mathbb{1} + \mathbb{U} \right]
            }\right\rangle
            \\
            n &\in \{0, 2\mathcal{V}\}
            \\
            s &\in \{-\mathcal{V}, +\mathcal{V}\}
        \end{align}

    where `N` and `Spin` are parameters of a :class:`.canonical.Sector`.
    The one-quantum-number projections :class:`.ProjectionN` and :class:`ProjectionS` have only one sum
    and are normalized by only one power of :math:`(2\mathcal{V}+1)` rather than 2, and
    :class:`.ProjectionN` ignores the :func:`.Sector.Spin` while :attr:`.ProjectionS` ignores the :attr:`.Sector.N`.

    An expectation value of an operator is given by

    .. math::
        \langle\mathcal{O}\rangle_{\texttt{N},\texttt{Spin}}
        =
        \frac{\langle\mathbb{P}_{\texttt{N},\texttt{Spin}}\mathcal{O}\rangle}{\langle\mathbb{P}_{\texttt{N},\texttt{Spin}}\rangle}

    where the operator :math:`\mathcal{O}` is evaluated 'under the sum' in :math:`\mathbb{P}`, and usually the operator must be
    evaluated with :math:`\exp\left\{ \frac{2\pi i}{2\mathcal{V}+1}\left(n + s \hat{h} \cdot \sigma\right)\right\} \mathbb{U}` rather
    than the normal :math:`\mathbb{U}`.


    Parameters
    ----------
    Action: :class:`.Action`
        A fermion action with a :class:`.Spacetime` and a :class:`.FermionMatrix` that can compute :func:`UU` and :func:`UU_tensor`.
    configurations: torch.tensor
        An ensemble of configurations.

    '''

    def __init__(self, Action, configurations):
        self.Action = Action
        self.configurations = configurations

        self.V = Action.Spacetime.Lattice.sites
        self.hhat = self.Action.FermionMatrix.hhat.to(torch.complex128)
        self.hhatsigma = torch.einsum('i,ist->st', self.hhat, tdg.PauliMatrix[1:])

    # These utility functions help turn a doubly-struck sausage UU into a tensor, and back.
    def _matrix_to_tensor(self, matrix):
        return matrix.unflatten(-2, (self.V, 2)).unflatten(-1, (self.V, 2)).transpose(-3,-2)

    def _tensor_to_matrix(self, tensor):
        return tensor.transpose(-3,-2).flatten(start_dim=-4, end_dim=-3).flatten(start_dim=-2, end_dim=-1)

    # Most of the properties needed for the projection are only evaluated lazily, due to computational cost.
    # Once they are evaluated, they're stored.
    # This makes the creation of a Projection object almost immediate.

    @cached_property
    def _UU(self):
        # A tensor for each configuration.
        return torch.stack(tuple(self.Action.FermionMatrix.UU_tensor(cfg) for cfg in self.configurations))

    # Each implementation must provide a _factors attribute (or @cached_property).
    # These factors are the exp(2πi/(2V+1) (n + s h.σ) that hit UU inside the determinant
    # inside the sum of PP.
    # For compatibility with _norm, the 0th element should be the identity matrix.
    def _factors(self):
        raise NotImplemented(f'{self.__class__} does not implement _factors')

    # The factors help us compute W, which is just each factor matrix-multiplied on each
    # configuration's UU and then made a matrix (rather than a tensor).
    @cached_property
    def _W(self):
        # UU is already a tensor, so the matrix multiplication of _factors broadcasts.
        return self._tensor_to_matrix(
                torch.stack(tuple(torch.matmul(factor,self._UU)
                                for factor in self._factors)
                                )).transpose(0,1)
                # but we have to transpose the first two dimensions to keep 
                # the configuration index on the outside.

    # Every term in the projector's sum gets a det(1+UU) in the downstairs
    # which comes from multiplying by det(1+UU) / det(1+UU) and using the upstairs
    # as part of the measure, to give a grand-canonical expectation value.
    # Also downstairs is one factor of (2V+1) for each quantum number projected.
    @cached_property
    def _norm(self):
        # A number for each configuration.
        # _W[:,0] is 'usual' UU for each configuration.
        return 1 / len(self._factors) / torch.det(torch.eye(2*self.V)+self._W[:,0])

    # Each terms in the sum needs to manipulate W+1
    @cached_property
    def _WPlusOne(self):
        # _W+1
        # A matrix for every factor and every configuration.
        return torch.eye(2*self.V) + self._W

    # In particular, its determinant shows up in the projector while
    @cached_property
    def _detWPlusOne(self):
        # A number for each factor and configuration.
        return torch.det(self._WPlusOne)

    # the inverse of W+1 shows up in observables.
    @cached_property
    def _WPlusOneInv(self):
        # A matrix for every factor and every configuration.
        return torch.linalg.inv(self._WPlusOne)

    # The '_canonical_weights' are the whole shebang:
    # the norm 1/(2V+1)^(# quantum numbers) / det(1+UU)
    # the phases exp(-2πi/(2V+1) (nN+2sS)), [omit the n or s if not projecting number/spin]
    def _phases(self, sector):
        raise NotImplemented(f'{self.__class__} does not implement _phases')
    # which correspond term-by-term to the _factors, implementation-dependent.
    # and the determinants of 1+W.
    #
    # These get reused 'under the sum' while computing observables.
    @cached
    def _canonical_weights(self, sector):
        # A matrix for every factor and every configuration.
        return torch.einsum('c,f,cf->cf', self._norm, self._phases(sector), self._detWPlusOne)

    @cached
    def weight(self, sector):
        r'''
        Parameters
        ----------
            sector: :class:`.canonical.Sector`
                Canonical sector of interest.
            
        Returns
        -------
            torch.tensor:
                The operator :math:`\mathbb{P}` evaluated on each configuration.
        '''
        return torch.sum(self._canonical_weights(sector), axis=(1,))

    def N(self, sector):
        r'''
        Parameters
        ----------
            sector: :class:`.canonical.Sector`
                Canonical sector of interest.
            
        Returns
        -------
            torch.tensor:
                The operator :math:`\mathbb{P}N` evaluated on each configuration.
                To get :math:`\langle N \rangle_{\texttt{sector}}` this should be averaged
                and divided by the mean :func:`.weight` of the same sector.

                They are computed separately to enable bootstrapping.
        '''
        try:
            self.n.shape
        except AttributeError:
            raise NotImplementedError(f'{self.__class__} cannot project the particle number.') from None

        trace = torch.einsum('cfxx->cf', torch.matmul(self._WPlusOneInv, self._W))
        return torch.einsum('cf,cf->c', self._canonical_weights(sector), trace)

    def S(self, sector):
        r'''
        Parameters
        ----------
            sector: :class:`.canonical.Sector`
                Canonical sector of interest.
            
        Returns
        -------
            torch.tensor:
                The operator :math:`\mathbb{P}S_h` evaluated on each configuration.
                The spin is automatically computed along the direction of the external field :math:`h` in the `Action`.
                To get :math:`\langle S_h \rangle_{\texttt{sector}}` this should be averaged
                and divided by the mean :func:`.weight` of the same sector.

                They are computed separately to enable bootstrapping.
        '''

        try:
            self.s.shape
        except AttributeError:
            raise NotImplementedError(f'{self.__class__} cannot project the total spin.') from None

        absh = self.Action.h.norm()
        if absh == 0:
            # pick the z axis
            mat = 0.5*tdg.PauliMatrix[3]
        else:
            mat = torch.einsum('k,kij->ij', 0.5*self.hhat, tdg.PauliMatrix[1:])
                
        trace = torch.einsum('cfxxss->cf', torch.matmul(
            self._matrix_to_tensor(torch.matmul(
                self._WPlusOneInv,
                self._W)),
            mat))
        return torch.einsum('cf,cf->c', self._canonical_weights(sector), trace)


class ProjectionN(Projection):
    r'''
    A :class:`.Projection` which only projects the particle number and ignores any spin
    specification in any passed :class:`.Sector`, to gain a factor of volume in speed.

    :func:`.S` will raise a runtime error.
    '''

    def __init__(self, Action, configurations):

        super(ProjectionN, self).__init__(Action, configurations)

        self.n = torch.arange(2*self.V+1)

    @cached_property
    def _factors(self):
        return torch.matrix_exp(torch.stack(tuple(
                +2j*torch.pi / (2*self.V+1) * (
                    n * tdg.PauliMatrix[0]
                    )
                for n in self.n
            )))

    @cached
    def _phases(self, sector):
        return torch.stack(tuple(torch.exp(-2j*torch.pi / (2*self.V+1) * (n * sector.N))
                                for n in self.n
                                ))

class ProjectionS(Projection):
    r'''
    A :class:`.Projection` which only projects the total spin and ignores any particle number
    specification in any passed :class:`.Sector`, to gain a factor of volume in speed.

    :func:`.N` will raise a runtime error.
    '''

    def __init__(self, Action, configurations):

        super(ProjectionS, self).__init__(Action, configurations)

        self.s = torch.arange(-self.V, self.V+1)

    @cached_property
    def _factors(self):
        return torch.matrix_exp(torch.stack(tuple(
                +2j*torch.pi / (2*self.V+1) * (
                    s * self.hhatsigma
                    )
                for s in self.s
            )))

    @cached
    def _phases(self, sector):
        return torch.stack(tuple(torch.exp(-2j*torch.pi / (2*self.V+1) * (2* s * sector.Spin))
                                for s in self.s
                                ))

class ProjectionNS(Projection):
    r'''
    A :class:`.Projection` which projects both the particle number and the total spin.
    A factor of lattice volume slower than the one-quantum-number projections
    :class:`.ProjectionN` and :class:`.ProjectionS`.
    '''

    def __init__(self, Action, configurations):

        super(ProjectionNS, self).__init__(Action, configurations)

        self.n = torch.arange(2*self.V+1)
        self.s = torch.arange(-self.V, self.V+1)

    @cached_property
    def _factors(self):
        return torch.matrix_exp(torch.stack(tuple(
                +2j*torch.pi / (2*self.V+1) * (
                    n * tdg.PauliMatrix[0] +
                    s * self.hhatsigma
                    )
                for n in self.n
                for s in self.s
            )))

    @cached
    def _phases(self, sector):
        return torch.stack(tuple(torch.exp(-2j*torch.pi / (2*self.V+1) * (n * sector.N + 2* s * sector.Spin))
                                for n in self.n
                                for s in self.s
                                ))

def _demo():
    import torch
    import tdg
    import tdg.HMC as HMC

    from tqdm import tqdm

    # Pick a small example
    lattice = tdg.Lattice(3)
    spacetime = tdg.Spacetime(8, lattice)
    # with uninteresting parameters
    beta = torch.tensor(1)
    mu = torch.tensor(-2.0)
    # but a nontrivial h as a check that everything is correctly h-dependent.
    h  = torch.tensor([0,0.5,0.5], dtype=torch.complex128)
    V = tdg.Potential((-5)*tdg.LegoSphere([0,0]))
    S = tdg.Action(spacetime, V, beta, mu, h)

    # so that HMC will go fast.
    H = HMC.Hamiltonian(S)
    integrator = HMC.Omelyan(H, 50, 1)
    hmc = HMC.MarkovChain(H, integrator)

    # just generate a few configurations.
    steps = 100 # configurations take about 10-15 seconds on my machine.
    configuration    = spacetime.vector(steps).to(torch.complex128)

    # do hot-start HMC
    configuration[0] = S.quenched_sample()
    for mcmc_step in tqdm(range(1,steps)):
         configuration[mcmc_step] = hmc.step(configuration[mcmc_step-1]).real

    # and now project the number
    np = ProjectionN(S, configuration)
    for N in range(1, 8):
        sector = Sector(N, None)
        n, d = np.N(sector), np.weight(sector)
        mean_n = torch.mean(n) / torch.mean(d)
        print(f'{str(sector):35s} {mean_n:+.4f}')

    # the total spin,
    sp = ProjectionS(S, configuration)
    for Spin in torch.arange(-3,4)/2:
        sector = Sector(None, Spin)
        s, d = sp.S(sector), sp.weight(sector)
        mean_s = torch.mean(s) / torch.mean(d)
        print(f'{str(sector):52s} {mean_s:+.4f}')

    # and both number and total spin.
    cp = ProjectionNS(S, configuration[::10])
    for N in range(1, 6):
        for Spin in torch.arange(-N,N+1,2)/2:
            sector = Sector(N, Spin)
            n, s, d = cp.N(sector), cp.S(sector), cp.weight(sector)
            mean_n = torch.mean(n) / torch.mean(d)
            mean_s = torch.mean(s) / torch.mean(d)
            print(f'{str(sector):35s} {mean_n:+.4f}  {mean_s:+.4f}') 

if __name__ == '__main__':
    _demo()
