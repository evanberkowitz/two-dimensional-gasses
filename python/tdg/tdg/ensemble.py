#!/usr/bin/env python

from functools import cached_property
from functools import lru_cache as cached

import torch
import functorch

import tdg

def _no_op(x):
    return x

class GrandCanonical:
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
        hhat = self.Action.FermionMatrix.hhat.to(torch.complex128)
        absh = self.Action.FermionMatrix.absh.to(torch.complex128)
        
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


def _demo(steps=100):

    S = tdg.action._demo()

    import tdg.HMC as HMC
    H = HMC.Hamiltonian(S)
    integrator = HMC.Omelyan(H, 50, 1)
    hmc = HMC.MarkovChain(H, integrator)

    from tqdm import tqdm
    ensemble = GrandCanonical(S).generate(steps, hmc, progress=tqdm)

    return ensemble

if __name__ == '__main__':
    ensemble = _demo()
    print(f"The fermionic estimator for the total particle number is {ensemble.N('fermionic').mean():+.4f}")
    print(f"The bosonic   estimator for the total particle number is {ensemble.N('bosonic'  ).mean():+.4f}")
    print(f"The Spin[0]   estimator for the total particle number is {ensemble.Spin[0].mean()       :+.4f}")
