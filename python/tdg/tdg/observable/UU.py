import torch
import functorch
import tdg.ensemble
from tdg.observable import observable

# These utility functions help turn a doubly-struck sausage UU into a tensor, and back.
def _matrix_to_tensor(ensemble, matrix):
    V = ensemble.Action.Spacetime.Lattice.sites
    return matrix.unflatten(-2, (V, 2)).unflatten(-1, (V, 2)).transpose(-3,-2)

def _tensor_to_matrix(ensemble, tensor):
    return tensor.transpose(-3,-2).flatten(start_dim=-4, end_dim=-3).flatten(start_dim=-2, end_dim=-1)

tdg.ensemble.GrandCanonical._matrix_to_tensor = _matrix_to_tensor
tdg.ensemble.GrandCanonical._tensor_to_matrix = _tensor_to_matrix

# Most of the intermediates needed for the observables are only evaluated lazily, due to computational cost.
# Once they are evaluated, they're stored.
# This makes the creation of an ensemble object almost immediate.

@observable
def _UU(ensemble):
    # A matrix for each configuration.
    return functorch.vmap(ensemble.Action.FermionMatrix.UU)(ensemble.configurations)

def _detUUPlusOne(ensemble):
    # One per configuration.
    UUPlusOne = torch.eye(2*ensemble.Action.Spacetime.Lattice.sites) + ensemble._UU
    return torch.det(UUPlusOne)

@observable
def _UUPlusOneInverseUU(ensemble):
    # A matrix for each configuration.
    UUPlusOne = torch.eye(2*ensemble.Action.Spacetime.Lattice.sites) + ensemble._UU

    # TODO: do this via a solve rather than a true inverse?
    inverse = torch.linalg.inv(UUPlusOne)
    return torch.matmul(inverse, ensemble._UU)

@observable
def G(ensemble):
    r'''
    The equal-time propagator that is the contraction of :math:`\psi^\dagger_{a\sigma} \psi_{b\tau}`
    where :math:`a` and :math:`b` are sites and :math:`\sigma` and :math:`\tau` are spins.

    .. math ::
       \mathcal{G} = [ \mathbb{U} (\mathbb{1} + \mathbb{U})^{-1} ]_{ab}^{\sigma\tau}
    
    A five-axis tensor: configurations slowest, then :math:`a`, :math:`b`, :math:`\sigma`, and :math:`\tau`.
    '''
    return ensemble._matrix_to_tensor(ensemble._UUPlusOneInverseUU)

@observable
def G_momentum(ensemble):
    r'''
    The equal-time propagator that is the contraction of :math:`N_x^{-2} \psi^\dagger_{k\sigma} \psi_{q\tau}`
    where :math:`k` and :math:`q` are integer momenta and :math:`\sigma` and :math:`\tau` are spins.

    .. math ::
       \mathcal{G}^{\sigma\tau}_{kq} = \frac{1}{N_x^2} \sum_{xy} e^{+2\pi i k x / N_x} \mathcal{G}^{\sigma\tau}_{xy} e^{-2\pi i q y / N_x}
    
    A five-axis tensor: configurations slowest, then :math:`k`, :math:`q`, :math:`\sigma`, and :math:`\tau`.
    '''
    
    L = ensemble.Action.Spacetime.Lattice

    # Checked via brute force:
    #
    #   # Warning: takes a long time even with this small ensemble!
    #   ensemble = tdg.ensemble._demo(nx=5, steps=3) 
    #   
    #   G = ensemble.G
    #   L = ensemble.Action.Spacetime.Lattice
    #   gp = torch.zeros_like(G)
    #   
    #   for c, g in enumerate(G):
    #       for a, k in enumerate(L.coordinates):
    #           for b, q in enumerate(L.coordinates):
    #               for i, x in enumerate(L.coordinates):
    #                   for j,y in enumerate(L.coordinates):
    #                       gp[c,a,b] += 1 / L.sites * g[i,j] * torch.exp(+2j*torch.pi * ( torch.dot(k,x) - torch.dot(q, y)) / L.nx)
    #   
    #   (gp - ensemble.G_momentum).abs().sum() < 1.e-12
    # 
    # which yields tensor(True).
    # However, because G is real in this test we may be tricking ourselves...

    return L.ifft(L.fft(ensemble.G, axis=2), axis=1)
