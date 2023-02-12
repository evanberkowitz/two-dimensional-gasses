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

@observable
def _UUPlusOne(ensemble):
    # A matrix for each configuration.
    return torch.eye(2*ensemble.Action.Spacetime.Lattice.sites) + ensemble._UU

@observable
def _detUUPlusOne(ensemble):
    return torch.det(ensemble._UUPlusOne)

@observable
def _UUPlusOneInverse(ensemble):
    # A matrix for each configuration.
    return torch.linalg.inv(ensemble._UUPlusOne)

@observable
def _UUPlusOneInverseUU(ensemble):
    # A matrix for each configuration.
    return torch.matmul(ensemble._UUPlusOneInverse,ensemble._UU)
