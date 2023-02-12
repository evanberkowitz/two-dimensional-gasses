import torch
import tdg
from tdg.observable import observable, callable_observable

####
#### Intensive
####

@observable
def _spin(ensemble):
    r'''
    The local spin density.
    Direction slowest, then configurations, then sites.  That makes it easy to do something with ``ensemble.s[1]``.

    .. note ::
        The indices on the spins match the indices on :data:`tdg.PauliMatrix`.
        The ``[0]`` entry matches ``ensemble.n('fermionic')``, a useful check.
    '''

    dt   = ensemble.Action.dt
    hhat = ensemble.Action.hhat + 0j
    absh = ensemble.Action.absh + 0j
    
    if absh == 0.:
        Pspin = torch.diag(torch.tensor((0,0,0.5))) + 0j
    else:
        Pspin = (
            0.5 * torch.outer(hhat,hhat)
            # sinh(x/2) * cosh(x/2) = 0.5 * sinh(x)
            + 0.5*torch.sinh(dt*absh) / (dt * absh) * (torch.eye(3) - torch.outer(hhat,hhat)) 
            # sinh^2(x/2) = 1/2 * (cosh(x) - 1)
            + 0.5j * (torch.cosh(0.5*dt*absh)-1) / (dt * absh) * torch.einsum('ijk,j->ik', tdg.epsilon + 0j, hhat)
           )
    
    # expand to include 'all four' "spin" matrices.
    Pspin = torch.nn.ConstantPad2d((1,0,1,0), 0)(Pspin)
    Pspin[0,0] = 1.
    # and construct a vector of paulis.
    Pspin = torch.einsum('ik,kab->iab', Pspin, tdg.PauliMatrix)
    
    return torch.einsum('cxxab,iba->icx', ensemble._matrix_to_tensor(ensemble._UUPlusOneInverseUU), Pspin)

@callable_observable
def spin(ensemble, direction):
    r'''
    The local spin density.  Configurations, then sites.
    
    Parameters
    ----------
        direction: int
            An integer indexing the spin direction, which matches the index of :data:`tdg.PauliMatrix`.  :code:`0` is equal to :code:`n('fermionic')`.
    
    Returns
    -------
        torch.tensor
    '''
    return ensemble._spin[direction]
    
####
#### Extensive
####

@observable
def _Spin(ensemble):
    return ensemble._spin.sum(-1)

@callable_observable
def Spin(ensemble, direction):
    r'''
    The total spin, summed over sites.  One per configuration.
    
    Parameters
    ----------
        direction: int
            An integer indexing the spin direction, which matches the index of :data:`tdg.PauliMatrix`.  :code:`0` is equal to :code:`N('fermionic')`.
    
    Returns
    -------
        torch.tensor
    '''
    return ensemble._Spin[direction]


