import torch
from tdg import PauliMatrix
from tdg.observable import observable, derived

@observable
def _cooper_pair_correlation(ensemble):
    r'''
    :math:`\sum_{xyzw} e^{+i(kx-ky+qz-qw)} \left\langle \tilde{\psi}^\dagger_{\alpha,x} \tilde{\psi}^\dagger_{\beta,y} \tilde{\psi}_{\sigma,z} \tilde{\psi}_{\tau,w} \right\rangle` which is a common ingredient in all pairing matrices.

    Configurations slowest, then two momenta, k and q.
    '''

    L = ensemble.Action.Spacetime.Lattice

    return L.sites**2 * (
    +   torch.einsum(   'ckqat,ckqbs->ckqabst',
                        L.ifft(L.fft(ensemble.G, axis=2), axis=1),
                        L.fft(L.ifft(ensemble.G, axis=2), axis=1),
                     )
    -   torch.einsum(   'ckqas,ckqbt->ckqabst',
                        L.ifft(L.ifft(ensemble.G, axis=2), axis=1),
                        L.fft (L.fft (ensemble.G, axis=2), axis=1),
                     )
    )

def _pairing_contraction(pairing_correlation, P):
    r'''
    This performs the contraction with the P matrices to give the Hermitian :math:`M^{S,m}_{k,q} \left\langle \tilde{\Delta}^\dagger_{S,m,k} \tilde{\Delta}_{S,m,q} - \delta_{kq} \tilde{\Delta}^\dagger_{S,m,k} \tilde{\Delta}_{S,m,k} \right\rangle
`
    '''

    unsubtracted = torch.einsum('ab,ckqabst,st->ckq', P.adjoint(), pairing_correlation, P)
    M = unsubtracted * (1 - torch.eye(*unsubtracted.shape[1:]).repeat(unsubtracted.shape[0], 1, 1))
    return 0.5 * (M + M.mH)


@observable
def pairing_singlet(ensemble):
    r'''
    :math:`P_{0,0} = \frac{-i \sigma_2}{\sqrt{2}}`.

    Bootstraps first, then two momenta, k and q.
    '''

    P = -1.j / torch.tensor(2.).sqrt() * PauliMatrix[2]
    return _pairing_contraction(ensemble._cooper_pair_correlation, P)

@observable
def pairing_triplet_plus(ensemble):
    r'''
    :math:`P_{1,+1} = \frac{1}{2}\left( \sigma_0 + \sigma_3 \right)`.

    Bootstraps first, then two momenta, k and q.
    '''

    P = 0.5 * ( PauliMatrix[0] + PauliMatrix[3] )
    return _pairing_contraction(ensemble._cooper_pair_correlation, P)

@observable
def pairing_triplet_zero(ensemble):
    r'''
    :math:`P_{1,0} = \frac{\sigma_1}{\sqrt{2}}`.

    Bootstraps first, then two momenta, k and q.
    '''
    
    P = PauliMatrix[1] / torch.tensor(2.).sqrt()
    return _pairing_contraction(ensemble._cooper_pair_correlation, P)

@observable
def pairing_triplet_minus(ensemble):
    r'''
    :math:`P_{1,-1} = \frac{1}{2}\left( \sigma_0 - \sigma_3 \right)`.

    Bootstraps first, then two momenta, k and q.
    '''

    P = 0.5 * ( PauliMatrix[0] - PauliMatrix[3] )
    return _pairing_contraction(ensemble._cooper_pair_correlation, P)

@observable
def pairing_up_down(ensemble):
    r'''
    :math:`P = (\sigma_1 + i \sigma_2)/2`, which matches Refs. :cite:`PhysRevA.92.033603,PhysRevLett.123.136402,PhysRevLett.129.076403`.

    Bootstraps first, then two momenta, k and q.
    '''

    P = 0.j + torch.tensor([[0., 1.], [0., 0.]])
    unsubtracted = torch.einsum('ab,ckqabst,st->ckq', P.adjoint(), ensemble._cooper_pair_correlation, P)
    M = unsubtracted * (1 - torch.eye(*unsubtracted.shape[1:]).repeat(unsubtracted.shape[0], 1, 1))
    return _pairing_contraction(ensemble._cooper_pair_correlation, P)

