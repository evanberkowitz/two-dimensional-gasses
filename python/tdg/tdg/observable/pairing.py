import torch
from tdg import PauliMatrix
from tdg.observable import observable, derived

# Throughout we use the projection matrices P for the spin singlet and triplet
# These assume that we're interested in pairing wavefunctions aligned along the z-axis.
# If h≠0 then the triplet will split and you'd better make sure that you're looking along the axis parallel to h.
# However
# TODO: We leave this for the future.  We can presumably think in two ways.
# First:  rotate these projectors before the computations below [presumably easy]
# Second: rotate the answers using Wigner D matrices [presumably complicated]
P_singlet = -1.j / torch.tensor(2.).sqrt() * PauliMatrix[2]
r''':math:`P_{0,0} = \frac{-i \sigma_2}{\sqrt{2}}`'''

P_triplet_plus = 0.5 * ( PauliMatrix[0] + PauliMatrix[3] )
r''':math:`P_{1,+1} = \frac{1}{2}\left( \sigma_0 + \sigma_3 \right)`.'''

P_triplet_zero = PauliMatrix[1] / torch.tensor(2.).sqrt()
r''':math:`P_{1,0} = \frac{\sigma_1}{\sqrt{2}}`'''

P_triplet_minus= 0.5 * ( PauliMatrix[0] - PauliMatrix[3] )
r''':math:`P_{1,-1} = \frac{1}{2}\left( \sigma_0 - \sigma_3 \right)`.'''

# We always need to do contractions that look exactly the same:
# P†_ab P_st [ G^{at}_{k,q} G^{bs}_{-k,-q} - G^{as}_{k,-q} G^{bt}_{-k,q} ]
# but the quantum expectation values group terms differently.
# In the connected case < ∆† ∆ > the contractions are performed for each configuration.
# In the disconnected case the contractions are performed but already on expectation values.

####
#### FERMION CONTRACTIONS
####

# We can think conceptually that first we combine the propagators into one object with momentum and spin indices
# and then reuse that result over and over again no matter P.
def _four_fermion_contraction(L, G):
    r'''
    Computes the Wick contractions needed for both the connected and disconnected pieces.
    This is a convenience function that is reused in both a stored (and reused) observable and in computing the quantum disconnected piece.

    Parameters
    ----------
    Lattice:    tdg.Lattice
        The lattice with which to fourier-transform G
    G:          torch.tensor
        Propagators, one for each configuration or bootstrap sample.
    '''

    # When G is from an ensemble think: this is quantum-connected.
    # When G is from a  bootstrap think: this is quantum-disconnected.

    return L.sites**2 * (
    +   torch.einsum(   'ckqat,ckqbs->ckqabst',
                        L.ifft(L.fft(G, axis=2), axis=1),
                        L.fft(L.ifft(G, axis=2), axis=1),
                     )
    -   torch.einsum(   'ckqas,ckqbt->ckqabst',
                        L.ifft(L.ifft(G, axis=2), axis=1),
                        L.fft (L.fft (G, axis=2), axis=1),
                     )
    )

# We can store this intermediate result as an observable to be further reduced.
# However, we should not just implement _four_fermion_contraction directly in this observable
# because we need exactly the same contraction routines for the quantum-disconnected terms.

# It may be wiser to eliminate this stored intermediate result and to just recompute the contractions
# for each spin projection matrix.  It's a trade-off between time and space.  This is the space-agnostic
# choice, but it might be smarter to make the time-agnostic choice.
@observable
def _cooper_pair_correlation(ensemble):
    r'''
    :math:`\sum_{xyzw} e^{+i(kx-ky+qz-qw)} \left\langle \tilde{\psi}^\dagger_{\alpha,x} \tilde{\psi}^\dagger_{\beta,y} \tilde{\psi}_{\sigma,z} \tilde{\psi}_{\tau,w} \right\rangle` which is a common ingredient in all pairing matrices and therefore stored.

    Configurations slowest, then two momenta, k and q.
    '''

    return _four_fermion_contraction(ensemble.Action.Spacetime.Lattice, ensemble.G)

####
#### SPIN PROJECTION
####

# Once we have the complete correlation we can reduce to any particular channel, given P.
# Then we are left with just the configuration or bootstrap index and two momenta.
def _pairing_contraction(pairing_correlation, P):
    r'''
    Compues the contraction with the P matrices to give the Hermitian :math:`M^{S,m}_{k,q} \left\langle \tilde{\Delta}^\dagger_{S,m,k} \tilde{\Delta}_{S,m,q} - \delta_{kq} \tilde{\Delta}^\dagger_{S,m,k} \tilde{\Delta}_{S,m,k} \right\rangle`
    '''

    c = torch.einsum('ab,ckqabst,st->ckq', P.adjoint(), pairing_correlation, P)

    # Enforce Hermiticity
    return 0.5 * (c + c.mH)

####
#### FINAL RESULTS
####

# Under the bootstrap we want to evaluate the difference
# P†_ab P_st [ G^{at}_{k,q}     G^{bs}_{-k,-q}   -   G^{as}_{k,-q}     G^{bt}_{-k,q} ]
#            <                                                                       >
#          -(<              > <                > - <               > <               >)
#
# The first term is a primary observable, but this combination has quantum-disconnected pieces
# which can only be evaluated bootstrap-by-bootstrap.

def _quantum_disconnected(bootstrapped, P):
    return _pairing_contraction(_four_fermion_contraction(bootstrapped.Action.Spacetime.Lattice, bootstrapped.G), P)

def _pairing_matrix(pair_pair, disconnected, N, L):
    r'''Generic :math:`k_F^4 M / (N/2)` '''
    return torch.einsum('bkq,b->bkq',
                        pair_pair - disconnected,
                        (8*torch.pi**2 / L.sites**2 ) * N,
                        )


# leggett:2006 defines the 'condenate fraction' as the largest eigenvalue of this difference normalized by N/2.
# PhysRevA.92.033603,PhysRevLett.123.136402,PhysRevLett.129.076403 call the corresponding eigenvector the 'pairing wavefunction'.
#
# There being only one extensive eigenvalue is NOT the only possibility.  leggett:2006 calls multiple extensive eigenvalues
# a 'fractured condensate', which is something we can investigate.  However, for now we just assume an unfractured condensate
# and look at the largest eigenvalue.
def _eigen_answer(pairing):

    eigenvalues, eigenvectors = torch.linalg.eigh(pairing)
    # All the eigenvalues are positive because we're looking into the square of something.
    _, indices = torch.max(eigenvalues, dim=-1, keepdim=True)

    leading_eigenvalue  = eigenvalues.gather(-1, indices)
    pairing_wavefunction= eigenvectors.gather(-1, indices.unsqueeze(-1).expand((*eigenvectors.shape[:-1], 1)))
    return leading_eigenvalue, pairing_wavefunction

# Now we are ready to use the above infrasturcture to compute channel-specific observables.
# In each channel in (singlet, triplet_plus, triplet_zero, triplet_minus) we will define 5 functions
# pair_pair_channel             is the quantum-connected piece
# pairing_channel               is the difference
# _pairing_channel_eigen        does the eigenvalue analysis and stores pairs
# condensate_fraction_channel   gives the intensive quantity
# pairing_wavefunction_channel  gives the paring wavefunction

# In docs/observables/pairing.rst we give the documentation in a channel-agnostic way.

####
#### SINGLET
####

@observable
def pair_pair_singlet(ensemble):
    return _pairing_contraction(ensemble._cooper_pair_correlation, P_singlet)

@derived
def pair_pair_eigenvalues_singlet(ensemble):

    L = ensemble.Action.Spacetime.Lattice

    return torch.einsum('bl,b->bl', torch.linalg.eigvalsh(ensemble.pair_pair_singlet), 2./L.sites**2 / ensemble.N)

@derived
def pairing_singlet(ensemble):
    r'''
    Bootstraps first, then two momenta, k and q.
    '''

    return _pairing_matrix(
            ensemble.pair_pair_singlet,
            _quantum_disconnected(ensemble, P_singlet),
            ensemble.N, ensemble.Action.Spacetime.Lattice)

@derived
def _pairing_singlet_eigen(ensemble):
    return _eigen_answer(ensemble.pairing_singlet)

@derived
def condensate_fraction_singlet(ensemble):
    return ensemble._pairing_singlet_eigen[0]

@derived
def pairing_wavefunction_singlet(ensemble):
    return ensemble._pairing_singlet_eigen[1]


####
#### TRIPLET PLUS
####

@observable
def pair_pair_triplet_plus(ensemble):
    return _pairing_contraction(ensemble._cooper_pair_correlation, P_triplet_plus)

@derived
def pair_pair_eigenvalues_triplet_plus(ensemble):

    L = ensemble.Action.Spacetime.Lattice

    return torch.einsum('bl,b->bl', torch.linalg.eigvalsh(ensemble.pair_pair_triplet_plus), 2./L.sites**2 / ensemble.N)

@derived
def pairing_triplet_plus(ensemble):
    r'''
    

    Bootstraps first, then two momenta, k and q.
    '''

    return _pairing_matrix(
            ensemble.pair_pair_triplet_plus,
            _quantum_disconnected(ensemble, P_triplet_plus),
            ensemble.N, ensemble.Action.Spacetime.Lattice)

@derived
def _pairing_triplet_plus_eigen(ensemble):
    return _eigen_answer(ensemble.pairing_triplet_plus)

@derived
def condensate_fraction_triplet_plus(ensemble):
    return ensemble._pairing_triplet_plus_eigen[0]

@derived
def pairing_wavefunction_triplet_plus(ensemble):
    return ensemble._pairing_triplet_plus_eigen[1]


####
#### TRIPLET ZERO
####

@observable
def pair_pair_triplet_zero(ensemble):
    return _pairing_contraction(ensemble._cooper_pair_correlation, P_triplet_plus)

@derived
def pair_pair_eigenvalues_triplet_zero(ensemble):

    L = ensemble.Action.Spacetime.Lattice

    return torch.einsum('bl,b->bl', torch.linalg.eigvalsh(ensemble.pair_pair_triplet_zero), 2./L.sites**2 / ensemble.N)

@derived
def pairing_triplet_zero(ensemble):
    r'''
    :math:`P_{1,+1} = \frac{1}{2}\left( \sigma_0 + \sigma_3 \right)`.

    Bootstraps first, then two momenta, k and q.
    '''

    return _pairing_matrix(
            ensemble.pair_pair_triplet_zero,
            _quantum_disconnected(ensemble, P_triplet_zero),
            ensemble.N, ensemble.Action.Spacetime.Lattice)

@derived
def _pairing_triplet_zero_eigen(ensemble):
    return _eigen_answer(ensemble.pairing_triplet_zero)

@derived
def condensate_fraction_triplet_zero(ensemble):
    return ensemble._pairing_triplet_zero_eigen[0]

@derived
def pairing_wavefunction_triplet_zero(ensemble):
    return ensemble._pairing_triplet_zero_eigen[1]


####
#### TRIPLET MINUS
####

@observable
def pair_pair_triplet_minus(ensemble):
    return _pairing_contraction(ensemble._cooper_pair_correlation, P_triplet_minus)

@derived
def pair_pair_eigenvalues_triplet_minus(ensemble):

    L = ensemble.Action.Spacetime.Lattice

    return torch.einsum('bl,b->bl', torch.linalg.eigvalsh(ensemble.pair_pair_triplet_minus), 2./L.sites**2 / ensemble.N)

@derived
def pairing_triplet_minus(ensemble):
    r'''
    

    Bootstraps first, then two momenta, k and q.
    '''

    return _pairing_matrix(
            ensemble.pair_pair_triplet_minus,
            _quantum_disconnected(ensemble, P_triplet_minus),
            ensemble.N, ensemble.Action.Spacetime.Lattice)

@derived
def _pairing_triplet_minus_eigen(ensemble):
    return _eigen_answer(ensemble.pairing_triplet_minus)

@derived
def condensate_fraction_triplet_minus(ensemble):
    return ensemble._pairing_triplet_minus_eigen[0]

@derived
def pairing_wavefunction_triplet_minus(ensemble):
    return ensemble._pairing_triplet_minus_eigen[1]


####
#### UP / DOWN
####

# PhysRevA.92.033603,PhysRevLett.123.136402,PhysRevLett.129.076403 use a different projector which
# gives an operator with definite up and down labels for the +momentum and -momentum fermions.

P_up_down = 0.j + torch.tensor([[0., 1.], [0., 0.]])

@observable
def pair_pair_up_down(ensemble):
    return _pairing_contraction(ensemble._cooper_pair_correlation, P_up_down)

@derived
def pair_pair_eigenvalues_up_down(ensemble):

    L = ensemble.Action.Spacetime.Lattice

    return torch.einsum('bl,b->bl', torch.linalg.eigvalsh(ensemble.pair_pair_up_down), 2./L.sites**2 / ensemble.N)

@derived
def pairing_up_down(ensemble):
    r'''
    :math:`P_{1,+1} = \frac{1}{2}\left( \sigma_0 + \sigma_3 \right)`.

    Bootstraps first, then two momenta, k and q.
    '''

    return _pairing_matrix(
            ensemble.pair_pair_up_down,
            _quantum_disconnected(ensemble, P_up_down),
            ensemble.N, ensemble.Action.Spacetime.Lattice)

@derived
def _pairing_up_down_eigen(ensemble):
    return _eigen_answer(ensemble.pairing_up_down)

@derived
def condensate_fraction_up_down(ensemble):
    return ensemble._pairing_up_down_eigen[0]

@derived
def pairing_wavefunction_up_down(ensemble):
    return ensemble._pairing_up_down_eigen[1]
