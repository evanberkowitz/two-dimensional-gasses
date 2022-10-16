#!/usr/bin/env python

from functools import cached_property
import numpy as np
import torch

def _dimension(n):
    '''

    Parameters
    ----------
        n:  int
            size of the dimension

    Returns
    -------
        an FFT-convention-compatible list of coordinates for a dimension of size n,
        ``[0, 1, 2, ... max, min ... -2, -1]``.
    '''
    return torch.tensor(list(range(0, n // 2 + 1)) + list(range( - n // 2 + 1, 0)), dtype=int)

class Lattice:

    def __init__(self, nx, ny=None):
        self.nx = nx

        if ny is None:
            self.ny = nx
        else:
            self.ny = ny

        if(self.nx != self.ny):
            # For the time being I will restrict to nx = ny.
            # The reason is that when nx ≠ ny more care about the kinetic matrix κ is required.
            # The main issue is that when nx ≠ ny, self.nx**2 ≠ self.sites and so the different
            # momentum components need different normalizations.
            # When nx == ny this is coincidentally correct.
            raise NotImplemented("Anisotropic lattices not currently supported")

        self.dims = torch.tensor([self.nx, self.ny])
        r'''
        The dimension sizes in order.

        >>> lattice = tdg.Lattice(5)
        >>> lattice.dims
        tensor([5, 5])
        '''
        self.sites = self.nx * self.ny
        r'''
        The total number of sites.

        >>> lattice = tdg.Lattice(5)
        >>> lattice.sites
        25
        '''

        # We want to go from -n/2 to +n/2
        self.x = _dimension(self.nx)
        r'''
        The coordinates in the x direction.

        >>> lattice = tdg.Lattice(5)
        >>> lattice.x
        tensor([ 0,  1,  2, -2, -1])
        '''
        self.y = _dimension(self.ny)
        r'''
        The coordinates in the y direction.

        >>> lattice = tdg.Lattice(5)
        >>> lattice.y
        tensor([ 0,  1,  2, -2, -1])
        '''

        # These are chosen so that Lattice(nx, ny)
        # has coordinate matrices of size (nx, ny)
        self.X = torch.tile( self.x, (self.ny, 1)).T
        r'''
        A tensor of size ``dims`` with the x coordinate as a value.

        >>> lattice = tdg.Lattice(5)
        >>> lattice.X
        tensor([[ 0,  0,  0,  0,  0],
                [ 1,  1,  1,  1,  1],
                [ 2,  2,  2,  2,  2],
                [-2, -2, -2, -2, -2],
                [-1, -1, -1, -1, -1]])
        '''
        self.Y = torch.tile( self.y, (self.nx, 1))
        r'''
        A tensor of size ``dims`` with the y coordinate as a value.

        >>> lattice = tdg.Lattice(5)
        >>> lattice.Y
        tensor([[ 0,  1,  2, -2, -1],
                [ 0,  1,  2, -2, -1],
                [ 0,  1,  2, -2, -1],
                [ 0,  1,  2, -2, -1],
                [ 0,  1,  2, -2, -1]])
        '''

        # Wavenumbers are the same
        self.kx = self.x
        self.ky = self.y
        self.KX = self.X
        self.KY = self.Y
        # To get the dimensionless wavenumber^2 in the fourier basis we can simply
        self.ksq = self.KX**2 + self.KY**2

        # We also construct a linearized list of coordinates.
        # The order matches self.X.ravel() and self.Y.ravel()
        self.coordinates = torch.stack((self.X.flatten(), self.Y.flatten())).T
        '''
        A tensor of size ``[sites, len(dims)]``.  Each row contains a pair of coordinates.  The order matches ``{X,Y}.flatten()``.

        >>> lattice = tdg.Lattice(5)
        >>> lattice.coordinates
        >>> lattice.coordinates
        tensor([[ 0,  0],
                [ 0,  1],
                [ 0,  2],
                [ 0, -2],
                [ 0, -1],
                [ 1,  0],
                [ 1,  1],
                [ 1,  2],
                [ 1, -2],
                [ 1, -1],
                [ 2,  0],
                [ 2,  1],
                [ 2,  2],
                [ 2, -2],
                [ 2, -1],
                [-2,  0],
                [-2,  1],
                [-2,  2],
                [-2, -2],
                [-2, -1],
                [-1,  0],
                [-1,  1],
                [-1,  2],
                [-1, -2],
                [-1, -1]])
        '''

    def __str__(self):
        return f'SquareLattice({self.nx},{self.ny})'

    def __repr__(self):
        return str(self)

    def mod(self, x):
        r'''
        Mod integer coordinates x into values on the lattice.

        Parameters
        ----------
            x:  torch.tensor
                The last dimension should be of size 2.

        Returns
        -------
            torch.tensor
                Each x is identified with an entry of ``coordinates`` by periodic boundary conditions.
        '''
        return torch.stack((
            self.x[torch.remainder(x.T[0],self.nx)],
            self.y[torch.remainder(x.T[1],self.ny)],
            )).T

    def distance_squared(self, a, b):
        r'''
        .. math::
            \texttt{distance_squared}(a,b) = \left| \texttt{mod}(a - b)\right|^2

        Parameters
        ----------
            a:  torch.tensor
                coordinates that need not be on the lattice
            b:  torch.tensor
                coordinates that need not be on the lattice

        Returns
        -------
            torch.tensor
                The distance between ``a`` and ``b`` on the lattice accounting for the fact that,
                because of periodic boundary conditions, the distance may shorter than naively expected.
        '''
        d = self.mod(a-b)
        return torch.sum(d.T**2, axis=(0,))

    def tensor(self, n=2):
        # can do matrix-vector via
        #   torch.einsum('ijab,ab',matrix,vector)
        # to get a new vector (with indices ij).
        return torch.zeros(np.tile(self.dims, n).tolist())

    def tensor_linearized(self, tensor):
        repeats = len(tensor.shape)
        return tensor.reshape(*np.tile(self.sites, int(repeats / len(self.dims))))

    def linearized_tensor(self, linearized):
        return linearized.reshape(*np.tile(self.dims, len(linearized.shape)))

    def coordinatize(self, v, dims=(-1,)):
        r'''
        Unflattens all the dims from a linear superindex to one index for each dimension in ``.dims``.
        
        Parameters
        ----------
            v: torch.tensor
            dims: tuple of integers
            
        Returns
        -------
            torch.tensor
                ``v`` but tensor more, shorter dimensions.  Dimensions specified by ``dims`` are unflattened.
        '''
        
        v_dims  = len(v.shape)

        # We'll build up the new shape by considering each index left-to-right.
        # So, for negative indices we need to mod them by the number of dimensions.
        to_reshape, _ = torch.sort(torch.remainder(torch.tensor(dims), v_dims))
        
        new_shape = tuple(torch.cat(tuple( # Assemble a tuple which has
                # the size s of the dimension if we're not unflattening it
                # or the dimensions of the lattice if we are unflattening.
                torch.tensor([s]) if i not in to_reshape else self.dims
                for i, s in enumerate(v.shape)) 
            ))
        return v.reshape(new_shape)

    def linearize(self, v, dims=(-1,)):
        r'''
        Flattens adjacent dimensions of v with shape ``.dims`` into a dimension of size ``.sites``.
        
        Parameters
        ----------
            v:  torch.tensor
            dims: tuples of integers that specify that dimensions *in the result* that come from flattening.
                Modded by the dimension of the resulting tensor so that any dimension is legal.
                However, one should take care to ensure that no two are the SAME index of the result;
                this causes a RuntimeError.
            
        Returns
        -------
            torch.tensor
                ``v`` but with fewer, larger dimensions

        .. note::
            The ``dims`` parameter may be a bit confusing.  This perhaps-peculiar convention is to make it easier to
            combine with ``coordinatize``.  ``linearize`` and ``coordinatize`` are inverses when they get *the same* 
            dims arguments.

            >>> import torch
            >>> import tdg
            >>> nx = 5
            >>> dims = (0, -1)
            >>> lattice = tdg.Lattice(5)
            >>> v = torch.arange(nx**(2*3)).reshape(nx**2, nx**2, nx**2)
            >>> u = lattice.coordinatize(v, dims)
            >>> u.shape
            torch.Size([5, 5, 25, 5, 5])
            >>> w = lattice.linearize(u, dims) # dims indexes into the dimensions of w, not u!
            >>> w.shape
            torch.Size([25, 25, 25])
            >>> (v == w).all()
            tensor(True)

        '''

        shape   = v.shape
        v_dims  = len(shape)
        
        dm = set(dims)
        
        future_dims = v_dims - (len(self.dims)-1) * len(dm)
        dm = set(d % future_dims for d in dm)
            
        new_shape = []
        idx = 0
        for i in range(future_dims):
            if i not in dm:
                new_shape += [shape[idx]]
                idx += 1
            else:
                new_shape += [self.sites]
                idx += len(self.dims)
        try:
            return v.reshape(new_shape)
        except RuntimeError as error:
            raise ValueError(f'''
            This happens when two indices to be linearized are accidentally the same.
            For example, for a lattice of size [5,5], if v has .shape [x, y, 5, 5]
            and you linearize(v, (2,-1)) the 2 axis and the -1 axis would refer to
            the same axis in [x, y, 25].
            
            Perhaps this happened with your vector of shape {v.shape} and {dims=}?
            ''') from error

    def vector(self):
        return self.tensor(1)

    def fft(self, vector, axes=(-2,-1), norm='ortho'):
        return torch.fft.fft2(vector, dim=axes, norm=norm)

    def ifft(self, vector, axes=(-2,-1), norm='ortho'):
        return torch.fft.ifft2(vector, dim=axes, norm=norm)

    @cached_property
    def adjacency_tensor(self):
        r'''
        The `adjacency_matrix` but the two superindices are transformed into coordinate indices.
        '''
        return self.linearized_tensor(self.adjacency_matrix)

    @cached_property
    def adjacency_matrix(self):
        r'''
        A matrix which is 1 if the corresponding ``coordinates`` are nearest neighbors (accounting for periodic boundary conditions) and 0 otherwise.
        '''
        return torch.stack(tuple(
            torch.where(self.distance_squared(a,self.coordinates) == 1, 1, 0)
            for a in self.coordinates
            ))

    @cached_property
    def kappa(self):
        r'''
        The kinetic :math:`\kappa` with a perfect dispersion relation, as a ``[sites, sites]`` matrix.
        '''
        # Makes an assumption that nx = ny

        # We know kappa_ab = 1/V Σ(k) (2πk)^2/2V exp( -2πik•(a-b)/Nx )
        # where a are spatial coordinates
        a = self.coordinates.clone().detach().to(torch.complex128)
        # and k are also integer doublets;
        # in the exponent we group the -2πi / Nx into the momenta
        p = (-2*torch.pi*1j / self.nx) * a
        # Separating out the unitary U_ak = 1/√V exp( - 2πik•a )
        U = torch.exp(torch.einsum('ax,kx->ak', a, p)) / np.sqrt(self.sites)

        # we can write isolate the eigenvalues
        #   kappa_kq = δ_kq ( 2π k / Nx )^2 / 2
        eigenvalues = ((2*torch.pi)**2 / self.sites) * self.ksq.flatten() / 2

        # via the unitary transformation
        # kappa_ab = Σ(kq) U_ak kappa_kq U*_qb
        #          = Σ(k)  U_ak [(2πk/Nx)^2/2] U_kb
        return torch.einsum('ak,k,kb->ab', U, eigenvalues, U.conj().transpose(0,1))
        #
        # Can be checked by eg.
        # ```
        # lattice = tdg.Lattice(11)
        # left = torch.linalg.eigvalsh(lattice.kappa).sort().values 
        # right = torch.tensor(4*np.pi**2 * lattice.ksq.ravel() / lattice.sites / 2).sort().values
        # (torch.abs(left-right) < 1e-6).all()
        # ```

if __name__ == "__main__":
    import doctest
    doctest.testmod()
