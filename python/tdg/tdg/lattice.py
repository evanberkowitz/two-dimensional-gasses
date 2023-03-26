#!/usr/bin/env python

from functools import cached_property
import numpy as np
import torch

from tdg.h5 import H5able

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

class Lattice(H5able):

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
                Either one coordinate pair of `.shape==torch.Size([2])` or a set of pairs `.shape==torch.Size([*,2])`
                The last dimension should be of size 2.

        Returns
        -------
            torch.tensor
                Each x is identified with an entry of ``coordinates`` by periodic boundary conditions.
                The output is the same shape as the input.
        '''

        if x.ndim == 1:
            return torch.tensor([
                    self.x[torch.remainder(x[0],self.nx)],
                    self.y[torch.remainder(x[1],self.ny)],
                ])

        return torch.stack((
            self.x[torch.remainder(x.T[0],self.nx)],
            self.y[torch.remainder(x.T[1],self.ny)],
            )).mT

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
                Either ``a`` and ``b`` are the same shape (a single or 1D-tensor of coordinate pairs) or one is a singlet and one is a tensor.
        '''
        d = self.mod(a-b)
        if d.ndim == 1:
            return torch.sum(d**2)

        return torch.sum(d**2, axis=(1,))

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

    def vector(self, *dims):
        r'''
        Parameters
        ----------
            dims:   tuple
                Specifies how how many vectors to produce.
                
        Returns
        -------
            torch.tensor:
                A ``dims``-dimensional stack of linearized zero vectors.
        '''
        return torch.zeros(self.sites).repeat(*dims, 1)

    def fft(self, vector, axis=-1, norm='backward'):
        r'''The Fourier transform on a linearized axis.

        Parameters
        ----------
            vector: torch.tensor
                A vector of data.
            axis:
                The axis along which to perform a 2D Fourier transform on the vector.
            norm:
                A `convention for the Fourier transform`_, one of ``"forward"``, ``"backward"``, or ``"ortho"``.
                The default is `"backward"`, to match our notes.

        Returns
        -------
            torch.tensor:
                F(vector) with the same shape as the input vector, transformed along the axis.

        .. _convention for the Fourier transform: https://pytorch.org/docs/stable/generated/torch.fft.fft2.html#torch.fft.fft2
        '''
        fft_axes = (axis-1, axis) if axis < 0 else (axis, axis+1)
        return self.linearize(torch.fft.fft2(self.coordinatize(vector, dims=(axis,)), dim=fft_axes, norm=norm), dims=(axis,))

    def ifft(self, vector, axis=-1, norm='backward'):
        r'''The Fourier inverse transform on a linearized axis.

        Parameters
        ----------
            vector: torch.tensor
                A vector of data.
            axis:
                The axis along which to perform a 2D inverse Fourier transform on the vector.
            norm:
                A `convention for the inverse Fourier transform`_, one of ``"forward"``, ``"backward"``, or ``"ortho"``.
                The default is `"backward"`, to match our notes.

        Returns
        -------
            torch.tensor:
                Inverse[F](vector) with the same shape as the input vector, transformed along the axis.

        .. _convention for the inverse Fourier transform: https://pytorch.org/docs/stable/generated/torch.fft.ifft2.html#torch.fft.ifft2
        '''
        fft_axes = (axis-1, axis) if axis < 0 else (axis, axis+1)
        return self.linearize(torch.fft.ifft2(self.coordinatize(vector, dims=(axis,)), dim=fft_axes, norm=norm), dims=(axis,))

    @cached_property
    def adjacency_tensor(self):
        r'''
        The `adjacency_matrix` but the two superindices are transformed into coordinate indices.
        '''
        return self.coordinatize(self.adjacency_matrix, (0,1))

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

        # We know kappa_ab = Σ(k) (2πk)^2/2V exp( -2πik•(a-b)/Nx )
        # where a are spatial coordinates
        a = self.coordinates +0j # cast to complex for einsum
        # and k are also integer doublets;
        # in the exponent we group the -2πi / Nx into the momenta
        p = (-2*torch.pi*1j / self.nx) * a
        # Separating out the unitary U_ak = 1/√V exp( - 2πik•a )
        U = torch.exp(torch.einsum('ax,kx->ak', a, p)) / np.sqrt(self.sites)

        # we can write isolate the eigenvalues
        #   kappa_kq = δ_kq ( 2π k )^2 / 2
        eigenvalues = ((2*torch.pi)**2 ) * self.ksq.flatten() / 2

        # via the unitary transformation
        # kappa_ab = Σ(kq) U_ak kappa_kq U*_qb
        #          = Σ(k)  U_ak [(2πk)^2/2] U_kb
        return torch.einsum('ak,k,kb->ab', U, eigenvalues, U.conj().transpose(0,1))
        #
        # Can be checked by eg.
        # ```
        # lattice = tdg.Lattice(11)
        # left = torch.linalg.eigvalsh(lattice.kappa).sort().values 
        # right = torch.tensor(4*np.pi**2 * lattice.ksq.ravel() / 2).sort().values
        # (torch.abs(left-right) < 1e-6).all()
        # ```

    @cached_property
    def convolver(self):
        r'''
        The convolution of two vectors :math:`u` and :math:`v` is :math:`u * v = \frac{1}{V} \sum_a u_a v_{a-r}`.

        The convolution can be computed quickly using fast fourier transforms.  However, sometimes it is useful to implement
        the convolution as part of a tensor contraction.

        We can introduce this via the *convolver*, which satisfies

        .. math::

           \frac{1}{V} \sum_a u_a v_{a-r} = \sum_{ab} u_a\, v_b\, \texttt{convolver}_{bra} \text{ (note the order)}

        which can be implemented via einsum,

        .. code::

           u * v = torch.einsum('a,b,bra->r', u, v, convolver) # order as above

        where ``u`` and ``v`` have one linearized spatial index.

        .. note::

           This **includes** the factor of volume!
        '''

        # Here is an obviously-correct implementation.
        #
        # for i,r in enumerate(self.coordinates):
        #     for j,a in enumerate(selself.coordinates):
        #         diff = self.mod(a-r)
        #         for k,b in enumerate(self.coordinates):
        #             if (b==diff).all():
        #                 direct[k,i,j] = 1./self.sites
        #
        # but we opt for a slightly more sophisticated and torch-native
        normalized_identity = torch.eye(self.sites)/self.sites # include the 1/V factor

        return torch.stack(
            list(
                self.linearize(
                    self.coordinatize(
                        normalized_identity,
                        (0,)
                        ).roll(shifts=tuple(r), dims=(0,1)),
                    (0,)
                )
                for r in self.coordinates # r slowest
            )
        ).permute((2,0,1)) # (b=a-r) r a order

def _demo(nx=7):
    return Lattice(nx)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
