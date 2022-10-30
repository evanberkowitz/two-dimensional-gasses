#!/usr/bin/env python3

import torch

PauliMatrix = torch.tensor([
    [[+1,0],[0,+1]],
    [[0,+1],[+1,0]],
    [[0,-1j],[+1j,0]],
    [[+1,0],[0,-1]]
    ], dtype=torch.complex128)
r'''
A :math:`4\times2\times2` tensor of `Pauli matrices`_ :math:`\sigma`.

.. math::
    \begin{align}
        \texttt{PauliMatrix[0]} &= \begin{pmatrix} +1 & 0 \\ 0 & +1 \end{pmatrix}
        &
        \texttt{PauliMatrix[1]} &= \begin{pmatrix} 0 & +1 \\ +1 & 0 \end{pmatrix}
        \\
        \texttt{PauliMatrix[2]} &= \begin{pmatrix} 0 & -i \\ +i & 0 \end{pmatrix}
        &
        \texttt{PauliMatrix[3]} &= \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
    \end{align}

They form a complete basis for :math:`2\times2` Hermitian matrices and satisfy

.. math::
    \begin{align}
        \sigma_j \sigma_k = \delta_{jk} \sigma_0 + i \epsilon_{jkl} \sigma_l
    \end{align}

for :math:`j,k,l \in {1,2,3}`.

.. _Pauli matrices: https://en.wikipedia.org/wiki/Pauli_matrices
'''
