# two-dimensional-gasses

We study lattice-field-theory-style Monte Carlo calculations of attractive fermionic gasses in two dimensions.

We offer three pieces,

 - A LaTeX document describing our formalism, ideas, and algorithms,
 - A Mathematica package `TDG` that constructs the two-body A1-projected Hamiltonian and uses the Lüscher quantization condition to tune Hamiltonian parameters to the desired scattering data / effective range expansion.  This functionality is also available in
 - A [python package `tdg`][docs] which can be used for quantum many-body simulations.  `tdg` is built on [pytorch][torch].  Torch is leveraged for GPU calculations, automatic differentiation, and its machine-learning capabilities.

In addition to answering interesting questions about the physics of two-dimensional cold atom traps, we hope to provide a completely open end-to-end physics workflow for [FAIR][bennett] research.




 [torch]: https://pytorch.org/
 [bennett]: https://edbennett.github.io/lattice2022-survey-talk/
 [docs]: https://two-dimensional-gasses.readthedocs.io
